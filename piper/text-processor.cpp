
#include "utf8.h"
#include "text-processor.hpp"

#include <array>
#include <cctype>
#include <string_view>
#include <vector>
#include <algorithm>
#include <cstring>  
#include <piper-phonemize/uni_algo.h>

namespace piper {

std::string normalizeWhitespace(const std::string &text) {
  std::string result;
  result.reserve(text.size());
  bool pendingSpace = false;
  for (unsigned char ch : text) {
    if (std::isspace(ch)) {
      if (!result.empty()) pendingSpace = true;
    } else {
      if (pendingSpace) { result.push_back(' '); pendingSpace = false; }
      result.push_back(static_cast<char>(ch));
    }
  }
  return result;
}

std::string normalizeUnicode(const std::string &text) {
  return una::norm::to_nfc_utf8(text);
}

std::string normalizeRomanian(const std::string &text) {
  std::string result = text;
  const std::pair<std::string, std::string> replacements[] = 
    { {"ş", "ș"}, {"ţ", "ț"}, {"Ş", "Ș"}, {"Ţ", "Ț"} };

  for (const auto &[from, to] : replacements) {
    std::size_t pos = 0;
    while ((pos = result.find(from, pos)) != std::string::npos) {
      result.replace(pos, from.size(), to);
      pos += to.size();
    }
  }

  return result;
}

namespace {

bool isAsciiDigit(char c) {
  return c >= '0' && c <= '9';
}

bool isAsciiLetter(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

bool isRomanNumeral(char c) {
  return c == 'I' || c == 'V' || c == 'X' || c == 'L' || c == 'C';
}

bool startsWith(const std::string &text, size_t pos, std::string_view value) {
  return pos + value.size() <= text.size() && text.compare(pos, value.size(), value) == 0;
}

bool startsWithIcaseAscii(const std::string &text, size_t pos, std::string_view value) {
  if (pos + value.size() > text.size()) return false;
  for (size_t i = 0; i < value.size(); ++i) {
    unsigned char a = static_cast<unsigned char>(text[pos + i]);
    unsigned char b = static_cast<unsigned char>(value[i]);
    if (std::tolower(a) != std::tolower(b)) return false;
  }
  return true;
}

bool isBoundaryBefore(const std::string &text, size_t pos) {
  if (pos == 0) return true;
  unsigned char c = static_cast<unsigned char>(text[pos - 1]);
  return !std::isalnum(c) && c != '_';
}

bool isBoundaryAfter(const std::string &text, size_t pos) {
  if (pos >= text.size()) return true;
  unsigned char c = static_cast<unsigned char>(text[pos]);
  return !std::isalnum(c) && c != '_';
}

size_t skipSpaces(const std::string &text, size_t pos) {
  while (pos < text.size() && text[pos] == ' ') ++pos;
  return pos;
}

// The input has already been whitespace-normalized and NFC-normalized.
bool isRomanianUpperAt(const std::string &text, size_t pos) {
  if (pos >= text.size()) return false;
  unsigned char c = static_cast<unsigned char>(text[pos]);
  if (c >= 'A' && c <= 'Z') return true;
  static constexpr std::array<std::string_view, 5> upper = 
    { "Ă", "Â", "Î", "Ș", "Ț" };
  for (auto ch : upper) 
    if (startsWith(text, pos, ch)) return true;
  return false;
}

bool isRomanianLetterAt(const std::string &text, size_t pos) {
  if (pos >= text.size()) return false;
  unsigned char c = static_cast<unsigned char>(text[pos]);
  if (isAsciiLetter(static_cast<char>(c))) return true;
  static constexpr std::array<std::string_view, 10> letters = 
    { "ă", "â", "î", "ș", "ț", "Ă", "Â", "Î", "Ș", "Ț" };
  for (auto ch : letters)
    if (startsWith(text, pos, ch)) return true;
  return false;
}

void replaceRange(std::string &text, size_t pos, size_t len, std::string_view replacement) {
  text.replace(pos, len, replacement);
}

std::string carryInitialCase(std::string replacement, const std::string &matched) {
  if (!matched.empty() && matched[0] >= 'A' && matched[0] <= 'Z' && !replacement.empty())
    replacement[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(replacement[0])));
  return replacement;
}

bool shouldKeepSentencePeriod(const std::string &text, size_t after) {
  size_t pos = skipSpaces(text, after);
  if (pos >= text.size()) return true;
  return isRomanianUpperAt(text, pos);
}

int romanToInt(std::string_view roman) {
  auto value = [](char c) -> int {
    switch (c) {
      case 'I': return 1;
      case 'V': return 5;
      case 'X': return 10;
      case 'L': return 50;
      case 'C': return 100;
      default: return 0;
    }
  };
  int total = 0;  int prev = 0;
  for (auto it = roman.rbegin(); it != roman.rend(); ++it) {
    int v = value(*it);
    if (v == 0) return 0;
    total += v >= prev ? v : -v;
    prev = std::max(prev, v);
  }
  return total;
}

std::string ordinalCentury(int n) {
  static const std::array<const char *, 31> ordinal = {
    "",
    "întâi",
    "al doilea",
    "al treilea",
    "al patrulea",
    "al cincilea",
    "al șaselea",
    "al șaptelea",
    "al optulea",
    "al nouălea",
    "al zecelea",
    "al unsprezecelea",
    "al doisprezecelea",
    "al treisprezecelea",
    "al paisprezecelea",
    "al cincisprezecelea",
    "al șaisprezecelea",
    "al șaptesprezecelea",
    "al optsprezecelea",
    "al nouăsprezecelea",
    "al douăzecilea",
    "al douăzeci și unulea",
    "al douăzeci și doilea",
    "al douăzeci și treilea",
    "al douăzeci și patrulea",
    "al douăzeci și cincilea",
    "al douăzeci și șaselea",
    "al douăzeci și șaptelea",
    "al douăzeci și optulea",
    "al douăzeci și nouălea",
    "al treizecilea"
  };
  if (n < 1 || n > 30) return {};
  return ordinal[n];
}

std::size_t utf8Length(const std::string &text) {
  return static_cast<std::size_t>(utf8::distance(text.begin(), text.end()));
}

std::string trimSpaces(const std::string &text) {
  const auto first = text.find_first_not_of(' ');
  if (first == std::string::npos) return {};
  const auto last = text.find_last_not_of(' ');
  return text.substr(first, last - first + 1);
}

bool endsWith(const std::string &text, std::string_view value) {
  return text.size() >= value.size() &&
    text.compare(text.size() - value.size(), value.size(), value) == 0;
}

bool startsRomanianClauseWord(const std::string &text, std::size_t pos) {
  static constexpr std::array<std::string_view, 12> words = 
    { "și", "Și", "dar", "Dar", "iar", "Iar", "însă", "Însă", "care", "Care", "pentru că", "Pentru că" };
  for (auto word : words) {
    if (!startsWith(text, pos, word)) continue;
    const std::size_t end = pos + word.size();
    if (end >= text.size() || text[end] == ' ' ||
        text[end] == ',' || text[end] == ';' ||
        text[end] == ':' || text[end] == '.' ||
        text[end] == '?' || text[end] == '!') {
      return true;
    }
  }
  return false;
}

bool previousWordBlocksRomanianSplit(const std::string &text, std::size_t spacePos) {
  if (spacePos == 0) return false;
  std::size_t end = spacePos;
  while (end > 0 && text[end - 1] == ' ') --end;
  std::size_t begin = end;
  while (begin > 0) {
    const char c = text[begin - 1];
    if (c == ' ' || c == ',' || c == ';' || c == ':' || c == '.' || c == '?' || c == '!') break;
    --begin;
  }
  const std::string word = text.substr(begin, end - begin);
  return word == "dar" || word == "Dar" || word == "ci"  || word == "Ci"  || word == "nu"  || word == "Nu";
}

std::vector<std::string> splitSentencesRomanian(const std::string &text) {
  static constexpr std::array<std::string_view, 39> abbreviations = 
    { "d-le", "d-na", "d-lui", "d-nei", "d-lor", "nr", "str", "bd", "al", "et", "cf", "pag", "prof",
      "dr", "ing", "arh", "ec", "av", "sec", "min", "vol", "cap", "fig", "tab", "tel", "fax", "dept",
      "div", "sf", "pt", "ex", "approx", "cca", "aprox", "km", "cm", "mm", "mg", "kg" };

  auto isProtectedPeriod = [&](std::size_t periodPos) {
    for (auto abbr : abbreviations) {
      if (periodPos < abbr.size()) continue;
      const std::size_t start = periodPos - abbr.size();
      if (!isBoundaryBefore(text, start) || !startsWithIcaseAscii(text, start, abbr)) continue;
      return true;
    }

    // Special multi-dot forms from Eduard's list.
    if (periodPos >= 2) {
      const std::size_t start = periodPos - 2;
      if (startsWithIcaseAscii(text, start, "i.e") || startsWithIcaseAscii(text, start, "e.g")) return true;
    }
    return false;
  };

  std::vector<std::string> sentences;  std::size_t begin = 0;

  for (std::size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    if (c != '.' && c != '?' && c != '!') continue;
    if (c == '.' && isProtectedPeriod(i)) continue;

    const bool atEnd = i + 1 >= text.size();
    const bool followedBySpace = !atEnd && text[i + 1] == ' ';
    if (!atEnd && !followedBySpace) continue;

    std::string sentence = trimSpaces(text.substr(begin, i - begin + 1));
    if (!sentence.empty()) sentences.push_back(std::move(sentence));

    begin = i + 1;
    while (begin < text.size() && text[begin] == ' ') { ++begin; }
    i = begin == 0 ? 0 : begin - 1;
  }

  if (begin < text.size()) {
    std::string sentence = trimSpaces(text.substr(begin));
    if (!sentence.empty()) sentences.push_back(std::move(sentence));
  }

  return sentences;
}

std::vector<std::string> splitClauses(const std::string &sentence, const std::string &language) {
  std::vector<std::string> parts;
  std::size_t begin = 0;  std::size_t pos = 0;

  while (pos < sentence.size()) {
    if (sentence[pos] != ' ') { ++pos; continue; }

    bool boundary = false;
    // Boundary after punctuation: ", " "; " ": "
    if (pos > begin) {
      const char prev = sentence[pos - 1];
      if (prev == ',' || prev == ';' || prev == ':') boundary = true;
      // UTF-8 em dash: —
      if (pos >= 3 && sentence.compare(pos - 3, 3, "—") == 0) boundary = true;
    }

    // Romanian linguistic boundaries.
    if (!boundary && language == "ro") {
      const std::size_t next = pos + 1;
      if (next < sentence.size() &&
          startsRomanianClauseWord(sentence, next) &&
          !previousWordBlocksRomanianSplit(sentence, pos)) {
        boundary = true;
      }
    }
    if (boundary) {
      std::string part = trimSpaces(sentence.substr(begin, pos - begin));
      if (!part.empty()) parts.push_back(std::move(part));
      begin = pos + 1;
    }
    ++pos;
  }

  std::string tail = trimSpaces(sentence.substr(begin));
  if (!tail.empty()) parts.push_back(std::move(tail));
  return parts;
}

} // namespace

std::string expandAbbreviationsRomanian(const std::string &text) {
  std::string result = text;

  // --------------------------------------------------------------------------
  // Century: "sec. XVIII" / "secolul XVIII" -> "secolul al optsprezecelea"
  // --------------------------------------------------------------------------

  for (size_t pos = 0; pos < result.size();) {
    bool upper = false;  size_t headLen = 0;

    if (startsWith(result, pos, "sec. ")) {
      headLen = 5;
    } else if (startsWith(result, pos, "Sec. ")) {
      headLen = 5;  upper = true;
    } else if (startsWith(result, pos, "secolul ")) {
      headLen = 8;
    } else if (startsWith(result, pos, "Secolul ")) {
      headLen = 8; upper = true;
    }
    if (headLen == 0 || !isBoundaryBefore(result, pos)) { ++pos; continue; }

    size_t numberPos = pos + headLen;
    size_t numberEnd = numberPos;
    while (numberEnd < result.size() && (isAsciiDigit(result[numberEnd]) || 
      isRomanNumeral(result[numberEnd]))) ++numberEnd;
    if (numberEnd == numberPos || !isBoundaryAfter(result, numberEnd)) { ++pos; continue; }

    std::string token = result.substr(numberPos, numberEnd - numberPos);
    int n = 0;  bool digits = true;
    for (char c : token) if (!isAsciiDigit(c)) { digits = false; break; }
    if (digits) n = std::stoi(token); else n = romanToInt(token);

    std::string ordinal = ordinalCentury(n);
    if (ordinal.empty()) { ++pos; continue; }

    std::string replacement = upper ? "Secolul " : "secolul ";
    replacement += ordinal;

    replaceRange(result, pos, numberEnd - pos, replacement);
    pos += replacement.size();
  }

  // --------------------------------------------------------------------------
  // Special dotted forms
  // --------------------------------------------------------------------------

  struct SpecialRule {
    const char *written;
    const char *spoken;
  };

  static const std::array<SpecialRule, 6> specialRules = {{
    {"î.Hr.", "înainte de Hristos"},
    {"Î.Hr.", "înainte de Hristos"},
    {"d.Hr.", "după Hristos"},
    {"D.Hr.", "după Hristos"},
    {"S.A.", "societate pe acțiuni"},
    {"S.R.L.", "societate cu răspundere limitată"}
  }};

  for (const auto &rule : specialRules) {
    size_t pos = 0;
    while ((pos = result.find(rule.written, pos)) != std::string::npos) {
      if (!isBoundaryBefore(result, pos)) { pos += std::strlen(rule.written); continue; }
      size_t len = std::strlen(rule.written);
      bool keepPeriod = shouldKeepSentencePeriod(result, pos + len);
      std::string replacement = rule.spoken;
      if (keepPeriod) replacement += '.';
      replaceRange(result, pos, len, replacement);
      pos += replacement.size();
    }
  }

  // "un S.A." / "un S.R.L." -> feminine article after expansion.
  {
    size_t pos = 0;
    while ((pos = result.find("un societate pe acțiuni", pos)) != std::string::npos) 
      { replaceRange(result, pos, 2, "o"); pos += 2; }
    pos = 0;
    while ((pos = result.find("un societate cu răspundere limitată", pos)) != std::string::npos) 
      { replaceRange(result, pos, 2, "o"); pos += 2; }
  }

  // --------------------------------------------------------------------------
  // Context-sensitive abbreviations
  // --------------------------------------------------------------------------

  enum class Guard {
    Anywhere,
    UpperOrDigit,
    Digit,
    DigitOrUpperSingle,
    Upper,
    ProfTitle,
    DigitOrOpenParen,
    Letter,
    DigitOrRoman
  };

  struct Rule {
    const char *written;
    const char *spoken;
    Guard guard;
    bool keepSentencePeriod;
  };

  static const std::array<Rule, 24> rules = {{
    {"nr. crt.", "numărul curent", Guard::Anywhere, false},
    {"str.", "strada", Guard::UpperOrDigit, false},
    {"bd.", "bulevardul", Guard::UpperOrDigit, false},
    {"nr.", "numărul", Guard::Digit, false},
    {"et.", "etajul", Guard::Digit, false},
    {"ap.", "apartamentul", Guard::Digit, false},
    {"sc.", "scara", Guard::DigitOrUpperSingle, false},
    {"jud.", "județul", Guard::Upper, false},
    {"dl.", "domnul", Guard::Upper, false},
    {"dna.", "doamna", Guard::Upper, false},
    {"prof.", "profesor", Guard::ProfTitle, false},
    {"dr.", "doctor", Guard::Upper, false},
    {"ing.", "inginer", Guard::Upper, false},
    {"dvs.", "dumneavoastră", Guard::Anywhere, true},
    {"art.", "articolul", Guard::Digit, false},
    {"alin.", "alineatul", Guard::DigitOrOpenParen, false},
    {"lit.", "litera", Guard::Letter, false},
    {"vol.", "volumul", Guard::DigitOrRoman, false},
    {"cap.", "capitolul", Guard::DigitOrRoman, false},
    {"pag.", "pagina", Guard::Digit, false},
    {"cca.", "circa", Guard::Digit, false},
    {"ed.", "editura", Guard::Upper, false},
    // Keep room for compatible additions without changing the mechanism.
    {nullptr, nullptr, Guard::Anywhere, false},
    {nullptr, nullptr, Guard::Anywhere, false},
  }};

  for (const auto &rule : rules) {
    if (!rule.written) continue;
    size_t pos = 0;
    size_t writtenLen = std::strlen(rule.written);
    while (pos < result.size()) {
      if (!startsWithIcaseAscii(result, pos, rule.written) ||
        !isBoundaryBefore(result, pos)) { ++pos; continue; }
      size_t after = pos + writtenLen;
      size_t next = skipSpaces(result, after);
      bool valid = false;

      switch (rule.guard) {
        case Guard::Anywhere:
          valid = true;
          break;

        case Guard::UpperOrDigit:
          valid = next > after && next < result.size() &&
            (isRomanianUpperAt(result, next) || isAsciiDigit(result[next]));
          break;

        case Guard::Digit:
          valid = next < result.size() && isAsciiDigit(result[next]);
          break;

        case Guard::DigitOrUpperSingle:
          if (next < result.size() && isAsciiDigit(result[next])) {
            valid = true;
          } else if (next < result.size() && isRomanianUpperAt(result, next)) {
            size_t p = next;
            while (p < result.size() && static_cast<unsigned char>(result[p]) >= 0x80) ++p;
            if (p == next) ++p;
            valid = isBoundaryAfter(result, p);
          }
          break;

        case Guard::Upper:
          valid = next > after && next < result.size() && isRomanianUpperAt(result, next);
          break;

        case Guard::ProfTitle:
          valid = next > after && (isRomanianUpperAt(result, next) || 
            startsWithIcaseAscii(result, next, "dr.") || startsWithIcaseAscii(result, next, "univ."));
          break;

        case Guard::DigitOrOpenParen:
          valid = next < result.size() && (isAsciiDigit(result[next]) || result[next] == '(');
          break;

        case Guard::Letter:
          valid = next > after && next < result.size() && isRomanianLetterAt(result, next);
          break;

        case Guard::DigitOrRoman:
          valid = next < result.size() && (isAsciiDigit(result[next]) || isRomanNumeral(result[next]));
          break;
      }
      if (!valid) { ++pos; continue; }

      std::string matched = result.substr(pos, writtenLen);
      std::string replacement = carryInitialCase(rule.spoken, matched);
      if (rule.keepSentencePeriod && shouldKeepSentencePeriod(result, after)) replacement += '.';
      replaceRange(result, pos, writtenLen, replacement);
      pos += replacement.size();
    }
  }

  // --------------------------------------------------------------------------
  // Acronyms that eSpeak would otherwise spell letter-by-letter
  // --------------------------------------------------------------------------

  struct Acronym {
    const char *written;
    const char *spoken;
  };

  static const std::array<Acronym, 3> acronyms = {{
    {"ONU", "Onu"},
    {"SUA", "Sua"},
    {"PIB", "Pib"},
  }};

  for (const auto &rule : acronyms) {
    size_t pos = 0;  size_t len = std::strlen(rule.written);
    while ((pos = result.find(rule.written, pos)) != std::string::npos) {
      if (!isBoundaryBefore(result, pos)) { pos += len; continue; }
      size_t after = pos + len;

      // PIB-ul -> Pibul
      if (std::strcmp(rule.written, "PIB") == 0 &&
          after < result.size() && result[after] == '-' &&
          after + 1 < result.size()) {
        replaceRange(result, pos, len + 1, rule.spoken);
        pos += std::strlen(rule.spoken);
        continue;
      }

      if (!isBoundaryAfter(result, after)) { pos += len; continue; }
      replaceRange(result, pos, len, rule.spoken);
      pos += std::strlen(rule.spoken);
    }
  }

  // --------------------------------------------------------------------------
  // Surface units: 1 mp / 65 mp / 20 ha
  // --------------------------------------------------------------------------

  for (size_t pos = 0; pos < result.size();) {
    if (!isAsciiDigit(result[pos]) || (pos > 0 && isAsciiDigit(result[pos - 1]))) { ++pos; continue; }

    size_t numberEnd = pos;
    while (numberEnd < result.size() && isAsciiDigit(result[numberEnd])) ++numberEnd;
    int n = std::stoi(result.substr(pos, numberEnd - pos));
    size_t unitPos = skipSpaces(result, numberEnd);

    std::string singular;  std::string plural;  size_t unitLen = 0;
    if (startsWith(result, unitPos, "mp") && isBoundaryAfter(result, unitPos + 2)) {
      singular = "metru pătrat";
      plural = "metri pătrați";
      unitLen = 2;
    } else if (startsWith(result, unitPos, "ha") && isBoundaryAfter(result, unitPos + 2)) {
      singular = "hectar";
      plural = "hectare";
      unitLen = 2;
    }
    if (unitLen == 0) { pos = numberEnd; continue; }

    std::string replacement = std::to_string(n);
    if (n == 1) {
      replacement += " " + singular;
    } else {
      if (!(n % 100 >= 1 && n % 100 <= 19)) replacement += " de";
      replacement += " " + plural;
    }
    replaceRange(result, pos, unitPos + unitLen - pos, replacement);
    pos += replacement.size();
  }

  return result;
}

std::vector<TextChunk> chunkText(const std::string &text, std::size_t maxChars, const std::string &language) {
  if (text.empty()) return {};
  if (maxChars == 0) return {{text, true}};

  std::vector<std::string> sentences;
  if (language == "ro") {
    sentences = splitSentencesRomanian(text);
  } else {
    // Generic fallback: same sentence splitter is safe after language-specific
    // abbreviation expansion is disabled, but linguistic clause words are not used.
    sentences = splitSentencesRomanian(text);
  }

  std::vector<TextChunk> chunks;

  for (const auto &sentence : sentences) {
    if (utf8Length(sentence) <= maxChars) { chunks.push_back({sentence, true}); continue; }

    auto parts = splitClauses(sentence, language);
    std::vector<std::string> sentenceChunks;  std::string buffer;

    for (const auto &part : parts) {
      std::string candidate = buffer.empty() ? part : buffer + " " + part;
      if (!buffer.empty() && utf8Length(candidate) > maxChars) {
        sentenceChunks.push_back(std::move(buffer));
        buffer = part;
      } else {
        buffer = std::move(candidate);
      }
    }

    if (!buffer.empty()) sentenceChunks.push_back(std::move(buffer));

    // If there was nowhere sensible to split, keep the sentence whole.
    if (sentenceChunks.empty()) sentenceChunks.push_back(sentence);

    for (std::size_t i = 0; i < sentenceChunks.size(); ++i)
      chunks.push_back({ std::move(sentenceChunks[i]), i + 1 == sentenceChunks.size() });
  }

  return chunks;
}

} // namespace piper
