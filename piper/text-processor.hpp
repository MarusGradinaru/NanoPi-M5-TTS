#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace piper {

struct TextChunk {
  std::string text;
  bool sentenceEnd = true;
};

std::string normalizeWhitespace(const std::string &text);
std::string normalizeUnicode(const std::string &text);
std::string normalizeRomanian(const std::string &text);
std::string expandAbbreviationsRomanian(const std::string &text);

std::vector<TextChunk> chunkText(const std::string &text,
  std::size_t maxChars, const std::string &language);

} // namespace piper
