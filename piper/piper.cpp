#include <array>
#include <chrono>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <espeak-ng/speak_lib.h>
#include <onnxruntime_cxx_api.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include "piper.hpp"
#include "text-processor.hpp"
#include "utf8.h"
#include "wavfile.hpp"

#ifdef USE_RKNN
#include "rknn-inferer.hpp"
#endif

#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

namespace piper {

#ifdef _PIPER_VERSION
// https://stackoverflow.com/questions/47346133/how-to-use-a-define-inside-a-format-string
#define _STR(x) #x
#define STR(x) _STR(x)
const std::string VERSION = STR(_PIPER_VERSION);
#else
const std::string VERSION = "";
#endif

// Maximum value for 16-bit signed WAV sample
const float MAX_WAV_VALUE = 32767.0f;

const std::string instanceName{"piper"};

std::string getVersion() { return VERSION; }

// True if the string is a single UTF-8 codepoint
bool isSingleCodepoint(std::string s) {
  return utf8::distance(s.begin(), s.end()) == 1;
}

// Get the first UTF-8 codepoint of a string
Phoneme getCodepoint(std::string s) {
  utf8::iterator character_iter(s.begin(), s.begin(), s.end());
  return *character_iter;
}

// Load JSON config information for phonemization
void parsePhonemizeConfig(json &configRoot, PhonemizeConfig &phonemizeConfig) {
  // {
  //     "espeak": {
  //         "voice": "<language code>"
  //     },
  //     "phoneme_type": "<espeak or text>",
  //     "phoneme_map": {
  //         "<from phoneme>": ["<to phoneme 1>", "<to phoneme 2>", ...]
  //     },
  //     "phoneme_id_map": {
  //         "<phoneme>": [<id1>, <id2>, ...]
  //     }
  // }

  if (configRoot.contains("espeak")) {
    auto espeakValue = configRoot["espeak"];
    if (espeakValue.contains("voice")) {
      phonemizeConfig.eSpeak.voice = espeakValue["voice"].get<std::string>();
    }
  }

  if (configRoot.contains("phoneme_type")) {
    auto phonemeTypeStr = configRoot["phoneme_type"].get<std::string>();
    if (phonemeTypeStr == "text") {
      phonemizeConfig.phonemeType = TextPhonemes;
    }
  }

  // phoneme to [id] map
  // Maps phonemes to one or more phoneme ids (required).
  if (configRoot.contains("phoneme_id_map")) {
    auto phonemeIdMapValue = configRoot["phoneme_id_map"];
    for (auto &fromPhonemeItem : phonemeIdMapValue.items()) {
      std::string fromPhoneme = fromPhonemeItem.key();
      if (!isSingleCodepoint(fromPhoneme)) {
        std::stringstream idsStr;
        for (auto &toIdValue : fromPhonemeItem.value()) {
          PhonemeId toId = toIdValue.get<PhonemeId>();
          idsStr << toId << ",";
        }

        spdlog::error("\"{}\" is not a single codepoint (ids={})", fromPhoneme,
                      idsStr.str());
        throw std::runtime_error(
            "Phonemes must be one codepoint (phoneme id map)");
      }

      auto fromCodepoint = getCodepoint(fromPhoneme);
      for (auto &toIdValue : fromPhonemeItem.value()) {
        PhonemeId toId = toIdValue.get<PhonemeId>();
        phonemizeConfig.phonemeIdMap[fromCodepoint].push_back(toId);
      }
    }
  }

  // phoneme to [phoneme] map
  // Maps phonemes to one or more other phonemes (not normally used).
  if (configRoot.contains("phoneme_map")) {
    if (!phonemizeConfig.phonemeMap) {
      phonemizeConfig.phonemeMap.emplace();
    }

    auto phonemeMapValue = configRoot["phoneme_map"];
    for (auto &fromPhonemeItem : phonemeMapValue.items()) {
      std::string fromPhoneme = fromPhonemeItem.key();
      if (!isSingleCodepoint(fromPhoneme)) {
        spdlog::error("\"{}\" is not a single codepoint", fromPhoneme);
        throw std::runtime_error(
            "Phonemes must be one codepoint (phoneme map)");
      }

      auto fromCodepoint = getCodepoint(fromPhoneme);
      for (auto &toPhonemeValue : fromPhonemeItem.value()) {
        std::string toPhoneme = toPhonemeValue.get<std::string>();
        if (!isSingleCodepoint(toPhoneme)) {
          throw std::runtime_error(
              "Phonemes must be one codepoint (phoneme map)");
        }

        auto toCodepoint = getCodepoint(toPhoneme);
        (*phonemizeConfig.phonemeMap)[fromCodepoint].push_back(toCodepoint);
      }
    }
  }

} /* parsePhonemizeConfig */

// Load JSON config for audio synthesis
void parseSynthesisConfig(json &configRoot, SynthesisConfig &synthesisConfig) {
  // {
  //     "audio": {
  //         "sample_rate": 22050
  //     },
  //     "inference": {
  //         "noise_scale": 0.667,
  //         "length_scale": 1,
  //         "noise_w": 0.8,
  //         "phoneme_silence": {
  //           "<phoneme>": <seconds of silence>,
  //           ...
  //         }
  //     }
  // }

  if (configRoot.contains("audio")) {
    auto audioValue = configRoot["audio"];
    if (audioValue.contains("sample_rate")) {
      // Default sample rate is 22050 Hz
      synthesisConfig.sampleRate = audioValue.value("sample_rate", 22050);
    }
  }

  if (configRoot.contains("inference")) {
    // Overrides default inference settings
    auto inferenceValue = configRoot["inference"];
    if (inferenceValue.contains("noise_scale")) {
      synthesisConfig.noiseScale = inferenceValue.value("noise_scale", 0.667f);
    }

    if (inferenceValue.contains("length_scale")) {
      synthesisConfig.lengthScale = inferenceValue.value("length_scale", 1.0f);
    }

    if (inferenceValue.contains("noise_w")) {
      synthesisConfig.noiseW = inferenceValue.value("noise_w", 0.8f);
    }

    if (inferenceValue.contains("phoneme_silence")) {
      // phoneme -> seconds of silence to add after
      synthesisConfig.phonemeSilenceSeconds.emplace();
      auto phonemeSilenceValue = inferenceValue["phoneme_silence"];
      for (auto &phonemeItem : phonemeSilenceValue.items()) {
        std::string phonemeStr = phonemeItem.key();
        if (!isSingleCodepoint(phonemeStr)) {
          spdlog::error("\"{}\" is not a single codepoint", phonemeStr);
          throw std::runtime_error(
              "Phonemes must be one codepoint (phoneme silence)");
        }

        auto phoneme = getCodepoint(phonemeStr);
        (*synthesisConfig.phonemeSilenceSeconds)[phoneme] =
            phonemeItem.value().get<float>();
      }

    } // if phoneme_silence

  } // if inference

} /* parseSynthesisConfig */

void parseModelConfig(json &configRoot, ModelConfig &modelConfig) {
  modelConfig.numSpeakers = configRoot["num_speakers"].get<SpeakerId>();
  if (configRoot.contains("speaker_id_map")) {
    if (!modelConfig.speakerIdMap) modelConfig.speakerIdMap.emplace();
    auto speakerIdMapValue = configRoot["speaker_id_map"];
    for (auto &speakerItem : speakerIdMapValue.items()) {
      std::string speakerName = speakerItem.key();
      (*modelConfig.speakerIdMap)[speakerName] = speakerItem.value().get<SpeakerId>();
    }
  }
} /* parseModelConfig */

void parseTextConfig(json &configRoot, TextConfig &textConfig) {
  if (!configRoot.contains("text")) return;
  auto textValue = configRoot["text"];
  textConfig.language = textValue.value("language", std::string());
  textConfig.normalizeWhitespace = textValue.value("normalize_whitespace", false);
  textConfig.normalizeUnicode = textValue.value("normalize_unicode", false);
  textConfig.expandAbbreviations = textValue.value("expand_abbreviations", false);
  textConfig.maxChunkChars = textValue.value("max_chunk_chars", static_cast<std::size_t>(0));
}

void initialize(PiperConfig &config) {
  if (config.useESpeak) {
    // Set up espeak-ng for calling espeak_TextToPhonemesWithTerminator
    // See: https://github.com/rhasspy/espeak-ng
    spdlog::debug("Initializing eSpeak with data path: {}", config.eSpeakDataPath);
    int result = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, /*buflength*/ 0,
      /*path*/ config.eSpeakDataPath.c_str(), /*options*/ 0);
    if (result < 0) throw std::runtime_error("Failed to initialize eSpeak-ng");
    spdlog::debug("Initialized eSpeak");
  }

  spdlog::info(" Initialized piper");
}

void terminate(PiperConfig &config) {
  if (config.useESpeak) {
    // Clean up espeak-ng
    espeak_Terminate();
    spdlog::debug("Terminated eSpeak");
  }
  spdlog::info(" Terminated piper");
}

// Load ONNX/RKNN voice models from voice pack
bool loadVoice(PiperConfig &config, const std::filesystem::path &voicePack, Voice &voice, 
  const std::optional<SpeakerId> &speakerId, const std::string &accelerator) 
{                
  auto modelConfigPath = voicePack / "config.json";
  auto encoderPath     = voicePack / "encoder.onnx";
  auto cpuDecoderPath  = voicePack / "decoder.onnx";
  auto rknnDecoderPath = voicePack / "decoder.rknn";
  auto espeakDataPath  = voicePack / "espeak-ng-data";  
  
  std::error_code ec;
  spdlog::debug("Parsing model voice pack at {}", voicePack.string());
  bool pk_config   = std::filesystem::is_regular_file(modelConfigPath, ec);
  if (pk_config)   spdlog::debug(" - model config : available"); else spdlog::debug(" - model config : not found");
  bool pk_cpu_enc  = std::filesystem::is_regular_file(encoderPath, ec);
  if (pk_cpu_enc)  spdlog::debug(" - onnx encoder : available"); else spdlog::debug(" - onnx encoder : not found"); 
  bool pk_cpu_dec  = std::filesystem::is_regular_file(cpuDecoderPath, ec);
  if (pk_cpu_dec)  spdlog::debug(" - onnx decoder : available"); else spdlog::debug(" - onnx decoder : not found"); 
  #ifdef USE_RKNN  
    bool pk_rknn_dec = std::filesystem::is_regular_file(rknnDecoderPath, ec);
    if (!config.rknnEnabled) spdlog::debug(" - rknn decoder : disabled");
      else if (pk_rknn_dec) spdlog::debug(" - rknn decoder : available"); 
        else spdlog::debug(" - rknn decoder : not found"); 
  #endif
  bool pk_espeak   = std::filesystem::is_directory(espeakDataPath, ec);
  if (pk_espeak)   spdlog::debug(" - eSpeak data  : available"); else spdlog::debug(" - eSpeak data  : not found"); 
  if (!(pk_config && pk_cpu_enc && pk_cpu_dec)) { spdlog::debug("Error: A required module is missing !"); return false; }

  std::ifstream modelConfigFile(modelConfigPath);
  voice.configRoot = json::parse(modelConfigFile);
  parsePhonemizeConfig(voice.configRoot, voice.phonemizeConfig);
  parseTextConfig(voice.configRoot, voice.textConfig);
  parseSynthesisConfig(voice.configRoot, voice.synthesisConfig);
  parseModelConfig(voice.configRoot, voice.modelConfig);
  
  if (voice.phonemizeConfig.phonemeType == eSpeakPhonemes && pk_config && pk_espeak) 
    config.eSpeakDataPath = espeakDataPath.string();

  if (voice.modelConfig.numSpeakers > 1) {
    // Multi-speaker model
    if (speakerId) voice.synthesisConfig.speakerId = speakerId;
      else voice.synthesisConfig.speakerId = 0;  // Default speaker
  }
  spdlog::debug("Voice contains {} speaker(s)", voice.modelConfig.numSpeakers);

  voice.encoder.load(encoderPath, accelerator);
  voice.cpuDecoder = std::make_unique<OnnxDecoderInferer>();
  voice.cpuDecoder->load(cpuDecoderPath, accelerator);
  #ifdef USE_RKNN
    if (config.rknnEnabled && pk_rknn_dec) {
      voice.rknnDecoder = std::make_unique<RknnDecoderInferer>();
      voice.rknnDecoder->load(rknnDecoderPath, accelerator);
    }
  #endif
  return true;
} /* loadVoice */

void OnnxDecoderInferer::load(std::string path, std::string accelerator)
{
  spdlog::debug("Loading decoder onnx model from {}", path);
  env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
  env.DisableTelemetryEvents();

  if (accelerator == "cuda") {
    // Use CUDA provider
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
    options.AppendExecutionProvider_CUDA(cuda_options);
  }
  else if (accelerator == "tensorrt") {
    // Use TensorRT provider
    OrtTensorRTProviderOptions tensorrt_options{};
    options.AppendExecutionProvider_TensorRT(tensorrt_options);
  }

  //options.DisableCpuMemArena();
  //options.DisableMemPattern();
  onnx = Ort::Session(env, path.c_str(), options);
}

std::vector<int16_t> OnnxDecoderInferer::infer(const xt::xarray<float>& z, 
  const xt::xarray<float>& y_mask, const std::optional<xt::xarray<float>>& g)
{
  auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  std::vector<Ort::Value> inputTensors;
  const std::array<std::string, 3> paramNames = {"z", "y_mask", "g"};
  for(auto& name : paramNames) {
    const xt::xarray<float>* ptr = nullptr;
    if(name == "z") ptr = &z;
    else if(name == "y_mask") ptr = &y_mask;
    else if(name == "g") {
      if(!g.has_value()) continue;
      ptr = &g.value();
    }
    else throw std::runtime_error("Invalid parameter name");
    auto& arr = *ptr;
    std::vector<int64_t> shape(arr.shape().begin(), arr.shape().end());
    inputTensors.push_back(
      Ort::Value::CreateTensor<float>(memoryInfo, (float*)arr.data(), arr.size(), shape.data(), shape.size())
    );
  }

  std::vector<const char*> inputNames = {"z", "y_mask"};
  if(g.has_value()) inputNames.push_back("g");
  std::array<const char *, 1> outputNames = {"output"};

  //auto startTime = std::chrono::steady_clock::now();
  auto outputTensors = onnx.Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(),
    inputTensors.size(), outputNames.data(), outputNames.size());
  //auto endTime = std::chrono::steady_clock::now();

  if ((outputTensors.size() != 1) || (!outputTensors.front().IsTensor()))
    throw std::runtime_error("Invalid output tensors");
  std::vector<int16_t> output;
  output.resize(outputTensors.front().GetTensorTypeAndShapeInfo().GetElementCount());
  auto ortOutPtr = outputTensors.front().GetTensorData<float>();
  for(size_t i = 0; i < output.size(); i++) {
    float val = std::min(std::max(ortOutPtr[i], -1.0f), 1.0f);
    output[i] = val * MAX_WAV_VALUE;
  }
  //spdlog::debug("Decoder inference took {} seconds", std::chrono::duration<double>(endTime - startTime).count());
  return output;
}

void EncoderInferer::load(std::string path, std::string accelerator)
{
  spdlog::debug("Loading encoder onnx model from {}", path);
  env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
  env.DisableTelemetryEvents();

  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  options.DisableProfiling();

#if 0
  // CUDA is slower than the CPU at running the encoder
  if (accelerator == "cuda") {
    // Use CUDA provider
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
    options.AppendExecutionProvider_CUDA(cuda_options);
  }
#endif
  
  //options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);   // Makes encoder slower
  onnx = Ort::Session(env, path.c_str(), options);
}

std::map<std::string, xt::xarray<float>> EncoderInferer::infer(const std::vector<int64_t> &phonemeIds,
  int64_t inputLength, std::optional<int64_t> sid, float noiseScale, float lengthScale, float noiseW)
{
  auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  // Allocate
  std::vector<int64_t> phonemeIdLengths{(int64_t)phonemeIds.size()};
  std::vector<float> scales{ noiseScale, lengthScale, noiseW };

  std::vector<Ort::Value> inputTensors;
  std::vector<int64_t> phonemeIdsShape{1, (int64_t)phonemeIds.size()};
  inputTensors.push_back(
    Ort::Value::CreateTensor<int64_t>(memoryInfo, (int64_t*)phonemeIds.data(), 
      phonemeIds.size(), phonemeIdsShape.data(), phonemeIdsShape.size())
  );
  std::vector<int64_t> phomemeIdLengthsShape{(int64_t)phonemeIdLengths.size()};
  inputTensors.push_back(
    Ort::Value::CreateTensor<int64_t>(memoryInfo, phonemeIdLengths.data(), 
      phonemeIdLengths.size(), phomemeIdLengthsShape.data(), phomemeIdLengthsShape.size())
  );
  std::vector<int64_t> scalesShape{(int64_t)scales.size()};
  inputTensors.push_back(
    Ort::Value::CreateTensor<float>(memoryInfo, scales.data(), scales.size(), scalesShape.data(), scalesShape.size())
  );

  // Add speaker id.
  // NOTE: These must be kept outside the "if" below to avoid being deallocated.
  std::vector<int64_t> speakerId{sid.value_or(0)};
  std::vector<int64_t> speakerIdShape{(int64_t)speakerId.size()};

  if (sid.has_value())
    inputTensors.push_back(
      Ort::Value::CreateTensor<int64_t>(memoryInfo, speakerId.data(), speakerId.size(),
        speakerIdShape.data(), speakerIdShape.size())
    );

  // From export_onnx.py
  std::array<const char *, 4> inputNames = {"input", "input_lengths", "scales", "sid"};

  std::vector<std::string> outputNames;
  for (size_t i=0;i<onnx.GetOutputCount();i++)
    outputNames.push_back(onnx.GetOutputNameAllocated(i, allocator).get());
  // TODO: Just use all outputs
  std::vector<const char*> outputNamePtrs;
  for(size_t i=0;i<outputNames.size();i++)
    outputNamePtrs.push_back(outputNames[i].c_str());

  // Infer
  //auto startTime = std::chrono::steady_clock::now();
  auto outputTensors = onnx.Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(),
    inputTensors.size(), outputNamePtrs.data(), outputNamePtrs.size());
  //auto endTime = std::chrono::steady_clock::now();

  if(outputTensors.size() != outputNames.size())
    throw std::runtime_error("Number of output tensors does not match number of output names");

  std::map<std::string, xt::xarray<float>> output;
  for(int i = 0; i < outputTensors.size(); i++) {
    if(!outputTensors[i].IsTensor()) throw std::runtime_error("Output tensor is not a tensor");
    xt::xarray<float> arr = xt::adapt(outputTensors[i].GetTensorMutableData<float>(),
      outputTensors[i].GetTensorTypeAndShapeInfo().GetShape());
    output[outputNames[i]] = std::move(arr);
  }

  //auto inferDuration = std::chrono::duration<double>(endTime - startTime);
  //auto inferSeconds = inferDuration.count();

  // clean up
  for (std::size_t i = 0; i < outputTensors.size(); i++)
    Ort::detail::OrtRelease(outputTensors[i].release());
  for (std::size_t i = 0; i < inputTensors.size(); i++)
    Ort::detail::OrtRelease(inputTensors[i].release());

  //spdlog::debug("Encoder inference took {} seconds", inferSeconds);
  return output;
}

// ----------------------------------------------------------------------------

// Phase 1: Phonemize text into phoneme IDs, split into sentences and phrases
PhonemeData phonemize(PiperConfig &config, Voice &voice, std::string text) {
  PhonemeData phonemeData;

  if (voice.textConfig.language == "ro") text = normalizeRomanian(text);
  if (voice.textConfig.normalizeUnicode) text = normalizeUnicode(text);
  if (voice.textConfig.normalizeWhitespace) text = normalizeWhitespace(text);
  if (voice.textConfig.language == "ro" && voice.textConfig.expandAbbreviations)
    text = expandAbbreviationsRomanian(text);

  auto textChunks = chunkText(text, voice.textConfig.maxChunkChars, voice.textConfig.language);

  // Phonemes for each sentence
  spdlog::info(" Phonemizing text...");
  std::vector<std::vector<Phoneme>> phonemes;
  std::vector<bool> sentenceSilence;

  if (voice.phonemizeConfig.phonemeType == eSpeakPhonemes) {
    static std::mutex espeakMutex;
    std::lock_guard<std::mutex> lock(espeakMutex);

    eSpeakPhonemeConfig eSpeakConfig;
    eSpeakConfig.voice = voice.phonemizeConfig.eSpeak.voice;

    for (std::size_t i = 0; i < textChunks.size(); ++i) {
      const auto &chunk = textChunks[i];
      spdlog::debug(" - {}/{} text chunk: {} chars, sentence end = {}, \"{}\"",
        i + 1, textChunks.size(), utf8::distance(chunk.text.begin(), chunk.text.end()),
        chunk.sentenceEnd ? "YES" : "NO", chunk.text);

      std::vector<std::vector<Phoneme>> chunkPhonemes;
      phonemize_eSpeak(chunk.text, eSpeakConfig, chunkPhonemes);

      for (std::size_t j = 0; j < chunkPhonemes.size(); ++j) {
        phonemes.push_back(std::move(chunkPhonemes[j]));
        const bool lastEspeakSentence = j + 1 == chunkPhonemes.size();
        sentenceSilence.push_back(lastEspeakSentence && chunk.sentenceEnd);
      }
    }
  } else {
    for (const auto &chunk : textChunks) {
      std::vector<std::vector<Phoneme>> chunkPhonemes;
      CodepointsPhonemeConfig codepointsConfig;

      phonemize_codepoints(chunk.text, codepointsConfig, chunkPhonemes);

      for (std::size_t j = 0; j < chunkPhonemes.size(); ++j) {
        phonemes.push_back(std::move(chunkPhonemes[j]));
        const bool lastSentence = j + 1 == chunkPhonemes.size();
        sentenceSilence.push_back(lastSentence && chunk.sentenceEnd);
      }
    }
  }

  // Convert each sentence's phonemes to IDs
  std::map<Phoneme, std::size_t> missingPhonemes;
  PhonemeIdConfig idConfig;
  idConfig.phonemeIdMap = std::make_shared<PhonemeIdMap>(voice.phonemizeConfig.phonemeIdMap);

  spdlog::info(" Converting phonemes to IDs...");
  for (std::size_t sentenceIdx = 0; sentenceIdx < phonemes.size(); ++sentenceIdx) {
    auto &sentencePhonemes = phonemes[sentenceIdx];
    PhonemeSentence sentence;
    sentence.addSilence = sentenceIdx < sentenceSilence.size() ? sentenceSilence[sentenceIdx] : true;

    if (spdlog::should_log(spdlog::level::debug)) {
      std::string phonemesStr;
      for (auto phoneme : sentencePhonemes) utf8::append(phoneme, std::back_inserter(phonemesStr));
      spdlog::debug(" - {}/{}: {}", sentenceIdx+1, phonemes.size(), phonemesStr);
    }

    // Split sentence into phrases at silence boundaries
    std::vector<std::shared_ptr<std::vector<Phoneme>>> phrasePhonemes;
    std::vector<float> phraseSilenceSeconds;

    if (voice.synthesisConfig.phonemeSilenceSeconds) {
      std::map<Phoneme, float> &phonemeSilenceSecondsMap = *voice.synthesisConfig.phonemeSilenceSeconds;

      auto currentPhrasePhonemes = std::make_shared<std::vector<Phoneme>>();
      phrasePhonemes.push_back(currentPhrasePhonemes);

      for (auto sentencePhonemesIter = sentencePhonemes.begin();
          sentencePhonemesIter != sentencePhonemes.end();
          sentencePhonemesIter++) {
        Phoneme &currentPhoneme = *sentencePhonemesIter;
        currentPhrasePhonemes->push_back(currentPhoneme);
        if (phonemeSilenceSecondsMap.count(currentPhoneme) > 0) {
          phraseSilenceSeconds.push_back(phonemeSilenceSecondsMap[currentPhoneme]);
          currentPhrasePhonemes = std::make_shared<std::vector<Phoneme>>();
          phrasePhonemes.push_back(currentPhrasePhonemes);
        }
      }
    } else {
      // Use all phonemes as a single phrase
      phrasePhonemes.push_back(std::make_shared<std::vector<Phoneme>>(sentencePhonemes));
    }

    // Pad silence to match phrase count
    while (phraseSilenceSeconds.size() < phrasePhonemes.size()) phraseSilenceSeconds.push_back(0);

    // Convert each phrase's phonemes to IDs
    for (size_t phraseIdx = 0; phraseIdx < phrasePhonemes.size(); phraseIdx++) {
      if (phrasePhonemes[phraseIdx]->size() <= 0) continue;

      PhonemePhrase phrase;
      phrase.silenceSeconds = phraseSilenceSeconds[phraseIdx];

      phonemes_to_ids(*(phrasePhonemes[phraseIdx]), idConfig, phrase.phonemeIds, missingPhonemes);

      //if (spdlog::should_log(spdlog::level::debug)) {
      //  std::stringstream phonemeIdsStr;
      //  for (auto phonemeId : phrase.phonemeIds) phonemeIdsStr << phonemeId << ", ";
      //  spdlog::debug("Converted {} phoneme(s) to {} phoneme id(s): {}",
      //    phrasePhonemes[phraseIdx]->size(), phrase.phonemeIds.size(), phonemeIdsStr.str());
      //}
      sentence.phrases.push_back(std::move(phrase));
    }

    phonemeData.sentences.push_back(std::move(sentence));
  }

  if (missingPhonemes.size() > 0) {
    spdlog::warn("Missing {} phoneme(s) from phoneme/id map!", missingPhonemes.size());
    for (auto phonemeCount : missingPhonemes) {
      std::string phonemeStr;
      utf8::append(phonemeCount.first, std::back_inserter(phonemeStr));
      spdlog::warn("Missing \"{}\" (\\u{:04X}): {} time(s)",
        phonemeStr, (uint32_t)phonemeCount.first, phonemeCount.second);
    }
  }

  return phonemeData;
} /* phonemize */

// Phase 2: Synthesize audio from pre-phonemized data
void synthesize(Voice &voice, const PhonemeData &phonemeData, std::vector<int16_t> &audioBuffer, 
  SynthesisResult &result, const std::function<void()> &audioCallback, std::optional<size_t> speakerId,
  std::optional<float> noiseScale, std::optional<float> lengthScale, std::optional<float> noiseW)
{
  std::size_t sentenceSilenceSamples = 0;
  if (voice.synthesisConfig.sentenceSilenceSeconds > 0) {
    sentenceSilenceSamples = (std::size_t)(
      voice.synthesisConfig.sentenceSilenceSeconds *
      voice.synthesisConfig.sampleRate * voice.synthesisConfig.channels);
  }

  spdlog::info(" Synthesizing voice...");
  for (auto &sentence : phonemeData.sentences) {
    for (size_t phraseIdx = 0; phraseIdx < sentence.phrases.size(); phraseIdx++) {
      auto &phrase = sentence.phrases[phraseIdx];

      // ------- Encoder inference -------

      spdlog::debug(" - encoding using ONNX; {} phonemes", phrase.phonemeIds.size());
      auto encode_start = std::chrono::steady_clock::now();
      std::optional<size_t> sid = speakerId;
      if(!sid && voice.synthesisConfig.speakerId)
        sid = voice.synthesisConfig.speakerId;
      auto params = voice.encoder.infer(phrase.phonemeIds, phrase.phonemeIds.size(), sid,
        noiseScale.value_or(voice.synthesisConfig.noiseScale),
        lengthScale.value_or(voice.synthesisConfig.lengthScale),
        noiseW.value_or(voice.synthesisConfig.noiseW));
      auto encode_end = std::chrono::steady_clock::now();
      float encode_seconds = std::chrono::duration<double>(encode_end - encode_start).count();
      std::optional<xt::xarray<float>> g;
      if(params.count("g")) g = std::move(params["g"]);
      auto& y_mask = params["y_mask"];
      auto& z = params["z"];
      size_t nslices = z.shape()[2];

      if(nslices != y_mask.shape()[2])
        throw std::runtime_error("z and y_mask must have the same number of slices");

      const size_t chunkSize = 45;
      const size_t padding = 5;

      const size_t audioStart = audioBuffer.size();
      float inferSeconds = encode_seconds;

      // ------- Decoder inference ------- 

      // Short phrases always use ONNX.
      // If no RKNN decoder is loaded, ONNX handles phrases of any size.
      if (nslices < chunkSize * 2 || !voice.rknnDecoder) {
        spdlog::debug(" - decoding using ONNX; {} slices", nslices);
        auto t0 = std::chrono::steady_clock::now();
        auto cpu_audio = voice.cpuDecoder->infer(z, y_mask, g);
        auto t1 = std::chrono::steady_clock::now();
        if (!cpu_audio.empty()) audioBuffer.insert(audioBuffer.end(), cpu_audio.begin(), cpu_audio.end());
        inferSeconds += std::chrono::duration<double>(t1 - t0).count();

      } else {  // Using RKNN decoder
        spdlog::debug(" - decoding using RKNN; {} slices", nslices);
        for(size_t i=0,idx=0;i<nslices;i+=chunkSize,idx++) {
          const size_t windowSize = chunkSize + padding * 2; // 55
          const size_t remaining = nslices - i;

          // If all remaining useful frames fit in one 55-frame window while
          // preserving at least 'padding' frames of left context, finish here.
          const bool finalChunk = remaining <= chunkSize + padding; // <= 50

          size_t useful_end;  size_t start;

          if(finalChunk) {
            start = nslices - windowSize;
            useful_end = nslices;
          } else {
            useful_end = i + chunkSize;
            start = i > padding ? i - padding : 0;
            start = std::min(start, nslices - windowSize);
          }

          size_t end = start + windowSize;
          auto z_chunk = xt::view(z, xt::all(), xt::all(), xt::range(start, end));
          auto y_mask_chunk = xt::view(y_mask, xt::all(), xt::all(), xt::range(start, end));

          auto t0 = std::chrono::steady_clock::now();
          auto chunk_audio = voice.rknnDecoder->infer(z_chunk, y_mask_chunk, g);
          auto t1 = std::chrono::steady_clock::now();

          auto real_start = chunk_audio.begin() + (i - start) * 256;

          // HACK: compare the end of the previous chunk and the start of the next chunk to determine
          // the best place to stitch them together. This is 99% good. Still get pops rarely.
          constexpr size_t compare_window = 24;
          constexpr size_t search_window = 44;
          static_assert(compare_window < search_window, "compare_window must be less than search_window");
          const bool do_depop = idx > 0 && audioBuffer.size() >= compare_window && 
            chunk_audio.size() >= search_window * 2;
          if(do_depop) {
            auto prev_chunk_end = audioBuffer.end() - compare_window;
            auto next_chunk_start = real_start;
            next_chunk_start -= std::min(std::distance(chunk_audio.begin(), next_chunk_start), (ptrdiff_t)compare_window);
            size_t min_diff = std::numeric_limits<size_t>::max();
            // increment by 2 to speed up the search
            for(size_t j=0;j<search_window*2;j+=2) {
              size_t diff = 0;
              for(size_t k=0;k<compare_window;k++)
                diff += std::abs(prev_chunk_end[k] - next_chunk_start[j+k]);
              if(diff < min_diff) {
                min_diff = diff;
                real_start = next_chunk_start + j + compare_window;
              }
            }
            // average the samples in the compare window to smooth out the transition even more
            auto prev_base_ptr = audioBuffer.end() - compare_window;
            auto next_base_ptr = real_start - compare_window;
            for(size_t j=0;j<compare_window;j++) {
              float weight = (float)j / (float)compare_window;
              prev_base_ptr[j] = prev_base_ptr[j] * (1.0f - weight) + next_base_ptr[j] * weight;
            }
          }

          auto real_end = chunk_audio.begin() + (useful_end - start) * 256;

          //spdlog::debug("chunk={} i={} start={} end={} useful_end={} real_start={} real_end={} audio_pos={}",
          //  idx, i, start, end, useful_end, std::distance(chunk_audio.begin(), real_start),
          //  std::distance(chunk_audio.begin(), real_end), audioBuffer.size());

          audioBuffer.insert(audioBuffer.end(), real_start, real_end);
          float chunk_infer_seconds = std::chrono::duration<double>(t1 - t0).count();

          if(audioCallback && audioBuffer.size() > compare_window) {
            std::vector<int16_t> tmp;
            tmp.insert(tmp.end(), audioBuffer.end() - compare_window, audioBuffer.end());
            audioBuffer.resize(audioBuffer.size() - compare_window);
            audioCallback();
            audioBuffer.resize(tmp.size());
            memcpy(audioBuffer.data(), tmp.data(), tmp.size() * sizeof(int16_t));
          }

          inferSeconds += chunk_infer_seconds;
          //spdlog::debug("Chunk {} took {} seconds", idx, chunk_infer_seconds);

          //if(i == 0 && phraseIdx == 0) {
          //  auto t = std::chrono::steady_clock::now();
          //  auto first_chunk_duration = std::chrono::duration<double>(t - encode_start).count();
          //  spdlog::debug("First chunk latency: {} seconds", first_chunk_duration);
          //}

          if(finalChunk) break;
        }
      }

      const double audioSeconds =
        static_cast<double>(audioBuffer.size() - audioStart) /
        static_cast<double>(voice.synthesisConfig.sampleRate);

      result.audioSeconds += audioSeconds;
      result.inferSeconds += inferSeconds;

      // Add end of phrase silence
      std::size_t phraseSilenceSamples = 
        (std::size_t)(phrase.silenceSeconds * voice.synthesisConfig.sampleRate * voice.synthesisConfig.channels);
      for (std::size_t i = 0; i < phraseSilenceSamples; i++) audioBuffer.push_back(0);
    }

    // Add end of sentence silence
    if (sentence.addSilence && sentenceSilenceSamples > 0 )
      for (std::size_t i = 0; i < sentenceSilenceSamples; i++) audioBuffer.push_back(0);

    if (audioCallback) {
      // Call back must copy audio since it is cleared afterwards.
      audioCallback();
      audioBuffer.clear();
    }
  }

  if (result.audioSeconds > 0)
    result.realTimeFactor = result.inferSeconds / result.audioSeconds;
} /* synthesize */

// Phonemize text and synthesize audio
void textToAudio(PiperConfig &config, Voice &voice, std::string text, std::vector<int16_t> &audioBuffer, 
  SynthesisResult &result, const std::function<void()> &audioCallback, std::optional<size_t> speakerId,
  std::optional<float> noiseScale, std::optional<float> lengthScale, std::optional<float> noiseW)
{
  spdlog::debug("--------------------");
  auto phonemeData = phonemize(config, voice, text);
  synthesize(voice, phonemeData, audioBuffer, result, audioCallback,
    speakerId, noiseScale, lengthScale, noiseW);
  spdlog::debug("--------------------");
} 

// Phonemize text and synthesize audio to WAV file
void textToWavFile(PiperConfig &config, Voice &voice, std::string text, std::ostream &audioFile, 
  SynthesisResult &result, std::optional<size_t> speakerId, std::optional<float> noiseScale,
  std::optional<float> lengthScale, std::optional<float> noiseW)
{
  std::vector<int16_t> audioBuffer;
  textToAudio(config, voice, text, audioBuffer, result, NULL, noiseScale, lengthScale, noiseW);
  // Write WAV
  auto synthesisConfig = voice.synthesisConfig;
  writeWavHeader(synthesisConfig.sampleRate, synthesisConfig.sampleWidth,
    synthesisConfig.channels, (int32_t)audioBuffer.size(), audioFile);
  audioFile.write((const char *)audioBuffer.data(), sizeof(int16_t) * audioBuffer.size());
} 

} // namespace piper

