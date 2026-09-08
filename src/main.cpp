#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "piper.hpp"

enum OutputType { OUTPUT_FILE, OUTPUT_DIRECTORY, OUTPUT_STDOUT, OUTPUT_RAW };

struct RunConfig {
  // Voice package directory
  std::filesystem::path voicePack;

  // Type of output to produce.
  // Default is to write a WAV file in the current directory.
  OutputType outputType = OUTPUT_DIRECTORY;

  // Path for output
  std::optional<std::filesystem::path> outputPath = std::filesystem::path(".");

  // Numerical id of the default speaker (multi-speaker voices)
  std::optional<piper::SpeakerId> speakerId;

  // Amount of noise to add during audio generation
  std::optional<float> noiseScale;

  // Speed of speaking (1 = normal, < 1 is faster, > 1 is slower)
  std::optional<float> lengthScale;

  // Variation in phoneme lengths
  std::optional<float> noiseW;

  // Seconds of silence to add after each sentence
  std::optional<float> sentenceSilenceSeconds;

  // Path to espeak-ng data directory
  std::optional<std::filesystem::path> eSpeakDataPath;

  // Seconds of extra silence to insert after a single phoneme
  std::optional<std::map<piper::Phoneme, float>> phonemeSilenceSeconds;

  // RKNN support
  bool rknnEnabled = true;

  // Set to whatever accelerator is available for ONNX. Ex: "cuda"
  // This has 0 affect if the underlying model is not handled by ONNX.
  std::string accelerator = "";
};

void parseArgs(int argc, char *argv[], RunConfig &runConfig);
// ----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  int exitCode = EXIT_SUCCESS;
  spdlog::set_default_logger(spdlog::stderr_color_st("rk3576-tts-demo"));

  RunConfig runConfig;  parseArgs(argc, argv, runConfig);
  spdlog::debug("Voice package: {}", runConfig.voicePack.string());

  piper::PiperConfig piperConfig;  piper::Voice voice;
  piperConfig.rknnEnabled = runConfig.rknnEnabled;

  auto startTime = std::chrono::steady_clock::now();
  if (!loadVoice(piperConfig, runConfig.voicePack, voice, runConfig.speakerId, runConfig.accelerator)) exit(1);
  auto endTime = std::chrono::steady_clock::now();
  spdlog::info(" Voice loaded in {} second(s)", std::chrono::duration<double>(endTime - startTime).count());

  // Get the path to the piper executable so we can locate espeak-ng-data, etc. next to it.
  auto exePath = std::filesystem::canonical("/proc/self/exe");

  if (voice.phonemizeConfig.phonemeType == piper::eSpeakPhonemes) {
    spdlog::debug("Voice uses eSpeak phonemes ({})", voice.phonemizeConfig.eSpeak.voice);
    if (!piperConfig.eSpeakDataPath.empty()) {
      spdlog::debug("Using voice pack path for 'espeak-ng-data'");    
    } else if (runConfig.eSpeakDataPath) {
      piperConfig.eSpeakDataPath = runConfig.eSpeakDataPath.value().string();
      spdlog::debug("Using command line provided path for 'espeak-ng-data'");
    } else {
      piperConfig.eSpeakDataPath = std::filesystem::absolute(
        exePath.parent_path().parent_path().parent_path() / 
        "piper-phonemize-install" / "share" / "espeak-ng-data").string();
      spdlog::debug("Using default path for 'espeak-ng-data'");
    }
  } else {
    // Not using eSpeak
    piperConfig.useESpeak = false;
  }

  piper::initialize(piperConfig);

  // Scales
  if (runConfig.noiseScale) voice.synthesisConfig.noiseScale = runConfig.noiseScale.value();
  if (runConfig.lengthScale) voice.synthesisConfig.lengthScale = runConfig.lengthScale.value();
  if (runConfig.noiseW) voice.synthesisConfig.noiseW = runConfig.noiseW.value();
  if (runConfig.sentenceSilenceSeconds) 
    voice.synthesisConfig.sentenceSilenceSeconds = runConfig.sentenceSilenceSeconds.value();

  if (runConfig.phonemeSilenceSeconds)
    if (!voice.synthesisConfig.phonemeSilenceSeconds) {
      // Overwrite
      voice.synthesisConfig.phonemeSilenceSeconds = runConfig.phonemeSilenceSeconds;
    } else {
      // Merge
      for (const auto &[phoneme, silenceSeconds] : *runConfig.phonemeSilenceSeconds)
        voice.synthesisConfig.phonemeSilenceSeconds->try_emplace(phoneme, silenceSeconds);
    }

  if (runConfig.outputType == OUTPUT_DIRECTORY) {
    runConfig.outputPath = std::filesystem::absolute(runConfig.outputPath.value());
    spdlog::info(" Output directory: {}", runConfig.outputPath.value().string());
  }

  std::string line;  piper::SynthesisResult result;

  while (getline(std::cin, line)) {
    auto outputType = runConfig.outputType;
    std::optional<std::filesystem::path> maybeOutputPath = runConfig.outputPath;

    // Timestamp is used for path to output WAV file
    const auto now = std::chrono::system_clock::now();
    const auto timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();

    if (outputType == OUTPUT_DIRECTORY) {
      // Generate path using timestamp
      std::stringstream outputName;
      outputName << timestamp << ".wav";
      std::filesystem::path outputPath = runConfig.outputPath.value();
      outputPath.append(outputName.str());
      // Output audio to automatically-named WAV file in a directory
      std::ofstream audioFile(outputPath.string(), std::ios::binary);
      piper::textToWavFile(piperConfig, voice, line, audioFile, result);
      spdlog::info(" Wrote {}", outputPath.string());
    } else if (outputType == OUTPUT_FILE) {
      if (!maybeOutputPath || maybeOutputPath->empty()) 
        { std::cerr << "Error: no output path provided" << std::endl; exitCode = EXIT_FAILURE; break; }
      std::filesystem::path outputPath = maybeOutputPath.value();
      // Output audio to WAV file
      std::ofstream audioFile(outputPath.string(), std::ios::binary);
      piper::textToWavFile(piperConfig, voice, line, audioFile, result);
      spdlog::info(" Audio data was written to: {}", outputPath.string());
    } else if (outputType == OUTPUT_STDOUT) {
      // Output WAV to stdout
      piper::textToWavFile(piperConfig, voice, line, std::cout, result);
    } else if (outputType == OUTPUT_RAW) {
      // Raw output to stdout
      std::vector<int16_t> audioBuffer;
      auto audioCallback = [&audioBuffer]() {
        // Signal thread that audio is ready
        std::cout.write((const char *)audioBuffer.data(), sizeof(int16_t) * audioBuffer.size());
        std::cout.flush();
      };
      piper::textToAudio(piperConfig, voice, line, audioBuffer, result, audioCallback);
      // Wait for audio output to finish
      spdlog::info(" Waiting for audio to finish playing...");
    }

    spdlog::info(" Real-time factor: {} (infer={} sec, audio={} sec)",
      result.realTimeFactor, result.inferSeconds, result.audioSeconds);
  } // for each line

  piper::terminate(piperConfig);
  return exitCode;
}

// ----------------------------------------------------------------------------

void printUsage(char *argv[]) {
  std::cerr << std::endl;
  std::cerr << "usage: " << argv[0] << " <voice-pack-path> [options]" << std::endl;
  std::cerr << std::endl;
  std::cerr << "voice-pack-path (DIR): voice model package directory (needed!)" << std::endl;
  std::cerr << std::endl;
  std::cerr << "options:" << std::endl;
  std::cerr << "   -f  FILE  --output-file FILE  path to output WAV file ('-' for stdout)" << std::endl;
  std::cerr << "   -d  DIR   --output-dir  DIR   path to output directory (default: cwd)" << std::endl;
  std::cerr << "   --output-raw                  output raw audio to stdout as it becomes available" << std::endl;
  std::cerr << "   -s  NUM   --speaker     NUM   id of speaker (default: 0)" << std::endl;
  std::cerr << "   --no-rknn                     disable RKNN support" << std::endl;
  std::cerr << "   --noise-scale           NUM   generator noise (default: 0.667)" << std::endl;
  std::cerr << "   --length-scale          NUM   phoneme length (default: 1.0)" << std::endl;
  std::cerr << "   --noise-w               NUM   phoneme width noise (default: 0.8)" << std::endl;
  std::cerr << "   --sentence-silence      NUM   seconds of silence after each sentence (default: 0.2)" << std::endl;
  std::cerr << "   --espeak-data           DIR   path to default espeak-ng data directory" << std::endl;
  std::cerr << "   --accelerator           STR   accelerator to use for ONNX (default: none, valid: cuda)" << std::endl;
  std::cerr << "   --debug                       print DEBUG messages to the console" << std::endl;
  std::cerr << "   -q        --quiet             disable logging" << std::endl;
  std::cerr << "   -h        --help              show this message and exit" << std::endl;
  std::cerr << std::endl;
}

void ensureArg(int argc, char *argv[], int argi) {
  if ((argi + 1) >= argc) { printUsage(argv); exit(1); }
}

// Parse command-line arguments
void parseArgs(int argc, char *argv[], RunConfig &runConfig) {
  if (argc < 2) { printUsage(argv); exit(1); }

  runConfig.voicePack = std::filesystem::absolute(argv[1]);
  if (!std::filesystem::is_directory(runConfig.voicePack)) {
    std::cerr << "Error: invalid voice pack path (not a directory): " << runConfig.voicePack.string() << std::endl;
    printUsage(argv); exit(1);
  }

  for (int i = 2; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-f" || arg == "--output-file") {
      ensureArg(argc, argv, i);
      std::string filePath = argv[++i];
      if (filePath == "-") {
        runConfig.outputType = OUTPUT_STDOUT;
        runConfig.outputPath = std::nullopt;
      } else {
        runConfig.outputType = OUTPUT_FILE;
        runConfig.outputPath = std::filesystem::path(filePath);
      }
    } else if (arg == "-d" || arg == "--output-dir") {
      ensureArg(argc, argv, i);
      runConfig.outputType = OUTPUT_DIRECTORY;
      runConfig.outputPath = std::filesystem::path(argv[++i]);
    } else if (arg == "--output-raw") {
      runConfig.outputType = OUTPUT_RAW;
    } else if (arg == "-s" || arg == "--speaker") {
      ensureArg(argc, argv, i);
      runConfig.speakerId = (piper::SpeakerId)std::stol(argv[++i]);
    } else if (arg == "--no-rknn") {
      runConfig.rknnEnabled = false;
    } else if (arg == "--noise-scale") {
      ensureArg(argc, argv, i);
      runConfig.noiseScale = std::stof(argv[++i]);
    } else if (arg == "--length-scale") {
      ensureArg(argc, argv, i);
      runConfig.lengthScale = std::stof(argv[++i]);
    } else if (arg == "--noise-w") {
      ensureArg(argc, argv, i);
      runConfig.noiseW = std::stof(argv[++i]);
    } else if (arg == "--sentence-silence") {
      ensureArg(argc, argv, i);
      runConfig.sentenceSilenceSeconds = std::stof(argv[++i]);
    } else if (arg == "--phoneme-silence") {
      ensureArg(argc, argv, i);
      ensureArg(argc, argv, i + 1);
      auto phonemeStr = std::string(argv[++i]);
      if (!piper::isSingleCodepoint(phonemeStr)) {
        std::cerr << "Phoneme '" << phonemeStr << "' is not a single codepoint (--phoneme-silence)" << std::endl;
        exit(1);
      }
      if (!runConfig.phonemeSilenceSeconds) runConfig.phonemeSilenceSeconds.emplace();
      auto phoneme = piper::getCodepoint(phonemeStr);
      (*runConfig.phonemeSilenceSeconds)[phoneme] = std::stof(argv[++i]);
    } else if (arg == "--espeak-data") {
      ensureArg(argc, argv, i);
      runConfig.eSpeakDataPath = std::filesystem::absolute(argv[++i]);
    } else if (arg == "--accelerator") {
      ensureArg(argc, argv, i);
      runConfig.accelerator = argv[++i];
    } else if (arg == "--version") {
      std::cout << piper::getVersion() << std::endl;
      exit(0);
    } else if (arg == "--debug") {
      spdlog::set_level(spdlog::level::debug);
    } else if (arg == "-q" || arg == "--quiet") {
      spdlog::set_level(spdlog::level::off);
    } else if (arg == "-h" || arg == "--help") {
      printUsage(argv); exit(0);
    } else {
      std::cerr << "Error: unknown option: " << arg << std::endl;
      printUsage(argv); exit(1);
    }
  }
}
