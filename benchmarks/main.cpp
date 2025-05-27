#include <SegmentationUtils.hpp>
#include <ArgsParser.hpp>
#include <inferencetools/EngineFactory.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <optional>
#include <fstream>

#include "ClassificationUtils.hpp"
#include "DataLoader.hpp"
#include "inferencetools/EngineNotFound.hpp"

#ifdef USE_FRUGALLY_DEEP
#include "inferencetools/FrugallyDeepEngine.hpp"
#endif


bool isTorch(const std::string_view& name)
{
    return name == "LibTorch";
}

bool isFrugally(const std::string_view& name)
{
    return name == "FrugallyDeep";
}

bool isTensorRT(const std::string_view& name)
{
    return name == "TensorRT";
}

std::string getTrueLabel(const std::string& inputPath)
{
    const auto parentFolder = inputPath.substr(0, inputPath.find_last_of("/"));
    const auto classSlashIndex = parentFolder.find_last_of("/");
    return parentFolder.substr(classSlashIndex + 1);
}

std::string getOutputFilePath(const std::string& outputDir, const std::string& engineName,
                              const InferenceType& inferenceType, const std::string& inputPath)
{
    const auto parentFolder = inputPath.substr(0, inputPath.find_last_of("/"));
    const auto classSlashIndex = parentFolder.find_last_of("/");
    const auto expectedClass = parentFolder.substr(classSlashIndex + 1);
    const auto inferenceTypeSuffix = inferenceType == InferenceType::CLASSIFICATION ? "classification" : "segmentation";
    const auto imageName = inputPath.substr(inputPath.find_last_of("/") + 1);
    const auto outputFolder = outputDir + "/" + engineName + "/" + inferenceTypeSuffix + "/" + expectedClass;
    std::filesystem::create_directories(outputFolder);
    if (inferenceType == InferenceType::CLASSIFICATION)
    {
        return outputFolder + "/" + imageName + ".txt";
    }
    return outputFolder + "/" + imageName;
}

std::vector<std::string> getEngineName(const std::string& modelPath)
{
    std::cout << "Getting engine name for model: " << modelPath << std::endl;
    const auto modelSuffixIndex = modelPath.find_last_of('.');
    const auto modelSuffix = modelPath.substr(modelSuffixIndex);
    if (modelSuffix == ".json")
    {
        std::cout << "Selected FrugallyDeep engine for .json model" << std::endl;
        return {"FrugallyDeep"};
    }
    if (modelSuffix == ".onnx")
    {
        std::cout << "Selected TensorRT engine for .onnx model" << std::endl;
        return {"TensorRT"};
    }
    if (modelSuffix == ".pt")
    {
        std::cout << "Selected LibTorch engine for .pt model" << std::endl;
        return {"LibTorch"};
    }
    std::cerr << "Warning: No suitable engine found for model extension: " << modelSuffix << std::endl;
    return {};
}

void segment(const std::vector<InferPathInput>& inputs,
             const InferenceEngineSequential* engine,
             const std::string& engineName,
             const std::string& outputDir,
             const bool parallel,
             const bool debug)
{
    std::cout << "\nStarting segmentation with " << engineName << " engine" << std::endl;
    std::vector<InferPathOutput> outputs;
    uint64_t totalMillis = 0;
    if (parallel)
    {
#ifdef USE_FRUGALLY_DEEP

        std::cout << "Running in parallel mode with FrugallyDeep engine" << std::endl;
        const auto frugallyDeepEngine = dynamic_cast<const FrugallyDeepEngine*>(engine);
        std::vector<InferInput> parallelInputs;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(parallelInputs),
                       [](const auto& input) { return input.input; });

        std::cout << "Processing " << inputs.size() << " images in parallel" << std::endl;
        const auto start = std::chrono::high_resolution_clock::now();
        const auto tensorOutputs = frugallyDeepEngine->predictAll(parallelInputs, FrugallyDeepEngine::ParallelMode::STD_THREADING);
        totalMillis = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
        for (int i = 0; i < inputs.size(); i++)
        {
            const auto outputFilePath = getOutputFilePath(outputDir, engineName, InferenceType::SEGMENTATION, inputs[i].path);
            outputs.emplace_back(InferPathOutput{outputFilePath, tensorOutputs[i]});
        }
#else
        std::cerr << "WARN: FrugallyDeep engine not available. Please compile with USE_FRUGALLY_DEEP=1" << std::endl;
#endif
    }
    else
    {
        std::cout << "Running in sequential mode" << std::endl;
        int lastPercentage = -1;
        int processedCount = 0;
        const int totalFiles = inputs.size();
        for (const auto& [path, input] : inputs)
        {
            if (debug)
            {
                const auto currentPercentage = static_cast<int>((processedCount + 1) * 100.0 / totalFiles);
                processedCount++;
                if (currentPercentage != lastPercentage)
                {
                    std::cout << "Progress: " << currentPercentage << "%" << std::endl;
                    lastPercentage = currentPercentage;
                }
            }
            const auto start = std::chrono::high_resolution_clock::now();
            const auto outputTensor = engine->predict(input);
            totalMillis += std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start).count();
            const auto outputFilePath = getOutputFilePath(outputDir, engineName, InferenceType::SEGMENTATION, path);
            outputs.emplace_back(InferPathOutput{outputFilePath, outputTensor});
        }
    }
    const auto seconds = static_cast<double>(totalMillis) / 1000.0;
    std::cout << "Total time: " << seconds << " seconds" << std::endl;
    std::cout << "Time per image: " << (inputs.size() / seconds) << " images per second" << std::endl;

    std::cout << "\nWriting classification results..." << std::endl;
    for (const auto& [path, output] : outputs)
    {
        if (debug)
        {
            std::cout << "Writing output to: " << path << std::endl;
        }
        const auto outputText = sample::postProcessSegmentation(output);
        cv::imwrite(path, outputText);
    }

    std::cout << "\nSegmentation completed" << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
}

void classify(const std::vector<InferPathInput>& inputs,
              const InferenceEngineSequential* engine,
              const std::string& engineName,
              const std::string& outputDir,
              const bool parallel,
              const bool debug)
{
    std::cout << "\nStarting classification with " << engineName << " engine" << std::endl;

    std::vector<std::string> trueLabels;
    std::vector<std::string> predictedLabels;
    std::vector<InferPathOutput> outputs;
    uint64_t totalMillis = 0;
    if (parallel)
    {
#ifdef USE_FRUGALLY_DEEP

        std::cout << "Running in parallel mode with FrugallyDeep engine" << std::endl;
        const auto frugallyDeepEngine = dynamic_cast<const FrugallyDeepEngine*>(engine);
        std::vector<InferInput> parallelInputs;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(parallelInputs),
                       [](const auto& input) { return input.input; });

        std::cout << "Processing " << inputs.size() << " images in parallel" << std::endl;
        const auto start = std::chrono::high_resolution_clock::now();
        const auto tensorOutputs = frugallyDeepEngine->predictAll(parallelInputs, FrugallyDeepEngine::ParallelMode::STD_THREADING);
        totalMillis = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
        for (int i = 0; i < inputs.size(); i++)
        {
            outputs.emplace_back(InferPathOutput{inputs[i].path, tensorOutputs[i]});
        }
#else
        std::cerr << "WARN: FrugallyDeep engine not available. Please compile with USE_FRUGALLY_DEEP=1" << std::endl;
#endif
    }
    else
    {
        std::cout << "Running in sequential mode" << std::endl;
        int lastPercentage = -1;
        int processedCount = 0;
        const int totalFiles = inputs.size();
        for (const auto& [path, input] : inputs)
        {
            if (debug)
            {
                const auto currentPercentage = static_cast<int>((processedCount + 1) * 100.0 / totalFiles);
                processedCount++;
                if (currentPercentage != lastPercentage)
                {
                    std::cout << "Progress: " << currentPercentage << "%" << std::endl;
                    lastPercentage = currentPercentage;
                }
            }
            const auto start = std::chrono::high_resolution_clock::now();
            const auto outputTensor = engine->predict(input);
            totalMillis += std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start).count();
            const auto outputFilePath = getOutputFilePath(outputDir, engineName, InferenceType::CLASSIFICATION, path);
            outputs.emplace_back(InferPathOutput{outputFilePath, outputTensor});

            if (debug)
            {
                predictedLabels.push_back(sample::getClassName(outputTensor));
                trueLabels.push_back(getTrueLabel(path));
            }
        }
    }
    const auto seconds = static_cast<double>(totalMillis) / 1000.0;
    std::cout << "Total time: " << seconds << " seconds" << std::endl;
    std::cout << "Time per image: " << (inputs.size() / seconds) << " images per second" << std::endl;

    std::cout << "\nWriting classification results..." << std::endl;
    for (const auto& [path, output] : outputs)
    {
        if (debug)
        {
            std::cout << "Writing output to: " << path << std::endl;
        }
        const auto outputText = sample::postProcessClassification(output);
        std::ofstream(path) << outputText << std::flush;
    }

    if (debug)
    {
        std::cout << "Generating confusion matrix..." << std::endl;
        sample::printConfusionMatrix(predictedLabels, trueLabels);
    }
    std::cout << "\nClassification completed" << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
}

std::vector<std::string> argsToVector(const int argv, char** argc)
{
    std::vector<std::string> argsVector;
    for (int i = 1; i < argv; ++i)
    {
        argsVector.emplace_back(argc[i]);
    }
    return argsVector;
}

void inference(const std::string& inputDir,
               const std::string& outputDir,
               const std::vector<std::string>& models,
               const InferenceType& inferenceType,
               const bool parallel,
               const bool debug)
{
    std::cout << "\nStarting inference process..." << std::endl;

    for (const auto& model : models)
    {
        std::cout << "\nProcessing model: " << model << std::endl;
        const auto engineNames = getEngineName(model);

        for (const auto& engineName : engineNames)
        {
            std::cout << "Using engine: " << engineName << std::endl;
            const auto libTorch = isTorch(model);

            auto allFilePaths = sample::getAllFiles(inputDir);
            allFilePaths = std::vector(allFilePaths.begin(), allFilePaths.begin() + 10);

            try
            {
                std::cout << "Initializing " << engineName << " engine..." << std::endl;
                const auto engine = EngineFactory::findEngine(engineName);
                std::cout << "" << "Loading model..." << std::endl;
                engine->loadModel(model);
                std::cout << "Preparing inference inputs..." << std::endl;
                std::cout << "Scanning input directory..." << std::endl;

                if (inferenceType == InferenceType::CLASSIFICATION)
                {
                    const auto allInferenceInputs = sample::getInferInputs(allFilePaths, libTorch, MODEL_PROPERTIES,
                                                                           MODEL_PROPERTIES, MODEL_PROPERTIES,
                                                                           MODEL_PROPERTIES);
                    classify(allInferenceInputs, engine.get(), engineName, outputDir, parallel && isFrugally(engineName), debug);
                }
                else
                {
                    const auto allInferenceInputs = sample::getInferInputs(allFilePaths, libTorch, 224, 224, 3, 224 * 224 * 1);
                    segment(allInferenceInputs, engine.get(), engineName, outputDir, parallel && isFrugally(engineName), debug);
                }
            }
            catch (const EngineNotFound& e)
            {
                std::cerr << "ERROR: Engine " << engineName << " not found" << std::endl;
            }
            catch (const std::exception& e)
            {
                std::cerr << "ERROR: Exception occurred while processing with " << engineName << ": " << e.what() <<
                    std::endl;
            }
        }
    }
    std::cout << "\nInference process completed" << std::endl;
}

void printUsage()
{
    std::cout << "Usage: ./InferenceBenchmark[.exe] [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --output_dir=<path>         Path to output directory [required|single]\n";
    std::cout << "  --inputDir=<path>           Path to input directory with images [required|single]\n";
    std::cout << "  --classification=<path>     Path to classification inference model [required|multiple]\n";
    std::cout << "  --segmentation=<path>       Path to segmentation inference model [required|multiple]\n";
    std::cout << "  --debug                     Allow debug logging [simple]\n";
    std::cout << "  --parallel                  If running Frugally Deep, run it parallelly [simple]\n";
    std::cout << "  --help                      Print this help message [simple]\n";
}


void handleArgs(const std::vector<std::string>& argsVector)
{
    std::cout << "Parsing command line arguments..." << std::endl;

    const auto outputDir = sample::extractArgsSingle(argsVector, "output_dir");
    const auto inputDir = sample::extractArgsSingle(argsVector, "input_dir");
    const auto classificationModels = sample::extractArgsMulti(argsVector, "classification");
    const auto segmentationModels = sample::extractArgsMulti(argsVector, "segmentation");
    const auto parallelIfAvailable = sample::extractArgsSimple(argsVector, "parallel");
    const auto debug = sample::extractArgsSimple(argsVector, "debug");

    if (sample::extractArgsSimple(argsVector, "help"))
    {
        std::cout << "Showing help information:" << std::endl;
        printUsage();
        exit(0);
    }

    // Validate required arguments
    if (!outputDir)
    {
        std::cerr << "ERROR: Output directory not specified" << std::endl;
        printUsage();
        exit(1);
    }
    if (!inputDir)
    {
        std::cerr << "ERROR: Input directory not specified" << std::endl;
        printUsage();
        exit(1);
    }
    if (!classificationModels && !segmentationModels)
    {
        std::cerr << "ERROR: No models specified" << std::endl;
        printUsage();
        exit(1);
    }

    std::cout << "\nConfiguration:" << std::endl
        << "- Input directory: " << inputDir.value() << std::endl
        << "- Output directory: " << outputDir.value() << std::endl
        << "- Parallel processing: " << (parallelIfAvailable ? "enabled" : "disabled") << std::endl
        << "- Classification models: " << (classificationModels ? classificationModels.value().size() : 0) << std::endl
        << "- Segmentation models: " << (segmentationModels ? segmentationModels.value().size() : 0) << std::endl;

    if (classificationModels)
    {
        inference(inputDir.value(), outputDir.value(), classificationModels.value(), InferenceType::CLASSIFICATION, parallelIfAvailable, debug);
    }
    if (segmentationModels)
    {
        inference(inputDir.value(), outputDir.value(), segmentationModels.value(), InferenceType::SEGMENTATION, parallelIfAvailable, debug);
    }
}

int main(const int argv, char** argc)
{
    std::cout << "=== Starting InferenceBenchmark ===" << std::endl;

    std::cout << "\nRegistering inference engines..." << std::endl;
    EngineFactory::registerAllEngines();

    std::cout << "\nProcessing command line arguments..." << std::endl;
    const auto argsVector = argsToVector(argv, argc);

    handleArgs(argsVector);

    std::cout << "\n=== InferenceBenchmark completed successfully ===" << std::endl;
    return 0;
}
