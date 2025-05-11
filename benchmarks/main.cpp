#include <ClassificationUtils.hpp>
#include <SegmentationUtils.hpp>
#include <ArgsParser.hpp>
#include <DataLoader.hpp>
#include <inferencetools/EngineFactory.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <vector>
#include <random>
#include <memory>
#include <optional>

#include "inferencetools/EngineNotFound.hpp"

#define HANDLE_LOAD(call)                   \
    try {                                   \
        call;                               \
    } catch (const std::exception & e) {    \
        std::cerr << e.what() << std::endl; \
    }


enum InferenceType
{
    CLASSIFICATION,
    SEGMENTATION
};


bool isTorch(const std::string& name)
{
    return name == "LibTorch";
}

bool isFrugally(const std::string& name)
{
    return name == "FrugallyDeep";
}

bool isTensorRT(const std::string& name)
{
    return name == "TensorRT";
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
    std::cout << "Warning: No suitable engine found for model extension: " << modelSuffix << std::endl;
    return {};
}

void doSegmentation(const std::string& outputDir, const std::string& inputDir,
                    const std::unique_ptr<InferenceEngineSequential>& engine,
                    const std::string& engineName,
                    const bool debug)
{
    std::cout << "\nStarting segmentation with " << engineName << " engine" << std::endl;
    std::cout << "Input directory: " << inputDir << std::endl;
    std::cout << "Output directory: " << outputDir << std::endl;
    constexpr int totalImages = 100;
    int lastPercentage = -1;
    for (int imageIndex = 0; imageIndex < totalImages; imageIndex++)
    {
        const int currentPercentage = imageIndex * 100 / totalImages;
        if (currentPercentage != lastPercentage)
        {
            std::cout << "Progress: " << currentPercentage << "%" << std::endl;
            lastPercentage = currentPercentage;
        }
        const auto imagePath = sample::getRandomFilePath(inputDir);
        if (debug)
        {
            std::cout << "Selected image: " << imagePath << std::endl;
        }
        const auto isLibTorch = isTorch(engineName);
        constexpr int SEGMENTATION_WIDTH = 224;
        constexpr int SEGMENTATION_HEIGHT = 224;
        constexpr int SEGMENTATION_DEPTH = 3;
        constexpr int OUTPUT_SIZE = SEGMENTATION_WIDTH * SEGMENTATION_HEIGHT;
        const auto image = sample::preprocessImage(cv::imread(imagePath, cv::IMREAD_COLOR), SEGMENTATION_WIDTH,
                                                   SEGMENTATION_HEIGHT, isLibTorch);
        const auto input = InferInput(sample::loadToVector(image, isLibTorch), MODEL_PROPERTIES,
                                      MODEL_PROPERTIES,
                                      MODEL_PROPERTIES, MODEL_PROPERTIES);
        const auto prediction = engine->predict(input);
        const auto outputImage = sample::postProcessSegmentation(prediction);

        const auto imageName = imagePath.substr(imagePath.find_last_of("/") + 1);
        const auto outputFolderPath = outputDir + "/" + engineName + "/segmentation/";
        const auto outputFilePath = outputFolderPath + imageName;

        std::filesystem::create_directories(outputFolderPath);
        if (debug)
        {
            std::cout << "Prediction completed. Writing output to: " << outputFilePath << std::endl;
        }
        imwrite(outputFilePath, outputImage);
    }
    std::cout << "Segmentation processing completed" << std::endl;
}

void doClassification(const std::string& outputDir, const std::string& inputDir,
                      const std::unique_ptr<InferenceEngineSequential>& engine,
                      const std::string& engineName,
                      bool debug)
{
    std::cout << "\nStarting classification with " << engineName << " engine" << std::endl;
    std::cout << "Input directory: " << inputDir << std::endl;
    std::cout << "Output directory: " << outputDir << std::endl;

    std::vector<std::string> trueLabels;
    std::vector<std::string> predictedLabels;
    bool isLibTorch = isTorch(engineName);

    const auto allFiles = sample::getAllFiles(inputDir);
    const int totalFiles = allFiles.size();
    int processedCount = 0;
    int lastPercentage = -1;

    for (const auto& imagePath : allFiles)
    {
        int currentPercentage = processedCount * 100 / totalFiles;
        if (currentPercentage != lastPercentage)
        {
            std::cout << "Progress: " << currentPercentage << "%" << std::endl;
            lastPercentage = currentPercentage;
        }

        const auto image = sample::preprocessImage(imread(imagePath, cv::IMREAD_COLOR), 224, 224, isLibTorch);
        const auto input = InferInput(sample::loadToVector(image, isLibTorch));

        const auto imageSlashIndex = imagePath.find_last_of("/");
        const auto parentFolder = imagePath.substr(0, imagePath.find_last_of("/"));
        const auto classSlashIndex = parentFolder.find_last_of("/");
        const auto expectedClass = parentFolder.substr(classSlashIndex + 1);

        const auto prediction = engine->predict(input);
        const auto predictionClass = sample::getClassName(prediction);
        const auto outputText = sample::postProcessClassification(prediction);

        trueLabels.push_back(expectedClass);
        predictedLabels.push_back(predictionClass);

        const auto imageName = imagePath.substr(imageSlashIndex + 1);
        const auto outputFolderPath = outputDir + "/" + engineName + "/classification/" + expectedClass + "/";
        const auto outputFilePath = outputFolderPath + imageName + ".txt";

        std::filesystem::create_directories(outputFolderPath);
        if (debug)
        {
            std::cout << "Prediction: " << predictionClass << " (Expected: " << expectedClass << ")" << std::endl;
            std::cout << "Writing results to: " << outputFilePath << std::endl;
        }
        std::ofstream(outputFilePath) << outputText;
        processedCount++;
    }
    std::cout << "\nClassification processing completed. Generating confusion matrix..." << std::endl;
    sample::printConfusionMatrix(predictedLabels, trueLabels);
}

void inference(const std::string& outputDir, const std::string& inputDir, const std::vector<std::string>& models,
               const bool debug,
               const InferenceType& type)
{
    if (models.empty())
    {
        return;
    }
    std::cout << "\nStarting inference process" << std::endl;
    std::cout << "Type: " << (type == CLASSIFICATION ? "Classification" : "Segmentation") << std::endl;
    std::cout << "Number of models to process: " << models.size() << std::endl;

    for (const auto& model : models)
    {
        std::cout << "\nProcessing model: " << model << std::endl;
        const auto modelEngineNames = getEngineName(model);
        for (const auto& modelEngineName : modelEngineNames)
        {
            std::cout << "Initializing engine: " << modelEngineName << std::endl;
            try
            {
                const auto engine = EngineFactory::findEngine(modelEngineName);
                std::cout << "Loading model..." << std::endl;
                engine->loadModel(model);

                if (type == CLASSIFICATION)
                {
                    doClassification(outputDir, inputDir, engine, modelEngineName, debug);
                }
                else if (type == SEGMENTATION)
                {
                    doSegmentation(outputDir, inputDir, engine, modelEngineName, debug);
                }
            }
            catch (const EngineNotFound&)
            {
                std::cerr << "Engine initialization failed for model [" << modelEngineName << "]: Engine not found." << std::endl;
            }
        }
    }
    std::cout << "Inference process completed" << std::endl;
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

void printUsage()
{
    std::cout << "Usage: ./InferenceBenchmark[.exe] [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --output_dir=<path>         Path to output directory [required|single]\n";
    std::cout << "  --inputDir=<path>           Path to input directory with images [required|single]\n";
    std::cout << "  --classification=<path>     Path to classification inference model [required|multiple]\n";
    std::cout << "  --segmentation=<path>       Path to segmentation inference model [required|multiple]\n";
    std::cout << "  --debug                     Allow debug logging [simple]\n";
    std::cout << "  --help                      Print this help message [simple]\n";
}

void handleArgs(const std::vector<std::string>& argsVector)
{
    const auto outputDir = sample::extractArgsSingle(argsVector, "output_dir");
    const auto inputDir = sample::extractArgsSingle(argsVector, "input_dir");
    const auto classificationModels = sample::extractArgsMulti(argsVector, "classification");
    const auto segmentationModels = sample::extractArgsMulti(argsVector, "segmentation");
    const auto debug = sample::extractArgsSimple(argsVector, "debug");
    if (sample::extractArgsSimple(argsVector, "help"))
    {
        printUsage();
        exit(0);
    }
    if (!outputDir)
    {
        printUsage();
        exit(1);
    }
    if (!inputDir)
    {
        printUsage();
        exit(1);
    }
    if (!classificationModels && !segmentationModels)
    {
        printUsage();
        exit(1);
    }
    if (segmentationModels)
    {
        inference(outputDir.value(), inputDir.value(), segmentationModels.value(), debug, SEGMENTATION);
    }
    if (classificationModels)
    {
        inference(outputDir.value(), inputDir.value(), classificationModels.value(), debug, CLASSIFICATION);
    }
}


int main(const int argv, char** argc)
{
    std::cout << "Starting InferenceBenchmark..." << std::endl;
    std::cout << "Registering available engines..." << std::endl;
    EngineFactory::registerAllEngines();

    std::cout << "Processing command line arguments..." << std::endl;
    const auto argsVector = argsToVector(argv, argc);
    handleArgs(argsVector);

    std::cout << "InferenceBenchmark completed successfully" << std::endl;
}
