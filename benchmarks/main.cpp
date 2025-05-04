#include <ClassificationUtils.hpp>
#include <filesystem>
#include <inferencetools/EngineFactory.hpp>
#include <fstream>

#include "SegmentationUtils.hpp"
#include "DataLoader.hpp"

const std::vector<std::string> IMAGE_PATHS = {
    "C:/data/graphs/area/PMC1464820___pgen.0020068.g002.jpg",
    "C:/data/graphs/area/PMC7219504___20.jpg",
    "C:/data/graphs/area/PMC3148245___g003.jpg",
    "C:/data/graphs/area/PMC8170440___17_HTML.jpg",
    "C:/data/graphs/area/PMC8328422___05.jpg",
    "C:/data/graphs/heatmap/PMC5982579___04.jpg",
    "C:/data/graphs/heatmap/PMC6281281___g002.jpg",
    "C:/data/graphs/heatmap/PMC6390891___20_HTML.jpg",
    "C:/data/graphs/heatmap/PMC7138785___1_HTML.jpg",
    "C:/data/graphs/heatmap/PMC8037530___g006.jpg",
    "C:/data/graphs/horizontal_bar/PMC8284307___4.jpg",
    "C:/data/graphs/horizontal_bar/PMC8287754___8_HTML.jpg",
    "C:/data/graphs/horizontal_bar/PMC8207607___3_HTML.jpg",
    "C:/data/graphs/horizontal_bar/PMC8288115___04.jpg",
    "C:/data/graphs/horizontal_bar/PMC8160898___6.jpg",
    "C:/data/graphs/scatter-line/PMC6018568___3.jpg",
    "C:/data/graphs/scatter-line/PMC6039173___10-figsupp1.jpg",
    "C:/data/graphs/scatter-line/PMC6102526___2.jpg",
    "C:/data/graphs/scatter-line/PMC6140717___4-figsupp3.jpg",
    "C:/data/graphs/scatter-line/PMC6164382___7.jpg",
    "C:/data/graphs/scatter-line/PMC6176658___g0005.jpg",
    "C:/data/graphs/vertical_box/PMC2483419___g002.jpg",
    "C:/data/graphs/vertical_box/PMC3411624___g005.jpg",
    "C:/data/graphs/vertical_box/PMC3748051___5.jpg",
    "C:/data/graphs/vertical_box/PMC5769657___g1.jpg",
    "C:/data/graphs/vertical_box/PMC5818887___08.jpg",

};


const std::vector<std::string> JSON_MODELS = {
    R"(C:/~Lokalni data/School/DP/DP_analyza_grafu/models/segmentation.json)",
    R"(C:/~Lokalni data/School/DP/DP_analyza_grafu/models/classification.json)"
};

const std::vector<std::string> ONNX_MODELS = {
    R"(C:/Code/DP/C++/DP_analyza_grafu/models/segmenation.onnx)",
    R"(C:/~Lokalni data/School/DP/DP_analyza_grafu/models/classification.onnx)"
};

const std::vector<std::string> PYTORCH_MODELS = {
    R"(C:/~Lokalni data/School/DP/DP_analyza_grafu/models/segmentation.tar)",
    R"(C:/~Lokalni data/School/DP/DP_analyza_grafu/models/classification.pt)"
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

std::string getSegmentationModelPath(const std::string& engineName)
{
    if (isTorch(engineName))
    {
        return PYTORCH_MODELS[0];
    }
    if (isFrugally(engineName))
    {
        return JSON_MODELS[0];
    }
    if (isTensorRT(engineName))
    {
        return ONNX_MODELS[0];
    }
    return "";
}

std::string getClassificationModelPath(const std::string& engineName)
{
    if (isTorch(engineName))
    {
        return PYTORCH_MODELS[1];
    }
    if (isFrugally(engineName))
    {
        return JSON_MODELS[1];
    }
    if (isTensorRT(engineName))
    {
        return ONNX_MODELS[1];
    }
    return "";
}

void doSegmentation(const std::string& outRoot)
{
    const auto engineNames = EngineFactory::getAvailableEngines();
    for (const auto& engineName : engineNames)
    {
        const auto modelPath = getSegmentationModelPath(engineName);
        std::cout << "Inferencing model (Segmentation): " << modelPath << std::endl;
        const auto engine = EngineFactory::findEngine(engineName);
        engine->loadModel(modelPath);
        for (const auto& imagePath : IMAGE_PATHS)
        {
            const auto isLibTorch = isTorch(engineName);
            constexpr int SEGMENTATION_WIDTH = 224;
            constexpr int SEGMENTATION_HEIGHT = 224;
            constexpr int SEGMENTATION_DEPTH = 3;
            constexpr int OUTPUT_SIZE = SEGMENTATION_WIDTH * SEGMENTATION_HEIGHT * SEGMENTATION_DEPTH;
            const auto image = sample::preprocessImage(imread(imagePath, cv::IMREAD_COLOR), SEGMENTATION_WIDTH,
                                                       SEGMENTATION_HEIGHT, isLibTorch);
            const auto input = InferInput(sample::loadToVector(image, isLibTorch), SEGMENTATION_WIDTH,
                                          SEGMENTATION_HEIGHT,
                                          SEGMENTATION_DEPTH, OUTPUT_SIZE);
            const auto prediction = engine->predict(input);
            const auto outputImage = sample::postProcessSegmentation(prediction);

            const auto imageName = imagePath.substr(imagePath.find_last_of("/") + 1);
            const auto outputFolderPath = outRoot + "/" + engineName + "/segmentation/";
            const auto outputFilePath = outputFolderPath + imageName;

            std::filesystem::create_directories(outputFolderPath);
            imwrite(outputFilePath, outputImage);
            std::cout << "Segmentation result written to: " << outputFilePath << std::endl;
        }
    }
}

void doClassification(const std::string& outRoot)
{
    const auto engineNames = EngineFactory::getAvailableEngines();
    for (const auto& engineName : engineNames)
    {
        const auto modelPath = getClassificationModelPath(engineName);
        std::cout << "Inferencing model (Classification): " << modelPath << std::endl;

        const auto isLibTorch = isTorch(engineName);
        const auto engine = EngineFactory::findEngine(engineName);
        engine->loadModel(modelPath);
        for (const auto& imagePath : IMAGE_PATHS)
        {
            const auto image = sample::preprocessImage(imread(imagePath, cv::IMREAD_COLOR), 224, 224, isLibTorch);
            const auto input = InferInput(sample::loadToVector(image, isLibTorch));

            const auto imageSlashIndex = imagePath.find_last_of("/");
            const auto parentFolder = imagePath.substr(0, imagePath.find_last_of("/"));
            const auto classSlashIndex = parentFolder.find_last_of("/");
            const auto expectedClass = parentFolder.substr(classSlashIndex + 1);

            const auto prediction = engine->predict(input);
            const auto outputText = sample::postProcessClassification(prediction);

            const auto imageName = imagePath.substr(imageSlashIndex + 1);
            const auto outputFolderPath = outRoot + "/" + engineName + "/classification/" + expectedClass + "/";
            const auto outputFilePath = outputFolderPath + imageName + ".txt";

            std::filesystem::create_directories(outputFolderPath);
            std::ofstream(outputFilePath) << outputText;
            std::cout << "Classification ("
                << std::distance(prediction.begin(), std::max_element(prediction.begin(), prediction.end()))
                << ") text written to: " << outputFilePath << std::endl;
        }
    }
}


int main(const int argv, char** argc)
{
    if (argv != 2)
    {
        std::cerr << "Provide output folder via command line." << std::endl;
        return 5;
    }
    const auto rootFolderPath = std::string(argc[1]);

    EngineFactory::registerAllEngines();
    doSegmentation(rootFolderPath);
    // doClassification(rootFolderPath);
}
