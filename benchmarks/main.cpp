#include <ClassificationUtils.hpp>
#include <inferencetools/FrugallyDeepEngine.hpp>
#include <inferencetools/EngineFactory.hpp>

#include "SegmentationUtils.hpp"
#include "DataLoader.hpp"


std::string getTypeName(const InferenceType& inference)
{
    if (inference == InferenceType::SEGMENTATION)
    {
        return "segmentation";
    }
    if (inference == InferenceType::CLASSIFICATION)
    {
        return "classification";
    }
    return "<unknown>";
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
    const auto engineNames = EngineFactory::getAvailableEngines();
    const std::vector<std::string> imagePaths = {
        "C:\\data\\graphs\\area\\PMC7219504___20.jpg",
        "C:\\data\\graphs\\area\\PMC3148245___g003.jpg",
        "C:\\data\\graphs\\heatmap\\PMC5982579___04.jpg",
        "C:\\data\\graphs\\pie\\PMC7913272___3_HTML.jpg",
        "C:\\data\\graphs\\manhattan\\PMC6706940___3_HTML.jpg",
        "C:\\data\\graphs\\vertical_interval\\PMC2744016___4.jpg"
    };
    const auto models = {
        EngineInfo(R"(C:\~Lokalni data\School\DP\DP_analyza_grafu\models\segmentation.json)",
                   InferenceType::SEGMENTATION),
        EngineInfo(R"(C:\~Lokalni data\School\DP\DP_analyza_grafu\models\classification.json)",
                   InferenceType::CLASSIFICATION)
    };
    for (const auto& model : models)
    {
        std::cout << "Inferencing model: " << model.path << std::endl;
        for (const auto& engineName : engineNames)
        {
            std::cout << "On engine: " << engineName << std::flush;
            const auto engine = EngineFactory::findEngine(engineName);
            engine->loadModel(model);

            for (const auto& imagePath : imagePaths)
            {
                auto image = imread(imagePath, cv::IMREAD_COLOR);
                image = sample::preprocessImage(image, 224, 224);
                const auto prediction = engine->predict(image, sample::loadToVector);

                const auto imageName = imagePath.substr(imagePath.find_last_of("\\") + 1);
                std::string outputFilePath = "<unknown>";

                std::string outputFolderPath = rootFolderPath + "/" + getTypeName(engine->getType()) + "/" + engineName;
                std::filesystem::create_directories(outputFolderPath);
                if (engine->getType() == InferenceType::SEGMENTATION)
                {
                    outputFilePath = outputFolderPath + "/" + imageName;
                    const auto outputContent = sample::postProcessSegmentation(*prediction);
                    imwrite(outputFilePath, outputContent);
                }

                else if (engine->getType() == InferenceType::CLASSIFICATION)
                {
                    outputFilePath = outputFolderPath + "/" + imageName + ".txt";
                    const auto outputContent = sample::postProcessClassification(*prediction);
                    std::ofstream(outputFilePath) << outputContent;
                }
                std::cout << "OK - " << outputFilePath << std::endl;
            }
        }
    }
}
