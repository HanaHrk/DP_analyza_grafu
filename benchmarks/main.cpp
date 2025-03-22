#include <inferencetools/FrugallyDeepEngine.hpp>
#include <inferencetools/EngineFactory.hpp>

#include "DataLoader.hpp"

int main()
{
    EngineFactory::registerAllEngines();
    const auto engines = EngineFactory::getAvailableEngines();
    const std::vector<std::string> imagePaths = {
        "C:\\data\\graphs\\area\\PMC7219504___20.jpg",
        "C:\\data\\graphs\\heatmap\\PMC5982579___04.jpg",
        "C:\\data\\graphs\\pie\\PMC7913272___3_HTML.jpg",
        "C:\\data\\graphs\\manhattan\\PMC6706940___3_HTML.jpg"
    };
    for (const auto& engineName : engines)
    {
        std::cout << engineName << std::endl;
        const auto engine = EngineFactory::findEngine(engineName);
        engine->loadModel("C:\\~Lokalni data\\School\\DP\\DP_analyza_grafu\\models\\classification.json");
        const auto imageSize = engine->getSize();
        for (const auto& imagePath : imagePaths)
        {
            const auto image = imread(imagePath, cv::IMREAD_COLOR);
            const auto tensor = sample::loadToVector(image, imageSize.width, imageSize.height);
            const auto classificationResult = engine->classify(tensor);
            std::cout << "Class: " << classificationResult->classId << std::endl;
        }
    }
}
