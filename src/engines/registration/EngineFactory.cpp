#include <inferencetools/EngineFactory.hpp>
#include <inferencetools/EngineNotFound.hpp>

#if USE_FRUGALLY_DEEP
#include <inferencetools/FrugallyDeepEngineSequential.hpp>
#elif USE_TENSORRT
#include <inferencetools/TensorRTEngineSequential.hpp>
#elif USE_LIBTORCH
#include "inferencetools/LibTorchEngineSequential.hpp"
#endif

std::map<std::string, EngineFactory::EngineBuilder> EngineFactory::engineMaps_;

EngineFactory::EnginePtr EngineFactory::findEngine(const std::string& engineName)
{
    if (engineMaps_[engineName] == nullptr)
    {
        throw EngineNotFound("Engine " + engineName + " not found");
    }
    return engineMaps_[engineName]();
}

std::vector<std::string> EngineFactory::getAvailableEngines()
{
    std::vector<std::string> engines;
    for (const auto& [fst, snd] : engineMaps_)
    {
        engines.push_back(fst);
    }
    return engines;
}


void EngineFactory::registerEngine(const std::string& engineName, const EngineBuilder& engineBuilder)
{
    engineMaps_[engineName] = engineBuilder;
}

void EngineFactory::registerAllEngines()
{
#if USE_FRUGALLY_DEEP
    registerEngine("FrugallyDeep", [] { return std::make_unique<FrugallyDeepEngineSequential>(); });
#endif

#if USE_TENSORRT
    registerEngine("TensorRT", [] { return std::make_unique<TensorRTEngineSequential>(); });
#endif

#if USE_LIBTORCH
    registerEngine("LibTorch", [] { return std::make_unique<LibtorchEngineSequential>(); });
#endif
}
