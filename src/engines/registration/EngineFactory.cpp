#include <inferencetools/EngineFactory.hpp>
#include <inferencetools/EngineNotFound.hpp>

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
    registerEngine("FrugallyDeep", [] { return std::make_unique<FrugallyDeepEngine>(); });
#endif
}
