#pragma once
#include <memory>
#include <string>
#include <inferencetools/InferenceEngine.hpp>


class EngineFactory
{
    using EnginePtr = std::unique_ptr<InferenceEngineSequential>;

    using EngineBuilder = std::function<EnginePtr()>;

public:
    static void registerAllEngines();

    static std::vector<std::string> getAvailableEngines();

    static EnginePtr findEngine(const std::string& engineName);

private:
    static void registerEngine(const std::string& engineName, const EngineBuilder& engineBuilder);

    struct EngineMapNode
    {
        const std::string& engineName_;
        const EngineBuilder& engineBuilder_;

        EngineMapNode(const std::string& engineName, const EngineBuilder& engineBuilder) :
            engineName_(engineName), engineBuilder_(engineBuilder)
        {
        }
    };

    static std::map<std::string, EngineBuilder> engineMaps_;
};
