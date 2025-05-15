#pragma once
#include <memory>
#include <string>
#include <functional>
#include <map>
#include <inferencetools/InferenceEngine.hpp>


/**
 * @brief Factory class responsible for creating and managing engine instances.
 *
 * The `EngineFactory` class provides methods for generating engine objects,
 * ensuring they are instantiated and initialized consistently. It serves as
 * a centralized point for managing engine creation logic.
 */
class EngineFactory
{
    using EnginePtr = std::unique_ptr<InferenceEngineSequential>;

    using EngineBuilder = std::function<EnginePtr()>;

    /**
     * @brief Registers all available engine implementations with the system.
     *
     * The `registerAllEngines` function ensures that all supported engines are
     * initialized and made ready for subsequent use within the application.
     */
public:
    static void registerAllEngines();

    /**
     * @brief Retrieves a list of available engine names.
     *
     * This method provides the names of all engines that are currently available.
     *
     * @return A list of strings representing the names of the available engines.
     */
    static std::vector<std::string> getAvailableEngines();

    /**
     * @brief Searches for and retrieves an engine associated with the given identifier.
     *
     * The `findEngine` method locates an engine based on the specified identifier
     * and returns it if found, or performs necessary handling if not.
     *
     * @param engineId The unique identifier of the engine to locate.
     * @return The engine object associated with the provided identifier, or a null/empty value if not found.
     */
    static EnginePtr findEngine(const std::string& engineName);

    /**
     * @brief Registers an engine with a specific identifier and its corresponding builder.
     *
     * This method associates the provided engine name with a builder object, enabling the system
     * to construct instances of the engine using the specified builder.
     *
     * @param engineName The name of the engine to be registered.
     * @param builder The builder responsible for constructing the engine instance.
     */
private:
    static void registerEngine(const std::string& engineName, const EngineBuilder& engineBuilder);

    /**
     * @brief Represents a mapping node that associates an engine's name with its corresponding builder.
     *
     * The `EngineMapNode` structure is used to encapsulate a pair consisting of an engine name
     * and an engine builder function. This association is used within engine management systems
     * to register and retrieve available inference engines dynamically.
     */
    struct EngineMapNode
    {
        /**
         * @brief Reference to the name of the inference engine.
         *
         * This variable holds a constant reference to the name of a specific inference engine.
         * It is used to uniquely identify the engine in operations such as engine registration
         * and retrieval in the engine factory mechanism. The engine name must be unique within
         * the scope of the engine factory to ensure correct mapping and functionality.
         *
         * @note The lifetime of the referenced string must outlive the usage of this variable.
         */
        const std::string& engineName_;
        /**
         * @brief Represents a reference to a function that constructs instances of `InferenceEngineSequential`.
         *
         * This member variable holds a reference to a callable object, which is used to build and instantiate
         * engines for inference tasks. The callable encapsulates the logic for creating a specific type
         * of `InferenceEngineSequential` object, enabling dynamic and flexible engine creation at runtime.
         *
         * The `engineBuilder_` is typically utilized within a structure or a factory mechanism to register
         * and manage available engine types for querying and instantiation.
         *
         * @note The actual implementation of the callable must ensure the proper construction
         *       of `InferenceEngineSequential` objects, potentially including initialization of required
         *       configurations or loading of necessary model files.
         */
        const EngineBuilder& engineBuilder_;

        /**
         * @brief Constructs an EngineMapNode with the specified engine name and engine builder.
         *
         * This constructor initializes an instance of EngineMapNode, which encapsulates the
         * engine name and its corresponding builder function.
         *
         * @param engineName The name of the engine associated with this map node.
         * @param engineBuilder A reference to the engine builder function which can
         *                      create instances of the engine.
         */
        EngineMapNode(const std::string& engineName, const EngineBuilder& engineBuilder) :
            engineName_(engineName), engineBuilder_(engineBuilder)
        {
        }
    };

    /**
     * @brief Stores a collection that maps engine identifiers to their respective configuration or behavior details.
     *
     * The `engineMaps_` variable serves as a centralized repository for managing associations
     * between engine names or keys and their relevant configurations, mappings, or handlers.
     */
    static std::map<std::string, EngineBuilder> engineMaps_;
};
