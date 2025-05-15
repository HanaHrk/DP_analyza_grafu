#pragma once

#include <stdexcept>
#include <utility>

/**
 * @class EngineRuntimeException
 * @brief Represents an exception that occurs within the engine's runtime.
 *
 * This class inherits from std::exception and provides a mechanism
 * to handle runtime errors specific to the engine. It stores and
 * retrieves a descriptive message for the occurring error.
 */
class EngineRuntimeException final : public std::exception
{
    /**
     * A constant string that stores the error message associated with the
     * exception. This message provides additional details about the runtime
     * error encountered in the engine.
     */
    const std::string message;

    /**
     * Constructs an EngineRuntimeException with a detailed error message.
     *
     * @param message A string containing the error message that describes the exception.
     * @return An instance of the EngineRuntimeException with the provided error message.
     */
public:
    explicit EngineRuntimeException(std::string message): message(std::move(message))
    {
    }

    /**
     * Retrieves the explanatory message of the exception.
     *
     * @return A constant pointer to a null-terminated character sequence (C-string) representing the exception message.
     */
    [[nodiscard]] const char* what() const noexcept override
    {
        return message.c_str();
    }
};
