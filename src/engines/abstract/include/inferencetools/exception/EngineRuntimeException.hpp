#pragma once

#include <stdexcept>
#include <utility>

class EngineRuntimeException : public std::exception
{
private:
    const std::string message;

public:
    explicit EngineRuntimeException(std::string message): message(std::move(message))
    {
    }

    [[nodiscard]] const char* what() const override
    {
        return message.c_str();
    }
};
