#pragma once

#include <stdexcept>
#include <utility>

class TensorRTRuntimeException final : public std::exception
{
    std::string message;

public:
    TensorRTRuntimeException() = default;

    explicit TensorRTRuntimeException(std::string message): message(std::move(message))
    {
    }

    [[nodiscard]] const char* what() const noexcept override
    {
        return message.c_str();
    }
};
