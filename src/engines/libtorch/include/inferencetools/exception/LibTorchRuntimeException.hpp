#pragma once

#include <stdexcept>
#include <utility>

class LibTorchRuntimeException final : public std::exception
{
    std::string message;

public:
    LibTorchRuntimeException() = default;

    explicit LibTorchRuntimeException(std::string message): message(std::move(message))
    {
    }

    [[nodiscard]] const char* what() const noexcept override
    {
        return message.c_str();
    }
};
