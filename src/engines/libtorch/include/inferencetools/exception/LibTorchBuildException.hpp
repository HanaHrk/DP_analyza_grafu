#pragma once

#include <exception>
#include <string>
#include <utility>

class LibTorchBuildException final : public std::exception
{
public:
    LibTorchBuildException() = default;

    explicit LibTorchBuildException(std::string message) : message(std::move(message))
    {
    }

    [[nodiscard]] const char* what() const noexcept override
    {
        return message.c_str();
    }

private:
    std::string message;
};
