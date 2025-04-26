#pragma once

#include <exception>
#include <string>
#include <utility>

class TensorRTBuildException final : public std::exception
{
public:
    TensorRTBuildException() = default;

    explicit TensorRTBuildException(std::string message) : message(std::move(message))
    {
    }

    [[nodiscard]] const char* what() const noexcept override
    {
        return message.c_str();
    }

private:
    std::string message;
};
