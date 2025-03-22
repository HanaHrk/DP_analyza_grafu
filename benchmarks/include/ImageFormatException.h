#pragma once

#include <exception>
#include <string>

class ImageFormatException : public std::exception {
std::string message;
  public:
    ImageFormatException(const std::string& message): message(message) {
    }

};