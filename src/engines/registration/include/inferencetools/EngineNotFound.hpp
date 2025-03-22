#pragma once
#include <exception>
#include <string>

class EngineNotFound: public std::exception {
private:
  std::string engine_name_;

public:

  EngineNotFound(const std::string& engine_name) : engine_name_(engine_name) {}

  const char* what() const noexcept override {
    return ("Engine not found: " + engine_name_).c_str();
  }
};