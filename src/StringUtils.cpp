#include "StringUtils.h"

#include <string>


std::wstring to_wstring(const std::string &str) {
    return std::wstring{str.begin(), str.end()};
}
