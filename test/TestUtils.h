#pragma once

#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include "ImageUtils.h"
#include "FileUtils.h"

inline std::pair<std::string, std::string> parse_argument(const std::string &arg) {
    if (arg.size() < 4 || arg.substr(0, 2) != "--") {
        return {"", ""}; // Invalid format
    }

    const size_t split_position = arg.find('=');
    size_t value_offset = 0;
    if (split_position == std::string::npos || split_position <= 2) {
        return {"", ""}; // Invalid format
    }
    const size_t start_quote = arg.find('"', split_position + 1);
    if (start_quote == split_position + 1) {
        value_offset = 1;
    }

    std::string name = arg.substr(2, split_position - 2);
    std::string value = arg.substr(split_position + 1 + value_offset, arg.size() - 1 - value_offset);
    return {name, value};
}

inline std::map<std::string, std::string> get_args(const int argv, char *argc[]) {
    std::map<std::string, std::string> arguments;
    for (int i = 1; i < argv; i++) {
        const auto arg = parse_argument(argc[i]);
        if (!arg.first.empty()) {
            arguments[arg.first] = arg.second;
        }
    }
    return arguments;
}

inline std::vector<cv::Mat> load_images(const std::string &data_root) {
    const std::vector<std::string> image_paths = find_all_images(data_root);
    std::vector<cv::Mat> images;
    images.reserve(image_paths.size());
    for (const auto &image_path: image_paths) {
        images.push_back(load_image(image_path));
    }
    return images;
}
