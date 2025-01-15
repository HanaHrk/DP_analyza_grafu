#pragma once

#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include "ImageUtils.h"
#include "FileUtils.h"

/**
 * Parses a single command-line argument in the format '--key=value'
 * and extracts the key-value pair.
 *
 * If the argument does not follow the expected format, an empty pair is returned.
 *
 * @param arg The command-line argument string to be parsed.
 * @return A pair where the first element is the key (without '--'), and the second element
 *         is the value. Returns {"", ""} if the argument is invalid or poorly formatted.
 */
inline std::pair<std::string, std::string> parse_argument(const std::string& arg)
{
    if (arg.size() < 4 || arg.substr(0, 2) != "--")
    {
        return {"", ""}; // Invalid format
    }

    const size_t split_position = arg.find('=');
    size_t value_offset = 0;
    if (split_position == std::string::npos || split_position <= 2)
    {
        return {"", ""}; // Invalid format
    }
    const size_t start_quote = arg.find('"', split_position + 1);
    if (start_quote == split_position + 1)
    {
        value_offset = 1;
    }

    std::string name = arg.substr(2, split_position - 2);
    std::string value = arg.substr(split_position + 1 + value_offset, arg.size() - 1 - value_offset);
    return {name, value};
}

/**
 * Parses command-line arguments into a map of key-value pairs.
 * Only arguments in the format '--key=value' are considered valid.
 *
 * @param argv The number of command-line arguments.
 * @param argc The array of command-line argument strings.
 * @return A map where each key is an argument name (without '--') and each value is the associated argument value.
 */
inline std::map<std::string, std::string> get_args(const int argv, char* argc[])
{
    std::map<std::string, std::string> arguments;
    for (int i = 1; i < argv; i++)
    {
        const auto arg = parse_argument(argc[i]);
        if (!arg.first.empty())
        {
            arguments[arg.first] = arg.second;
        }
    }
    return arguments;
}

/**
 * Loads all image files from a specified directory and returns them as a vector of cv::Mat objects.
 *
 * @param data_root The root directory containing the images to be loaded.
 * @return A vector of loaded images represented as cv::Mat objects.
 */
inline std::vector<cv::Mat> load_images(const std::string& data_root)
{
    const std::vector<std::string> image_paths = find_all_images(data_root);
    std::vector<cv::Mat> images;
    images.reserve(image_paths.size());
    for (const auto& image_path : image_paths)
    {
        images.push_back(load_image(image_path));
    }
    return images;
}

template <template<class,class,class...> class C, typename K, typename V, typename... Args>
/**
 * Retrieves the value associated with a given key from a container. If the key
 * does not exist in the container, a default value is returned.
 *
 * @param m The container from which to retrieve the value. It must support
 *          a `find` method and provide an iterator to key-value pairs.
 * @param key The key to search for in the container.
 * @param defval The default value to return if the key is not found.
 * @return The value associated with the specified key if it exists in the container;
 *         otherwise, the provided default value.
 */
V get_with_def(const C<K, V, Args...>& m, K const& key, const V& defval)
{
    typename C<K, V, Args...>::const_iterator it = m.find(key);
    if (it == m.end())
        return defval;
    return it->second;
}

/**
 * Checks if all required keys exist in the provided map of arguments.
 *
 * @param args A map containing key-value pairs of arguments.
 * @param required_keys A vector of keys that are required to be present in the map.
 * @return True if all required keys are present in the map, otherwise false.
 */
inline bool has_args(const std::map<std::string, std::string>& args,
                     const std::vector<std::string>& required_keys)
{
    return std::all_of(required_keys.begin(), required_keys.end(), [&args](const auto& key)
    {
        return args.find(key) != args.end();
    });
}
