#pragma once
#include <string>
#include <vector>
#include <optional>
#include <algorithm>

namespace sample
{
    /**
     * Extract arguments given from command line and put them into vector.
     * @param argc Number of arguments.
     * @param argv Pointer to arguments.
     * @return Vector of arguments.
     */
    std::vector<std::string> extractArgs(int argc, char** argv);

    /**
     * This function extracts only values for given option.
     * If option is not found, empty optional is returned.
     * Vector of values is created by multiple occurences of option.
     *
     * Options are stored in format:
     * <p><code>--optionName=optionValue</code></p>
     *
     * @param argsVector Vector of command line arguments.
     * @param optionName Name of option to extract.
     * @return Vector of values for option if any found.
     */
    std::optional<std::vector<std::string>> extractArgsMulti(const std::vector<std::string>& argsVector,
                                                             const std::string& optionName);

    /**
     * Extract argument given from command line and return it as string.
     * If option is not found, empty optional is returned.
     *
     * Options are stored in format:
     * <p><code>--optionName=optionValue</code></p>
     *
     * @param argsVector Vector of command line arguments.
     * @param optionName Name of option to extract.
     * @return Value for option if any found.
     */
    std::optional<std::string> extractArgsSingle(const std::vector<std::string>& argsVector,
                                                 const std::string& optionName);

    /**
     * Signalize if option is icluded in argsVector.
     *
     * @param argsVector Vector of command line arguments.
     * @param optionName Name of option to check.
     * @return True of option is found, false otherwise.
     */
    bool extractArgsSimple(const std::vector<std::string>& argsVector, const std::string& optionName);
}
