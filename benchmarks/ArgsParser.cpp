#include <ArgsParser.hpp>

std::vector<std::string> sample::extractArgs(const int argc, char** argv)
{
    std::vector<std::string> args;
    if (argc > 1)
    {
        args.reserve(argc - 1); // Reserve space for efficiency, excluding program name
        for (int i = 1; i < argc; ++i)
        {
            // Start from 1 to skip program name
            args.emplace_back(argv[i]);
        }
    }
    return args;
}

std::optional<std::vector<std::string>> sample::extractArgsMulti(const std::vector<std::string>& argsVector,
                                                                 const std::string& optionName)
{
    const std::string prefix = "--" + optionName + "=";
    std::vector<std::string> values;

    for (const auto& arg : argsVector)
    {
        if (arg.compare(0, prefix.length(), prefix) == 0)
        {
            values.push_back(arg.substr(prefix.length()));
        }
    }

    if (values.empty())
    {
        return std::nullopt;
    }
    return values;
}


std::optional<std::string> sample::extractArgsSingle(const std::vector<std::string>& argsVector,
                                                     const std::string& optionName)
{
    const std::string prefix = "--" + optionName + "=";

    const auto it = std::find_if(argsVector.begin(), argsVector.end(),
                           [&prefix](const std::string& arg)
                           {
                               return arg.compare(0, prefix.length(), prefix) == 0;
                           });

    if (it != argsVector.end())
    {
        return it->substr(prefix.length());
    }
    return std::nullopt;
}

bool sample::extractArgsSimple(const std::vector<std::string>& argsVector, const std::string& optionName)
{
    const std::string option = "--" + optionName;
    return std::find(argsVector.begin(), argsVector.end(), option) != argsVector.end();
}
