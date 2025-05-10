#include <DataLoader.hpp>
#include <ImageFormatException.h>
#include <filesystem>
#include <random>


cv::Mat sample::preprocessImage(const cv::Mat& image, const int width, const int height, bool normalize)
{
    if (width <= 0 || height <= 0)
    {
        throw ImageFormatException("Invalid image dimensions.");
    }
    // 1. Resize the image to the target dimensions
    cv::Mat resizedImage;
    resize(image, resizedImage, cv::Size(width, height));

    // 2. Convert to float32 and normalize
    cv::Mat floatImage;
    if (normalize)
    {
        resizedImage.convertTo(floatImage, CV_32F, 1.0 / 255.0);
    }
    else
    {
        resizedImage.convertTo(floatImage, CV_32F, 1.0);
    }

    // 3. Subtract mean and divide by std deviation
    cv::Mat mean = (cv::Mat_<float>(1, 3) << 0.485, 0.456, 0.406);
    cv::Mat stddev = (cv::Mat_<float>(1, 3) << 0.229, 0.224, 0.225);

    std::array<cv::Mat, 3> channels;
    split(floatImage, channels);
    for (int i = 0; i < 3; ++i)
    {
        //channels[i] = (channels[i] - mean.at<float>(0, i)) / stddev.at<float>(0, i);
    }

    // 4. Merge channels back into a single image
    cv::Mat preprocessedImage;
    merge(channels.data(), 3, preprocessedImage);

    return preprocessedImage;
}

std::vector<float> sample::loadToVector(const cv::Mat& image, const bool useBGR)
{
    std::vector<float> data;
    data.reserve(image.channels() * image.cols * image.rows); // Reserve space for 224x224x3 elements

    // Iterate over the image and store pixel values in the vector
    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            const auto& pixel = image.at<cv::Vec3f>(y, x);
            if (useBGR)
            {
                // BGR -> RGB
                data.push_back(pixel[2]); // R channel
                data.push_back(pixel[1]); // G channel
                data.push_back(pixel[0]); // B channel
            }
            else
            {
                // RGB -> RGB
                data.push_back(pixel[0]); // R channel
                data.push_back(pixel[1]); // G channel
                data.push_back(pixel[2]); // B channel
            }
        }
    }

    return data;
}

std::vector<std::string> sample::getAllFiles(const std::filesystem::path& rootPath)
{
    std::vector<std::filesystem::path> allFiles;
    std::vector<std::string> allFilesPaths;
    std::function<void(const std::filesystem::path&)> findFiles =
        [&](const std::filesystem::path& currentPath)
    {
        for (const auto& entry : std::filesystem::directory_iterator(currentPath))
        {
            if (std::filesystem::is_regular_file(entry.status()))
            {
                allFiles.push_back(entry.path());
                auto path = std::filesystem::absolute(entry).string();
                std::replace(path.begin(), path.end(), '\\', '/');
                allFilesPaths.push_back(path);
            }
            else if (std::filesystem::is_directory(entry.status()))
            {
                findFiles(entry.path());
            }
        }
    };
    findFiles(rootPath);
    return allFilesPaths;
}

std::string sample::getRandomFilePath(const std::filesystem::path& folderPath)
{
    if (!std::filesystem::exists(folderPath) || !std::filesystem::is_directory(folderPath))
    {
        throw std::runtime_error("Folder does not exist");
    }

    std::vector<std::filesystem::path> allFiles;
    std::function<void(const std::filesystem::path&)> findFiles =
        [&](const std::filesystem::path& currentPath)
    {
        for (const auto& entry : std::filesystem::directory_iterator(currentPath))
        {
            if (std::filesystem::is_regular_file(entry.status()))
            {
                allFiles.push_back(entry.path());
            }
            else if (std::filesystem::is_directory(entry.status()))
            {
                findFiles(entry.path());
            }
        }
    };

    findFiles(folderPath);

    if (allFiles.empty())
    {
        throw std::runtime_error("No files found");
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, allFiles.size() - 1);
    auto absolutePath = std::filesystem::absolute(allFiles[distrib(gen)]).string();
    std::replace(absolutePath.begin(), absolutePath.end(), '\\', '/');
    return absolutePath;
}
