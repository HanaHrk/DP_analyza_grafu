#include "AbstractInference.h"
#include "ImageUtils.h"

std::vector<float> AbstractInference::convert_image(const cv::Mat& image, const uint32_t width, const uint32_t height)
{
    return to_vector_input(modify_image(image, static_cast<int>(width), static_cast<int>(height)),
                           static_cast<int>(width), static_cast<int>(height));
}

std::chrono::time_point<std::chrono::steady_clock> AbstractInference::get_time()
{
    return std::chrono::steady_clock::now();
}

float AbstractInference::get_elapsed_time(const std::chrono::time_point<std::chrono::steady_clock>& start)
{
    return std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
}
