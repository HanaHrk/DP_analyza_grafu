#pragma once


#include <opencv2/opencv.hpp>

cv::Mat load_image(const std::string &path);

cv::Mat modify_image(const cv::Mat &image, int width, int height);

std::vector<float> to_vector_input(const cv::Mat &image, int width, int height);
