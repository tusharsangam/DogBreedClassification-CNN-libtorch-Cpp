#pragma once
#include "pch.h"
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <time.h>
#include <stdlib.h>


void plotImage(cv::Mat& image);
cv::Mat read_image(const std::string& path);
torch::Tensor convert_to_tensor(cv::Mat& image);
void Resize(int newsize, cv::Mat& image);
void CenterCrop(const int cropSize, cv::Mat& image);
void RandomRotate(double angle, cv::Mat& src);
void RandomHorizontalFlip(int p, cv::Mat& image);




