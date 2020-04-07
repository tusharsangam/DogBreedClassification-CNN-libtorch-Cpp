#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

struct Example {
	torch::Tensor data;
	torch::Tensor target;
	int batch_idx;
	Example() {}
	Example(torch::Tensor& data, torch::Tensor& target, int batch_idx = -1) :data(data), target(target), batch_idx(batch_idx) {}
};