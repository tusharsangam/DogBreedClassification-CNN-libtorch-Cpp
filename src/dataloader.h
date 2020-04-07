#pragma once
#include "pch.h"
#include "dogbreeddataset.h"



class DataLoader {

private:
	DogBreedDataset dataset;
	size_t batchsize;
	torch::data::samplers::RandomSampler sampler;
public:
	explicit DataLoader(DogBreedDataset& dataset, size_t batchsize);
	torch::optional<Example> next();
	void reset();
};


