#include "pch.h"
#include "dataloader.h"






DataLoader::DataLoader(DogBreedDataset& _dataset, size_t batchsize):dataset(_dataset), sampler(dataset.size()), batchsize(batchsize)
{
}


torch::optional<Example> DataLoader::next()
{	
	torch::optional<std::vector<size_t>> batchindices = sampler.next(batchsize);
	if (batchindices) {
		return std::move(dataset.get_batch(batchindices.value(), sampler.index()/batchsize));
	}
	else {
		return {};
	}
}

void DataLoader::reset()
{
	sampler.reset();
}
