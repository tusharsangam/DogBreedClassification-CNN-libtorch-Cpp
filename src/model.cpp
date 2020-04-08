#include "pch.h"
#include "model.h"
#include "transforms.h"
#include <fstream>

Sequential make_feature_extractor( size_t level, size_t step)
{
	size_t kernel = 3;
	size_t pad = 1;
	size_t input_channel = 3;
	size_t out_channel = step;
	Sequential features;
	for (size_t i = 1; i < level + 1; i++)
	{
		features->push_back(Conv2d(Conv2dOptions(input_channel, out_channel, kernel).padding(1)));
		features->push_back(ReLU{ true });
		features->push_back(MaxPool2d(MaxPool2dOptions(2).stride(2)));
		features->push_back(BatchNorm2d(BatchNorm2dOptions(out_channel)));
		input_channel = out_channel;
		out_channel *= 2;

	}
	features->push_back(AdaptiveAvgPool2d(AdaptiveAvgPool2dOptions({ 1,1 })));
	return features;
}



cnnNetImpl::cnnNetImpl(std::string savedir):
	savepath(savedir),
	features(register_module("features", make_feature_extractor(5, 16))),
	dp(register_module("dropout", Dropout(DropoutOptions(0.25)))),
	inference(register_module("inference", Linear(256, 133)))
{
	savepath += "/cnnNet-weights.bin";
}



torch::Tensor cnnNetImpl::forward(torch::Tensor x)
{	
	x = features->forward(x);
	x = x.reshape({ x.size(0), x.size(1) * x.size(2) * x.size(3) });
	x = dp->forward(x);
	x = inference->forward(x);
	return x;
}

void cnnNetImpl::load()
{
	torch::autograd::GradMode::set_enabled(false);
	auto params = named_parameters(true);
	auto buffers = named_buffers(true);
	auto start = std::chrono::high_resolution_clock::now();
	std::ifstream fs(savepath, std::ios::binary);
	if (fs.is_open()) {
		for (auto& pair : params) {
			
			auto* p = params.find(pair.key());
			long length = p->numel() * p->storage().elementSize();
			char* bytearray = new char[length];
			fs.read(bytearray, length);
			auto tensor = torch::from_blob(reinterpret_cast<void*>(bytearray), p->sizes(), p->options());
			p->copy_(tensor);
			
		}
		for (auto& pair : buffers) {
			auto* p = buffers.find(pair.key());
			long length = p->numel() * p->storage().elementSize();
			char* bytearray = new char[length];
			fs.read(bytearray, length);
			auto tensor = torch::from_blob(reinterpret_cast<void*>(bytearray), p->sizes(), p->options());
			p->copy_(tensor);
			
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Weight Loading took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << "\n";
	}
	else {
		std::cout << "Error reading file" << savepath << std::endl;	
	}
	fs.close();
	torch::autograd::GradMode::set_enabled(true);
}

void cnnNetImpl::save(bool ismodeloncuda)
{
	if (ismodeloncuda) {
		to(torch::kCPU);
	}
	torch::autograd::GradMode::set_enabled(false);
	auto params = named_parameters(true);
	auto buffers = named_buffers(true);
	auto start = std::chrono::high_resolution_clock::now();
	std::ofstream wf;
	wf.open(savepath, std::ios::trunc | std::ios::out | std::ios::binary);
	if (wf.is_open()) {
		for (auto& pair : params) {
			//std::cout << "Trying to dump " << pair.key() << "\n";
			auto* p = params.find(pair.key());
			wf.write(reinterpret_cast<char*>(p->data_ptr()), p->numel() * p->storage().elementSize());
		}
		for (auto& pair : buffers) {
			//std::cout << "Trying to dump " << pair.key() << "\n";
			auto* p = buffers.find(pair.key());
			wf.write(reinterpret_cast<char*>(p->data_ptr()), p->numel() * p->storage().elementSize());
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Weight Dumping took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << "\n";
	}
	else {
		std::cout << "Error in opening file for writing" << std::endl;
		
	}
	wf.close();
	torch::autograd::GradMode::set_enabled(true);
	if (ismodeloncuda) {
		to(torch::kCUDA);
	}
}


std::vector<int>vgg16featureconfiguration{ 64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1 };




Sequential make_features() {
	Sequential features;
	int64_t input_channel = 3;
	int64_t kernel = 3;
	int64_t stride = 1;
	int64_t pad = 1;

	for (int& v : vgg16featureconfiguration) {
		if (v == -1) {
			features->push_back(MaxPool2d(MaxPool2dOptions(2).stride(2)));
		}
		else {

			features->push_back(Conv2d(Conv2dOptions(input_channel, v, kernel).padding(pad).stride(stride)));
			features->push_back(ReLU(ReLUOptions(true)));
			input_channel = v;
		}

	}
	return features;
}

torch::Tensor VGG16Impl::forward(torch::Tensor input) {
	//std::cout << "Insider forward" << std::endl;
	//std::cout << input << std::endl;
	input = features->forward(input);
	//std::cout << "Input Features of size " << input.size(0) << " x " << input.size(1) << " x " << input.size(2) << " x " << input.size(3) << std::endl;
	input = avgpool->forward(input);
	//std::cout << "Input Features of size " << input.size(0) << " x " << input.size(1) << " x " << input.size(2) << " x " << input.size(3) << std::endl;
	input = input.view({ input.size(0), -1 });
	//std::cout << "input dimensions are " << input.size(0)<< " x "<< input.size(1) << std::endl;
	input = classifier->forward(input);

	return input;

}



void VGG16Impl::load()
{
	torch::autograd::GradMode::set_enabled(false);	
	//std::cout << "Yah ";
	auto start = std::chrono::high_resolution_clock::now();
	std::ifstream fs(filename, std::ios::binary);
	if (fs) {
		auto params = named_parameters(true);
		for (auto& pair : params) {
			auto name = pair.key();
			auto* p = params.find(name);
			//std::cout << p->storage().elementSize() << std::endl;
			long length = p->numel() * p->storage().elementSize();
			char* bytearray = new char[length];
			fs.read(bytearray, length);
			auto tensor = torch::from_blob(reinterpret_cast<void*>(bytearray), p->sizes(), torch::TensorOptions().dtype(torch::kFloat32));
			p->set_data(tensor);
			p->requires_grad_(false);
		}
		fs.close();
		torch::autograd::GradMode::set_enabled(true);
		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Weight Loading took " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()<< " s" << "\n";
	}
	else {
		std::cout << "Error reading file" << filename << std::endl;
		
	}
	fs.close();
	torch::autograd::GradMode::set_enabled(true);
	//
	return;
}

void VGG16Impl::save() {
	torch::autograd::GradMode::set_enabled(false);
	auto params = named_parameters(true);
	auto buffers = named_buffers(true);
	auto start = std::chrono::high_resolution_clock::now();
	std::ofstream wf;
	wf.open(filename, std::ios::trunc | std::ios::binary);
	if (wf.is_open()) {
		for (auto& pair : params) {
			//std::cout << "Trying to dump " << pair.key() << "\n";
			auto* p = params.find(pair.key());
			wf.write(reinterpret_cast<char*>(p->data_ptr()), p->numel() * p->storage().elementSize());
		}
		for (auto& pair : buffers) {
			//std::cout << "Trying to dump " << pair.key() << "\n";
			auto* p = buffers.find(pair.key());
			wf.write(reinterpret_cast<char*>(p->data_ptr()), p->numel() * p->storage().elementSize());
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Weight Dumping took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << "\n";
	}
	else {
		std::cout << "Error in opening file for writing" << std::endl;

	}
	wf.close();
	torch::autograd::GradMode::set_enabled(true);
}



torch::Tensor TransferLearningImpl::forward(torch::Tensor input)
{
	auto convop = features->forward(input);
	
	
	//std::cout << "Input Features of size " << input.size(0) << " x " << input.size(1) << " x " << input.size(2) << " x " << input.size(3) << std::endl;
	auto avgpoolop = avgpool->forward(convop);
	//std::cout << avgpoolop;
	//std::cout << "Input Features of size " << input.size(0) << " x " << input.size(1) << " x " << input.size(2) << " x " << input.size(3) << std::endl;
	avgpoolop = avgpoolop.view({ input.size(0), -1 });
	//std::cout << "input dimensions are " << input.size(0)<< " x "<< input.size(1) << std::endl;
	//input = classifier->forward(input);
	auto output = last_layer->forward(avgpoolop);

	return output;
}

void TransferLearningImpl::load()
{
	torch::autograd::GradMode::set_enabled(false);
	std::ifstream fs(featuresfilename, std::ios::binary);
	if (fs) {
		auto params = features->named_parameters(true);
		auto start = std::chrono::high_resolution_clock::now();
		for (auto& pair : params) {
			auto name = pair.key();
			auto* p = params.find(name);
			long length = p->numel() * p->storage().elementSize();
			char* bytearray = new char[length];
			fs.read(bytearray, length);
			auto tensor = torch::from_blob(reinterpret_cast<void*>(bytearray), p->sizes(), p->options());
			p->copy_(tensor);
			p->requires_grad_(false);
		}
		/*params = classifier->named_parameters(true);
		for (auto& pair : params) {
			auto name = pair.key();
			auto* p = params.find(name);
			long length = p->numel() * p->storage().elementSize();
			char* bytearray = new char[length];
			fs.read(bytearray, length);
			auto tensor = torch::from_blob(reinterpret_cast<void*>(bytearray), p->sizes(), p->options());
			p->copy_(tensor);
			//p->requires_grad_(false);
		}*/
		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Features Loading took " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " s" << "\n";
	}
	else {
		std::cout << "Error reading features file" << featuresfilename << std::endl;
	}
	fs.close();

	
	fs.open(classifierfilename, std::ios::binary);
	if (fs) {
		auto params = last_layer->named_parameters(true);
		auto buffers = named_buffers(true);
		auto start = std::chrono::high_resolution_clock::now();
		
		for (auto& pair : params) {
			auto name = pair.key();
			auto* p = params.find(name);
			long length = p->numel() * p->storage().elementSize();
			char* bytearray = new char[length];
			fs.read(bytearray, length);
			auto tensor = torch::from_blob(reinterpret_cast<void*>(bytearray), p->sizes(), p->options());
			p->set_data(tensor);
		}
		
		for (auto& pair : buffers) {
			auto name = pair.key();
			auto* p = params.find(name);
			long length = p->numel() * p->storage().elementSize();
			char* bytearray = new char[length];
			fs.read(bytearray, length);
			auto tensor = torch::from_blob(reinterpret_cast<void*>(bytearray), p->sizes(), p->options());
			p->set_data(tensor);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Classifier Loading took " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " s" << "\n";
	}
	fs.close();

	torch::autograd::GradMode::set_enabled(true);
}

void TransferLearningImpl::save(bool isongpu)
{	
	if (isongpu) {
		last_layer->to(torch::kCPU);
	}
	torch::autograd::GradMode::set_enabled(false);
	auto params = last_layer->named_parameters(true);
	auto buffers = named_buffers(true);
	auto start = std::chrono::high_resolution_clock::now();
	std::ofstream wf;
	wf.open(classifierfilename, std::ios::trunc | std::ios::binary);
	if (wf.is_open()) {
		for (auto& pair : params) {
			auto* p = params.find(pair.key());
			wf.write(reinterpret_cast<char*>(p->data_ptr()), p->numel() * p->storage().elementSize());
		}
		for (auto& pair : buffers) {
			//std::cout << "Trying to dump " << pair.key() << "\n";
			auto* p = buffers.find(pair.key());
			wf.write(reinterpret_cast<char*>(p->data_ptr()), p->numel() * p->storage().elementSize());
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Weight Dumping took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << "\n";
	}
	else {
		std::cout << "Error in opening file for writing" << std::endl;

	}
	wf.close();
	torch::autograd::GradMode::set_enabled(true);
	if (isongpu) {
		last_layer->to(torch::kCUDA);
	}
	

}

void TransferLearningImpl::freeze_features()
{
	auto params = features->named_parameters(true);
	for (auto& pair : params) {
		auto* p = params.find(pair.key());
		p->requires_grad_(false);
	}
	/*params = classifier->named_parameters(true);
	for (auto& pair : params) {
		auto* p = params.find(pair.key());
		p->requires_grad_(false);
	}*/
}