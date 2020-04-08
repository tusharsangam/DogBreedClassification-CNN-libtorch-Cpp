#pragma once
#include "pch.h"

using namespace torch::nn;

Sequential make_feature_extractor(size_t level=5, size_t step=16);



struct cnnNetImpl : Module {
	Sequential features;
	Linear inference;
	Dropout dp;
	std::string savepath;
	cnnNetImpl(std::string savedir);
	torch::Tensor forward(torch::Tensor x);
	void load();
	void save(bool isoncuda);
	
};



Sequential make_features();

//Sequential make_classifier(size_t output_classes);


struct VGG16Impl :Module {
	Sequential features;
	Sequential classifier;
	AdaptiveAvgPool2d avgpool;
	std::string filename;
	VGG16Impl(std::string filedir_) :
		features(register_module("features", make_features())),
		classifier(
			register_module("classifier",
				Sequential(
					Linear(512 * 7 * 7, 4096),
					ReLU(ReLUOptions(true)),
					Dropout(),
					Linear(4096, 4096),
					ReLU(ReLUOptions(true)),
					Dropout(),
					Linear(4096, 1000)
				)
			)
		),
		avgpool(register_module("avgpool", AdaptiveAvgPool2d(AdaptiveAvgPool2dOptions({ 7,7 }))))
	{	
		filename = filedir_+"/vgg16-weights.bin";
		//std::cout << filename << std::endl;
		load();
	}
	torch::Tensor forward(torch::Tensor input);
	void load();
	void save();
	
};


struct TransferLearningImpl : torch::nn::Module {
	Sequential features;
	Sequential classifier;
	AdaptiveAvgPool2d avgpool;
	Sequential last_layer;
	std::string featuresfilename;
	std::string classifierfilename;
	TransferLearningImpl(std::string filename) :
		featuresfilename(filename),
		features(register_module("features", make_features())),
		/*classifier(
			register_module("classifier",
				Sequential(
					Linear(512 * 7 * 7, 4096),
					ReLU(ReLUOptions(true)),
					Dropout(),
					Linear(4096, 4096),
					ReLU(ReLUOptions(true)),
					Dropout()
				)
			)
		),*/
		avgpool(register_module("avgpool", AdaptiveAvgPool2d(AdaptiveAvgPool2dOptions({ 1,1 })))),
		last_layer(register_module("last_layer", Sequential(
			Linear(512 * 1* 1, 256),
			ReLU(ReLUOptions(true)),
			Dropout(0.2),
			Linear(256, 133)
		)
		))
		
	{
		classifierfilename = filename + "/vgg16-transfer-learning-model.bin";
		featuresfilename = filename+"/vgg16-weights.bin";
		load();
		//freeze_features();
	}
	torch::Tensor forward(torch::Tensor input);
	void load();
	void save(bool isongpu);
	void freeze_features();
	
};

TORCH_MODULE(VGG16);
TORCH_MODULE(cnnNet);
TORCH_MODULE(TransferLearning);

//void load_module_weights(VGG16& vgg16);