#include "pch.h"
#include <iostream>
#include <fstream>


const char* exporter(torch::Tensor& originaltensor) {
	//auto originaltensor = torch::randn({ 10,56 }, torch::kFloat);
	const char* filename = "..\\python\\model.weight.bin";
	std::ofstream wf;
	wf.open(filename, std::ios::out | std::ios::binary);
	if (wf.is_open()) {
		std::cout << originaltensor.storage().elementSize() << std::endl;
		wf.write(reinterpret_cast<char*>(originaltensor.data_ptr()), originaltensor.numel()*originaltensor.storage().elementSize());
		wf.close();
		std::cout << "file written\n";
		return filename;
	}

}

torch::Tensor importer(const char* filename, c10::IntArrayRef sizes, torch::TensorOptions& options) {
	std::ifstream fs(filename, std::ios::binary);
	if (fs) {
		fs.seekg(0, fs.end);
		int length = fs.tellg();
		fs.seekg(0, fs.beg);
		char* bytearray = new char[length];
		fs.read(bytearray, length);
		fs.close();
		torch::Tensor tensor = torch::from_blob(reinterpret_cast<void*>(bytearray), sizes, options);
		std::cout << "file read" << std::endl;
		return std::move(tensor);
	}
}

