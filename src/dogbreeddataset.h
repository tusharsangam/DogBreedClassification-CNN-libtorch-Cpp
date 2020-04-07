#pragma once
#include "pch.h"
#include "transforms.h"





std::string& getClassNameByClassId(int64 index);
 


class DogBreedDataset 
{
    //friend std::string& getClassNameByClassId(int index);
    
    private:
        std::vector<std::string> _image_files_list;
        std::vector<int> _labels_list;
        void (*transform_lambda)(cv::Mat& image);
        size_t numberofworkers;
       
    public:
        //override methods parent class
        explicit DogBreedDataset(std::string type, void (*transform_func)(cv::Mat& image));
        
        Example get(size_t index);
        
        std::vector<Example> get_minibatch(std::vector<size_t>::iterator start, std::vector<size_t>::iterator end);
        
        Example get_batch(std::vector<size_t>& indices, int batch_idx);
        
        int64_t size()const ;
        
        void setTransforms(void (*transform_func)(cv::Mat& image));
    private:
        //my methods and customizations 
        void read_directory(const std::string& name);
        int extractNumberFromString(std::string& convert);
};

/*auto testdataset = torch::data::datasets::make_shared_dataset<DogBreedDataset>(DogBreedDataset("test", lambda_transform_valid)).map(torch::data::transforms::Stack<>());



    //{ 0.485,0.456,0.406 }, { 0.229,0.224,0.225 }
    //torch::data::datasets::make_shared_dataset<DogBreedDataset>(validationdataset);

    //auto sharedtrainingset = torch::data::datasets::make_shared_dataset<DogBreedDataset>(trainingdataset)

    auto data_loader = torch::data::make_data_loader( std::move(trainingdataset), torch::data::DataLoaderOptions(kBatchSize).workers(4));

    auto validation_data_loader = torch::data::make_data_loader(std::move(validationdataset), torch::data::DataLoaderOptions(kBatchSize).workers(4));

    auto test_data_loader = torch::data::make_data_loader(std::move(testdataset), torch::data::DataLoaderOptions(kBatchSize).workers(4));
    */