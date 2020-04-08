#include "pch.h"
#include "transforms.h"
#include "dataloader.h"
#include  "model.h"

template <typename DataLoader, typename Model, typename Optimizer>
void Train(DataLoader& data_loader, DataLoader& test_data_loader, Model& model, Optimizer& optimizer) {
    bool isCUDAAvailable = torch::cuda::is_available();
    size_t epochs = 20;
    double validation_loss_min = INFINITY;

    if (isCUDAAvailable) {
        model->to(torch::kCUDA);
    }

    std::cout << "Starting Training \n";
    for (size_t iteration = 1; iteration < epochs + 1; iteration++) {
        double train_loss = 0.0;
        double valid_loss = 0.0;
        long correct = 0; long total = 0;
        auto start = std::chrono::high_resolution_clock::now();
        //train
        model->train();

        auto batch = data_loader.next();
        while (batch) {
            optimizer.zero_grad();
            auto data = batch.value().data;
            //std::cout << data;
            auto labels = batch.value().target.squeeze_();

            if (isCUDAAvailable) {
                data = data.to(torch::kCUDA);
                labels = labels.to(torch::kCUDA);
            }
            auto output = model->forward(data);

            auto loss = torch::nn::functional::cross_entropy(output, labels, torch::nn::functional::CrossEntropyFuncOptions().ignore_index(-100).reduction(torch::kMean));
            loss.backward();
            optimizer.step();
            loss = loss.to(torch::kCPU);
            int batch_idx = batch.value().batch_idx;
            float* lossf = reinterpret_cast<float*>(loss.data_ptr());
            train_loss = train_loss + ((1 / (batch_idx)) * (*lossf - train_loss));
            //std::cout <<"average loss " <<train_loss << " for batch_idx "<< batch_idx << std::endl;
            batch = data_loader.next();
        }
        //validate
        model->eval();
        auto testbatch = test_data_loader.next();
        while (testbatch) {

            auto data = testbatch.value().data;
            auto labels = testbatch.value().target.squeeze_();
            if (isCUDAAvailable) {
                data = data.to(torch::kCUDA);
                labels = labels.to(torch::kCUDA);
            }
            auto output = model->forward(data);
            auto loss = torch::nn::functional::cross_entropy(output, labels, torch::nn::functional::CrossEntropyFuncOptions().ignore_index(-100).reduction(torch::kMean));
            loss = loss.to(torch::kCPU);
            int batch_idx = testbatch.value().batch_idx;
            float* lossf = reinterpret_cast<float*>(loss.data_ptr());
            valid_loss = valid_loss + ((1 / (batch_idx)) * (*lossf - valid_loss));
            auto pred = output.data().argmax(1);
            auto correctionmatrix = torch::eq(pred, labels);
            auto sum = torch::sum(correctionmatrix).to(torch::kCPU);
            int64_t* sumd = reinterpret_cast<int64_t*>(sum.data_ptr());
            correct += *sumd;
            total += data.size(0);
            //std::cout <<"average loss " << valid_loss << " for batch_idx " << batch_idx << std::endl;
            testbatch = test_data_loader.next();
        }
        std::cout << "\nTest Accuracy: " << (100. * correct / total) << " %  " << correct << " , " << total << "\n";
        std::cout << "Epoch: " << iteration << " \tTraining Loss: " << train_loss << " \tValidation Loss: " << valid_loss << "\n";
        if (valid_loss < validation_loss_min) {
            model->save(isCUDAAvailable);
            validation_loss_min = valid_loss;
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Time took for " << iteration << " th iterations is " << std::chrono::duration_cast<std::chrono::minutes>(end - start).count() << " mins\n";
        data_loader.reset();
        test_data_loader.reset();
    }

}
template <typename DataLoader, typename Model>
void Test(DataLoader& test_data_loader, Model& model) {
    bool isCUDAAvailable = torch::cuda::is_available();
    if (isCUDAAvailable) {
        model->to(torch::kCUDA);
    }
    model->eval();
    std::cout << "Starting Testing \n";
    auto start = std::chrono::high_resolution_clock::now();
    double valid_loss = 0.0;
    long total = 0, correct = 0;
    auto testbatch = test_data_loader.next();
    while (testbatch) {

        auto data = testbatch.value().data;
        auto labels = testbatch.value().target.squeeze_();
        if (isCUDAAvailable) {
            data = data.to(torch::kCUDA);
            labels = labels.to(torch::kCUDA);
        }
        auto output = model->forward(data);
        auto loss = torch::nn::functional::cross_entropy(output, labels, torch::nn::functional::CrossEntropyFuncOptions().ignore_index(-100).reduction(torch::kMean));
        loss = loss.to(torch::kCPU);
        int batch_idx = testbatch.value().batch_idx;
        float* lossf = reinterpret_cast<float*>(loss.data_ptr());
        valid_loss = valid_loss + ((1 / (batch_idx)) * (*lossf - valid_loss));
        
        auto pred = output.data().argmax(1);
        
        
        auto correctionmatrix = torch::eq(pred, labels);
        auto sum = torch::sum(correctionmatrix).to(torch::kCPU);
        int64_t* sumd = reinterpret_cast<int64_t*>(sum.data_ptr());
        correct += *sumd;
        total += data.size(0);
        testbatch = test_data_loader.next();
    }
    std::cout << "\nTest Accuracy: " << (100. * correct / total) << " %  " << correct << " , " << total << "\n";
    std::cout << " Average Validation Loss: " << valid_loss << "\n";
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time took for test is " << std::chrono::duration_cast<std::chrono::minutes>(end - start).count() << " mins\n";
}

template <typename Model>
void Evaluate(Model& model, bool ispretrained=false) {
    bool isCUDAAvailable = torch::cuda::is_available();
    boost::filesystem::path p("../evaluation");
    boost::filesystem::directory_iterator start(p);
    boost::filesystem::directory_iterator end;
    std::vector<std::string> filenames;
    for (auto it = start; it != end; ++it) {
        if (boost::filesystem::is_regular_file(it->path())) {
            filenames.emplace_back(it->path().string());
        }
    }
    auto normalize = torch::data::transforms::Normalize<>({ 0.485, 0.456, 0.406 }, { 0.229, 0.224, 0.225 });
    torch::nn::Softmax smax(torch::nn::SoftmaxOptions(1));
    if (!filenames.empty()) {
        for (std::string& filename : filenames) {
            //auto output = model->predict(filename);
            auto image = read_image(filename);
            Resize(224, image);
            auto image_tensor = convert_to_tensor(image);
            
            image_tensor = normalize(image_tensor);
            image_tensor = image_tensor.unsqueeze_(0);
            //std::cout << image_tensor.size(0) << " x " << image_tensor.size(1) << " x " << image_tensor.size(1) << " x " << image_tensor.size(2) << std::endl;
            model->eval();
            if (isCUDAAvailable) {
                model->to(torch::kCUDA);
                image_tensor = image_tensor.to(torch::kCUDA);
            }
            
            auto output = model->forward(image_tensor);
            output = smax->forward(output);
            output = output.argmax(1);
            output = output.to(torch::kCPU);
            int64_t* prediction = reinterpret_cast<int64_t*>(output.data_ptr());
            if (ispretrained) {
                if (151 <= *prediction <= 268) {
                    std::cout << "Predicted bit is " << *prediction << " which means " << filename << " is of a dog" << std::endl;
                }
                else {
                    std::cout << "Predicted bit is " << *prediction << " which means " << filename << " is not of a dog" << std::endl;
                }
            }
            else {
                std::cout << "Predicted bit is " << *prediction << " which means " << filename << " resembles " << getClassNameByClassId(*prediction) << "\n";
            }   
        }
    }
}   


int main() {
    size_t kBatchSize = 32;
    std::cout << "Program started " << std::endl;
    auto lambda_transform_train = [](cv::Mat& image) {
        //apply transforms acording to need
        Resize(224, image);
        //CenterCrop(224, image);
        //srand(time(0));
        //double randrotation = (rand() % (30 + 30 + 1)) - 30;
        //std::cout << "Random ROate " << randrotation << "\n";
        //RandomRotate(randrotation, image);
        //srand(time(0));
        //RandomHorizontalFlip(rand() % 2, image);
    };
   
    auto lambda_transform_valid = [](cv::Mat& image) {
        //apply transforms acording to need
        Resize(224, image);
        //CenterCrop(224, image);
    };
  
    
    DogBreedDataset dataset("train", lambda_transform_train);
    DataLoader data_loader(dataset, kBatchSize);
    
    DogBreedDataset testdataset("valid", lambda_transform_valid);
    DataLoader valid_data_loader(testdataset, kBatchSize);
    
    DogBreedDataset validdataset("test", lambda_transform_valid);
    DataLoader test_data_loader(validdataset, kBatchSize);
    
    
    bool isCUDAAvailable = torch::cuda::is_available();
    std::cout << "Is Cuda Available ? "<< isCUDAAvailable << std::endl;
    

    std::cout << "Please Choose the Network you want to use " << std::endl;
    std::cout << "0. For Transfer Learning Network" << std::endl;
    std::cout << "1. For Network Built from scratch" << std::endl;
    std::cout << "2. For Network VGG-16 network" << std::endl;
    
    int kNetwork = -1; int kOperation = -1;
    std::cin >> kNetwork;
    std::cout << "Choose the operation "<<std::endl;
    std::cout << "0. Evaluation" << std::endl;
    std::cout << "1. Test" << std::endl;
    std::cout << "2. Train" << std::endl;
    std::cin >> kOperation;
    
    
    
    if (kNetwork == 0) {
        auto model = TransferLearning("../weights");
        if (kOperation == 0) {
            Evaluate<TransferLearning>(model);
        }
        else if(kOperation == 1){
            
            Test<DataLoader, TransferLearning>(valid_data_loader, model);
        }
        else if (kOperation == 2) {
            
            torch::optim::RMSprop optimizer(model->last_layer->parameters(), torch::optim::RMSpropOptions(0.001));
            Train<DataLoader, TransferLearning, torch::optim::RMSprop>(data_loader, test_data_loader, model, optimizer);
        }
        else {
            std::cout << "sorry action option not recognised" << std::endl;
        }
    }else if (kNetwork == 1) {
        auto model = cnnNet("../weights");
        if (kOperation == 0) {
            Evaluate<cnnNet>(model);
        }
        else if (kOperation == 1) {

            Test<DataLoader, cnnNet>(valid_data_loader, model);
        }
        else if (kOperation == 2) {

            torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.01));
            Train<DataLoader, cnnNet, torch::optim::Adam>(data_loader, test_data_loader, model, optimizer);
        }
        else {
            std::cout << "sorry action option not recognised" << std::endl;
        }
    }else if (kNetwork == 3) {
        auto model = VGG16("../weights");
        if (kOperation == 0) {
            Evaluate<VGG16>(model, true);
        }
        else if (kOperation == 1) {

            Test<DataLoader, VGG16>(valid_data_loader, model);
        }
        else if (kOperation == 2) {
            std::cout << "Sorry can't train on VGG-16 net" << std::endl;   
        }
         else {
            std::cout << "sorry action option not recognised" << std::endl;
        }
    
    }
    else {
        std::cout << "Network option not recognised" << std::endl;
    }
    
    std::cout << "Press Any Key to Shutdown program" << std::endl;
    std::cin.get();
    std::cin.get();
}

