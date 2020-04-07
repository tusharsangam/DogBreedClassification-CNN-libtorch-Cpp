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
            train_loss = train_loss + ((1 / (batch_idx)) * (*loss.data_ptr<float>() - train_loss));
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
            valid_loss = valid_loss + ((1 / (batch_idx)) * (*loss.data_ptr<float>() - valid_loss));
            auto pred = output.data().argmax(1);
            auto correctionmatrix = torch::eq(pred, labels);
            auto sum = torch::sum(correctionmatrix).to(torch::kCPU);
            correct += *sum.data_ptr<int64>();
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
        valid_loss = valid_loss + ((1 / (batch_idx)) * (*loss.data_ptr<float>() - valid_loss));
        
        auto pred = output.data().argmax(1);
        
        
        auto correctionmatrix = torch::eq(pred, labels);
        auto sum = torch::sum(correctionmatrix).to(torch::kCPU);
        correct += *sum.data_ptr<int64>();
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
    boost::filesystem::path p("..\\evaluation");
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
            std::cout << image_tensor.size(0) << " x " << image_tensor.size(1) << " x " << image_tensor.size(1) << " x " << image_tensor.size(2) << std::endl;
            model->eval();
            model->to(torch::kCUDA);
            image_tensor = image_tensor.to(torch::kCUDA);
            
            auto output = model->forward(image_tensor);
            output = smax->forward(output);
            output = output.argmax(1);
            output = output.to(torch::kCPU);
            
            std::cout << "Predicted bit is " << *output.data_ptr<int64>() << " which means " << filename << " resembles " << getClassNameByClassId(*output.data_ptr<int64>()) << "\n";
            
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
    DogBreedDataset validdataset("test", lambda_transform_valid);
    DogBreedDataset testdataset("valid", lambda_transform_valid);
    DataLoader data_loader(dataset, kBatchSize);
    DataLoader test_data_loader(validdataset, kBatchSize);
    DataLoader valid_data_loader(testdataset, kBatchSize);
    bool isCUDAAvailable = torch::cuda::is_available();
    std::cout << "Is Cuda Available ? "<< isCUDAAvailable << std::endl;
      
    
    
    auto model = cnnNet("..\\weights");
    model->load();
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.01));
    
    Train<DataLoader, cnnNet, torch::optim::Adam>(data_loader, test_data_loader, model, optimizer);
    //Test<DataLoader, TransferLearning>(valid_data_loader, model);
    //Evaluate<TransferLearning>(model);
    
         
    std::cin.get();
}

