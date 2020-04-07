#include "pch.h"
#include "dogbreeddataset.h"
#include "transforms.h"
#include <future>
#include <mutex>


auto normalize = torch::data::transforms::Normalize<>({ 0.485,0.456,0.406 }, { 0.229,0.224,0.225 });//torch::data::transforms::Normalize<>(0.5, 0.5);


std::unordered_map<int, std::string >_inttostringlabels;
std::unordered_map<std::string, int >_stringtointlabels;
bool _islookupset = false;
std::string _parentdir = "C:\\dogImages";

void DogBreedDataset::read_directory(const std::string& name)
{
    int kdogtype = 0;
    boost::filesystem::path p(name);
    boost::filesystem::directory_iterator start(p);
    boost::filesystem::directory_iterator end;
    for (auto it = start; it != end; ++it) {
        std::string dogtype = (*it).path().leaf().string();
        kdogtype = extractNumberFromString(dogtype) -1;
        if (!_islookupset) {
            
            _stringtointlabels[dogtype] = kdogtype;
            _inttostringlabels[kdogtype] = dogtype;
        }

        boost::filesystem::path childp(name + "\\" + dogtype);
        boost::filesystem::directory_iterator startp(childp);

        for (auto childit = startp; childit != end; ++childit) {
            //std::cout << (*childit).path().string() << std::endl;
            _image_files_list.emplace_back((*childit).path().string());
            _labels_list.emplace_back(kdogtype);
        }
        //std::cout << dogtype << std::endl;
        kdogtype++;
    }
    _islookupset = true;
}

int DogBreedDataset::extractNumberFromString(std::string& convert)
{
    std::istringstream iss(convert);
    std::string number;
    std::getline(iss, number, '.');
    return std::stoi(number);
}

DogBreedDataset::DogBreedDataset(std::string type, void (*transform_func)(cv::Mat& image)):transform_lambda(transform_func)
{
    
    
    if (type == "train") {
        _image_files_list.reserve(6680);
        _labels_list.reserve(6680);
    }
    else if (type == "test") {
        _image_files_list.reserve(836);
        _labels_list.reserve(836);
    }
    else if (type == "valid") {
        _image_files_list.reserve(836);
        _labels_list.reserve(836);
    }
    if (!_islookupset) {
        _stringtointlabels.reserve(133);
        _inttostringlabels.reserve(133);
    }
    read_directory(_parentdir + "\\" + type);
    
    numberofworkers = 4;

    /*auto ittoint = _labels_list.begin();
    for (auto it = _image_files_list.begin(); it != _image_files_list.end(); ++it) {
        std::cout << *it << "- >" << *ittoint << std::endl;
        ++ittoint;
    }*/
    
    
}


Example DogBreedDataset::get(size_t index) {

    //get cv image
    //std::cout << _image_files_list[index] << "\n";
    cv::Mat image = read_image(_image_files_list[index]);
    transform_lambda(image);
    auto target_tensor = torch::zeros(1, torch::TensorOptions().dtype(torch::kInt64));
    target_tensor[0] = _labels_list[index];
    /*mtx.lock();
    std::cout << " INdex we are looking for " << index << " corresponding data " << _labels_list[index] << "  filename " << _image_files_list[index] << std::endl;
    mtx.unlock();*/
    torch::Tensor tensor_image = normalize(convert_to_tensor(image));
    return  std::move(Example{ tensor_image , target_tensor });
}

std::vector<Example> DogBreedDataset::get_minibatch(std::vector<size_t>::iterator start, std::vector<size_t>::iterator end)
{   
    
    std::vector<Example> examplevec;
    examplevec.reserve(end-start);
    
    for (auto it = start; it != end; ++it) {
        examplevec.push_back(get(*it));
    }

    return std::move(examplevec);
}


Example DogBreedDataset::get_batch(std::vector<size_t>& indices, int batch_idx)
{   
   
    //std::vector<torch::Dtype> newindices = indices.value();
    int batchsize = indices.size();
    auto start = indices.begin();
    auto end = indices.end();
    size_t numberoftasks = numberofworkers - 1;
    size_t stride = floor(batchsize / numberofworkers);
    std::vector<std::future< std::vector<Example>>>_workers;
    
    auto newstart = start;
    for (size_t i = 0; i < numberoftasks; i++) {
       // std::cout << "Sending " << (newstart - start) <<" to " <<(stride + newstart - start) << std::endl;
        _workers.emplace_back(std::async(std::launch::async, &DogBreedDataset::get_minibatch, this,  newstart,  newstart+stride ) );
        newstart = newstart + stride;
    }
    std::vector<Example> maintaskresult = get_minibatch(newstart, end);
    std::vector<torch::Tensor> datavec;
    datavec.reserve(batchsize);
    std::vector<torch::Tensor> targetvec;
    targetvec.reserve(batchsize);
    for (auto& worker : _workers) {
        std::vector<Example> examples = worker.get();
        //std::cout << examples.size() << std::endl;
        for (auto& example : examples) {
            datavec.push_back(std::move(example.data));
            targetvec.push_back(std::move(example.target));
        }
    }
    for (auto& main :maintaskresult) {
        datavec.push_back(std::move(main.data));
        targetvec.push_back(std::move(main.target));
    }
    //std::cout << dats
    auto data = torch::stack(datavec);
    auto targets = torch::stack(targetvec);
    return std::move(Example{data, targets, batch_idx });

    
}

int64_t DogBreedDataset::size() const {
    return _image_files_list.size();
}


void DogBreedDataset::setTransforms(void (*transform_func)(cv::Mat& image)) {
    transform_lambda = transform_func;
}

std::string& getClassNameByClassId(int64 index)
{
	// TODO: insert return statement here
    return _inttostringlabels[index]; 
}

/*std::vector<torch::data::Example<>> DogBreedDataset::get_minibatch(std::vector<size_t>::iterator start, std::vector<size_t>::iterator end)
{   
    std::vector<torch::data::Example<>> res;
    for (start; start != end; ++start) {
        res.emplace_back(this->get(*start));
    }
    return std::move(res);
}*/

/*std::vector<torch::data::Example<>> DogBreedDataset::get_batch(torch::ArrayRef<size_t> indices)  {
    std::vector<torch::data::Example<>> batch;
    batch.reserve(indices.size());
    
    for (const auto i : indices) {
        batch.emplace_back(get(i));
    }
    
    return batch;
}*/