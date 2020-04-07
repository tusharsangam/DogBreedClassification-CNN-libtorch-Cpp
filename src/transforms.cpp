#include "pch.h"
#include "transforms.h"

void plotImage(cv::Mat& image) {
    cv::namedWindow("image", cv::WINDOW_NORMAL);
    cv::imshow("image", image);
    cv::waitKey(10);
}

cv::Mat read_image(const std::string& path) {
    cv::Mat img = cv::imread(path); //reads BGR image
    if (img.empty()) {
        std::cout << "this image is problematic " << path << std::endl;
    }
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB); // change from BGR to RGB channel reordering
    img.convertTo(img, CV_32FC3, 1.0f / 255.0f);
    return std::move(img);
}

torch::Tensor convert_to_tensor(cv::Mat& image) {
    
    //bool isChar = (image.type() & 0xF) < 2;
    //std::cout << image << std::endl;
    torch::Tensor tensor_image = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kFloat).clone();
    //tensor_image = tensor_image.to(at::kFloat);
    // Transpose the image for[channels, rows, columns] format of pytorch tensor
    
    
    tensor_image = torch::transpose(tensor_image, 0, 1);
    tensor_image = torch::transpose(tensor_image, 0, 2);
    
    
    return std::move(tensor_image);
}
void Resize(int newsize, cv::Mat& image)
{
    /*int height = image.rows;
    int width = image.cols;
    int newheight, newwidth;
    if (width < height) {
        newwidth = newsize;
        newheight = int(newsize * height / width);
    }
    else {
        newheight = newsize;
        newwidth = int(newsize * width / height);
    }*/
    //resize
    cv::resize(image, image, { newsize, newsize });
}
void CenterCrop(const int cropSize, cv::Mat& image)
{
    if (image.cols > cropSize && image.rows > cropSize) {
        const int offsetW = (image.cols - cropSize) / 2;
        const int offsetH = (image.rows - cropSize) / 2;
        const cv::Rect roi(offsetW, offsetH, cropSize, cropSize);
        image = image(roi);

    }
    else {
        cv::resize(image, image, { cropSize, cropSize });
    }

}

void RandomRotate(double angle, cv::Mat& src)
{

    //double angle_random = (double)*(static_cast<float*>(randint(-angle, angle, IntArrayRef{ 1 }).storage().data()));
    //std::cout <<" random angle  " << angle << std::endl;
    cv::Point2f pt(src.cols / 2., src.rows / 2.);
    cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
    cv::warpAffine(src, src, r, cv::Size(src.cols, src.rows));
}

void RandomHorizontalFlip(int p, cv::Mat& image)
{

    //std::cout << "Flip Probability " << p  << std::endl;
    if (p == 1) {
        //do flip else pass
        cv::flip(image, image, 1);
    }
}
