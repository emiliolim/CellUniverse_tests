#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
#include "prepare.cpp"


int main()
{
    // Specify the path to your multi-page TIFF file
    std::string filePath = "squish_cell07.tif";

    // Create a vector to store all 2D slices
    std::vector<cv::Mat> tiffSlices;

    // Read the multi-page TIFF file
    bool success = cv::imreadmulti(filePath, tiffSlices, cv::IMREAD_ANYDEPTH | cv::IMREAD_COLOR);

    // Check if the file was read successfully
    if (!success || tiffSlices.empty()) {
        std::cerr << "Error: Could not open or read the multi-page TIFF file!" << std::endl;
        return -1;
    }
    std::cout << "Image type: " << tiffSlices[0].type() << " (type before processing)" << std::endl;


    // OpenCV Processing of TiffSlices
    std::vector<cv::Mat> pTiffSlices;
    processImage(pTiffSlices, tiffSlices);

    // // Testing a 2D slice
    cv::Mat squish = pTiffSlices[19];  // No need to convert; it’s already CV_8UC1
    test2DSlice(squish, pTiffSlices);
    


  return 0;
}
