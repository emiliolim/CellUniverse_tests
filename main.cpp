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
    for (int i = 0; i < tiffSlices.size(); ++i) {
        cv::Mat slice = tiffSlices[i].clone();

        cv::cvtColor(slice, slice, cv::COLOR_BGR2GRAY);

        cv::Size kernel_size = cv::Size(5, 5); // Kernel size (width, height). Should be odd and positive

        // Apply Gaussian blur
        cv::GaussianBlur(slice, slice, kernel_size, 0, 0);

        // // Threshold to create a binary image
        cv::threshold(slice, slice, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        // Add to the processed slices vector
        pTiffSlices.push_back(slice);
    }

    // // Testing a 2D slice
    cv::Mat squish = pTiffSlices[19];  // No need to convert; it’s already CV_8UC1
    squish.convertTo(squish, CV_8UC1);
    vector<vector<cv::Point> > contours;
    findContours(squish, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    cv::Mat squish_copy;
    cv::cvtColor(squish, squish_copy, cv::COLOR_GRAY2BGR);

    for (size_t i = 0; i < contours.size(); i++)
    {
        // Calculate the area of each contour
        double area = contourArea(contours[i]);
        // // Ignore contours that are too small or too large
        if (area < 1e2 || 1e5 < area) continue; 
        // // Draw each contour only for visualisation purposes
        // // Find the orientation of each shape

        std::cout << "contour area " << i << ": " << area << std::endl;
        cv::drawContours(squish_copy, contours, static_cast<int>(i), cv::Scalar(0, 0, 255), 2); 
        getOrientation(contours[i], squish_copy);
    }  
    // Print the number of slices read
    std::cout << "Number of slices: " << pTiffSlices.size() << std::endl;

    // Example: Print the dimensions of the first slice
    if (!pTiffSlices.empty()) {
        std::cout << "First slice dimensions: " 
                << pTiffSlices[0].rows << " x " << pTiffSlices[0].cols << std::endl;
    }

    // Save the processed slices to a new multi-page TIFF file
    std::string outputFilePath = "processed.tif";
    bool saveSuccess = cv::imwritemulti(outputFilePath, pTiffSlices);
    std::string outputFilePath2D = "processed2D.tif";
    bool saveSuccess2D = cv::imwritemulti(outputFilePath2D, squish_copy);

    // Check if the file was saved successfully
    if (!saveSuccess2D) {
        std::cerr << "Error: Could not save the processed TIFF file!" << std::endl;
        return -1;
    }

    std::cout << "Processed slices saved to: " << outputFilePath << std::endl;

    // Create a vector to store 3D points
    // std::vector<cv::Point3f> points;
    // const int threshold = 180;
    // // Iterate over each slice (z-axis)
    // for (size_t z = 0; z < pTiffSlices.size(); z++) {
    //     cv::Mat slice = pTiffSlices[z];

    //     // Iterate over each pixel in the slice (x, y)
    //     for (int y = 0; y < slice.rows; y++) {
    //         for (int x = 0; x < slice.cols; x++) {
    //             // Get the pixel intensity (for grayscale images)
    //             float intensity = slice.at<uchar>(y, x);

    //             // Create a 3D point: (x, y, z)
    //             cv::Point3f point(x, y, z);
    //             // Add the point to the vector
    //             points.push_back(point);
                
    //         }
    //     }
    // }

    // Print the number of 3D points created
    // std::cout << "Number of 3D points: " << points.size() << std::endl;

    // Perform PCA to get eigenvalues
    //std::vector<std::pair<float, cv::Vec3f>> eigenvalues = performPCA(points);
    // std::cout << "Eigenvalues with most variance:" << std::endl;
    // for (size_t i = 0; i < pca.eigenvalues.size(); ++i) {
    //     std::cout << "Eigenvalue " << i + 1 << ": " << pca.eigenvalues[i].first << " " << pca.eigenvalues[i].second << std::endl;
    // }

  return 0;
}
