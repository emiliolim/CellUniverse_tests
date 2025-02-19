#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>

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

    // OpenCV Processing of TiffSlices
    std::vector<cv::Mat> pTiffSlices;
    for(int i = 0; i < tiffSlices.size(); ++i)
    {
        cv::Mat slice = tiffSlices[i].clone();
        cv::cvtColor(slice, slice, cv::COLOR_BGR2GRAY);
        //cv::Mat processedImg = processImage(slice);
        cv::bitwise_not(slice, slice);
        cv::GaussianBlur(slice, slice, cv::Size(0, 0), 1.5);
        //cv::threshold(slice, slice, 128, 255, cv::THRESH_BINARY);
        pTiffSlices.push_back(slice);
    }

    // Print the number of slices read
    std::cout << "Number of slices: " << pTiffSlices.size() << std::endl;

    // Example: Print the dimensions of the first slice
    std::cout << "First slice dimensions: " << pTiffSlices[0].rows << " x " << pTiffSlices[0].cols << std::endl;


    // Testing a 2D slice
    cv::Mat squish {pTiffSlices[19]};

    // Save the processed slices to a new multi-page TIFF file
    std::string outputFilePath = "processed.tif";
    bool saveSuccess = cv::imwritemulti(outputFilePath, pTiffSlices);
    std::string outputFilePath2D = "processed2D.tif";
    bool saveSuccess2D = cv::imwritemulti(outputFilePath2D, squish);

    // Check if the file was saved successfully
    if (!saveSuccess) {
        std::cerr << "Error: Could not save the processed TIFF file!" << std::endl;
        return -1;
    }

    std::cout << "Processed slices saved to: " << outputFilePath << std::endl;

    cv::Mat data = squish;
    // Step 2: Perform mean subtraction (centering the data)
    cv::Mat mean;
    data.convertTo(data, CV_32F);
    cv::reduce(data, mean, 0, cv::REDUCE_AVG); // Compute mean for each column (feature)
    cv::Mat centered = data - cv::repeat(mean, data.rows, 1);

    // Step 3: Perform PCA
    cv::PCA pca(centered, cv::Mat(), cv::PCA::DATA_AS_ROW); // Rows are samples

    // Step 4: Print eigenvalues and eigenvectors
    std::cout << "Eigenvalues:" << std::endl << pca.eigenvalues << std::endl;
    //std::cout << "Eigenvectors:" << std::endl << pca.eigenvectors << std::endl;
    // unsigned numTiffSlices = pTiffSlices.size();
    // assert(numTiffSlices == 33);
    // const int expandFactor = 3; 
    // unsigned numSynthSlices = expandFactor * (numTiffSlices-1) + 1;
    // std::vector<cv::Mat> iTiffSlices;

    // for (int synthSlice = 0; synthSlice < numSynthSlices; ++synthSlice) {
    //     int tiffSlice = int(synthSlice / expandFactor); // "real" slice index 
    //     if(synthSlice % expandFactor == 0) 
    //     { // copy the real slice to the synth one, verbatim
    //     iTiffSlices.push_back(pTiffSlices[tiffSlice]);
    //     } 
    //     else if (synthSlice % expandFactor == 1) {
    //         // Interpolate between realTiff[tiffSlice] and realTiff[tiffSlice + 1]
    //         interpolateSlices(pTiffSlices[tiffSlice], 
    //                         pTiffSlices[tiffSlice + 1], 
    //                         iTiffSlices, 
    //                         expandFactor - 1);
    //     }
    // }
    // Create a vector to store 3D points
    std::vector<cv::Point3f> points;
    const int threshold = 180;
    // Iterate over each slice (z-axis)
    for (size_t z = 0; z < pTiffSlices.size(); z++) {
        cv::Mat slice = pTiffSlices[z];

        // Iterate over each pixel in the slice (x, y)
        for (int y = 0; y < slice.rows; y++) {
            for (int x = 0; x < slice.cols; x++) {
                // Get the pixel intensity (for grayscale images)
                float intensity = slice.at<uchar>(y, x);

                // Create a 3D point: (x, y, z)
                cv::Point3f point(x, y, z);
                // Add the point to the vector
                points.push_back(point);
                
            }
        }
    }

    // Print the number of 3D points created
    std::cout << "Number of 3D points: " << points.size() << std::endl;

    // Perform PCA to get eigenvalues
    std::vector<std::pair<float, cv::Vec3f>> eigenvalues = performPCA(points);
    std::cout << "Eigenvalues with most variance:" << std::endl;
    for (size_t i = 0; i < eigenvalues.size(); ++i) {
        std::cout << "Eigenvalue " << i + 1 << ": " << eigenvalues[i].first << " " << eigenvalues[i].second << std::endl;
    }

  return 0;
}
