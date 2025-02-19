#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
using namespace std;

// Function to interpolate between two slices
void interpolateSlices(const cv::Mat& slice1, const cv::Mat& slice2, 
    std::vector<cv::Mat>& processedSlices, int numInterpolations) {
    // Ensure the two slices have the same size and type
    if (slice1.size() != slice2.size() || slice1.type() != slice2.type()) {
        throw std::invalid_argument("Slices must have the same size and type for interpolation!");
    }

    // Perform interpolation
    for (int i = 1; i <= numInterpolations; ++i) {
        double t = static_cast<double>(i) / (numInterpolations + 1);
        cv::Mat interpolatedSlice = (1.0 - t) * slice1 + t * slice2;
        processedSlices.push_back(interpolatedSlice);
    }
}

std::vector<std::pair<float, cv::Vec3f>> performPCA(const std::vector<cv::Point3f> &points)
{
    if (points.empty())
    {
        throw std::invalid_argument("No points provided for PCA.");
    }

    // Create a matrix from the points
    cv::Mat data(points.size(), 3, CV_32F);
    for (size_t i = 0; i < points.size(); ++i)
    {
        data.at<float>(i, 0) = points[i].x;
        data.at<float>(i, 1) = points[i].y;
        data.at<float>(i, 2) = points[i].z;
    }

    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);

    // Extract the eigenvalues and eigenvectors
    cv::Mat eigenvalues = pca.eigenvalues;
    cv::Mat eigenvectors = pca.eigenvectors;

    // Prepare the result
    std::vector<std::pair<float, cv::Vec3f>> eigenPairs;
    for (int i = 0; i < eigenvalues.rows; ++i)
    {
        float eigenvalue = eigenvalues.at<float>(i);
        cv::Vec3f eigenvector(
            eigenvectors.at<float>(i, 0),
            eigenvectors.at<float>(i, 1),
            eigenvectors.at<float>(i, 2)
        );
        eigenPairs.emplace_back(eigenvalue, eigenvector);
    }

    return eigenPairs;
}

std::vector<std::pair<float, cv::Vec3f>> performPCAAlongAxes(const std::vector<cv::Point3f>& points) {
    if (points.empty()) {
        throw std::invalid_argument("No points provided for PCA.");
    }

    // Create matrices for each axis (x, y, z)
    cv::Mat dataX(points.size(), 1, CV_32F);
    cv::Mat dataY(points.size(), 1, CV_32F);
    cv::Mat dataZ(points.size(), 1, CV_32F);

    for (size_t i = 0; i < points.size(); ++i) {
        dataX.at<float>(i, 0) = points[i].x;
        dataY.at<float>(i, 0) = points[i].y;
        dataZ.at<float>(i, 0) = points[i].z;
    }

    // Perform PCA for each axis
    cv::PCA pcaX(dataX, cv::Mat(), cv::PCA::DATA_AS_ROW);
    cv::PCA pcaY(dataY, cv::Mat(), cv::PCA::DATA_AS_ROW);
    cv::PCA pcaZ(dataZ, cv::Mat(), cv::PCA::DATA_AS_ROW);

    // Extract eigenvalues and eigenvectors for each axis
    float eigenvalueX = pcaX.eigenvalues.at<float>(0);
    float eigenvalueY = pcaY.eigenvalues.at<float>(0);
    float eigenvalueZ = pcaZ.eigenvalues.at<float>(0);

    cv::Vec3f eigenvectorX(pcaX.eigenvectors.at<float>(0, 0), 0, 0);
    cv::Vec3f eigenvectorY(0, pcaY.eigenvectors.at<float>(0, 0), 0);
    cv::Vec3f eigenvectorZ(0, 0, pcaZ.eigenvectors.at<float>(0, 0));

    // Prepare the result
    std::vector<std::pair<float, cv::Vec3f>> eigenPairs;
    eigenPairs.emplace_back(eigenvalueX, eigenvectorX);
    eigenPairs.emplace_back(eigenvalueY, eigenvectorY);
    eigenPairs.emplace_back(eigenvalueZ, eigenvectorZ);

    return eigenPairs;
}

cv::Mat processImage(const cv::Mat &image)
{
    cv::Mat processedImage;

    if (image.channels() == 3)
    {
        cv::cvtColor(image, processedImage, cv::COLOR_RGB2GRAY);
    }
    else
    {
        processedImage = image.clone();
    }

    processedImage.convertTo(processedImage, CV_32F, 1.0 / 255.0);

    cv::GaussianBlur(processedImage, processedImage, cv::Size(0, 0), 1.5);

    return processedImage;
}