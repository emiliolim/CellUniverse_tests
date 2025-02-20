#include <iostream>
#include <vector>
#include <math.h>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
using namespace std;

constexpr double PI = 3.1415926535897932384626433832795;

void drawAxis(cv::Mat& img, cv::Point p, cv::Point q, cv::Scalar colour, const float scale)
{
    double angle = atan2( (double) p.y - q.y, (double) p.x - q.x ); // angle in radians
    double hypotenuse = sqrt( (double) (p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));
    // Here we lengthen the arrow by a factor of scale
    q.x = (int) (p.x - scale * hypotenuse * cos(angle));
    q.y = (int) (p.y - scale * hypotenuse * sin(angle));
    cv::line(img, p, q, colour, 1, cv::LINE_AA);

    // // create the arrow hooks
    // p.x = (int) (q.x + 9 * cos(angle + PI / 4));
    // p.y = (int) (q.y + 9 * sin(angle + PI / 4));
    // cv::line(img, p, q, colour, 1, cv::LINE_AA);
    
    // p.x = (int) (q.x + 9 * cos(angle - PI / 4));
    // p.y = (int) (q.y + 9 * sin(angle - PI / 4));
    // cv::line(img, p, q, colour, 1, cv::LINE_AA);
}

double getOrientation(const std::vector<cv::Point> &pts, cv::Mat &img)
{
    //Construct a buffer used by the pca analysis
    int sz = static_cast<int>(pts.size());
    cv::Mat data_pts = cv::Mat(sz, 2, CV_64F);
    for (int i = 0; i < data_pts.rows; i++)
    {
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
    }
    //Perform PCA analysis
    cv::PCA pca_analysis(data_pts, cv::Mat(), cv::PCA::DATA_AS_ROW);

    //Store the center of the object
    cv::Point cntr = cv::Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
    static_cast<int>(pca_analysis.mean.at<double>(0, 1)));

    //Store the eigenvalues and eigenvectors
    std::vector<cv::Point2d> eigen_vecs(2);
    std::vector<double> eigen_val(2);
    for (int i = 0; i < 2; i++)
    {
    eigen_vecs[i] = cv::Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                    pca_analysis.eigenvectors.at<double>(i, 1));

    eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);
    std::cout << "eigenval " << i << ": " << eigen_val[i] << std::endl;
    std::cout << "eigenvec " << i << ": " << eigen_vecs[i] << std::endl;

    }
    // Draw the principal components
    cv::circle(img, cntr, 3, cv::Scalar(255, 0, 255), 2);
    cv::Point p1 = cntr + 0.02 * cv::Point(static_cast<int>(eigen_vecs[0].x * eigen_val[0]), static_cast<int>(eigen_vecs[0].y * eigen_val[0]));
    cv::Point p2 = cntr - 0.02 * cv::Point(static_cast<int>(eigen_vecs[1].x * eigen_val[1]), static_cast<int>(eigen_vecs[1].y * eigen_val[1]));
    
    drawAxis(img, cntr, p1, cv::Scalar(0, 255, 0), 5);
    drawAxis(img, cntr, p2, cv::Scalar(255, 255, 0), 10);
    double angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x); // orientation in radians
    return angle;
}

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