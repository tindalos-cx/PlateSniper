#pragma once

#include <opencv2/core.hpp>
#include "core/detection.h"

namespace platesniper {

class PlateCorrector {
public:
    PlateCorrector();

    cv::Mat correct(const cv::Mat& image, const Detection& det);
    cv::Mat correct(const cv::Mat& image, const std::vector<cv::Point2f>& corners);

    void setOutputSize(int width, int height);
    void setBluePlateSize(int width, int height);
    void setGreenPlateSize(int width, int height);

private:
    int output_width_;
    int output_height_;
    int blue_width_;
    int blue_height_;
    int green_width_;
    int green_height_;

    std::vector<cv::Point2f> estimateCorners(const cv::Rect& bbox, float aspect_ratio);
    cv::Mat applyPerspectiveTransform(const cv::Mat& image,
                                       const std::vector<cv::Point2f>& src_points,
                                       const std::vector<cv::Point2f>& dst_points);
};

} // namespace platesniper
