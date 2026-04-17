#include "core/plate_corrector.h"
#include <algorithm>

namespace platesniper {

PlateCorrector::PlateCorrector()
    : output_width_(136)
    , output_height_(36)
    , blue_width_(136)
    , blue_height_(36)
    , green_width_(136)
    , green_height_(40)
{}

cv::Mat PlateCorrector::correct(const cv::Mat& image, const Detection& det) {
    float aspect_ratio = static_cast<float>(green_width_) / green_height_;
    auto corners = estimateCorners(det.bbox, aspect_ratio);
    return correct(image, corners);
}

cv::Mat PlateCorrector::correct(const cv::Mat& image, const std::vector<cv::Point2f>& corners) {
    if (corners.size() != 4 || image.empty()) {
        return cv::Mat();
    }

    std::vector<cv::Point2f> dst_points = {
        cv::Point2f(0, 0),
        cv::Point2f(static_cast<float>(output_width_ - 1), 0),
        cv::Point2f(static_cast<float>(output_width_ - 1), static_cast<float>(output_height_ - 1)),
        cv::Point2f(0, static_cast<float>(output_height_ - 1))
    };

    return applyPerspectiveTransform(image, corners, dst_points);
}

void PlateCorrector::setOutputSize(int width, int height) {
    output_width_ = width;
    output_height_ = height;
}

void PlateCorrector::setBluePlateSize(int width, int height) {
    blue_width_ = width;
    blue_height_ = height;
}

void PlateCorrector::setGreenPlateSize(int width, int height) {
    green_width_ = width;
    green_height_ = height;
}

std::vector<cv::Point2f> PlateCorrector::estimateCorners(const cv::Rect& bbox, float aspect_ratio) {
    float x = static_cast<float>(bbox.x);
    float y = static_cast<float>(bbox.y);
    float w = static_cast<float>(bbox.width);
    float h = static_cast<float>(bbox.height);

    float center_x = x + w / 2.0f;
    float center_y = y + h / 2.0f;

    float scale = 1.1f;
    float new_h = h * scale;
    float new_w = new_h * aspect_ratio;

    float new_x = center_x - new_w / 2.0f;
    float new_y = center_y - new_h / 2.0f;

    std::vector<cv::Point2f> corners = {
        cv::Point2f(new_x, new_y),
        cv::Point2f(new_x + new_w, new_y),
        cv::Point2f(new_x + new_w, new_y + new_h),
        cv::Point2f(new_x, new_y + new_h)
    };

    return corners;
}

cv::Mat PlateCorrector::applyPerspectiveTransform(const cv::Mat& image,
                                                   const std::vector<cv::Point2f>& src_points,
                                                   const std::vector<cv::Point2f>& dst_points) {
    cv::Mat transform = cv::getPerspectiveTransform(src_points, dst_points);

    cv::Mat result;
    cv::warpPerspective(image, result, transform,
                       cv::Size(output_width_, output_height_),
                       cv::INTER_LINEAR,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(0, 0, 0));

    return result;
}

} // namespace platesniper
