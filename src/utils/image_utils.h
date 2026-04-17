#pragma once

#include <opencv2/core.hpp>

namespace platesniper {

class ImageUtils {
public:
    static cv::Mat resize(const cv::Mat& image, int width, int height,
                          bool keep_aspect = false);
    static cv::Mat normalize(const cv::Mat& image, float scale = 1.0f / 255.0f);
    static cv::Mat convertColor(const cv::Mat& image, int code);

    static cv::Mat preprocessForDetection(const cv::Mat& image, int target_width, int target_height);
    static cv::Mat preprocessForRecognition(const cv::Mat& image, int target_width, int target_height);

    static void drawDetection(cv::Mat& image, const cv::Rect& bbox,
                             float confidence, const cv::Scalar& color = cv::Scalar(0, 255, 0));
    static void drawText(cv::Mat& image, const cv::Point& pos,
                        const std::string& text, const cv::Scalar& color = cv::Scalar(255, 255, 255));
};

} // namespace platesniper
