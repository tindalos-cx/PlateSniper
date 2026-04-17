#include "utils/image_utils.h"
#include <algorithm>

namespace platesniper {

cv::Mat ImageUtils::resize(const cv::Mat& image, int width, int height, bool keep_aspect) {
    if (image.empty()) {
        return cv::Mat();
    }

    if (!keep_aspect) {
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(width, height));
        return resized;
    }

    float scale_x = static_cast<float>(width) / image.cols;
    float scale_y = static_cast<float>(height) / image.rows;
    float scale = std::min(scale_x, scale_y);

    int new_width = static_cast<int>(image.cols * scale);
    int new_height = static_cast<int>(image.rows * scale);

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_width, new_height));

    cv::Mat result(height, width, image.type(), cv::Scalar(0, 0, 0));

    int x_offset = (width - new_width) / 2;
    int y_offset = (height - new_height) / 2;

    resized.copyTo(result(cv::Rect(x_offset, y_offset, new_width, new_height)));

    return result;
}

cv::Mat ImageUtils::normalize(const cv::Mat& image, float scale) {
    cv::Mat normalized;
    image.convertTo(normalized, CV_32FC3, scale);
    return normalized;
}

cv::Mat ImageUtils::convertColor(const cv::Mat& image, int code) {
    cv::Mat converted;
    cv::cvtColor(image, converted, code);
    return converted;
}

cv::Mat ImageUtils::preprocessForDetection(const cv::Mat& image, int target_width, int target_height) {
    if (image.empty()) {
        return cv::Mat();
    }

    cv::Mat resized = resize(image, target_width, target_height, true);

    cv::Mat blob;
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels.data());

    for (int i = 0; i < 3; ++i) {
        channels[i].convertTo(channels[i], CV_32FC1, 1.0 / 255.0);
    }
    cv::merge(channels, blob);

    return blob;
}

cv::Mat ImageUtils::preprocessForRecognition(const cv::Mat& image, int target_width, int target_height) {
    if (image.empty()) {
        return cv::Mat();
    }

    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(target_width, target_height));

    cv::Mat normalized;
    resized.convertTo(normalized, CV_32FC1, 1.0 / 255.0);

    return normalized;
}

void ImageUtils::drawDetection(cv::Mat& image, const cv::Rect& bbox,
                               float confidence, const cv::Scalar& color) {
    if (image.empty()) {
        return;
    }

    cv::rectangle(image, bbox, color, 2);

    std::string label = cv::format("%.2f", confidence);
    int baseline;
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                         0.5, 1, &baseline);

    cv::Point text_pos(bbox.x, bbox.y - 5);
    if (text_pos.y - text_size.height < 0) {
        text_pos.y = bbox.y + bbox.height + text_size.height + 5;
    }

    cv::rectangle(image,
                 cv::Point(text_pos.x, text_pos.y - text_size.height),
                 cv::Point(text_pos.x + text_size.width, text_pos.y),
                 color, cv::FILLED);

    cv::putText(image, label, text_pos, cv::FONT_HERSHEY_SIMPLEX,
               0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}

void ImageUtils::drawText(cv::Mat& image, const cv::Point& pos,
                         const std::string& text, const cv::Scalar& color) {
    if (image.empty()) {
        return;
    }

    cv::putText(image, text, pos, cv::FONT_HERSHEY_SIMPLEX,
               1.0, color, 2, cv::LINE_AA);
}

} // namespace platesniper
