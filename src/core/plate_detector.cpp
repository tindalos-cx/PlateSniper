#include "core/plate_detector.h"
#include "models/onnx_session.h"
#include <algorithm>
#include <cmath>

namespace platesniper {

struct PlateDetector::Impl {
    OnnxSession session;
    int input_width_ = 640;
    int input_height_ = 640;
    std::string error_message;
    bool loaded_ = false;

    std::vector<Detection> detect(const cv::Mat& image, float conf_threshold) {
        std::vector<Detection> detections;

        if (!loaded_ || image.empty()) {
            return detections;
        }

        cv::Mat resized;
        cv::resize(image, resized, cv::Size(input_width_, input_height_));

        cv::Mat blob = cv::dnn::blobFromImage(resized, 1.0 / 255.0,
                                              cv::Size(input_width_, input_height_),
                                              cv::Scalar(0, 0, 0), true, false);

        std::vector<cv::Mat> input_mats;
        std::vector<cv::Mat> channels;
        for (int i = 0; i < 3; ++i) {
            channels.push_back(blob.at<cv::Mat>(0, i));
        }
        cv::merge(channels, blob);

        input_mats.push_back(blob);

        auto outputs = session.inference(input_mats);
        if (outputs.empty()) {
            return detections;
        }

        cv::Mat& output = outputs[0];
        auto shape = session.getOutputShape();

        if (shape.size() >= 3) {
            int num_proposals = shape[2];
            float scale_x = static_cast<float>(image.cols) / input_width_;
            float scale_y = static_cast<float>(image.rows) / input_height_;

            for (int i = 0; i < num_proposals; ++i) {
                float* data = output.ptr<float>(0, i);

                float cx = data[0] * input_width_;
                float cy = data[1] * input_height_;
                float w = data[2] * input_width_;
                float h = data[3] * input_height_;
                float conf = data[4];

                if (conf < conf_threshold) {
                    continue;
                }

                int x = static_cast<int>((cx - w / 2) * scale_x);
                int y = static_cast<int>((cy - h / 2) * scale_y);
                int width = static_cast<int>(w * scale_x);
                int height = static_cast<int>(h * scale_y);

                x = std::max(0, x);
                y = std::max(0, y);
                width = std::min(width, image.cols - x);
                height = std::min(height, image.rows - y);

                Detection det;
                det.bbox = cv::Rect(x, y, width, height);
                det.confidence = conf;
                det.classId = 0;

                detections.push_back(det);
            }
        }

        return nonMaxSuppression(detections, 0.5f);
    }

    std::vector<Detection> nonMaxSuppression(const std::vector<Detection>& dets, float iou_threshold) {
        std::vector<Detection> result;
        if (dets.empty()) return result;

        std::vector<Detection> sorted_dets = dets;
        std::sort(sorted_dets.begin(), sorted_dets.end(),
                 [](const Detection& a, const Detection& b) {
                     return a.confidence > b.confidence;
                 });

        std::vector<bool> suppressed(sorted_dets.size(), false);

        for (size_t i = 0; i < sorted_dets.size(); ++i) {
            if (suppressed[i]) continue;

            result.push_back(sorted_dets[i]);

            for (size_t j = i + 1; j < sorted_dets.size(); ++j) {
                if (suppressed[j]) continue;

                float iou = calculateIoU(sorted_dets[i].bbox, sorted_dets[j].bbox);
                if (iou > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }

        return result;
    }

    float calculateIoU(const cv::Rect& a, const cv::Rect& b) {
        cv::Rect inter = a & b;
        float inter_area = static_cast<float>(inter.area());
        float union_area = static_cast<float>(a.area() + b.area() - inter.area());

        if (union_area <= 0) return 0.0f;
        return inter_area / union_area;
    }
};

PlateDetector::PlateDetector() : pImpl_(std::make_unique<Impl>()) {}
PlateDetector::~PlateDetector() = default;

bool PlateDetector::loadModel(const std::string& modelPath) {
    bool success = pImpl_->session.loadModel(modelPath);
    pImpl_->loaded_ = success;
    if (!success) {
        pImpl_->error_message = pImpl_->session.lastError();
    }
    return success;
}

void PlateDetector::unload() {
    pImpl_->session.unload();
    pImpl_->loaded_ = false;
}

std::vector<Detection> PlateDetector::detect(const cv::Mat& image, float confThreshold) {
    return pImpl_->detect(image, confThreshold);
}

void PlateDetector::setInputSize(int width, int height) {
    pImpl_->input_width_ = width;
    pImpl_->input_height_ = height;
}

bool PlateDetector::isLoaded() const {
    return pImpl_->loaded_;
}

std::string PlateDetector::lastError() const {
    return pImpl_->error_message;
}

} // namespace platesniper
