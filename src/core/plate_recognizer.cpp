#include "core/plate_recognizer.h"
#include "models/onnx_session.h"
#include <algorithm>
#include <cctype>

namespace platesniper {

struct PlateRecognizer::Impl {
    OnnxSession session;
    std::vector<std::string> charset_;
    std::string error_message;
    bool loaded_ = false;

    static const std::vector<std::string> DEFAULT_CHARSET;

    Impl() {
        charset_ = DEFAULT_CHARSET;
    }

    std::string recognize(const cv::Mat& plate_image) {
        if (!loaded_ || plate_image.empty()) {
            return "";
        }

        cv::Mat gray;
        if (plate_image.channels() == 3) {
            cv::cvtColor(plate_image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = plate_image.clone();
        }

        cv::Mat resized;
        cv::resize(gray, resized, cv::Size(94, 24));

        cv::Mat normalized;
        resized.convertTo(normalized, CV_32FC1, 1.0 / 255.0);

        cv::Mat input;
        normalized.convertTo(input, CV_32FC3);
        cv::Mat blob[3];
        for (int i = 0; i < 3; ++i) {
            blob[i] = normalized;
        }
        cv::merge(blob, 3, input);

        std::vector<cv::Mat> inputs = {input};
        auto outputs = session.inference(inputs);

        if (outputs.empty()) {
            return "";
        }

        return decodeCTCGreedy(outputs[0]);
    }

    std::string decodeCTCGreedy(const cv::Mat& output) {
        std::string result;

        int seq_len = output.rows;
        int num_classes = output.cols;

        std::vector<int> best_indices(seq_len);
        for (int i = 0; i < seq_len; ++i) {
            float max_val = -1.0f;
            int max_idx = 0;
            const float* row = output.ptr<float>(i);
            for (int j = 0; j < num_classes; ++j) {
                if (row[j] > max_val) {
                    max_val = row[j];
                    max_idx = j;
                }
            }
            best_indices[i] = max_idx;
        }

        int last_idx = -1;
        for (int i = 0; i < seq_len; ++i) {
            int idx = best_indices[i];

            if (idx != 0 && idx != last_idx) {
                if (idx > 0 && idx <= static_cast<int>(charset_.size())) {
                    result += charset_[idx - 1];
                }
                last_idx = idx;
            }
        }

        return result;
    }
};

const std::vector<std::string> PlateRecognizer::Impl::DEFAULT_CHARSET = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z",
    "\u5186",
    "\u4EBA",
    "\u54E5",
    "\u5E94",
    "\u8D75",
    "\u5B81",
    "\u4E61",
    "\u8FB0",
    "\u897F",
    "\u9752",
    "\u5C71",
    "\u5E7F",
    "\u5F20",
    "\u6B66",
    "\u9ED1",
    "\u9F99",
    "\u5185",
    "\u65E5",
    "\u8F68",
    "\u8D35",
    "\u9519",
    "\u5408",
    "\u9C81",
    "\u5409",
    "\u5B81",
    "\u9519",
    "\u82CF",
    "\u6D77",
    "\u6E1D",
    "\u5B9C",
    "\u5609",
    "\u9526",
    "\u7CA4",
    "\u6B22",
    "\u8F6E",
    "\u81EA",
    "\u6C5F",
    "\u6CD5",
    "\u793E"
};

PlateRecognizer::PlateRecognizer() : pImpl_(std::make_unique<Impl>()) {}
PlateRecognizer::~PlateRecognizer() = default;

bool PlateRecognizer::loadModel(const std::string& modelPath) {
    bool success = pImpl_->session.loadModel(modelPath);
    pImpl_->loaded_ = success;
    if (!success) {
        pImpl_->error_message = pImpl_->session.lastError();
    }
    return success;
}

void PlateRecognizer::unload() {
    pImpl_->session.unload();
    pImpl_->loaded_ = false;
}

std::string PlateRecognizer::recognize(const cv::Mat& plateImage) {
    return pImpl_->recognize(plateImage);
}

void PlateRecognizer::setCharset(const std::vector<std::string>& charset) {
    pImpl_->charset_ = charset;
}

bool PlateRecognizer::isLoaded() const {
    return pImpl_->loaded_;
}

std::string PlateRecognizer::lastError() const {
    return pImpl_->error_message;
}

} // namespace platesniper
