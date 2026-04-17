#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>

namespace platesniper {

class OnnxSession {
public:
    OnnxSession();
    ~OnnxSession();

    bool loadModel(const std::string& modelPath);
    void unload();

    std::vector<cv::Mat> inference(const std::vector<cv::Mat>& inputs);
    std::vector<int64_t> getInputShape() const;
    std::vector<int64_t> getOutputShape() const;

    bool isLoaded() const;
    std::string lastError() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace platesniper
