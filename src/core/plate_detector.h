#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "core/detection.h"

namespace platesniper {

class PlateDetector {
public:
    PlateDetector();
    ~PlateDetector();

    bool loadModel(const std::string& modelPath);
    void unload();

    std::vector<Detection> detect(const cv::Mat& image, float confThreshold = 0.5f);
    void setInputSize(int width, int height);

    bool isLoaded() const;
    std::string lastError() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace platesniper
