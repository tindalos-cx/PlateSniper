#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace platesniper {

class PlateRecognizer {
public:
    PlateRecognizer();
    ~PlateRecognizer();

    bool loadModel(const std::string& modelPath);
    void unload();

    std::string recognize(const cv::Mat& plateImage);
    void setCharset(const std::vector<std::string>& charset);

    bool isLoaded() const;
    std::string lastError() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace platesniper
