#pragma once

#include <string>
#include <memory>
#include "core/plate_detector.h"
#include "core/plate_recognizer.h"

namespace platesniper {

class ModelManager {
public:
    ModelManager();
    ~ModelManager();

    bool initialize(const std::string& detectModel, const std::string& recModel);
    bool initialize(const std::string& configPath);
    void shutdown();

    PlateDetector* detector();
    PlateRecognizer* recognizer();

    bool isReady() const;
    std::string lastError() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace platesniper
