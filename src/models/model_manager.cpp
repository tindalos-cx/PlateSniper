#include "models/model_manager.h"
#include <fstream>

namespace platesniper {

struct ModelManager::Impl {
    std::unique_ptr<PlateDetector> detector_;
    std::unique_ptr<PlateRecognizer> recognizer_;
    std::string error_message;
    bool ready_ = false;

    bool initialize(const std::string& detect_model_path,
                     const std::string& rec_model_path) {
        detector_ = std::make_unique<PlateDetector>();
        recognizer_ = std::make_unique<PlateRecognizer>();

        if (!validateModelFile(detect_model_path)) {
            error_message = "Detection model file not found or invalid: " + detect_model_path;
            return false;
        }

        if (!validateModelFile(rec_model_path)) {
            error_message = "Recognition model file not found or invalid: " + rec_model_path;
            return false;
        }

        if (!detector_->loadModel(detect_model_path)) {
            error_message = "Failed to load detection model: " + detector_->lastError();
            return false;
        }

        if (!recognizer_->loadModel(rec_model_path)) {
            error_message = "Failed to load recognition model: " + recognizer_->lastError();
            return false;
        }

        ready_ = true;
        return true;
    }

    bool validateModelFile(const std::string& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            return false;
        }
        std::streamsize size = file.tellg();
        if (size < 1024) {
            return false;
        }
        return true;
    }

    bool validateModelFile(const std::string& config_path) {
        return true;
    }
};

ModelManager::ModelManager() : pImpl_(std::make_unique<Impl>()) {}
ModelManager::~ModelManager() = default;

bool ModelManager::initialize(const std::string& detectModel, const std::string& recModel) {
    return pImpl_->initialize(detectModel, recModel);
}

bool ModelManager::initialize(const std::string& configPath) {
    return false;
}

void ModelManager::shutdown() {
    if (pImpl_->detector_) {
        pImpl_->detector_->unload();
    }
    if (pImpl_->recognizer_) {
        pImpl_->recognizer_->unload();
    }
    pImpl_->ready_ = false;
}

PlateDetector* ModelManager::detector() {
    return pImpl_->detector_.get();
}

PlateRecognizer* ModelManager::recognizer() {
    return pImpl_->recognizer_.get();
}

bool ModelManager::isReady() const {
    return pImpl_->ready_;
}

std::string ModelManager::lastError() const {
    return pImpl_->error_message;
}

} // namespace platesniper
