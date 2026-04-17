#include "models/onnx_session.h"
#include <onnxruntime_cxx_api.h>
#include <stdexcept>
#include <cstring>

namespace platesniper {

struct OnnxSession::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "PlateSniper"};
    Ort::Session session_{nullptr};
    Ort::AllocatorWithDefaultOptions allocator_;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;

    std::string error_message;
    bool loaded = false;

    bool initialize(const std::string& model_path) {
        try {
            Ort::SessionOptions session_options;
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef _WIN32
            std::wstring wmodel_path = std::wstring(model_path.begin(), model_path.end());
            session_ = Ort::Session(env, wmodel_path.c_str(), session_options);
#else
            session_ = Ort::Session(env, model_path.c_str(), session_options);
#endif

            size_t num_input_nodes = session_.GetInputCount();
            size_t num_output_nodes = session_.GetOutputCount();

            input_names.resize(num_input_nodes);
            input_shapes.resize(num_input_nodes);
            output_names.resize(num_output_nodes);
            output_shapes.resize(num_output_nodes);

            for (size_t i = 0; i < num_input_nodes; ++i) {
                auto input_name = session_.GetInputNameAllocated(i, allocator_);
                input_names[i] = input_name.get();

                auto input_shape = session_.GetInputShape(i);
                input_shapes[i] = input_shape;
            }

            for (size_t i = 0; i < num_output_nodes; ++i) {
                auto output_name = session_.GetOutputNameAllocated(i, allocator_);
                output_names[i] = output_name.get();

                auto output_shape = session_.GetOutputShape(i);
                output_shapes[i] = output_shape;
            }

            loaded = true;
            return true;
        } catch (const Ort::Exception& e) {
            error_message = std::string("ONNX Error: ") + e.what();
            return false;
        } catch (const std::exception& e) {
            error_message = std::string("Error: ") + e.what();
            return false;
        }
    }

    std::vector<Ort::Value> run(const std::vector<Ort::Value>& input_values) {
        return session_.Run(Ort::RunOptions{nullptr},
                           input_names.data(), input_values.data(), input_names.size(),
                           output_names.data(), output_names.size());
    }
};

OnnxSession::OnnxSession() : pImpl_(std::make_unique<Impl>()) {}

OnnxSession::~OnnxSession() = default;

bool OnnxSession::loadModel(const std::string& modelPath) {
    return pImpl_->initialize(modelPath);
}

void OnnxSession::unload() {
    pImpl_->session_.release();
    pImpl_->loaded = false;
}

std::vector<cv::Mat> OnnxSession::inference(const std::vector<cv::Mat>& inputs) {
    std::vector<cv::Mat> outputs;

    if (!pImpl_->loaded || inputs.empty()) {
        return outputs;
    }

    try {
        std::vector<Ort::Value> input_values;
        std::vector<std::vector<float>> input_buffers;
        std::vector<Ort::MemoryInfo> memory_info;

        for (size_t i = 0; i < inputs.size(); ++i) {
            const cv::Mat& mat = inputs[i];

            memory_info.push_back(Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault));

            std::vector<int64_t> shape = pImpl_->input_shapes[i];

            if (mat.channels() == 3) {
                cv::Mat rgb_mat;
                cv::cvtColor(mat, rgb_mat, cv::COLOR_BGR2RGB);
                shape[1] = rgb_mat.rows;
                shape[2] = rgb_mat.cols;
                shape[3] = rgb_mat.channels();

                std::vector<float> buffer(shape[1] * shape[2] * shape[3]);
                rgb_mat.convertTo(rgb_mat, CV_32FC3);
                std::memcpy(buffer.data(), rgb_mat.data,
                           buffer.size() * sizeof(float));
                input_buffers.push_back(std::move(buffer));
            } else {
                std::vector<float> buffer(shape[1] * shape[2]);
                mat.convertTo(mat, CV_32FC1);
                std::memcpy(buffer.data(), mat.data,
                           buffer.size() * sizeof(float));
                input_buffers.push_back(std::move(buffer));
            }

            input_values.push_back(Ort::Value::CreateTensor<float>(
                memory_info.back(),
                input_buffers.back().data(),
                input_buffers.back().size(),
                shape.data(), shape.size()));
        }

        auto output_values = pImpl_->run(input_values);

        for (auto& value : output_values) {
            auto tensor_info = value.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            auto data = value.GetTensorData<float>();

            cv::Mat output_mat(static_cast<int>(shape[2]),
                              static_cast<int>(shape[3]),
                              CV_32F, const_cast<float*>(data));
            output_mat = output_mat.clone();
            outputs.push_back(output_mat);
        }
    } catch (const Ort::Exception& e) {
        pImpl_->error_message = std::string("Inference Error: ") + e.what();
    }

    return outputs;
}

std::vector<int64_t> OnnxSession::getInputShape() const {
    if (!pImpl_->input_shapes.empty()) {
        return pImpl_->input_shapes[0];
    }
    return {};
}

std::vector<int64_t> OnnxSession::getOutputShape() const {
    if (!pImpl_->output_shapes.empty()) {
        return pImpl_->output_shapes[0];
    }
    return {};
}

bool OnnxSession::isLoaded() const {
    return pImpl_->loaded;
}

std::string OnnxSession::lastError() const {
    return pImpl_->error_message;
}

} // namespace platesniper
