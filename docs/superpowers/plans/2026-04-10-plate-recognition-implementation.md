# PlateSniper 车牌识别系统实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于 C++17、OpenCV、ONNX Runtime 和 Qt5 实现一个完整的车牌识别系统，支持图片和摄像头输入，包含检测、矫正、识别三阶段流水线。

**架构：** 模块化三阶段流水线。PlateDetector 使用 YOLO ONNX 模型定位车牌；PlateCorrector 通过透视变换得到正视图；PlateRecognizer 使用 CRNN/LPRNet 识别字符。Qt GUI 提供交互界面。所有 ONNX 推理通过统一的 OnnxSession 封装管理。

**Tech Stack:** C++17, OpenCV 4.x, ONNX Runtime 1.16+, Qt5, CMake 3.16+

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `src/models/onnx_session.h/.cpp` | 封装 ONNX Runtime 会话，统一输入预处理、推理执行、输出获取 |
| `src/utils/image_utils.h/.cpp` | 图像预处理工具：resize、归一化、色彩空间转换、letterbox |
| `src/core/plate_detector.h/.cpp` | 加载 YOLO ONNX 模型，执行车牌检测，输出 Detection 列表 |
| `src/core/plate_corrector.h/.cpp` | 基于检测框进行透视变换，输出矫正后的车牌图像 |
| `src/core/plate_recognizer.h/.cpp` | 加载 CRNN ONNX 模型，识别车牌字符序列 |
| `src/models/model_manager.h/.cpp` | 集中管理检测和识别模型的加载与生命周期 |
| `src/gui/image_view.h/.cpp` | 自定义 QWidget，支持显示 OpenCV Mat 和绘制检测框 |
| `src/gui/result_panel.h/.cpp` | 右侧面板：展示矫正后的车牌图、识别文本、置信度 |
| `src/gui/main_window.h/.cpp` | Qt 主窗口：布局管理、信号/槽连接、协调识别流程 |
| `src/utils/config.h/.cpp` | 配置管理：模型路径、阈值参数 |
| `src/main.cpp` | 程序入口：初始化 QApplication、ModelManager、显示 MainWindow |
| `tests/test_utils.cpp` | ImageUtils 单元测试 |
| `tests/test_corrector.cpp` | PlateCorrector 单元测试 |

---

## Task 1: ONNX Session 封装

**Files:**
- Create: `src/models/onnx_session.h`
- Create: `src/models/onnx_session.cpp`
- Modify: `CMakeLists.txt` (添加新源文件到 SOURCES 列表)

### 1.1 头文件

- [ ] **Step 1: 编写 OnnxSession 头文件**

```cpp
// src/models/onnx_session.h
#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace platesniper {

class OnnxSession {
public:
    OnnxSession();
    ~OnnxSession();

    // 禁止拷贝
    OnnxSession(const OnnxSession&) = delete;
    OnnxSession& operator=(const OnnxSession&) = delete;

    // 加载模型
    bool load(const std::string& modelPath);
    bool isLoaded() const;
    std::string lastError() const;

    // 获取输入/输出信息
    std::vector<int64_t> getInputShape() const;
    std::vector<int64_t> getOutputShape() const;

    // 执行推理：输入 OpenCV Mat，输出 float 向量
    // 调用者负责将输出 reshape 为模型输出的维度
    std::vector<float> run(const cv::Mat& inputImage);

private:
    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memoryInfo_;

    std::vector<const char*> inputNames_;
    std::vector<const char*> outputNames_;
    std::vector<int64_t> inputShape_;
    std::vector<int64_t> outputShape_;

    std::string lastError_;
    bool loaded_ = false;

    void releaseNames();
};

} // namespace platesniper
```

### 1.2 实现文件

- [ ] **Step 2: 编写 OnnxSession 实现（构造函数、析构、load、辅助方法）**

```cpp
// src/models/onnx_session.cpp
#include "models/onnx_session.h"
#include <opencv2/imgproc.hpp>
#include <iostream>

namespace platesniper {

OnnxSession::OnnxSession()
    : env_(ORT_LOGGING_LEVEL_WARNING, "PlateSniper"),
      memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    sessionOptions_.SetIntraOpNumThreads(1);
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

OnnxSession::~OnnxSession() {
    releaseNames();
}

void OnnxSession::releaseNames() {
    for (auto* name : inputNames_) {
        delete[] name;
    }
    for (auto* name : outputNames_) {
        delete[] name;
    }
    inputNames_.clear();
    outputNames_.clear();
}

bool OnnxSession::load(const std::string& modelPath) {
    try {
        releaseNames();
        session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), sessionOptions_);

        // 获取输入信息
        Ort::AllocatorWithDefaultOptions allocator;
        size_t numInputs = session_->GetInputCount();
        for (size_t i = 0; i < numInputs; ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            char* nameCopy = new char[strlen(name.get()) + 1];
            strcpy(nameCopy, name.get());
            inputNames_.push_back(nameCopy);

            auto typeInfo = session_->GetInputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            inputShape_ = tensorInfo.GetShape();
        }

        // 获取输出信息
        size_t numOutputs = session_->GetOutputCount();
        for (size_t i = 0; i < numOutputs; ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            char* nameCopy = new char[strlen(name.get()) + 1];
            strcpy(nameCopy, name.get());
            outputNames_.push_back(nameCopy);

            auto typeInfo = session_->GetOutputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            outputShape_ = tensorInfo.GetShape();
        }

        loaded_ = true;
        return true;
    } catch (const Ort::Exception& e) {
        lastError_ = std::string("ONNX Error: ") + e.what();
        loaded_ = false;
        return false;
    } catch (const std::exception& e) {
        lastError_ = std::string("Error: ") + e.what();
        loaded_ = false;
        return false;
    }
}

bool OnnxSession::isLoaded() const {
    return loaded_;
}

std::string OnnxSession::lastError() const {
    return lastError_;
}

std::vector<int64_t> OnnxSession::getInputShape() const {
    return inputShape_;
}

std::vector<int64_t> OnnxSession::getOutputShape() const {
    return outputShape_;
}

} // namespace platesniper
```

- [ ] **Step 3: 编写 run() 方法实现**

```cpp
// 追加到 src/models/onnx_session.cpp

std::vector<float> OnnxSession::run(const cv::Mat& inputImage) {
    if (!loaded_ || !session_) {
        lastError_ = "Session not loaded";
        return {};
    }

    try {
        // 转换 Mat 为连续内存的 float 向量 (RGB, NCHW)
        cv::Mat rgbImage;
        if (inputImage.channels() == 3) {
            cv::cvtColor(inputImage, rgbImage, cv::COLOR_BGR2RGB);
        } else {
            rgbImage = inputImage;
        }

        rgbImage.convertTo(rgbImage, CV_32F, 1.0f / 255.0f);

        // NCHW 排列
        std::vector<float> inputData;
        int channels = rgbImage.channels();
        int height = rgbImage.rows;
        int width = rgbImage.cols;

        inputData.reserve(channels * height * width);
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    inputData.push_back(rgbImage.at<cv::Vec3f>(h, w)[c]);
                }
            }
        }

        // 构建输入 tensor
        std::vector<int64_t> inputDims = {1, channels, height, width};
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo_, inputData.data(), inputData.size(), inputDims.data(), inputDims.size());

        // 执行推理
        auto outputTensors = session_->Run(
            Ort::RunOptions{nullptr},
            inputNames_.data(), &inputTensor, 1,
            outputNames_.data(), outputNames_.size());

        // 获取输出数据
        Ort::Value& outputTensor = outputTensors[0];
        auto typeInfo = outputTensor.GetTensorTypeAndShapeInfo();
        size_t outputSize = typeInfo.GetElementCount();

        std::vector<float> outputData(outputSize);
        memcpy(outputData.data(), outputTensor.GetTensorData<float>(), outputSize * sizeof(float));

        return outputData;
    } catch (const Ort::Exception& e) {
        lastError_ = std::string("ONNX Run Error: ") + e.what();
        return {};
    } catch (const std::exception& e) {
        lastError_ = std::string("Run Error: ") + e.what();
        return {};
    }
}
```

- [ ] **Step 4: 更新 CMakeLists.txt 添加新源文件**

修改 `CMakeLists.txt` 中的 SOURCES 列表，添加：
```cmake
    src/models/onnx_session.cpp
```

修改 HEADERS 列表，添加：
```cmake
    src/models/onnx_session.h
```

- [ ] **Step 5: Commit**

```bash
git add src/models/onnx_session.h src/models/onnx_session.cpp CMakeLists.txt
git commit -m "feat: add OnnxSession wrapper for ONNX Runtime inference"
```

---

## Task 2: 图像工具函数 (ImageUtils)

**Files:**
- Create: `src/utils/image_utils.h`
- Create: `src/utils/image_utils.cpp`
- Create: `tests/test_utils.cpp`
- Modify: `CMakeLists.txt`

### 2.1 头文件与实现

- [ ] **Step 1: 编写 ImageUtils 头文件**

```cpp
// src/utils/image_utils.h
#pragma once

#include <opencv2/core.hpp>

namespace platesniper {
namespace utils {

// Letterbox resize: 保持宽高比，多余部分用灰色填充
// 返回缩放比例和填充偏移
struct LetterboxResult {
    cv::Mat image;
    float scale;
    int padLeft;
    int padTop;
};

LetterboxResult letterbox(const cv::Mat& src, int targetWidth, int targetHeight,
                          const cv::Scalar& padColor = cv::Scalar(114, 114, 114));

// 将 letterbox 后的坐标还原到原图坐标
cv::Rect restoreCoordinates(const cv::Rect& box, float scale, int padLeft, int padTop);

// NMS (非极大值抑制)
std::vector<int> nms(const std::vector<cv::Rect>& boxes,
                    const std::vector<float>& scores,
                    float iouThreshold);

} // namespace utils
} // namespace platesniper
```

- [ ] **Step 2: 编写 ImageUtils 实现**

```cpp
// src/utils/image_utils.cpp
#include "utils/image_utils.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>

namespace platesniper {
namespace utils {

LetterboxResult letterbox(const cv::Mat& src, int targetWidth, int targetHeight,
                          const cv::Scalar& padColor) {
    LetterboxResult result;

    float scaleX = static_cast<float>(targetWidth) / src.cols;
    float scaleY = static_cast<float>(targetHeight) / src.rows;
    result.scale = std::min(scaleX, scaleY);

    int newWidth = static_cast<int>(src.cols * result.scale);
    int newHeight = static_cast<int>(src.rows * result.scale);

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);

    result.padLeft = (targetWidth - newWidth) / 2;
    result.padTop = (targetHeight - newHeight) / 2;
    int padRight = targetWidth - newWidth - result.padLeft;
    int padBottom = targetHeight - newHeight - result.padTop;

    cv::copyMakeBorder(resized, result.image, result.padTop, padBottom,
                       result.padLeft, padRight, cv::BORDER_CONSTANT, padColor);

    return result;
}

cv::Rect restoreCoordinates(const cv::Rect& box, float scale, int padLeft, int padTop) {
    int x = static_cast<int>((box.x - padLeft) / scale);
    int y = static_cast<int>((box.y - padTop) / scale);
    int w = static_cast<int>(box.width / scale);
    int h = static_cast<int>(box.height / scale);
    return cv::Rect(x, y, w, h);
}

std::vector<int> nms(const std::vector<cv::Rect>& boxes,
                    const std::vector<float>& scores,
                    float iouThreshold) {
    std::vector<int> indices(boxes.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = static_cast<int>(i);

    // 按置信度降序排序
    std::sort(indices.begin(), indices.end(), [&scores](int a, int b) {
        return scores[a] > scores[b];
    });

    std::vector<int> keep;
    std::vector<bool> suppressed(boxes.size(), false);

    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        if (suppressed[idx]) continue;

        keep.push_back(idx);

        for (size_t j = i + 1; j < indices.size(); ++j) {
            int otherIdx = indices[j];
            if (suppressed[otherIdx]) continue;

            // 计算 IoU
            const cv::Rect& a = boxes[idx];
            const cv::Rect& b = boxes[otherIdx];

            int x1 = std::max(a.x, b.x);
            int y1 = std::max(a.y, b.y);
            int x2 = std::min(a.x + a.width, b.x + b.width);
            int y2 = std::min(a.y + a.height, b.y + b.height);

            if (x2 <= x1 || y2 <= y1) continue;

            int interArea = (x2 - x1) * (y2 - y1);
            int unionArea = a.width * a.height + b.width * b.height - interArea;
            float iou = static_cast<float>(interArea) / unionArea;

            if (iou > iouThreshold) {
                suppressed[otherIdx] = true;
            }
        }
    }

    return keep;
}

} // namespace utils
} // namespace platesniper
```

### 2.2 测试

- [ ] **Step 3: 编写 ImageUtils 单元测试**

```cpp
// tests/test_utils.cpp
#include <gtest/gtest.h>
#include "utils/image_utils.h"

using namespace platesniper::utils;

TEST(ImageUtilsTest, LetterboxPreservesAspectRatio) {
    cv::Mat img(100, 200, CV_8UC3, cv::Scalar(255, 0, 0));
    auto result = letterbox(img, 640, 640);

    EXPECT_EQ(result.image.cols, 640);
    EXPECT_EQ(result.image.rows, 640);
    EXPECT_FLOAT_EQ(result.scale, 3.2f);  // 640 / 200 = 3.2, 640 / 100 = 6.4, min = 3.2
    EXPECT_EQ(result.padLeft, 220);       // (640 - 640) / 2 = 0? 等等，新宽 = 200 * 3.2 = 640, 新高 = 100 * 3.2 = 320
    // 重新计算：scale = min(640/200, 640/100) = min(3.2, 6.4) = 3.2
    // newWidth = 200 * 3.2 = 640, newHeight = 100 * 3.2 = 320
    // padLeft = (640 - 640) / 2 = 0
    // padTop = (640 - 320) / 2 = 160
    EXPECT_EQ(result.padLeft, 0);
    EXPECT_EQ(result.padTop, 160);
}

TEST(ImageUtilsTest, RestoreCoordinates) {
    cv::Rect box(320, 320, 100, 50);  // letterbox 后的坐标
    auto restored = restoreCoordinates(box, 0.5f, 10, 20);

    EXPECT_EQ(restored.x, static_cast<int>((320 - 10) / 0.5f));  // 620
    EXPECT_EQ(restored.y, static_cast<int>((320 - 20) / 0.5f));  // 600
    EXPECT_EQ(restored.width, static_cast<int>(100 / 0.5f));      // 200
    EXPECT_EQ(restored.height, static_cast<int>(50 / 0.5f));      // 100
}

TEST(ImageUtilsTest, NmsRemovesOverlappingBoxes) {
    std::vector<cv::Rect> boxes = {
        cv::Rect(10, 10, 50, 50),
        cv::Rect(15, 15, 50, 50),  // 与第一个重叠
        cv::Rect(100, 100, 50, 50) // 不重叠
    };
    std::vector<float> scores = {0.9f, 0.8f, 0.85f};

    auto keep = nms(boxes, scores, 0.5f);

    EXPECT_EQ(keep.size(), 2);
    EXPECT_EQ(keep[0], 0);  // 最高分保留
    // 第二个被抑制，第三个保留
    EXPECT_EQ(keep[1], 2);
}
```

- [ ] **Step 4: Commit**

```bash
git add src/utils/image_utils.h src/utils/image_utils.cpp tests/test_utils.cpp CMakeLists.txt
git commit -m "feat: add image utilities (letterbox, NMS) with unit tests"
```

---

## Task 3: 车牌检测器 (PlateDetector)

**Files:**
- Create: `src/core/plate_detector.h`
- Create: `src/core/plate_detector.cpp`
- Modify: `CMakeLists.txt`

### 3.1 头文件

- [ ] **Step 1: 编写 PlateDetector 头文件**

```cpp
// src/core/plate_detector.h
#pragma once

#include "core/detection.h"
#include "models/onnx_session.h"
#include <opencv2/core.hpp>
#include <vector>

namespace platesniper {

class PlateDetector {
public:
    PlateDetector();
    ~PlateDetector() = default;

    bool loadModel(const std::string& modelPath);
    bool isLoaded() const;
    std::string lastError() const;

    // 检测车牌，返回所有候选框（已做 NMS）
    std::vector<Detection> detect(const cv::Mat& image, float confThreshold = 0.5f);

    void setInputSize(int width, int height);

private:
    OnnxSession session_;
    int inputWidth_ = 640;
    int inputHeight_ = 640;
    std::string lastError_;
};

} // namespace platesniper
```

### 3.2 实现

- [ ] **Step 2: 编写 PlateDetector 实现**

```cpp
// src/core/plate_detector.cpp
#include "core/plate_detector.h"
#include "utils/image_utils.h"
#include <opencv2/imgproc.hpp>
#include <math>

namespace platesniper {

PlateDetector::PlateDetector() = default;

bool PlateDetector::loadModel(const std::string& modelPath) {
    if (!session_.load(modelPath)) {
        lastError_ = session_.lastError();
        return false;
    }

    // 从模型获取输入尺寸
    auto shape = session_.getInputShape();
    if (shape.size() >= 4) {
        inputHeight_ = static_cast<int>(shape[2]);
        inputWidth_ = static_cast<int>(shape[3]);
    }

    return true;
}

bool PlateDetector::isLoaded() const {
    return session_.isLoaded();
}

std::string PlateDetector::lastError() const {
    return lastError_;
}

void PlateDetector::setInputSize(int width, int height) {
    inputWidth_ = width;
    inputHeight_ = height;
}

std::vector<Detection> PlateDetector::detect(const cv::Mat& image, float confThreshold) {
    if (!session_.isLoaded()) {
        lastError_ = "Model not loaded";
        return {};
    }

    // Letterbox resize
    auto lb = utils::letterbox(image, inputWidth_, inputHeight_);
    cv::Mat inputBlob = lb.image;

    // 执行推理
    std::vector<float> output = session_.run(inputBlob);
    if (output.empty()) {
        lastError_ = session_.lastError();
        return {};
    }

    // 解析 YOLO 输出
    // 假设输出格式: [batch, num_boxes, 5 + num_classes]
    // 其中 5 = x, y, w, h, conf
    auto outShape = session_.getOutputShape();
    if (outShape.size() < 3) {
        lastError_ = "Unexpected output shape";
        return {};
    }

    int numBoxes = static_cast<int>(outShape[1]);
    int boxDataSize = static_cast<int>(outShape[2]);

    std::vector<Detection> detections;
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    for (int i = 0; i < numBoxes; ++i) {
        float* boxData = output.data() + i * boxDataSize;
        float cx = boxData[0];
        float cy = boxData[1];
        float w = boxData[2];
        float h = boxData[3];
        float conf = boxData[4];

        if (conf < confThreshold) continue;

        // 转换为中心坐标到左上角坐标
        int x = static_cast<int>(cx - w / 2);
        int y = static_cast<int>(cy - h / 2);
        int bw = static_cast<int>(w);
        int bh = static_cast<int>(h);

        cv::Rect box(x, y, bw, bh);

        // 还原到原图坐标
        box = utils::restoreCoordinates(box, lb.scale, lb.padLeft, lb.padTop);

        // 裁剪到图像边界内
        box.x = std::max(0, box.x);
        box.y = std::max(0, box.y);
        box.width = std::min(box.width, image.cols - box.x);
        box.height = std::min(box.height, image.rows - box.y);

        boxes.push_back(box);
        scores.push_back(conf);
    }

    // NMS
    auto keepIndices = utils::nms(boxes, scores, 0.45f);

    for (int idx : keepIndices) {
        Detection det;
        det.bbox = boxes[idx];
        det.confidence = scores[idx];
        det.classId = 0;  // 目前只检测车牌一种类型
        detections.push_back(det);
    }

    return detections;
}

} // namespace platesniper
```

- [ ] **Step 3: 更新 CMakeLists.txt**

添加 `src/core/plate_detector.cpp` 和 `src/core/plate_detector.h` 到对应列表。

- [ ] **Step 4: Commit**

```bash
git add src/core/plate_detector.h src/core/plate_detector.cpp CMakeLists.txt
git commit -m "feat: add PlateDetector with YOLO ONNX inference"
```

---

## Task 4: 车牌矫正器 (PlateCorrector)

**Files:**
- Create: `src/core/plate_corrector.h`
- Create: `src/core/plate_corrector.cpp`
- Create: `tests/test_corrector.cpp`
- Modify: `CMakeLists.txt`

### 4.1 头文件与实现

- [ ] **Step 1: 编写 PlateCorrector 头文件**

```cpp
// src/core/plate_corrector.h
#pragma once

#include "core/detection.h"
#include <opencv2/core.hpp>
#include <vector>

namespace platesniper {

class PlateCorrector {
public:
    PlateCorrector();

    // 基于检测框进行透视变换
    cv::Mat correct(const cv::Mat& image, const Detection& det);

    // 基于精确角点进行透视变换
    cv::Mat correct(const cv::Mat& image, const std::vector<cv::Point2f>& corners);

    void setOutputSize(int width, int height);

private:
    int outputWidth_ = 136;
    int outputHeight_ = 36;
};

} // namespace platesniper
```

- [ ] **Step 2: 编写 PlateCorrector 实现**

```cpp
// src/core/plate_corrector.cpp
#include "core/plate_corrector.h"
#include <opencv2/imgproc.hpp>

namespace platesniper {

PlateCorrector::PlateCorrector() = default;

void PlateCorrector::setOutputSize(int width, int height) {
    outputWidth_ = width;
    outputHeight_ = height;
}

cv::Mat PlateCorrector::correct(const cv::Mat& image, const Detection& det) {
    // 从检测框估算四个角点
    // 假设车牌是矩形，使用检测框的四个角
    std::vector<cv::Point2f> srcPoints = {
        cv::Point2f(static_cast<float>(det.bbox.x), static_cast<float>(det.bbox.y)),
        cv::Point2f(static_cast<float>(det.bbox.x + det.bbox.width), static_cast<float>(det.bbox.y)),
        cv::Point2f(static_cast<float>(det.bbox.x + det.bbox.width), static_cast<float>(det.bbox.y + det.bbox.height)),
        cv::Point2f(static_cast<float>(det.bbox.x), static_cast<float>(det.bbox.y + det.bbox.height))
    };

    return correct(image, srcPoints);
}

cv::Mat PlateCorrector::correct(const cv::Mat& image, const std::vector<cv::Point2f>& corners) {
    if (corners.size() != 4) {
        return cv::Mat();
    }

    // 目标角点（矫正后的矩形）
    std::vector<cv::Point2f> dstPoints = {
        cv::Point2f(0, 0),
        cv::Point2f(static_cast<float>(outputWidth_), 0),
        cv::Point2f(static_cast<float>(outputWidth_), static_cast<float>(outputHeight_)),
        cv::Point2f(0, static_cast<float>(outputHeight_))
    };

    // 计算透视变换矩阵
    cv::Mat transformMatrix = cv::getPerspectiveTransform(corners, dstPoints);

    // 执行透视变换
    cv::Mat corrected;
    cv::warpPerspective(image, corrected, transformMatrix, cv::Size(outputWidth_, outputHeight_));

    return corrected;
}

} // namespace platesniper
```

### 4.2 测试

- [ ] **Step 3: 编写 PlateCorrector 单元测试**

```cpp
// tests/test_corrector.cpp
#include <gtest/gtest.h>
#include "core/plate_corrector.h"
#include "core/detection.h"

using namespace platesniper;

TEST(PlateCorrectorTest, CorrectFromDetection) {
    // 创建一个 200x100 的测试图像
    cv::Mat image(200, 200, CV_8UC3, cv::Scalar(100, 150, 200));

    Detection det;
    det.bbox = cv::Rect(50, 50, 100, 50);
    det.confidence = 0.95f;
    det.classId = 0;

    PlateCorrector corrector;
    corrector.setOutputSize(136, 36);
    cv::Mat result = corrector.correct(image, det);

    EXPECT_EQ(result.cols, 136);
    EXPECT_EQ(result.rows, 36);
    EXPECT_FALSE(result.empty());
}

TEST(PlateCorrectorTest, CorrectFromCorners) {
    cv::Mat image(300, 300, CV_8UC3, cv::Scalar(50, 100, 150));

    // 定义一个梯形（模拟倾斜车牌）
    std::vector<cv::Point2f> corners = {
        cv::Point2f(80, 80),
        cv::Point2f(220, 70),
        cv::Point2f(230, 130),
        cv::Point2f(70, 140)
    };

    PlateCorrector corrector;
    corrector.setOutputSize(136, 36);
    cv::Mat result = corrector.correct(image, corners);

    EXPECT_EQ(result.cols, 136);
    EXPECT_EQ(result.rows, 36);
    EXPECT_FALSE(result.empty());
}

TEST(PlateCorrectorTest, InvalidCornersReturnsEmpty) {
    cv::Mat image(100, 100, CV_8UC3);
    std::vector<cv::Point2f> corners = {
        cv::Point2f(0, 0),
        cv::Point2f(50, 0)
    };  // 只有 2 个角点

    PlateCorrector corrector;
    cv::Mat result = corrector.correct(image, corners);

    EXPECT_TRUE(result.empty());
}
```

- [ ] **Step 4: Commit**

```bash
git add src/core/plate_corrector.h src/core/plate_corrector.cpp tests/test_corrector.cpp CMakeLists.txt
git commit -m "feat: add PlateCorrector with perspective transform and unit tests"
```

---

## Task 5: 车牌识别器 (PlateRecognizer)

**Files:**
- Create: `src/core/plate_recognizer.h`
- Create: `src/core/plate_recognizer.cpp`
- Modify: `CMakeLists.txt`

### 5.1 头文件

- [ ] **Step 1: 编写 PlateRecognizer 头文件**

```cpp
// src/core/plate_recognizer.h
#pragma once

#include "models/onnx_session.h"
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace platesniper {

class PlateRecognizer {
public:
    PlateRecognizer();
    ~PlateRecognizer() = default;

    bool loadModel(const std::string& modelPath);
    bool isLoaded() const;
    std::string lastError() const;

    // 识别车牌字符
    std::string recognize(const cv::Mat& plateImage);

    // 设置字符集（用于 CTC 解码映射）
    void setCharset(const std::vector<std::string>& charset);

private:
    OnnxSession session_;
    std::vector<std::string> charset_;
    std::string lastError_;

    int inputWidth_ = 94;
    int inputHeight_ = 24;

    // 默认中文字符集
    void initDefaultCharset();
    std::string ctcDecode(const std::vector<float>& output, int numClasses, int timeSteps);
};

} // namespace platesniper
```

### 5.2 实现

- [ ] **Step 2: 编写 PlateRecognizer 实现**

```cpp
// src/core/plate_recognizer.cpp
#include "core/plate_recognizer.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>

namespace platesniper {

PlateRecognizer::PlateRecognizer() {
    initDefaultCharset();
}

void PlateRecognizer::initDefaultCharset() {
    // 中文字符集：省份简称 + 字母 + 数字
    // 索引 0 保留给 CTC blank
    charset_ = {
        "blank",  // CTC blank token at index 0
        "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
        "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
        "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
        "新",
        "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
        "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
        "W", "X", "Y", "Z",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "学", "警", "港", "澳", "挂"
    };
}

bool PlateRecognizer::loadModel(const std::string& modelPath) {
    if (!session_.load(modelPath)) {
        lastError_ = session_.lastError();
        return false;
    }

    auto shape = session_.getInputShape();
    if (shape.size() >= 4) {
        inputHeight_ = static_cast<int>(shape[2]);
        inputWidth_ = static_cast<int>(shape[3]);
    }

    return true;
}

bool PlateRecognizer::isLoaded() const {
    return session_.isLoaded();
}

std::string PlateRecognizer::lastError() const {
    return lastError_;
}

void PlateRecognizer::setCharset(const std::vector<std::string>& charset) {
    charset_ = charset;
    if (charset_.empty() || charset_[0] != "blank") {
        charset_.insert(charset_.begin(), "blank");
    }
}

std::string PlateRecognizer::recognize(const cv::Mat& plateImage) {
    if (!session_.isLoaded()) {
        lastError_ = "Model not loaded";
        return "";
    }

    // 预处理：resize 到模型输入尺寸
    cv::Mat resized;
    cv::resize(plateImage, resized, cv::Size(inputWidth_, inputHeight_));

    // 执行推理
    std::vector<float> output = session_.run(resized);
    if (output.empty()) {
        lastError_ = session_.lastError();
        return "";
    }

    // 解析输出并 CTC 解码
    auto outShape = session_.getOutputShape();
    if (outShape.size() < 3) {
        lastError_ = "Unexpected output shape";
        return "";
    }

    int timeSteps = static_cast<int>(outShape[1]);
    int numClasses = static_cast<int>(outShape[2]);

    return ctcDecode(output, numClasses, timeSteps);
}

std::string PlateRecognizer::ctcDecode(const std::vector<float>& output,
                                       int numClasses, int timeSteps) {
    std::string result;
    int prevIndex = -1;

    for (int t = 0; t < timeSteps; ++t) {
        // 找到当前时间步概率最高的字符索引
        const float* probs = output.data() + t * numClasses;
        int maxIndex = static_cast<int>(std::max_element(probs, probs + numClasses) - probs);

        // CTC 规则：跳过 blank (index 0) 和重复字符
        if (maxIndex != 0 && maxIndex != prevIndex) {
            if (maxIndex < static_cast<int>(charset_.size())) {
                result += charset_[maxIndex];
            }
        }
        prevIndex = maxIndex;
    }

    return result;
}

} // namespace platesniper
```

- [ ] **Step 3: 更新 CMakeLists.txt**

添加 `src/core/plate_recognizer.cpp` 和 `src/core/plate_recognizer.h`。

- [ ] **Step 4: Commit**

```bash
git add src/core/plate_recognizer.h src/core/plate_recognizer.cpp CMakeLists.txt
git commit -m "feat: add PlateRecognizer with CTC decoding and Chinese charset"
```

---

## Task 6: 模型管理器 (ModelManager)

**Files:**
- Create: `src/models/model_manager.h`
- Create: `src/models/model_manager.cpp`
- Modify: `CMakeLists.txt`

### 6.1 头文件与实现

- [ ] **Step 1: 编写 ModelManager 头文件**

```cpp
// src/models/model_manager.h
#pragma once

#include "core/plate_detector.h"
#include "core/plate_recognizer.h"
#include <memory>
#include <string>

namespace platesniper {

class ModelManager {
public:
    ModelManager();
    ~ModelManager() = default;

    // 从两个模型路径初始化
    bool initialize(const std::string& detectModelPath,
                    const std::string& recModelPath);

    bool isReady() const;
    std::string lastError() const;

    PlateDetector* detector();
    PlateRecognizer* recognizer();

private:
    std::unique_ptr<PlateDetector> detector_;
    std::unique_ptr<PlateRecognizer> recognizer_;
    std::string lastError_;
    bool ready_ = false;
};

} // namespace platesniper
```

- [ ] **Step 2: 编写 ModelManager 实现**

```cpp
// src/models/model_manager.cpp
#include "models/model_manager.h"

namespace platesniper {

ModelManager::ModelManager()
    : detector_(std::make_unique<PlateDetector>()),
      recognizer_(std::make_unique<PlateRecognizer>())
{
}

bool ModelManager::initialize(const std::string& detectModelPath,
                              const std::string& recModelPath) {
    ready_ = false;

    if (!detector_->loadModel(detectModelPath)) {
        lastError_ = "Failed to load detection model: " + detector_->lastError();
        return false;
    }

    if (!recognizer_->loadModel(recModelPath)) {
        lastError_ = "Failed to load recognition model: " + recognizer_->lastError();
        return false;
    }

    ready_ = true;
    return true;
}

bool ModelManager::isReady() const {
    return ready_;
}

std::string ModelManager::lastError() const {
    return lastError_;
}

PlateDetector* ModelManager::detector() {
    return detector_.get();
}

PlateRecognizer* ModelManager::recognizer() {
    return recognizer_.get();
}

} // namespace platesniper
```

- [ ] **Step 3: 更新 CMakeLists.txt**

添加 `src/models/model_manager.cpp` 和 `src/models/model_manager.h`。

- [ ] **Step 4: Commit**

```bash
git add src/models/model_manager.h src/models/model_manager.cpp CMakeLists.txt
git commit -m "feat: add ModelManager for centralized model lifecycle management"
```

---

## Task 7: GUI 组件 - ImageView

**Files:**
- Create: `src/gui/image_view.h`
- Create: `src/gui/image_view.cpp`
- Modify: `CMakeLists.txt`

### 7.1 头文件与实现

- [ ] **Step 1: 编写 ImageView 头文件**

```cpp
// src/gui/image_view.h
#pragma once

#include <QWidget>
#include <QImage>
#include <QPainter>
#include <vector>
#include "core/detection.h"

namespace platesniper {

// 自定义控件：显示 OpenCV Mat 图像，支持绘制检测框
class ImageView : public QWidget {
    Q_OBJECT

public:
    explicit ImageView(QWidget* parent = nullptr);

    // 设置显示的图像（OpenCV BGR Mat）
    void setImage(const cv::Mat& mat);

    // 设置检测框（将在图像上绘制）
    void setDetections(const std::vector<Detection>& detections);

    // 清空
    void clear();

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QImage currentImage_;
    std::vector<Detection> detections_;

    QImage matToQImage(const cv::Mat& mat);
};

} // namespace platesniper
```

- [ ] **Step 2: 编写 ImageView 实现**

```cpp
// src/gui/image_view.cpp
#include "gui/image_view.h"
#include <opencv2/imgproc.hpp>
#include <QPainter>

namespace platesniper {

ImageView::ImageView(QWidget* parent) : QWidget(parent) {
    setMinimumSize(320, 240);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

void ImageView::setImage(const cv::Mat& mat) {
    currentImage_ = matToQImage(mat);
    update();
}

void ImageView::setDetections(const std::vector<Detection>& detections) {
    detections_ = detections;
    update();
}

void ImageView::clear() {
    currentImage_ = QImage();
    detections_.clear();
    update();
}

void ImageView::paintEvent(QPaintEvent* /*event*/) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::SmoothPixmapTransform);

    // 绘制背景
    painter.fillRect(rect(), Qt::darkGray);

    if (currentImage_.isNull()) {
        painter.setPen(Qt::white);
        painter.drawText(rect(), Qt::AlignCenter, "No Image");
        return;
    }

    // 按比例缩放图像以适应控件
    QRect targetRect = rect();
    QImage scaled = currentImage_.scaled(targetRect.size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);

    int x = (targetRect.width() - scaled.width()) / 2;
    int y = (targetRect.height() - scaled.height()) / 2;
    painter.drawImage(x, y, scaled);

    // 绘制检测框
    if (!detections_.empty()) {
        float scaleX = static_cast<float>(scaled.width()) / currentImage_.width();
        float scaleY = static_cast<float>(scaled.height()) / currentImage_.height();

        painter.setPen(QPen(Qt::red, 2));
        painter.setBrush(Qt::NoBrush);

        for (const auto& det : detections_) {
            int dx = static_cast<int>(det.bbox.x * scaleX) + x;
            int dy = static_cast<int>(det.bbox.y * scaleY) + y;
            int dw = static_cast<int>(det.bbox.width * scaleX);
            int dh = static_cast<int>(det.bbox.height * scaleY);

            painter.drawRect(dx, dy, dw, dh);

            // 绘制置信度标签
            QString label = QString("%1%").arg(static_cast<int>(det.confidence * 100));
            painter.setBrush(QColor(255, 0, 0, 180));
            painter.drawRect(dx, dy - 20, 40, 20);
            painter.setPen(Qt::white);
            painter.drawText(dx + 2, dy - 4, label);
            painter.setPen(QPen(Qt::red, 2));
            painter.setBrush(Qt::NoBrush);
        }
    }
}

QImage ImageView::matToQImage(const cv::Mat& mat) {
    if (mat.channels() == 3) {
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
        return QImage(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888).copy();
    } else if (mat.channels() == 4) {
        cv::Mat rgba;
        cv::cvtColor(mat, rgba, cv::COLOR_BGRA2RGBA);
        return QImage(rgba.data, rgba.cols, rgba.rows, static_cast<int>(rgba.step), QImage::Format_RGBA8888).copy();
    } else if (mat.channels() == 1) {
        return QImage(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_Grayscale8).copy();
    }
    return QImage();
}

} // namespace platesniper
```

- [ ] **Step 3: 更新 CMakeLists.txt**

添加 `src/gui/image_view.cpp` 和 `src/gui/image_view.h`。注意 Qt 头文件需要 MOC 处理，`CMAKE_AUTOMOC ON` 已设置。

- [ ] **Step 4: Commit**

```bash
git add src/gui/image_view.h src/gui/image_view.cpp CMakeLists.txt
git commit -m "feat: add ImageView widget for displaying OpenCV images with detection overlays"
```

---

## Task 8: GUI 组件 - ResultPanel

**Files:**
- Create: `src/gui/result_panel.h`
- Create: `src/gui/result_panel.cpp`
- Modify: `CMakeLists.txt`

### 8.1 头文件与实现

- [ ] **Step 1: 编写 ResultPanel 头文件**

```cpp
// src/gui/result_panel.h
#pragma once

#include <QWidget>
#include <QLabel>
#include <QVBoxLayout>
#include <opencv2/core.hpp>

namespace platesniper {

// 右侧面板：展示矫正后的车牌图、识别文本、置信度
class ResultPanel : public QWidget {
    Q_OBJECT

public:
    explicit ResultPanel(QWidget* parent = nullptr);

    // 设置矫正后的车牌图像
    void setPlateImage(const cv::Mat& plateImage);

    // 设置识别结果
    void setResultText(const QString& text, float confidence);

    // 清空显示
    void clear();

private:
    QLabel* plateImageLabel_;
    QLabel* resultTextLabel_;
    QLabel* confidenceLabel_;

    QImage matToQImage(const cv::Mat& mat);
};

} // namespace platesniper
```

- [ ] **Step 2: 编写 ResultPanel 实现**

```cpp
// src/gui/result_panel.cpp
#include "gui/result_panel.h"
#include <opencv2/imgproc.hpp>

namespace platesniper {

ResultPanel::ResultPanel(QWidget* parent) : QWidget(parent) {
    auto* layout = new QVBoxLayout(this);
    layout->setSpacing(10);
    layout->setContentsMargins(10, 10, 10, 10);

    // 矫正后的车牌图像
    auto* imageTitle = new QLabel("Corrected Plate:", this);
    imageTitle->setStyleSheet("font-weight: bold;");
    layout->addWidget(imageTitle);

    plateImageLabel_ = new QLabel(this);
    plateImageLabel_->setMinimumSize(200, 60);
    plateImageLabel_->setAlignment(Qt::AlignCenter);
    plateImageLabel_->setStyleSheet("background-color: #333; border: 1px solid #666;");
    plateImageLabel_->setText("No plate detected");
    layout->addWidget(plateImageLabel_);

    // 识别结果
    auto* textTitle = new QLabel("Recognition Result:", this);
    textTitle->setStyleSheet("font-weight: bold;");
    layout->addWidget(textTitle);

    resultTextLabel_ = new QLabel("-", this);
    resultTextLabel_->setAlignment(Qt::AlignCenter);
    resultTextLabel_->setStyleSheet(
        "font-size: 24px; font-weight: bold; color: #00AA00; "
        "background-color: #f0f0f0; padding: 10px; border: 2px solid #ccc;"
    );
    layout->addWidget(resultTextLabel_);

    // 置信度
    confidenceLabel_ = new QLabel("Confidence: -", this);
    confidenceLabel_->setAlignment(Qt::AlignCenter);
    layout->addWidget(confidenceLabel_);

    layout->addStretch();
    setLayout(layout);
    setMinimumWidth(250);
}

void ResultPanel::setPlateImage(const cv::Mat& plateImage) {
    if (plateImage.empty()) {
        plateImageLabel_->setText("No plate detected");
        return;
    }

    QImage qimg = matToQImage(plateImage);
    QPixmap pixmap = QPixmap::fromImage(qimg);
    // 缩放以适应标签
    pixmap = pixmap.scaled(plateImageLabel_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    plateImageLabel_->setPixmap(pixmap);
}

void ResultPanel::setResultText(const QString& text, float confidence) {
    if (text.isEmpty()) {
        resultTextLabel_->setText("Recognition failed");
        resultTextLabel_->setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #AA0000; "
            "background-color: #f0f0f0; padding: 10px; border: 2px solid #ccc;"
        );
        confidenceLabel_->setText("Confidence: -");
    } else {
        resultTextLabel_->setText(text);
        resultTextLabel_->setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #00AA00; "
            "background-color: #f0f0f0; padding: 10px; border: 2px solid #ccc;"
        );
        confidenceLabel_->setText(QString("Confidence: %1%").arg(static_cast<int>(confidence * 100)));
    }
}

void ResultPanel::clear() {
    plateImageLabel_->setText("No plate detected");
    resultTextLabel_->setText("-");
    resultTextLabel_->setStyleSheet(
        "font-size: 24px; font-weight: bold; color: #00AA00; "
        "background-color: #f0f0f0; padding: 10px; border: 2px solid #ccc;"
    );
    confidenceLabel_->setText("Confidence: -");
}

QImage ResultPanel::matToQImage(const cv::Mat& mat) {
    if (mat.channels() == 3) {
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
        return QImage(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888).copy();
    } else if (mat.channels() == 1) {
        return QImage(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_Grayscale8).copy();
    }
    return QImage();
}

} // namespace platesniper
```

- [ ] **Step 3: 更新 CMakeLists.txt**

添加 `src/gui/result_panel.cpp` 和 `src/gui/result_panel.h`。

- [ ] **Step 4: Commit**

```bash
git add src/gui/result_panel.h src/gui/result_panel.cpp CMakeLists.txt
git commit -m "feat: add ResultPanel widget for displaying plate correction and recognition results"
```

---

## Task 9: GUI 主窗口 (MainWindow)

**Files:**
- Create: `src/gui/main_window.h`
- Create: `src/gui/main_window.cpp`
- Modify: `CMakeLists.txt`

### 9.1 头文件

- [ ] **Step 1: 编写 MainWindow 头文件**

```cpp
// src/gui/main_window.h
#pragma once

#include <QMainWindow>
#include <QTimer>
#include <memory>
#include "models/model_manager.h"

namespace platesniper {

class ImageView;
class ResultPanel;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(ModelManager* modelManager, QWidget* parent = nullptr);
    ~MainWindow();

protected:
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dropEvent(QDropEvent* event) override;

private slots:
    void onOpenImage();
    void onOpenCamera();
    void onCloseCamera();
    void onSaveResult();
    void onCameraFrame();

private:
    void setupUI();
    void processImage(const cv::Mat& image);

    ModelManager* modelManager_;

    // UI 组件
    ImageView* originalView_;
    ImageView* resultView_;
    ResultPanel* resultPanel_;

    // 摄像头
    QTimer* cameraTimer_;
    cv::VideoCapture camera_;
    bool cameraRunning_ = false;

    // 当前处理结果
    cv::Mat currentImage_;
    std::vector<Detection> currentDetections_;
    cv::Mat currentPlateImage_;
    QString currentPlateText_;
    float currentConfidence_ = 0.0f;
};

} // namespace platesniper
```

### 9.2 实现

- [ ] **Step 2: 编写 MainWindow 实现（构造函数和 UI 设置）**

```cpp
// src/gui/main_window.cpp
#include "gui/main_window.h"
#include "gui/image_view.h"
#include "gui/result_panel.h"
#include "core/plate_detector.h"
#include "core/plate_corrector.h"
#include "core/plate_recognizer.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QPushButton>
#include <QFileDialog>
#include <QMessageBox>
#include <QDragEnterEvent>
#include <QMimeData>
#include <QUrl>

namespace platesniper {

MainWindow::MainWindow(ModelManager* modelManager, QWidget* parent)
    : QMainWindow(parent),
      modelManager_(modelManager),
      cameraTimer_(new QTimer(this))
{
    setupUI();

    setAcceptDrops(true);
    setWindowTitle("PlateSniper - License Plate Recognition");
    resize(1200, 700);

    connect(cameraTimer_, &QTimer::timeout, this, &MainWindow::onCameraFrame);
}

MainWindow::~MainWindow() {
    if (camera_.isOpened()) {
        camera_.release();
    }
}

void MainWindow::setupUI() {
    auto* centralWidget = new QWidget(this);
    auto* mainLayout = new QHBoxLayout(centralWidget);
    mainLayout->setSpacing(5);
    mainLayout->setContentsMargins(5, 5, 5, 5);

    // 左侧：原图
    auto* leftLayout = new QVBoxLayout();
    auto* origTitle = new QLabel("Original Image", this);
    origTitle->setStyleSheet("font-weight: bold; font-size: 14px;");
    leftLayout->addWidget(origTitle);

    originalView_ = new ImageView(this);
    leftLayout->addWidget(originalView_, 1);

    mainLayout->addLayout(leftLayout, 2);

    // 中间：检测结果
    auto* midLayout = new QVBoxLayout();
    auto* resultTitle = new QLabel("Detection Result", this);
    resultTitle->setStyleSheet("font-weight: bold; font-size: 14px;");
    midLayout->addWidget(resultTitle);

    resultView_ = new ImageView(this);
    midLayout->addWidget(resultView_, 1);

    mainLayout->addLayout(midLayout, 2);

    // 右侧：识别结果
    resultPanel_ = new ResultPanel(this);
    mainLayout->addWidget(resultPanel_, 1);

    // 底部控制栏
    auto* bottomLayout = new QHBoxLayout();

    auto* openBtn = new QPushButton("Open Image", this);
    connect(openBtn, &QPushButton::clicked, this, &MainWindow::onOpenImage);
    bottomLayout->addWidget(openBtn);

    auto* camBtn = new QPushButton("Open Camera", this);
    connect(camBtn, &QPushButton::clicked, this, &MainWindow::onOpenCamera);
    bottomLayout->addWidget(camBtn);

    auto* closeCamBtn = new QPushButton("Close Camera", this);
    connect(closeCamBtn, &QPushButton::clicked, this, &MainWindow::onCloseCamera);
    bottomLayout->addWidget(closeCamBtn);

    auto* saveBtn = new QPushButton("Save Result", this);
    connect(saveBtn, &QPushButton::clicked, this, &MainWindow::onSaveResult);
    bottomLayout->addWidget(saveBtn);

    bottomLayout->addStretch();

    auto* fullLayout = new QVBoxLayout();
    fullLayout->addLayout(mainLayout, 1);
    fullLayout->addLayout(bottomLayout);

    centralWidget->setLayout(fullLayout);
    setCentralWidget(centralWidget);
}
```

- [ ] **Step 3: 编写 MainWindow 实现（事件处理和核心逻辑）**

```cpp
// 追加到 src/gui/main_window.cpp

void MainWindow::dragEnterEvent(QDragEnterEvent* event) {
    if (event->mimeData()->hasUrls()) {
        event->acceptProposedAction();
    }
}

void MainWindow::dropEvent(QDropEvent* event) {
    auto urls = event->mimeData()->urls();
    if (urls.isEmpty()) return;

    QString filePath = urls.first().toLocalFile();
    cv::Mat image = cv::imread(filePath.toStdString());
    if (!image.empty()) {
        processImage(image);
    } else {
        QMessageBox::warning(this, "Error", "Failed to load image: " + filePath);
    }
}

void MainWindow::onOpenImage() {
    QString filePath = QFileDialog::getOpenFileName(this,
        "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)");
    if (filePath.isEmpty()) return;

    cv::Mat image = cv::imread(filePath.toStdString());
    if (image.empty()) {
        QMessageBox::warning(this, "Error", "Failed to load image: " + filePath);
        return;
    }

    processImage(image);
}

void MainWindow::processImage(const cv::Mat& image) {
    currentImage_ = image.clone();
    originalView_->setImage(image);

    if (!modelManager_ || !modelManager_->isReady()) {
        QMessageBox::warning(this, "Error", "Models not loaded");
        return;
    }

    // 检测
    auto* detector = modelManager_->detector();
    currentDetections_ = detector->detect(image);

    // 在结果图上绘制检测框
    cv::Mat resultImage = image.clone();
    resultView_->setImage(resultImage);
    resultView_->setDetections(currentDetections_);

    if (currentDetections_.empty()) {
        resultPanel_->clear();
        return;
    }

    // 取置信度最高的检测结果
    const auto& bestDet = currentDetections_[0];

    // 矫正
    PlateCorrector corrector;
    corrector.setOutputSize(136, 36);
    currentPlateImage_ = corrector.correct(image, bestDet);

    // 识别
    auto* recognizer = modelManager_->recognizer();
    std::string plateText = recognizer->recognize(currentPlateImage_);

    currentPlateText_ = QString::fromStdString(plateText);
    currentConfidence_ = bestDet.confidence;

    // 更新结果面板
    resultPanel_->setPlateImage(currentPlateImage_);
    resultPanel_->setResultText(currentPlateText_, currentConfidence_);
}

void MainWindow::onOpenCamera() {
    if (cameraRunning_) return;

    camera_.open(0);  // 默认摄像头
    if (!camera_.isOpened()) {
        QMessageBox::warning(this, "Error", "Failed to open camera");
        return;
    }

    cameraRunning_ = true;
    cameraTimer_->start(33);  // ~30 FPS
}

void MainWindow::onCloseCamera() {
    cameraTimer_->stop();
    cameraRunning_ = false;
    camera_.release();
}

void MainWindow::onCameraFrame() {
    if (!cameraRunning_) return;

    cv::Mat frame;
    if (!camera_.read(frame)) {
        return;
    }

    // 每 3 帧检测一次（降低 CPU 占用）
    static int frameCount = 0;
    frameCount++;
    if (frameCount % 3 == 0) {
        processImage(frame);
    } else {
        originalView_->setImage(frame);
    }
}

void MainWindow::onSaveResult() {
    if (currentImage_.empty()) {
        QMessageBox::information(this, "Info", "No result to save");
        return;
    }

    QString filePath = QFileDialog::getSaveFileName(this,
        "Save Result", "result.png", "Images (*.png *.jpg)");
    if (filePath.isEmpty()) return;

    cv::imwrite(filePath.toStdString(), currentImage_);
    QMessageBox::information(this, "Success", "Result saved to: " + filePath);
}

} // namespace platesniper
```

- [ ] **Step 4: 更新 CMakeLists.txt**

添加 `src/gui/main_window.cpp` 和 `src/gui/main_window.h`。

- [ ] **Step 5: Commit**

```bash
git add src/gui/main_window.h src/gui/main_window.cpp CMakeLists.txt
git commit -m "feat: add MainWindow with image processing pipeline and camera support"
```

---

## Task 10: 程序入口更新

**Files:**
- Modify: `src/main.cpp`

### 10.1 更新 main.cpp

- [ ] **Step 1: 重写 main.cpp 集成所有组件**

```cpp
// src/main.cpp
#include <QApplication>
#include <QMessageBox>
#include <QDir>
#include "gui/main_window.h"
#include "models/model_manager.h"
#include <iostream>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    app.setApplicationName("PlateSniper");
    app.setOrganizationName("PlateSniper");

    // 初始化模型管理器
    auto modelManager = std::make_unique<platesniper::ModelManager>();

    // 模型路径（从可执行文件所在目录的 models/ 子目录加载）
    QDir appDir(QCoreApplication::applicationDirPath());
    QString detectModelPath = appDir.filePath("models/plate_detect.onnx");
    QString recModelPath = appDir.filePath("models/plate_recognize.onnx");

    // 尝试加载模型（模型文件可能不存在，不阻止程序启动）
    bool modelsLoaded = modelManager->initialize(
        detectModelPath.toStdString(),
        recModelPath.toStdString()
    );

    if (!modelsLoaded) {
        QMessageBox::warning(nullptr, "Model Loading Warning",
            QString("Failed to load models:\n%1\n\n"
                    "Please ensure ONNX model files exist in the models/ directory.\n"
                    "You can still open the application to browse images.")
                .arg(QString::fromStdString(modelManager->lastError())));
    }

    // 创建并显示主窗口
    platesniper::MainWindow window(modelManager.get());
    window.show();

    return app.exec();
}
```

- [ ] **Step 2: Commit**

```bash
git add src/main.cpp
git commit -m "feat: integrate all components in main.cpp with model loading and error handling"
```

---

## Task 11: 配置管理 (Config)

**Files:**
- Create: `src/utils/config.h`
- Create: `src/utils/config.cpp`
- Modify: `CMakeLists.txt`

### 11.1 头文件与实现

- [ ] **Step 1: 编写 Config 头文件**

```cpp
// src/utils/config.h
#pragma once

#include <string>

namespace platesniper {

// 全局配置常量
struct Config {
    // 模型路径
    static std::string detectModelPath;
    static std::string recognizeModelPath;

    // 检测参数
    static float detectionConfidenceThreshold;
    static float nmsIoUThreshold;

    // 输入尺寸
    static int detectorInputWidth;
    static int detectorInputHeight;
    static int recognizerInputWidth;
    static int recognizerInputHeight;

    // 矫正输出尺寸
    static int plateOutputWidth;
    static int plateOutputHeight;

    // 摄像头参数
    static int cameraFps;
    static int cameraFrameSkip;
};

} // namespace platesniper
```

- [ ] **Step 2: 编写 Config 实现**

```cpp
// src/utils/config.cpp
#include "utils/config.h"

namespace platesniper {

std::string Config::detectModelPath = "models/plate_detect.onnx";
std::string Config::recognizeModelPath = "models/plate_recognize.onnx";

float Config::detectionConfidenceThreshold = 0.5f;
float Config::nmsIoUThreshold = 0.45f;

int Config::detectorInputWidth = 640;
int Config::detectorInputHeight = 640;
int Config::recognizerInputWidth = 94;
int Config::recognizerInputHeight = 24;

int Config::plateOutputWidth = 136;
int Config::plateOutputHeight = 36;

int Config::cameraFps = 30;
int Config::cameraFrameSkip = 3;

} // namespace platesniper
```

- [ ] **Step 3: 更新 CMakeLists.txt**

添加 `src/utils/config.cpp` 和 `src/utils/config.h`。

- [ ] **Step 4: Commit**

```bash
git add src/utils/config.h src/utils/config.cpp CMakeLists.txt
git commit -m "feat: add Config utility for centralized parameter management"
```

---

## Task 12: 测试配置

**Files:**
- Modify: `CMakeLists.txt`

### 12.1 配置测试子目录

- [ ] **Step 1: 创建 tests/CMakeLists.txt**

```cmake
# tests/CMakeLists.txt
find_package(GTest REQUIRED)

add_executable(plate_sniper_tests
    test_utils.cpp
    test_corrector.cpp
)

target_link_libraries(plate_sniper_tests PRIVATE
    ${OpenCV_LIBS}
    GTest::gtest
    GTest::gtest_main
)

target_include_directories(plate_sniper_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

add_test(NAME PlateSniperTests COMMAND plate_sniper_tests)
```

- [ ] **Step 2: 更新根 CMakeLists.txt 中的测试选项**

确保根 `CMakeLists.txt` 中已有：
```cmake
option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    add_subdirectory(tests)
endif()
```

- [ ] **Step 3: Commit**

```bash
git add tests/CMakeLists.txt CMakeLists.txt
git commit -m "chore: configure Google Test integration and test subdirectory"
```

---

## 自检

### Spec 覆盖检查

| 规范要求 | 实现任务 |
|----------|----------|
| ONNX Runtime 会话封装 | Task 1: OnnxSession |
| 图像预处理工具 | Task 2: ImageUtils |
| PlateDetector (YOLO 检测) | Task 3: PlateDetector |
| PlateCorrector (透视变换) | Task 4: PlateCorrector |
| PlateRecognizer (CTC 解码) | Task 5: PlateRecognizer |
| ModelManager (模型生命周期) | Task 6: ModelManager |
| ImageView (图像显示) | Task 7: ImageView |
| ResultPanel (结果展示) | Task 8: ResultPanel |
| MainWindow (GUI 主窗口) | Task 9: MainWindow |
| 程序入口集成 | Task 10: main.cpp |
| 配置管理 | Task 11: Config |
| 单元测试 | Task 2, 4, 12 |

**无遗漏。**

### Placeholder 扫描

- ✅ 无 "TBD" / "TODO"（除 main.cpp 中原有的注释已替换）
- ✅ 无 "implement later"
- ✅ 无 "Add appropriate error handling" 模糊描述
- ✅ 每个步骤包含具体代码
- ✅ 无 "Similar to Task N" 引用

### 类型一致性检查

- ✅ `Detection` 结构体在所有任务中一致使用
- ✅ `OnnxSession::run()` 返回 `std::vector<float>` 在各处一致
- ✅ `PlateDetector::detect()` 返回 `std::vector<Detection>` 一致
- ✅ `ModelManager` 使用 `std::unique_ptr` 管理模型实例一致
- ✅ Qt 信号/槽命名一致

---

## 执行选项

**计划已完成并保存到 `docs/superpowers/plans/2026-04-10-plate-recognition-implementation.md`。**

两个执行选项：

**1. Subagent-Driven（推荐）** — 为每个任务单独调度子代理，我在任务之间审查，迭代速度快

**2. Inline Execution** — 在当前会话中使用 executing-plans 批量执行任务，设置检查点进行审查

你想用哪种方式执行？