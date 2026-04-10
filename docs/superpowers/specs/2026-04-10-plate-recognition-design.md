# PlateSniper 车牌识别项目设计文档

**日期**：2026-04-10  
**版本**：v1.0  
**状态**：待实现

---

## 1. 项目概述

PlateSniper 是一个基于 C++、OpenCV 和 ONNX Runtime 的车牌识别系统，采用 Qt 构建图形界面。项目支持静态图片和实时摄像头视频流两种输入方式，能够检测多种中国车牌类型（蓝牌、绿牌、警牌等），并进行字符识别。

### 1.1 目标与范围

**目标**：
- 构建一个可运行的车牌识别演示系统，用于学习 OpenCV 和深度学习推理技术
- 采用模块化架构，各阶段独立清晰，便于理解和替换
- 提供友好的 Qt GUI，支持图片选择和摄像头预览

**范围**：
- 检测层：使用轻量级 YOLO ONNX 模型定位车牌区域
- 矫正层：对检测到的车牌进行透视变换，得到正面正视图
- 识别层：使用 CRNN/LPRNet ONNX 模型识别车牌字符序列
- GUI 层：Qt 界面支持图片处理、摄像头实时预览、结果展示
- 输入：静态图片（文件选择/拖拽）、实时摄像头视频流

**不在范围内**：
- 模型训练（使用预训练 ONNX 模型）
- 多语言车牌支持（仅限中文车牌）
- 车牌归属地/违章查询等后端业务功能
- 移动端部署

### 1.2 技术栈

| 组件 | 技术选型 | 版本/说明 |
|------|----------|-----------|
| 编程语言 | C++17 | MSVC/GCC/Clang |
| 图像处理 | OpenCV | 4.x |
| 深度学习推理 | ONNX Runtime | 1.16+ |
| GUI 框架 | Qt | 5.15 或 6.x |
| 构建系统 | CMake | 3.16+ |
| 测试框架 | Google Test | 可选 |

---

## 2. 架构设计

### 2.1 系统架构

项目采用 **模块化三阶段流水线** 架构：

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   输入层    │ ──▶ │   检测层    │ ──▶ │   矫正层    │ ──▶ │   识别层    │
│ 图片/摄像头 │     │ PlateDetector│     │PlateCorrector│     │PlateRecognizer│
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                   │
                                                                   ▼
                                                            ┌─────────────┐
                                                            │   输出层    │
                                                            │ Qt GUI 展示 │
                                                            └─────────────┘
```

**核心原则**：
- 每个阶段职责单一，通过清晰的 C++ 接口交互
- 阶段之间只传递 `cv::Mat` 和标准容器，无隐式状态依赖
- 任意阶段可独立替换而不影响其他阶段

### 2.2 文件结构

```
PlateSniper/
├── src/
│   ├── core/
│   │   ├── plate_detector.h/.cpp       # 车牌检测器
│   │   ├── plate_recognizer.h/.cpp     # 车牌识别器
│   │   ├── plate_corrector.h/.cpp      # 车牌矫正器
│   │   └── detection.h                 # Detection 结构体定义
│   ├── models/
│   │   ├── model_manager.h/.cpp        # 模型管理器
│   │   └── onnx_session.h/.cpp         # ONNX Runtime 会话封装
│   ├── gui/
│   │   ├── main_window.h/.cpp          # Qt 主窗口
│   │   ├── image_view.h/.cpp           # 图像显示组件
│   │   └── result_panel.h/.cpp         # 结果展示面板
│   ├── utils/
│   │   ├── image_utils.h/.cpp          # 图像预处理工具
│   │   └── config.h/.cpp               # 配置管理
│   └── main.cpp                        # 程序入口
├── models/
│   ├── plate_detect.onnx               # 车牌检测模型
│   └── plate_recognize.onnx            # 车牌识别模型
├── tests/
│   ├── test_corrector.cpp              # 矫正器单元测试
│   ├── test_utils.cpp                  # 工具函数测试
│   └── test_images/                    # 测试图片集
├── CMakeLists.txt
├── README.md
└── .gitignore
```

---

## 3. 组件设计

### 3.1 Detection 结构体

```cpp
struct Detection {
    cv::Rect bbox;           // 边界框 (x, y, width, height)
    float confidence;        // 置信度 0.0 ~ 1.0
    int classId;             // 类别 ID（预留多类别支持）
};
```

### 3.2 PlateDetector（车牌检测器）

**职责**：加载 YOLO ONNX 模型，对输入图像进行车牌位置检测。

**接口**：
```cpp
class PlateDetector {
public:
    bool loadModel(const std::string& modelPath);
    std::vector<Detection> detect(const cv::Mat& image, float confThreshold = 0.5f);
    void setInputSize(int width, int height);  // 模型输入尺寸，默认 640x640
};
```

**内部处理流程**：
1. 将输入图像 resize 到模型输入尺寸（默认 640x640）
2. 色彩空间转换（BGR → RGB）
3. 归一化（除以 255.0）
4. ONNX Runtime 推理
5. 解析输出（边界框 + 置信度 + NMS 去重）
6. 将坐标映射回原图尺寸

### 3.3 PlateCorrector（车牌矫正器）

**职责**：基于检测框进行透视变换，得到正面正视的车牌图像。

**接口**：
```cpp
class PlateCorrector {
public:
    cv::Mat correct(const cv::Mat& image, const Detection& det);
    cv::Mat correct(const cv::Mat& image, const std::vector<cv::Point2f>& corners);
    // 输出尺寸：蓝牌默认 136x36，绿牌默认 136x40
    void setOutputSize(int width, int height);
};
```

**处理策略**：
- 第一阶段：使用检测框的四个角点进行简单透视变换
- 进阶优化：通过颜色分割或边缘检测提取精确的四边形角点，再矫正

### 3.4 PlateRecognizer（车牌识别器）

**职责**：加载 CRNN/LPRNet ONNX 模型，对矫正后的车牌图像进行字符序列识别。

**接口**：
```cpp
class PlateRecognizer {
public:
    bool loadModel(const std::string& modelPath);
    std::string recognize(const cv::Mat& plateImage);
    // 字符集：包含中文省份简称、字母、数字、新能源专用字符
    void setCharset(const std::vector<std::string>& charset);
};
```

**内部处理流程**：
1. 灰度化/二值化（根据模型要求）
2. Resize 到模型输入尺寸（如 LPRNet 的 94x24）
3. 归一化
4. ONNX Runtime 推理
5. CTC 解码（去除重复字符和空白符）
6. 映射到字符集，输出文本

**字符集**：包含 31 个中文省份简称 + 26 个英文字母 + 10 个数字 + 新能源专用字符（如 "D"、"F"、"G" 等）。

### 3.5 ModelManager（模型管理器）

**职责**：集中管理所有 ONNX 模型的加载、校验和生命周期。

**接口**：
```cpp
class ModelManager {
public:
    bool initialize(const std::string& configPath);  // 从配置文件加载
    bool initialize(const std::string& detectModel, const std::string& recModel);
    
    PlateDetector* detector();
    PlateRecognizer* recognizer();
    
    bool isReady() const;
    std::string lastError() const;
};
```

**校验项**：
- 模型文件是否存在
- 文件大小是否合理（非空文件）
- ONNX Runtime 能否成功创建推理会话

### 3.6 MainWindow（Qt 主窗口）

**职责**：提供用户交互界面，协调各组件完成识别流程。

**界面布局**：
- **左侧区域**：原图显示（支持拖拽打开图片）
- **中间区域**：检测结果图（标注车牌边界框和置信度）
- **右侧区域**：
  - 矫正后的车牌图像
  - 识别出的车牌文本（大字体显示）
  - 置信度信息
- **底部控制栏**：
  - "打开图片" 按钮
  - "打开摄像头" / "关闭摄像头" 按钮
  - "保存结果" 按钮
  - 处理状态显示

**核心信号/槽**：
- 图片加载完成 → 触发检测流程
- 摄像头新帧 → 可选触发检测（控制 FPS）
- 检测/识别完成 → 更新界面显示

---

## 4. 数据流

### 4.1 单张图片处理流程

```
用户点击"打开图片"或拖拽图片到界面
              │
              ▼
    MainWindow::loadImage(path)
              │
              ▼
    cv::Mat image = cv::imread(path)
              │
              ▼
    PlateDetector::detect(image)
              │
              ▼
    std::vector<Detection> dets
              │
              ▼
    对每个 det in dets:
        PlateCorrector::correct(image, det)
                  │
                  ▼
        cv::Mat correctedPlate
                  │
                  ▼
        PlateRecognizer::recognize(correctedPlate)
                  │
                  ▼
        std::string plateText
                  │
                  ▼
    MainWindow 更新界面:
        - 左侧：原图
        - 中间：标注检测框的结果图
        - 右侧：矫正图 + plateText
```

### 4.2 摄像头实时处理流程

```
用户点击"打开摄像头"
              │
              ▼
    VideoCaptureThread 启动
              │
              ▼
    循环读取帧:
        cv::Mat frame = capture.read()
              │
              ▼
    可选：每 N 帧执行一次检测（降低 CPU 占用）
              │
              ▼
    PlateDetector::detect(frame) → 得到 dets
              │
              ▼
    对每个 det:
        correct → recognize → 得到 (text, confidence)
              │
              ▼
    通过 Qt 信号发送结果到主线程
              │
              ▼
    MainWindow 更新显示（线程安全）
```

### 4.3 跨线程安全

- 摄像头捕获在独立线程中运行
- 检测结果通过 `Qt::QueuedConnection` 信号发送到主线程更新 GUI
- ONNX 推理默认在捕获线程中同步执行（后续可优化为异步/批处理）

---

## 5. 错误处理

| 场景 | 检测时机 | 处理方式 |
|------|----------|----------|
| ONNX 模型文件缺失 | 启动时 `ModelManager::initialize()` | 返回 false，弹窗提示用户检查 `models/` 目录，提供模型下载链接 |
| ONNX 模型损坏/不兼容 | 启动时创建推理会话 | 捕获 ONNX Runtime 异常，记录详细错误信息，提示用户 |
| 图片读取失败 | `cv::imread()` 后 | 提示"无法读取图片：路径"，清空显示区 |
| 未检测到车牌 | `PlateDetector::detect()` 返回空 | 在中间区域显示"未检测到车牌"，不执行后续矫正/识别 |
| 检测到但识别失败 | `PlateRecognizer::recognize()` 返回空字符串 | 展示检测框和矫正图，但文本显示"识别失败" |
| 摄像头打开失败 | `cv::VideoCapture::open()` | 提示"无法访问摄像头"，自动回退到图片模式 |
| 摄像头读取帧失败 | `cv::VideoCapture::read()` | 连续失败 N 次后提示"摄像头断开"，停止捕获 |

**错误信息展示**：统一使用 `QMessageBox`，核心模块通过返回值（bool/std::string）传递状态，不使用 C++ 异常（避免破坏实时性能）。

---

## 6. 模型来源与构建计划

### 6.1 预训练模型

| 模型 | 来源 | 说明 |
|------|------|------|
| 检测模型 | PaddleOCR `ch_PP-OCRv4_det` 或 YOLOv8-nano 车牌检测 | 支持多尺度车牌检测 |
| 识别模型 | 基于 CCPD 数据集训练的 LPRNet 或 CRNN | 支持蓝牌、绿牌、警牌等 |

**模型获取方式**：
- 在 README.md 中提供官方/社区预训练模型的下载链接
- 提供 Python 转换脚本（如从 PyTorch `.pt` 导出为 ONNX）
- 将转换好的 ONNX 模型文件放入 `models/` 目录即可运行

### 6.2 模型文件

```
models/
├── plate_detect.onnx      # 检测模型 (~5-20 MB)
└── plate_recognize.onnx   # 识别模型 (~1-5 MB)
```

### 6.3 训练扩展（可选进阶）

- 使用 Python + PyTorch/PaddlePaddle 训练自定义模型
- 数据集：CCPD（Chinese City Parking Dataset）或其他中文车牌数据集
- 导出：训练完成后导出为 ONNX 格式，替换 `models/` 下文件即可无缝切换

---

## 7. 测试策略

### 7.1 单元测试

| 测试目标 | 测试内容 | 框架 |
|----------|----------|------|
| `PlateCorrector` | 测试透视变换对已知角点的输出是否正确 | Google Test |
| `ImageUtils` | 测试 resize、归一化、色彩空间转换等函数 | Google Test |
| `Detection` NMS | 测试非极大值抑制逻辑 | Google Test |

### 7.2 集成测试

- 准备 10-20 张测试图片，覆盖：
  - 不同光照条件（白天、夜晚、逆光）
  - 不同拍摄角度（正面、侧面、俯视）
  - 不同车牌类型（蓝牌、绿牌、警牌、双层牌）
  - 不同车牌状态（清晰、模糊、污损、遮挡）
- 运行完整流水线，验证输出文本是否正确

### 7.3 性能测试

- 单张图片端到端耗时（检测 + 识别）
- 摄像头模式下平均 FPS
- 不同输入分辨率（640x480、1280x720、1920x1080）下的性能对比

---

## 8. 依赖与构建

### 8.1 依赖安装

**Windows**：
- OpenCV 4.x：通过 vcpkg 或官网下载预编译库
- ONNX Runtime：通过 vcpkg 或 GitHub Releases 下载
- Qt 5.15/6.x：通过 Qt 在线安装器

**Linux**：
```bash
# Ubuntu/Debian
sudo apt install libopencv-dev qtbase5-dev
# ONNX Runtime 需手动下载或使用 vcpkg
```

**vcpkg 方式（推荐跨平台）**：
```bash
vcpkg install opencv4 onnxruntime qt5-base
```

### 8.2 构建步骤

```bash
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=[vcpkg]/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```

---

## 9. 风险评估

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| 预训练模型不可用或转换失败 | 中 | 高 | 准备备选模型来源，提供手动转换指南 |
| ONNX Runtime 与 Qt 符号冲突 | 低 | 高 | 使用动态链接，确保编译器/CRT一致 |
| 实时摄像头性能不足 | 中 | 中 | 支持跳帧检测，提供分辨率/帧率配置 |
| 倾斜/遮挡车牌识别率低 | 高 | 中 | 明确文档说明限制，建议拍摄角度 |
| C++ 跨平台编译问题 | 中 | 低 | 主要支持 Windows，Linux/macOS 作为后续扩展 |

---

## 10. 后续扩展

- [ ] 支持更多车牌类型（港澳台、军牌、使馆牌）
- [ ] GPU 加速（CUDA/TensorRT）
- [ ] 批量图片处理功能
- [ ] 识别结果导出（CSV/JSON）
- [ ] REST API 服务接口
- [ ] 多摄像头支持
