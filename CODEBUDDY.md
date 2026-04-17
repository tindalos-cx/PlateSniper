# CODEBUDDY.md 本文件为 CodeBuddy 在此仓库工作时提供指导。

## 构建命令

**标准构建**：
```bash
mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build . --config Release
```

**使用 vcpkg 工具链**：
```bash
cmake .. -DCMAKE_TOOLCHAIN_FILE=[vcpkg]/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```

**启用测试**：
```bash
cmake .. -DBUILD_TESTS=ON
cmake --build . --config Release
./PlateSniper  # 运行测试：ctest
```

## 架构设计

PlateSniper 采用**三阶段流水线**架构进行中文车牌识别：

```
输入（图片/视频） → 检测 → 矫正 → 识别 → GUI 输出
```

### 核心流水线模块（`src/core/`）

| 模块 | 职责 |
|------|------|
| `PlateDetector` | 加载 YOLO ONNX 模型，检测车牌边界框及置信度 |
| `PlateCorrector` | 透视变换获取车牌正面正视图 |
| `PlateRecognizer` | CRNN/LPRNet 推理解码车牌文本 |

各阶段间通过 `cv::Mat` 和标准容器传递数据，每个阶段可独立替换。

### 模型层（`src/models/`）

- `ModelManager`：集中管理 ONNX 模型生命周期（加载、校验）
- `OnnxSession`：ONNX Runtime 会话创建和推理的轻量封装

### GUI 层（`src/gui/`）

- `MainWindow`：协调流水线，处理图片/视频输入和结果显示
- `ImageView`：支持拖拽的图片显示组件
- `ResultPanel`：识别结果展示面板

### 线程模型

- 摄像头捕获在独立线程运行
- ONNX 推理在捕获线程同步执行
- GUI 更新通过 Qt 信号/槽配合 `Qt::QueuedConnection` 实现线程安全

## 关键约定

- **命名空间**：所有代码位于 `platesniper`
- **错误处理**：使用返回值（`bool`/`std::string`），不使用异常
- **Qt 自动化**：CMake 启用 MOC/UIC/RCC（`CMAKE_AUTOMOC`、`CMAKE_AUTORCC`、`CMAKE_AUTOUIC`）
- **模型文件**：放入 `models/` 目录（`plate_detect.onnx`、`plate_recognize.onnx`）

## 项目状态

**✅ 全部模块已实现**

| 模块 | 文件 | 状态 |
|------|------|------|
| ONNX 会话 | `src/models/onnx_session.h/.cpp` | ✅ |
| 车牌检测器 | `src/core/plate_detector.h/.cpp` | ✅ |
| 车牌矫正器 | `src/core/plate_corrector.h/.cpp` | ✅ |
| 车牌识别器 | `src/core/plate_recognizer.h/.cpp` | ✅ |
| 模型管理器 | `src/models/model_manager.h/.cpp` | ✅ |
| 图像工具 | `src/utils/image_utils.h/.cpp` | ✅ |
| 配置管理 | `src/utils/config.h/.cpp` | ✅ |
| 主窗口 | `src/gui/main_window.h/.cpp` | ✅ |
| 图像视图 | `src/gui/image_view.h/.cpp` | ✅ |
| 结果面板 | `src/gui/result_panel.h/.cpp` | ✅ |
| 单元测试 | `tests/` | ✅ |

**⚠️ 需手动准备**：
- ONNX 模型文件（`models/plate_detect.onnx`、`models/plate_recognize.onnx`）

---

## 快速开始

### 1. 获取 ONNX 模型文件

**方式一：使用 PaddleOCR 预训练模型**
```bash
# 克隆 PaddleOCR
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR

# 下载检测模型（PP-OCRv4）
wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar.gz
tar -xf ch_PP-OCRv4_det_infer.tar.gz

# 下载识别模型
wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar.gz
tar -xf ch_PP-OCRv4_rec_infer.tar.gz

# 转换为 ONNX（需要 paddle2onnx）
pip install paddle2onnx
paddle2onnx --model_dir ch_PP-OCRv4_det_infer --save_file ../models/plate_detect.onnx
paddle2onnx --model_dir ch_PP-OCRv4_rec_infer --save_file ../models/plate_recognize.onnx
```

**方式二：使用 Ultralytics YOLOv8**
```bash
# 安装 yolo
pip install ultralytics

# 下载预训练模型并导出
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.export(format='onnx')"
```

### 2. 构建项目

```bash
# 创建构建目录
mkdir -p build && cd build

# 配置（指定 ONNX Runtime 路径）
cmake .. -DCMAKE_TOOLCHAIN_FILE=[vcpkg]/scripts/buildsystems/vcpkg.cmake

# 如果不用 vcpkg，手动指定 ONNX Runtime
cmake .. -DONNXRUNTIME_ROOT=/usr/local/onnxruntime

# 编译
cmake --build . --config Release
```

### 3. 运行测试

```bash
# 启用测试构建
cmake .. -DBUILD_TESTS=ON -DONNXRUNTIME_ROOT=/usr/local/onnxruntime
cmake --build . --config Release

# 运行所有测试
ctest --output-on-failure

# 运行单个测试
./PlateSniperTests --gtest_filter=DetectionTest.*
```

### 4. 运行程序

```bash
# 确保模型文件在正确位置
ls models/
# plate_detect.onnx  plate_recognize.onnx

# 运行
./PlateSniper
```
