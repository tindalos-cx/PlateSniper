# CLAUDE.md

本文档为 Claude Code (claude.ai/code) 提供本仓库的编码指导。

## 项目概述

PlateSniper 是一个基于 C++17、OpenCV、Qt5 和 ONNX Runtime 的中文车牌识别系统。采用三阶段流水线架构：检测（YOLO）→ 矫正（透视变换）→ 识别（CRNN/LPRNet）。

## 构建系统

项目使用 CMake (3.16+)，主要依赖如下：
- OpenCV 4.x
- Qt5（Core、Gui、Widgets）
- ONNX Runtime 1.16+

### 构建命令

```bash
# 标准构建
mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build . --config Release

# 使用 vcpkg 工具链
cmake .. -DCMAKE_TOOLCHAIN_FILE=[vcpkg]/scripts/buildsystems/vcpkg.cmake

# 带测试构建
cmake .. -DBUILD_TESTS=ON
cmake --build . --config Release
```

### 主要 CMake 选项

- `ONNXRUNTIME_ROOT`：ONNX Runtime 安装路径（默认：`third_party/onnxruntime`）
- `BUILD_TESTS`：启用 Google Test 单元测试（默认：OFF）

## 架构

### 三阶段流水线

```
输入（图片/摄像头） → 检测 → 矫正 → 识别 → GUI 输出
```

**核心模块**（`src/core/`）：
- `plate_detector.h/cpp`：基于 YOLO 的车牌检测，输出 `std::vector<Detection>`
- `plate_corrector.h/cpp`：透视变换得到正面正视图
- `plate_recognizer.h/cpp`：CRNN/LPRNet 字符识别
- `detection.h`：Detection 结构体，包含 bbox、confidence、classId

**模型层**（`src/models/`）：
- `model_manager.h/cpp`：集中管理 ONNX 模型生命周期
- `onnx_session.h/cpp`：ONNX Runtime 会话封装

**GUI 层**（`src/gui/`）：
- `main_window.h/cpp`：Qt 主窗口，协调整个流水线
- `image_view.h/cpp`：图像显示组件，支持拖拽
- `result_panel.h/cpp`：识别结果展示面板

**工具模块**（`src/utils/`）：
- `image_utils.h/cpp`：图像预处理（resize、归一化、色彩空间转换）
- `config.h/cpp`：配置管理

### 命名空间

所有代码位于 `platesniper` 命名空间下。

## 模型文件

ONNX 模型需放入 `models/` 目录：
- `models/plate_detect.onnx` - 检测模型
- `models/plate_recognize.onnx` - 识别模型

构建时会自动将模型文件复制到构建目录。

## 开发注意事项

- CMake 中启用了 Qt MOC/UIC/RCC（CMAKE_AUTOMOC、CMAKE_AUTORCC、CMAKE_AUTOUIC）
- 错误处理使用返回值（bool/std::string），不使用异常
- 跨线程 GUI 更新使用 Qt 信号/槽，配合 Qt::QueuedConnection
- 摄像头捕获在独立线程中运行；推理在该线程中同步执行

## 当前状态

**已实现全部模块**：

| 模块 | 文件 | 状态 |
|------|------|------|
| 核心数据结构 | `src/core/detection.h/.cpp` | ✅ |
| 车牌检测器 | `src/core/plate_detector.h/.cpp` | ✅ |
| 车牌矫正器 | `src/core/plate_corrector.h/.cpp` | ✅ |
| 车牌识别器 | `src/core/plate_recognizer.h/.cpp` | ✅ |
| ONNX 会话 | `src/models/onnx_session.h/.cpp` | ✅ |
| 模型管理器 | `src/models/model_manager.h/.cpp` | ✅ |
| 图像工具 | `src/utils/image_utils.h/.cpp` | ✅ |
| 配置管理 | `src/utils/config.h/.cpp` | ✅ |
| 主窗口 | `src/gui/main_window.h/.cpp` | ✅ |
| 图像视图 | `src/gui/image_view.h/.cpp` | ✅ |
| 结果面板 | `src/gui/result_panel.h/.cpp` | ✅ |

**待完成**：
- 添加测试文件（`tests/`）
- 准备预训练 ONNX 模型文件
