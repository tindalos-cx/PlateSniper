# PlateSniper 车牌识别系统

基于 C++、OpenCV 和 ONNX Runtime 的车牌识别项目，采用 Qt 构建图形界面。支持静态图片和实时摄像头视频流两种输入方式。

## 功能特性

- **车牌检测**：使用 YOLO ONNX 模型定位车牌区域
- **车牌矫正**：透视变换得到正面正视图
- **字符识别**：CRNN/LPRNet 模型识别车牌文本
- **图形界面**：Qt 界面支持图片选择、拖拽和摄像头预览
- **多类型支持**：中国蓝牌、绿牌、警牌等常见车牌

## 技术栈

| 组件 | 版本 |
|------|------|
| C++ | 17 |
| OpenCV | 4.x |
| ONNX Runtime | 1.16+ |
| Qt | 5.15 / 6.x |
| CMake | 3.16+ |

## 快速开始

### 1. 克隆仓库

```bash
git clone <repository-url>
cd PlateSniper
```

### 2. 安装依赖

**Windows (vcpkg)**:
```bash
vcpkg install opencv4 onnxruntime qt5-base
```

**Linux (Ubuntu)**:
```bash
sudo apt install libopencv-dev qtbase5-dev
# ONNX Runtime 需手动下载
```

### 3. 下载模型

将预训练 ONNX 模型放入 `models/` 目录：

```
models/
├── plate_detect.onnx      # 车牌检测模型
└── plate_recognize.onnx   # 车牌识别模型
```

模型下载链接详见 [docs/superpowers/specs/2026-04-10-plate-recognition-design.md](docs/superpowers/specs/2026-04-10-plate-recognition-design.md)。

### 4. 构建项目

```bash
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=[vcpkg]/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```

### 5. 运行

```bash
./PlateSniper
```

## 项目结构

```
PlateSniper/
├── src/
│   ├── core/          # 检测、矫正、识别核心模块
│   ├── models/        # ONNX 模型封装
│   ├── gui/           # Qt 界面代码
│   └── utils/         # 工具函数
├── models/            # ONNX 模型文件
├── tests/             # 单元测试
├── docs/              # 设计文档
└── CMakeLists.txt
```

## 设计文档

详细设计规格见：[docs/superpowers/specs/2026-04-10-plate-recognition-design.md](docs/superpowers/specs/2026-04-10-plate-recognition-design.md)

## 许可证

MIT License
