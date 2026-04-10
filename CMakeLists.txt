cmake_minimum_required(VERSION 3.16)
project(PlateSniper VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖
find_package(OpenCV 4 REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets)

# ONNX Runtime 路径（可通过命令行指定）
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime" CACHE PATH "ONNX Runtime root directory")

# 源文件
set(SOURCES
    src/main.cpp
    src/core/detection.cpp
    src/core/plate_detector.cpp
    src/core/plate_recognizer.cpp
    src/core/plate_corrector.cpp
    src/models/model_manager.cpp
    src/models/onnx_session.cpp
    src/gui/main_window.cpp
    src/gui/image_view.cpp
    src/gui/result_panel.cpp
    src/utils/image_utils.cpp
    src/utils/config.cpp
)

set(HEADERS
    src/core/detection.h
    src/core/plate_detector.h
    src/core/plate_recognizer.h
    src/core/plate_corrector.h
    src/models/model_manager.h
    src/models/onnx_session.h
    src/gui/main_window.h
    src/gui/image_view.h
    src/gui/result_panel.h
    src/utils/image_utils.h
    src/utils/config.h
)

# 自动处理 Qt MOC
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# 可执行文件
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# 包含目录
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_ROOT}/include
)

# 链接库
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
)

# ONNX Runtime 库
if(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib
    )
else()
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
    )
endif()

# 复制模型文件到构建目录
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/models $<TARGET_FILE_DIR:${PROJECT_NAME}>/models
)

# 测试（可选）
option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    add_subdirectory(tests)
endif()
