cmake_minimum_required(VERSION 3.16)
project(PlateSniperTests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)

set(TEST_SOURCES
    test_detection.cpp
    test_corrector.cpp
    test_utils.cpp
)

add_executable(${PROJECT_NAME} ${TEST_SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    GTest::gtest
    GTest::gtest_main
    ${OpenCV_LIBS}
)

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})
