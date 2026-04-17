#include <gtest/gtest.h>
#include "utils/image_utils.h"
#include "utils/config.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace platesniper;

class ImageUtilsTest : public ::testing::Test {
protected:
    void SetUp() override {
        color_image = cv::Mat(100, 200, CV_8UC3, cv::Scalar(100, 150, 200));
        gray_image = cv::Mat(100, 200, CV_8UC1, cv::Scalar(128));
    }
    cv::Mat color_image;
    cv::Mat gray_image;
};

TEST_F(ImageUtilsTest, ResizeWithoutKeepAspect) {
    cv::Mat resized = ImageUtils::resize(color_image, 50, 50, false);

    EXPECT_EQ(resized.cols, 50);
    EXPECT_EQ(resized.rows, 50);
    EXPECT_EQ(resized.type(), color_image.type());
}

TEST_F(ImageUtilsTest, ResizeKeepAspect) {
    cv::Mat resized = ImageUtils::resize(color_image, 100, 100, true);

    EXPECT_LE(resized.cols, 100);
    EXPECT_LE(resized.rows, 100);
    EXPECT_EQ(resized.type(), color_image.type());
}

TEST_F(ImageUtilsTest, ResizeEmptyImage) {
    cv::Mat empty;
    cv::Mat resized = ImageUtils::resize(empty, 50, 50);

    EXPECT_TRUE(resized.empty());
}

TEST_F(ImageUtilsTest, Normalize) {
    cv::Mat normalized = ImageUtils::normalize(color_image, 1.0f / 255.0f);

    EXPECT_EQ(normalized.type(), CV_32FC3);
}

TEST_F(ImageUtilsTest, ConvertColorBGR2RGB) {
    cv::Mat rgb = ImageUtils::convertColor(color_image, cv::COLOR_BGR2RGB);

    EXPECT_EQ(rgb.cols, color_image.cols);
    EXPECT_EQ(rgb.rows, color_image.rows);
}

TEST_F(ImageUtilsTest, PreprocessForDetection) {
    cv::Mat processed = ImageUtils::preprocessForDetection(color_image, 640, 640);

    EXPECT_FALSE(processed.empty());
}

TEST_F(ImageUtilsTest, PreprocessForRecognition) {
    cv::Mat processed = ImageUtils::preprocessForRecognition(color_image, 94, 24);

    EXPECT_FALSE(processed.empty());
    EXPECT_EQ(processed.cols, 94);
    EXPECT_EQ(processed.rows, 24);
}

TEST_F(ImageUtilsTest, DrawDetection) {
    cv::Mat result = color_image.clone();
    cv::Rect bbox(10, 20, 50, 30);

    ImageUtils::drawDetection(result, bbox, 0.95f);

    EXPECT_FALSE(result.empty());
}

TEST_F(ImageUtilsTest, DrawText) {
    cv::Mat result = color_image.clone();
    cv::Point pos(50, 50);

    ImageUtils::drawText(result, pos, "TEST");

    EXPECT_FALSE(result.empty());
}

TEST_F(ImageUtilsTest, DrawDetectionOnEmptyImage) {
    cv::Mat empty;
    cv::Rect bbox(10, 20, 50, 30);

    ImageUtils::drawDetection(empty, bbox, 0.95f);

    EXPECT_TRUE(empty.empty());
}

class ConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        Config::instance().reset();
    }
};

TEST_F(ConfigTest, GetSetString) {
    Config::instance().setString("test_key", "test_value");

    std::string value = Config::instance().getString("test_key");
    EXPECT_EQ(value, "test_value");
}

TEST_F(ConfigTest, GetStringDefault) {
    std::string value = Config::instance().getString("nonexistent", "default");
    EXPECT_EQ(value, "default");
}

TEST_F(ConfigTest, GetSetInt) {
    Config::instance().setInt("int_key", 42);

    int value = Config::instance().getInt("int_key", 0);
    EXPECT_EQ(value, 42);
}

TEST_F(ConfigTest, GetIntDefault) {
    int value = Config::instance().getInt("nonexistent", -1);
    EXPECT_EQ(value, -1);
}

TEST_F(ConfigTest, GetSetFloat) {
    Config::instance().setFloat("float_key", 3.14f);

    float value = Config::instance().getFloat("float_key", 0.0f);
    EXPECT_FLOAT_EQ(value, 3.14f);
}

TEST_F(ConfigTest, GetSetBool) {
    Config::instance().setBool("bool_key", true);

    bool value = Config::instance().getBool("bool_key", false);
    EXPECT_TRUE(value);
}

TEST_F(ConfigTest, BoolFalseValue) {
    Config::instance().setBool("bool_key", false);

    bool value = Config::instance().getBool("bool_key", true);
    EXPECT_FALSE(value);
}

TEST_F(ConfigTest, Reset) {
    Config::instance().setString("test_key", "value");
    Config::instance().reset();

    std::string value = Config::instance().getString("test_key", "none");
    EXPECT_EQ(value, "none");
}

TEST_F(ConfigTest, SaveAndLoad) {
    Config::instance().setString("key1", "value1");
    Config::instance().setInt("key2", 123);
    Config::instance().setFloat("key3", 1.5f);
    Config::instance().setBool("key4", true);

    bool saved = Config::instance().save("test_config.ini");
    EXPECT_TRUE(saved);

    Config::instance().reset();
    bool loaded = Config::instance().load("test_config.ini");
    EXPECT_TRUE(loaded);

    EXPECT_EQ(Config::instance().getString("key1"), "value1");
    EXPECT_EQ(Config::instance().getInt("key2"), 123);
    EXPECT_FLOAT_EQ(Config::instance().getFloat("key3"), 1.5f);
    EXPECT_TRUE(Config::instance().getBool("key4"));
}
