#include <gtest/gtest.h>
#include "core/plate_corrector.h"
#include "core/detection.h"
#include <opencv2/core.hpp>

using namespace platesniper;

class PlateCorrectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        corrector = std::make_unique<PlateCorrector>();
        test_image = cv::Mat(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    }

    std::unique_ptr<PlateCorrector> corrector;
    cv::Mat test_image;
};

TEST_F(PlateCorrectorTest, DefaultOutputSize) {
    cv::Rect bbox(100, 100, 200, 80);
    Detection det;
    det.bbox = bbox;

    cv::Mat result = corrector->correct(test_image, det);

    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.cols, 136);
    EXPECT_EQ(result.rows, 36);
}

TEST_F(PlateCorrectorTest, CustomOutputSize) {
    corrector->setOutputSize(200, 60);

    Detection det;
    det.bbox = cv::Rect(100, 100, 200, 80);

    cv::Mat result = corrector->correct(test_image, det);

    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.cols, 200);
    EXPECT_EQ(result.rows, 60);
}

TEST_F(PlateCorrectorTest, EmptyImage) {
    cv::Mat empty;
    Detection det;
    det.bbox = cv::Rect(100, 100, 200, 80);

    cv::Mat result = corrector->correct(empty, det);

    EXPECT_TRUE(result.empty());
}

TEST_F(PlateCorrectorTest, EmptyDetection) {
    Detection det;
    det.bbox = cv::Rect();

    cv::Mat result = corrector->correct(test_image, det);

    EXPECT_FALSE(result.empty());
}

TEST_F(PlateCorrectorTest, CustomCornerPoints) {
    std::vector<cv::Point2f> corners = {
        cv::Point2f(100.0f, 100.0f),
        cv::Point2f(236.0f, 100.0f),
        cv::Point2f(236.0f, 136.0f),
        cv::Point2f(100.0f, 136.0f)
    };

    cv::Mat result = corrector->correct(test_image, corners);

    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.cols, 136);
    EXPECT_EQ(result.rows, 36);
}

TEST_F(PlateCorrectorTest, BluePlateSize) {
    corrector->setBluePlateSize(440, 140);

    Detection det;
    det.bbox = cv::Rect(100, 100, 220, 70);

    cv::Mat result = corrector->correct(test_image, det);

    EXPECT_FALSE(result.empty());
}

TEST_F(PlateCorrectorTest, GreenPlateSize) {
    corrector->setGreenPlateSize(160, 50);

    Detection det;
    det.bbox = cv::Rect(100, 100, 160, 50);

    cv::Mat result = corrector->correct(test_image, det);

    EXPECT_FALSE(result.empty());
}

TEST_F(PlateCorrectorTest, InvalidCornerPoints) {
    std::vector<cv::Point2f> invalid_corners = {
        cv::Point2f(100.0f, 100.0f),
        cv::Point2f(200.0f, 100.0f)
    };

    cv::Mat result = corrector->correct(test_image, invalid_corners);

    EXPECT_TRUE(result.empty());
}
