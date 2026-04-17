#include <gtest/gtest.h>
#include "core/detection.h"
#include <opencv2/core.hpp>

using namespace platesniper;

TEST(DetectionTest, DefaultConstruction) {
    Detection det;
    EXPECT_EQ(det.bbox.x, 0);
    EXPECT_EQ(det.bbox.y, 0);
    EXPECT_EQ(det.bbox.width, 0);
    EXPECT_EQ(det.bbox.height, 0);
    EXPECT_FLOAT_EQ(det.confidence, 0.0f);
    EXPECT_EQ(det.classId, 0);
}

TEST(DetectionTest, Initialization) {
    Detection det;
    det.bbox = cv::Rect(10, 20, 100, 50);
    det.confidence = 0.95f;
    det.classId = 0;

    EXPECT_EQ(det.bbox.x, 10);
    EXPECT_EQ(det.bbox.y, 20);
    EXPECT_EQ(det.bbox.width, 100);
    EXPECT_EQ(det.bbox.height, 50);
    EXPECT_FLOAT_EQ(det.confidence, 0.95f);
    EXPECT_EQ(det.classId, 0);
}

TEST(DetectionTest, AreaCalculation) {
    Detection det;
    det.bbox = cv::Rect(0, 0, 100, 50);

    int area = det.bbox.area();
    EXPECT_EQ(area, 5000);
}

TEST(DetectionTest, BboxIntersection) {
    Detection det1;
    det1.bbox = cv::Rect(0, 0, 100, 100);

    Detection det2;
    det2.bbox = cv::Rect(50, 50, 100, 100);

    cv::Rect intersection = det1.bbox & det2.bbox;
    EXPECT_EQ(intersection.x, 50);
    EXPECT_EQ(intersection.y, 50);
    EXPECT_EQ(intersection.width, 50);
    EXPECT_EQ(intersection.height, 50);
}
