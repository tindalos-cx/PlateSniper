#pragma once

#include <QWidget>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QString>
#include <opencv2/core.hpp>

namespace platesniper {

class ResultPanel : public QWidget {
    Q_OBJECT

public:
    explicit ResultPanel(QWidget* parent = nullptr);
    ~ResultPanel() override;

    void setPlateImage(const cv::Mat& image);
    void setPlateText(const QString& text);
    void setConfidence(float confidence);
    void setStatus(const QString& status);

    void clear();

signals:
    void saveClicked();

private:
    QLabel* plate_image_label_;
    QLabel* plate_text_label_;
    QLabel* confidence_label_;
    QLabel* status_label_;

    void updatePlateImage(const cv::Mat& image);
};

} // namespace platesniper
