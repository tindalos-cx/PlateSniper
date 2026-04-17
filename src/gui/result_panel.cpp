#include "gui/result_panel.h"
#include <QFont>
#include <QLabel>
#include <QFrame>

namespace platesniper {

ResultPanel::ResultPanel(QWidget* parent)
    : QWidget(parent)
{
    QVBoxLayout* main_layout = new QVBoxLayout(this);
    main_layout->setSpacing(10);
    main_layout->setContentsMargins(10, 10, 10, 10);

    plate_image_label_ = new QLabel(this);
    plate_image_label_->setAlignment(Qt::AlignCenter);
    plate_image_label_->setMinimumSize(200, 60);
    plate_image_label_->setFrameStyle(QFrame::Box | QFrame::Sunken);
    plate_image_label_->setStyleSheet("QLabel { background-color: #1a1a1a; border: 1px solid #444; }");
    plate_image_label_->setText("No plate detected");
    main_layout->addWidget(plate_image_label_);

    QLabel* text_label = new QLabel("Recognized Plate:", this);
    text_label->setStyleSheet("QLabel { color: #aaaaaa; font-size: 12px; }");
    main_layout->addWidget(text_label);

    plate_text_label_ = new QLabel(this);
    plate_text_label_->setAlignment(Qt::AlignCenter);

    QFont text_font("Microsoft YaHei", 24, QFont::Bold);
    plate_text_label_->setFont(text_font);
    plate_text_label_->setStyleSheet("QLabel { color: #00ff00; background-color: #1a1a1a; "
                                    "padding: 10px; border-radius: 5px; }");
    plate_text_label_->setText("-");
    main_layout->addWidget(plate_text_label_);

    QLabel* conf_label = new QLabel("Confidence:", this);
    conf_label->setStyleSheet("QLabel { color: #aaaaaa; font-size: 12px; }");
    main_layout->addWidget(conf_label);

    confidence_label_ = new QLabel(this);
    confidence_label_->setAlignment(Qt::AlignCenter);
    QFont conf_font("Consolas", 14);
    confidence_label_->setFont(conf_font);
    confidence_label_->setStyleSheet("QLabel { color: #ffcc00; background-color: #1a1a1a; "
                                     "padding: 5px; border-radius: 3px; }");
    confidence_label_->setText("0.00%");
    main_layout->addWidget(confidence_label_);

    main_layout->addStretch();

    QLabel* status_title = new QLabel("Status:", this);
    status_title->setStyleSheet("QLabel { color: #aaaaaa; font-size: 12px; }");
    main_layout->addWidget(status_title);

    status_label_ = new QLabel(this);
    status_label_->setAlignment(Qt::AlignCenter);
    status_label_->setStyleSheet("QLabel { color: #ffffff; background-color: #333333; "
                                 "padding: 8px; border-radius: 3px; }");
    status_label_->setText("Ready");
    main_layout->addWidget(status_label_);
}

ResultPanel::~ResultPanel() = default;

void ResultPanel::setPlateImage(const cv::Mat& image) {
    updatePlateImage(image);
}

void ResultPanel::setPlateText(const QString& text) {
    plate_text_label_->setText(text.isEmpty() ? "-" : text);
}

void ResultPanel::setConfidence(float confidence) {
    confidence_label_->setText(QString::number(confidence * 100, 'f', 2) + "%");
}

void ResultPanel::setStatus(const QString& status) {
    status_label_->setText(status);
}

void ResultPanel::clear() {
    plate_image_label_->setText("No plate detected");
    plate_text_label_->setText("-");
    confidence_label_->setText("0.00%");
    status_label_->setText("Ready");
}

void ResultPanel::updatePlateImage(const cv::Mat& image) {
    if (image.empty()) {
        plate_image_label_->setText("No plate detected");
        return;
    }

    cv::Mat rgb_image;
    if (image.channels() == 3) {
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
    } else {
        cv::cvtColor(image, rgb_image, cv::COLOR_GRAY2RGB);
    }

    QImage qimage(rgb_image.data, rgb_image.cols, rgb_image.rows,
                 rgb_image.step, QImage::Format_RGB888);

    QPixmap pixmap = QPixmap::fromImage(qimage);
    QPixmap scaled = pixmap.scaled(plate_image_label_->size(), Qt::KeepAspectRatio,
                                  Qt::SmoothTransformation);

    plate_image_label_->setPixmap(scaled);
}

} // namespace platesniper
