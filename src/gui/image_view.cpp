#include "gui/image_view.h"
#include <QMimeData>
#include <QFileInfo>
#include <QImage>
#include <opencv2/imgcodecs.hpp>

namespace platesniper {

ImageView::ImageView(QWidget* parent)
    : QLabel(parent)
    , has_image_(false)
{
    setAlignment(Qt::AlignCenter);
    setMinimumSize(200, 150);
    setFrameStyle(QFrame::Box | QFrame::Sunken);
    setStyleSheet("QLabel { background-color: #2b2b2b; color: #ffffff; }");

    setAcceptDrops(true);
    setText("Drag & drop image here\nor click to select");
}

ImageView::~ImageView() = default;

void ImageView::setImage(const cv::Mat& image) {
    if (image.empty()) {
        clear();
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

    QPixmap scaled_pixmap = pixmap.scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);

    setPixmap(scaled_pixmap);
    has_image_ = true;
    last_file_path_.clear();
}

void ImageView::clear() {
    QLabel::clear();
    setText("Drag & drop image here\nor click to select");
    has_image_ = false;
    last_file_path_.clear();
}

QString ImageView::selectedFilePath() const {
    return last_file_path_;
}

void ImageView::dragEnterEvent(QDragEnterEvent* event) {
    if (event->mimeData()->hasUrls()) {
        event->acceptProposedAction();
        setStyleSheet("QLabel { background-color: #3b3b3b; border: 2px dashed #4a9eff; }");
    }
}

void ImageView::dragLeaveEvent(QDragLeaveEvent* event) {
    setStyleSheet("QLabel { background-color: #2b2b2b; color: #ffffff; }");
    event->accept();
}

void ImageView::dropEvent(QDropEvent* event) {
    setStyleSheet("QLabel { background-color: #2b2b2b; color: #ffffff; }");

    const QMimeData* mime_data = event->mimeData();
    if (mime_data->hasUrls()) {
        QList<QUrl> urls = mime_data->urls();
        if (!urls.isEmpty()) {
            QString file_path = urls.first().toLocalFile();
            QFileInfo file_info(file_path);

            if (file_info.exists() && file_info.isFile()) {
                loadImageFromPath(file_path);
                emit imageDropped(file_path);
            }
        }
    }

    event->acceptProposedAction();
}

void ImageView::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        if (!last_file_path_.isEmpty()) {
            emit imageClicked(last_file_path_);
        }
    }
    QLabel::mousePressEvent(event);
}

void ImageView::loadImageFromPath(const QString& path) {
    cv::Mat image = cv::imread(path.toStdString());
    if (!image.empty()) {
        setImage(image);
        last_file_path_ = path;
    }
}

cv::Mat ImageView::qImageToMat(const QImage& qimage) {
    cv::Mat mat(qimage.height(), qimage.width(), CV_8UC3,
               const_cast<uchar*>(qimage.bits()),
               static_cast<size_t>(qimage.bytesPerLine()));
    return mat.clone();
}

} // namespace platesniper
