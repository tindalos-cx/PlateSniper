#pragma once

#include <QLabel>
#include <QMouseEvent>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QString>

class QPixmap;

namespace platesniper {

class ImageView : public QLabel {
    Q_OBJECT

public:
    explicit ImageView(QWidget* parent = nullptr);
    ~ImageView() override;

    void setImage(const cv::Mat& image);
    void clear();

    QString selectedFilePath() const;

signals:
    void imageDropped(const QString& file_path);
    void imageClicked(const QString& file_path);

protected:
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dragLeaveEvent(QDragLeaveEvent* event) override;
    void dropEvent(QDropEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;

private:
    QString last_file_path_;
    bool has_image_ = false;

    void loadImageFromPath(const QString& path);
    cv::Mat qImageToMat(const QImage& qimage);
};

} // namespace platesniper
