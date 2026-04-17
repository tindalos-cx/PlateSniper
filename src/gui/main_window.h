#pragma once

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QThread>
#include <QTimer>
#include <QMutex>
#include <memory>
#include <opencv2/core.hpp>

namespace platesniper {

class PlateDetector;
class PlateRecognizer;
class PlateCorrector;
class ModelManager;
class ImageView;
class ResultPanel;

class VideoCaptureThread;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

private slots:
    void onOpenImage();
    void onOpenCamera();
    void onCloseCamera();
    void onSaveResult();
    void onImageDropped(const QString& file_path);
    void onVideoFrameReady(const cv::Mat& frame);

private:
    std::unique_ptr<ModelManager> model_manager_;

    ImageView* original_view_;
    ImageView* detection_view_;
    ResultPanel* result_panel_;

    QPushButton* btn_open_image_;
    QPushButton* btn_open_camera_;
    QPushButton* btn_save_;
    QLabel* status_bar_label_;

    VideoCaptureThread* video_thread_;
    QTimer* process_timer_;

    cv::Mat current_frame_;
    cv::Mat current_detection_frame_;
    QMutex frame_mutex_;

    bool camera_active_;

    void setupUi();
    void createMenuBar();
    void createCentralWidget();
    void createStatusBar();

    bool loadModels();
    void processImage(const cv::Mat& image);
    void updateDetectionView(const cv::Mat& original,
                            const std::vector<Detection>& detections);

    std::string detectAndRecognize(const cv::Mat& image,
                                  std::vector<Detection>& detections,
                                  float& best_confidence);
};

class VideoCaptureThread : public QThread {
    Q_OBJECT

public:
    explicit VideoCaptureThread(QObject* parent = nullptr);
    ~VideoCaptureThread() override;

    void startCapture(int camera_index = 0);
    void stopCapture();

signals:
    void frameReady(const cv::Mat& frame);
    void errorOccurred(const QString& error);

protected:
    void run() override;

private:
    int camera_index_;
    volatile bool running_;
};

} // namespace platesniper
