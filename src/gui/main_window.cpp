#include "gui/main_window.h"
#include "gui/image_view.h"
#include "gui/result_panel.h"
#include "models/model_manager.h"
#include "core/plate_detector.h"
#include "core/plate_recognizer.h"
#include "core/plate_corrector.h"
#include "core/detection.h"
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QFileDialog>
#include <QMessageBox>
#include <QStatusBar>
#include <QToolBar>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QCoreApplication>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace platesniper {

VideoCaptureThread::VideoCaptureThread(QObject* parent)
    : QThread(parent)
    , camera_index_(0)
    , running_(false)
{}

VideoCaptureThread::~VideoCaptureThread() {
    stopCapture();
    wait();
}

void VideoCaptureThread::startCapture(int camera_index) {
    camera_index_ = camera_index;
    running_ = true;
    start();
}

void VideoCaptureThread::stopCapture() {
    running_ = false;
}

void VideoCaptureThread::run() {
    cv::VideoCapture cap(camera_index_);
    if (!cap.isOpened()) {
        emit errorOccurred("Cannot open camera " + QString::number(camera_index_));
        return;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    cv::Mat frame;
    while (running_) {
        if (cap.read(frame)) {
            if (!frame.empty()) {
                emit frameReady(frame.clone());
            }
        }
        QThread::msleep(33);
    }

    cap.release();
}

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , video_thread_(nullptr)
    , process_timer_(nullptr)
    , camera_active_(false)
{
    setWindowTitle("PlateSniper - 车牌识别系统");
    setMinimumSize(1200, 700);

    setupUi();
    loadModels();
}

MainWindow::~MainWindow() {
    if (video_thread_) {
        video_thread_->stopCapture();
        delete video_thread_;
    }
    if (model_manager_) {
        model_manager_->shutdown();
    }
}

void MainWindow::setupUi() {
    createMenuBar();
    createCentralWidget();
    createStatusBar();
}

void MainWindow::createMenuBar() {
    QMenuBar* menu_bar = this->menuBar();

    QMenu* file_menu = menu_bar->addMenu("文件");

    QAction* open_action = new QAction("打开图片", this);
    open_action->setShortcut(QKeySequence::Open);
    connect(open_action, &QAction::triggered, this, &MainWindow::onOpenImage);
    file_menu->addAction(open_action);

    QAction* save_action = new QAction("保存结果", this);
    save_action->setShortcut(QKeySequence::Save);
    connect(save_action, &QAction::triggered, this, &MainWindow::onSaveResult);
    file_menu->addAction(save_action);

    file_menu->addSeparator();

    QAction* exit_action = new QAction("退出", this);
    exit_action->setShortcut(QKeySequence::Quit);
    connect(exit_action, &QAction::triggered, this, &QWidget::close);
    file_menu->addAction(exit_action);

    QMenu* camera_menu = menu_bar->addMenu("摄像头");

    QAction* open_cam_action = new QAction("打开摄像头", this);
    connect(open_cam_action, &QAction::triggered, this, &MainWindow::onOpenCamera);
    camera_menu->addAction(open_cam_action);

    QAction* close_cam_action = new QAction("关闭摄像头", this);
    connect(close_cam_action, &QAction::triggered, this, &MainWindow::onCloseCamera);
    camera_menu->addAction(close_cam_action);

    QMenu* help_menu = menu_bar->addMenu("帮助");

    QAction* about_action = new QAction("关于", this);
    connect(about_action, &QAction::triggered, []() {
        QMessageBox::information(nullptr, "关于 PlateSniper",
                               "PlateSniper v1.0\n\n"
                               "基于 OpenCV 和 ONNX Runtime 的车牌识别系统\n"
                               "支持图片和实时摄像头输入");
    });
    help_menu->addAction(about_action);
}

void MainWindow::createCentralWidget() {
    QWidget* central_widget = new QWidget(this);
    setCentralWidget(central_widget);

    QHBoxLayout* main_layout = new QHBoxLayout(central_widget);
    main_layout->setSpacing(10);
    main_layout->setContentsMargins(10, 10, 10, 10);

    QGroupBox* original_group = new QGroupBox("原图", central_widget);
    original_group->setMinimumWidth(400);
    QVBoxLayout* original_layout = new QVBoxLayout(original_group);
    original_view_ = new ImageView(original_group);
    original_layout->addWidget(original_view_);
    connect(original_view_, &ImageView::imageDropped, this, &MainWindow::onImageDropped);
    main_layout->addWidget(original_group);

    QGroupBox* detection_group = new QGroupBox("检测结果", central_widget);
    detection_group->setMinimumWidth(400);
    QVBoxLayout* detection_layout = new QVBoxLayout(detection_group);
    detection_view_ = new ImageView(detection_group);
    detection_layout->addWidget(detection_view_);
    main_layout->addWidget(detection_group);

    QGroupBox* result_group = new QGroupBox("识别结果", central_widget);
    result_group->setMinimumWidth(300);
    QVBoxLayout* result_layout = new QVBoxLayout(result_group);
    result_panel_ = new ResultPanel(result_group);
    result_layout->addWidget(result_panel_);

    QVBoxLayout* btn_layout = new QVBoxLayout();

    btn_open_image_ = new QPushButton("打开图片", this);
    btn_open_image_->setMinimumHeight(40);
    connect(btn_open_image_, &QPushButton::clicked, this, &MainWindow::onOpenImage);
    btn_layout->addWidget(btn_open_image_);

    btn_open_camera_ = new QPushButton("打开摄像头", this);
    btn_open_camera_->setMinimumHeight(40);
    connect(btn_open_camera_, &QPushButton::clicked, this, &MainWindow::onOpenCamera);
    btn_layout->addWidget(btn_open_camera_);

    btn_save_ = new QPushButton("保存结果", this);
    btn_save_->setMinimumHeight(40);
    connect(btn_save_, &QPushButton::clicked, this, &MainWindow::onSaveResult);
    btn_layout->addWidget(btn_save_);

    btn_layout->addStretch();

    result_layout->addLayout(btn_layout);
    main_layout->addWidget(result_group);
}

void MainWindow::createStatusBar() {
    QStatusBar* status_bar = this->statusBar();
    status_bar_label_ = new QLabel("Ready", this);
    status_bar->addWidget(status_bar_label_);
}

bool MainWindow::loadModels() {
    model_manager_ = std::make_unique<ModelManager>();

    QString detect_model = "models/plate_detect.onnx";
    QString rec_model = "models/plate_recognize.onnx";

    if (!model_manager_->initialize(detect_model.toStdString(),
                                    rec_model.toStdString())) {
        status_bar_label_->setText("Model load failed: " +
                                   QString::fromStdString(model_manager_->lastError()));
        return false;
    }

    status_bar_label_->setText("Models loaded successfully");
    return true;
}

void MainWindow::onOpenImage() {
    QString file_path = QFileDialog::getOpenFileName(
        this, "选择图片", QString(),
        "Images (*.png *.jpg *.jpeg *.bmp *.tiff)");

    if (!file_path.isEmpty()) {
        onImageDropped(file_path);
    }
}

void MainWindow::onOpenCamera() {
    if (camera_active_) {
        return;
    }

    video_thread_ = new VideoCaptureThread(this);
    connect(video_thread_, &VideoCaptureThread::frameReady,
            this, &MainWindow::onVideoFrameReady);
    connect(video_thread_, &VideoCaptureThread::errorOccurred, [](const QString& error) {
        QMessageBox::warning(nullptr, "Camera Error", error);
    });

    video_thread_->startCapture(0);
    camera_active_ = true;

    btn_open_camera_->setText("关闭摄像头");
    status_bar_label_->setText("Camera opened");

    process_timer_ = new QTimer(this);
    connect(process_timer_, &QTimer::timeout, [this]() {
        if (!camera_active_) return;

        cv::Mat frame;
        {
            QMutexLocker locker(&frame_mutex_);
            if (current_frame_.empty()) return;
            frame = current_frame_.clone();
        }

        processImage(frame);
    });
    process_timer_->start(100);
}

void MainWindow::onCloseCamera() {
    if (!camera_active_) {
        return;
    }

    camera_active_ = false;

    if (video_thread_) {
        video_thread_->stopCapture();
        delete video_thread_;
        video_thread_ = nullptr;
    }

    if (process_timer_) {
        process_timer_->stop();
        delete process_timer_;
        process_timer_ = nullptr;
    }

    btn_open_camera_->setText("打开摄像头");
    status_bar_label_->setText("Camera closed");
}

void MainWindow::onSaveResult() {
    if (current_detection_frame_.empty()) {
        QMessageBox::information(this, "提示", "没有可保存的结果");
        return;
    }

    QString file_path = QFileDialog::getSaveFileName(
        this, "保存结果", QString(),
        "Images (*.png *.jpg *.bmp)");

    if (!file_path.isEmpty()) {
        cv::imwrite(file_path.toStdString(), current_detection_frame_);
        status_bar_label_->setText("Result saved to: " + file_path);
    }
}

void MainWindow::onImageDropped(const QString& file_path) {
    cv::Mat image = cv::imread(file_path.toStdString());
    if (!image.empty()) {
        original_view_->setImage(image);
        processImage(image);
    } else {
        QMessageBox::warning(this, "错误", "无法读取图片: " + file_path);
    }
}

void MainWindow::onVideoFrameReady(const cv::Mat& frame) {
    QMutexLocker locker(&frame_mutex_);
    current_frame_ = frame.clone();

    QMetaObject::invokeMethod(original_view_, [this, frame]() {
        original_view_->setImage(frame);
    });
}

void MainWindow::processImage(const cv::Mat& image) {
    if (image.empty() || !model_manager_ || !model_manager_->isReady()) {
        return;
    }

    PlateCorrector corrector;

    std::vector<Detection> detections;
    auto* detector = model_manager_->detector();
    if (detector) {
        detections = detector->detect(image, 0.5f);
    }

    updateDetectionView(image, detections);

    if (detections.empty()) {
        result_panel_->setStatus("No plate detected");
        return;
    }

    auto* recognizer = model_manager_->recognizer();
    if (!recognizer) {
        result_panel_->setStatus("Recognizer not loaded");
        return;
    }

    Detection best_detection;
    float best_confidence = 0.0f;
    std::string best_plate;

    for (const auto& det : detections) {
        cv::Mat corrected = corrector.correct(image, det);
        if (corrected.empty()) {
            continue;
        }

        std::string plate_text = recognizer->recognize(corrected);

        if (!plate_text.empty() && det.confidence > best_confidence) {
            best_detection = det;
            best_confidence = det.confidence;
            best_plate = plate_text;
            result_panel_->setPlateImage(corrected);
        }
    }

    if (!best_plate.empty()) {
        result_panel_->setPlateText(QString::fromStdString(best_plate));
        result_panel_->setConfidence(best_confidence);
        result_panel_->setStatus("Recognition complete");
    } else {
        result_panel_->setPlateText("-");
        result_panel_->setStatus("Recognition failed");
    }
}

void MainWindow::updateDetectionView(const cv::Mat& original,
                                     const std::vector<Detection>& detections) {
    cv::Mat result = original.clone();

    for (const auto& det : detections) {
        cv::rectangle(result, det.bbox, cv::Scalar(0, 255, 0), 2);

        std::string label = cv::format("%.2f", det.confidence);
        cv::putText(result, label,
                   cv::Point(det.bbox.x, det.bbox.y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5,
                   cv::Scalar(0, 255, 0), 2);
    }

    current_detection_frame_ = result.clone();
    detection_view_->setImage(result);

    result_panel_->setStatus(QString("Detected %1 plate(s)").arg(detections.size()));
}

} // namespace platesniper
