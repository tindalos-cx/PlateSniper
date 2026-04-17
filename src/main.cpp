#include <QApplication>
#include <QMessageBox>
#include <iostream>
#include "gui/main_window.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    app.setApplicationName("PlateSniper");
    app.setOrganizationName("PlateSniper");
    app.setApplicationVersion("1.0.0");

    platesniper::MainWindow window;
    window.show();

    return app.exec();
}
