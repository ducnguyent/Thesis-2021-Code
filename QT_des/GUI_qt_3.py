# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI_app.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1572, 846)
        MainWindow.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setStyleSheet("")
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1548, 771))
        self.scrollAreaWidgetContents.setMouseTracking(True)
        self.scrollAreaWidgetContents.setTabletTracking(True)
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.name_app = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.name_app.setGeometry(QtCore.QRect(290, 20, 971, 151))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(28)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.name_app.setFont(font)
        self.name_app.setStyleSheet("color:rgb(48, 44, 184);\n"
                                    "font: 75 28pt \"MS Shell Dlg 2\";\n"
                                    "font-weight: bold;\n"
                                    "")
        self.name_app.setObjectName("name_app")
        self.logo = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.logo.setGeometry(QtCore.QRect(40, 20, 151, 151))
        self.logo.setText("")
        self.logo.setPixmap(QtGui.QPixmap(":/newPrefix/BK.png"))
        self.logo.setScaledContents(True)
        self.logo.setObjectName("logo")
        self.line_14 = QtWidgets.QFrame(self.scrollAreaWidgetContents)
        self.line_14.setGeometry(QtCore.QRect(1830, 350, 20, 341))
        self.line_14.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_14.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_14.setObjectName("line_14")
        self.input_main = QtWidgets.QLineEdit(self.scrollAreaWidgetContents)
        self.input_main.setGeometry(QtCore.QRect(280, 420, 601, 41))
        self.input_main.setStyleSheet("font: 10pt \"MS Shell Dlg 2\";")
        self.input_main.setObjectName("input_main")
        self.browser_main = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.browser_main.setGeometry(QtCore.QRect(900, 420, 111, 41))
        self.browser_main.setStyleSheet("Background-color:rgb(191, 191, 191);\n"
                                        "font: 75 12pt \"MS Shell Dlg 2\";")
        self.browser_main.setObjectName("browser_main")
        self.yolo_btn = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.yolo_btn.setGeometry(QtCore.QRect(280, 590, 221, 61))
        self.yolo_btn.setStyleSheet("Background-color:rgb(191, 191, 191);\n"
                                    "font: 20pt \"MS Shell Dlg 2\";")
        self.yolo_btn.setObjectName("yolo_btn")
        self.eff_btn = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.eff_btn.setGeometry(QtCore.QRect(280, 680, 221, 61))
        self.eff_btn.setStyleSheet("Background-color:rgb(191, 191, 191);\n"
                                   "font: 20pt \"MS Shell Dlg 2\";")
        self.eff_btn.setObjectName("eff_btn")
        self.input_sp = QtWidgets.QLineEdit(self.scrollAreaWidgetContents)
        self.input_sp.setGeometry(QtCore.QRect(280, 490, 601, 41))
        self.input_sp.setStyleSheet("font: 10pt \"MS Shell Dlg 2\";")
        self.input_sp.setObjectName("input_sp")
        self.browser_sp = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.browser_sp.setGeometry(QtCore.QRect(900, 490, 111, 41))
        self.browser_sp.setStyleSheet("Background-color:rgb(191, 191, 191);\n"
                                      "font: 75 12pt \"MS Shell Dlg 2\";")
        self.browser_sp.setObjectName("browser_sp")
        self.line_4 = QtWidgets.QFrame(self.scrollAreaWidgetContents)
        self.line_4.setGeometry(QtCore.QRect(1040, 270, 20, 501))
        self.line_4.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.new_data = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.new_data.setGeometry(QtCore.QRect(320, 290, 351, 51))
        self.new_data.setStyleSheet("color:rgb(0, 0, 255);\n"
                                    "font: 26pt \"MS Shell Dlg 2\";\n"
                                    "font-weight: bold;")
        self.new_data.setObjectName("new_data")
        self.img_path_2 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.img_path_2.setGeometry(QtCore.QRect(40, 590, 221, 59))
        self.img_path_2.setStyleSheet("Background-color:rgb(44, 118, 255);\n"
                                      "font: 75 18pt \"MS Shell Dlg 2\";\n"
                                      "color:rgb(255, 255, 255);")
        self.img_path_2.setObjectName("img_path_2")
        self.img_path_3 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.img_path_3.setGeometry(QtCore.QRect(40, 680, 221, 59))
        self.img_path_3.setStyleSheet("Background-color:rgb(44, 118, 255);\n"
                                      "font: 75 18pt \"MS Shell Dlg 2\";\n"
                                      "color:rgb(255, 255, 255);\n"
                                      "")
        self.img_path_3.setObjectName("img_path_3")
        self.img_path = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.img_path.setGeometry(QtCore.QRect(40, 420, 221, 39))
        self.img_path.setStyleSheet("Background-color:rgb(44, 118, 255);\n"
                                    "font: 75 18pt \"MS Shell Dlg 2\";\n"
                                    "color:rgb(255, 255, 255);")
        self.img_path.setObjectName("img_path")
        self.img_path_4 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.img_path_4.setGeometry(QtCore.QRect(40, 490, 221, 39))
        self.img_path_4.setStyleSheet("Background-color:rgb(44, 118, 255);\n"
                                      "font: 75 18pt \"MS Shell Dlg 2\";\n"
                                      "color:rgb(255, 255, 255);")
        self.img_path_4.setObjectName("img_path_4")
        self.dateEdit = QtWidgets.QDateEdit(self.scrollAreaWidgetContents)
        self.dateEdit.setGeometry(QtCore.QRect(1310, 530, 211, 41))
        self.dateEdit.setStyleSheet("font: 15pt \"MS Shell Dlg 2\";")
        self.dateEdit.setObjectName("dateEdit")
        self.choose_time = QtWidgets.QComboBox(self.scrollAreaWidgetContents)
        self.choose_time.setGeometry(QtCore.QRect(1310, 640, 211, 41))
        self.choose_time.setStyleSheet("font: 15pt \"MS Shell Dlg 2\";")
        self.choose_time.setObjectName("choose_time")
        self.old_data = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.old_data.setGeometry(QtCore.QRect(1140, 290, 351, 51))
        self.old_data.setStyleSheet("color:rgb(0, 0, 255);\n"
                                    "font: 26pt \"MS Shell Dlg 2\";\n"
                                    "font-weight: bold;")
        self.old_data.setObjectName("old_data")
        self.img_path_5 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.img_path_5.setGeometry(QtCore.QRect(1090, 530, 201, 41))
        self.img_path_5.setStyleSheet("Background-color:rgb(44, 118, 255);\n"
                                      "font: 75 18pt \"MS Shell Dlg 2\";\n"
                                      "color:rgb(255, 255, 255);")
        self.img_path_5.setObjectName("img_path_5")
        self.img_path_6 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.img_path_6.setGeometry(QtCore.QRect(1090, 640, 201, 41))
        self.img_path_6.setStyleSheet("Background-color:rgb(44, 118, 255);\n"
                                      "font: 75 18pt \"MS Shell Dlg 2\";\n"
                                      "color:rgb(255, 255, 255);")
        self.img_path_6.setObjectName("img_path_6")
        self.apply_model = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.apply_model.setGeometry(QtCore.QRect(1390, 470, 131, 41))
        self.apply_model.setStyleSheet("Background-color:rgb(191, 191, 191);\n"
                                       "font: 15pt \"MS Shell Dlg 2\";")
        self.apply_model.setObjectName("apply_model")
        self.img_path_7 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.img_path_7.setGeometry(QtCore.QRect(1090, 420, 201, 41))
        self.img_path_7.setStyleSheet("Background-color:rgb(44, 118, 255);\n"
                                      "font: 75 18pt \"MS Shell Dlg 2\";\n"
                                      "color:rgb(255, 255, 255);")
        self.img_path_7.setObjectName("img_path_7")
        self.choose_model = QtWidgets.QComboBox(self.scrollAreaWidgetContents)
        self.choose_model.setGeometry(QtCore.QRect(1310, 420, 211, 41))
        self.choose_model.setStyleSheet("font: 15pt \"MS Shell Dlg 2\";")
        self.choose_model.setObjectName("choose_model")
        self.apply_date = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.apply_date.setGeometry(QtCore.QRect(1390, 580, 131, 41))
        self.apply_date.setStyleSheet("Background-color:rgb(191, 191, 191);\n"
                                      "font: 15pt \"MS Shell Dlg 2\";")
        self.apply_date.setObjectName("apply_date")
        self.search = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.search.setGeometry(QtCore.QRect(1390, 700, 131, 41))
        self.search.setStyleSheet("Background-color:rgb(191, 191, 191);\n"
                                  "font: 15pt \"MS Shell Dlg 2\";")
        self.search.setObjectName("search")
        self.label = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label.setGeometry(QtCore.QRect(1330, 20, 181, 151))
        self.label.setStyleSheet("background-image:url(:/newPrefix/test_xml_31.jpg);")
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap(":/newPrefix/test_xml_31.jpg"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.line = QtWidgets.QFrame(self.scrollAreaWidgetContents)
        self.line.setGeometry(QtCore.QRect(0, 260, 1551, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.name_app.raise_()
        self.logo.raise_()
        self.line_14.raise_()
        self.line_4.raise_()
        self.input_main.raise_()
        self.input_sp.raise_()
        self.browser_sp.raise_()
        self.browser_main.raise_()
        self.eff_btn.raise_()
        self.img_path_2.raise_()
        self.img_path_3.raise_()
        self.img_path.raise_()
        self.img_path_4.raise_()
        self.new_data.raise_()
        self.yolo_btn.raise_()
        self.dateEdit.raise_()
        self.choose_time.raise_()
        self.old_data.raise_()
        self.img_path_5.raise_()
        self.img_path_6.raise_()
        self.apply_model.raise_()
        self.img_path_7.raise_()
        self.choose_model.raise_()
        self.apply_date.raise_()
        self.search.raise_()
        self.label.raise_()
        self.line.raise_()
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout.addWidget(self.scrollArea, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1572, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.name_app.setText(_translate("MainWindow", "         STATE RECOGNITION OF \n"
                                                       "         DISCONNECT SWITCHES"))
        self.browser_main.setText(_translate("MainWindow", "Browser"))
        self.yolo_btn.setText(_translate("MainWindow", "YOLOV4"))
        self.eff_btn.setText(_translate("MainWindow", "EfficientDet"))
        self.browser_sp.setText(_translate("MainWindow", "Browser"))
        self.new_data.setText(_translate("MainWindow", "     New Data"))
        self.img_path_2.setText(_translate("MainWindow", "    Main Model   "))
        self.img_path_3.setText(_translate("MainWindow", " Support Model "))
        self.img_path.setText(_translate("MainWindow", "    Main Input    "))
        self.img_path_4.setText(_translate("MainWindow", " Support Input  "))
        self.old_data.setText(_translate("MainWindow", "    Old Data"))
        self.img_path_5.setText(_translate("MainWindow", " Choose Date"))
        self.img_path_6.setText(_translate("MainWindow", " Choose Time"))
        self.apply_model.setText(_translate("MainWindow", "Apply"))
        self.img_path_7.setText(_translate("MainWindow", " Choose Model"))
        self.apply_date.setText(_translate("MainWindow", "Apply"))
        self.search.setText(_translate("MainWindow", "Search"))


import logo
import test_rc

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
