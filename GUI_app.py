import sys
import os
import numpy as np
from datetime import datetime, date, time
from pprint import pprint
import xlrd
from xlsxwriter.utility import xl_rowcol_to_cell

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QApplication, QPushButton
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from QT_des.GUI_qt_3 import Ui_MainWindow
from QT_des.eff import Ui_eff_Window
from QT_des.yolo import Ui_yolo_Window

from save_file import save_data

from effi_detect.EfficientDet_Detection import Detect_Effi
from yolo_detect.Yolov4_Detection import Detect_yolo

# Path to EfficientDet model material
PATH_TO_MODEL = "./EfficientDet_Material/test_model/new_d1/saved_model-d1-newdata-30000/"
PATH_TO_LABELS = "./EfficientDet_Material/DCL_label_map.pbtxt"
PATH_TO_RESULT_IMAGES_DIR = "./output/"
load_model = None

# Path to YOLOv4 model material
YOLO_WEIGHTS = "./YOLO_Material/ver4_27.5/yolov4_DS2_last.weights"
YOLO_CFG = "./YOLO_Material/ver4_27.5/yolov4_DS2.cfg"
NAMES = "./YOLO_Material/DS.names"

yolo_confidence_thresh = 0.7

yolo_detector = Detect_yolo(YOLO_WEIGHTS, YOLO_CFG, NAMES, yolo_confidence_thresh, "",
                            PATH_TO_RESULT_IMAGES_DIR, "")
# efficientnet_detector = Detect_Effi(PATH_TO_MODEL, PATH_TO_LABELS, "", PATH_TO_RESULT_IMAGES_DIR, "")


class project(QMainWindow, Ui_MainWindow, Ui_eff_Window, Detect_Effi, Detect_yolo, Ui_yolo_Window, save_data):
    def __init__(self, verbose=1):
        super(project, self).__init__()
        self.ui = Ui_MainWindow()
        self.setupUi(self)
        self.choose_model.addItem("main")
        self.choose_model.addItem("support")

        self.eff_win = QtWidgets.QMainWindow()
        self.ui2 = Ui_eff_Window()
        self.ui2.setupUi(self.eff_win)

        self.yolo_win = QtWidgets.QMainWindow()
        self.ui3 = Ui_yolo_Window()
        self.ui3.setupUi(self.yolo_win)

        self.yolo_win = QtWidgets.QMainWindow()
        self.ui3 = Ui_yolo_Window()
        self.ui3.setupUi(self.yolo_win)

        self.input_sp.setFont(QFont('Arial', 10))
        self.input_main.setFont(QFont('Arial', 10))

        self.ui2.state_in.setFont(QFont('Arial', 24))
        self.ui2.score_in.setFont(QFont('Arial', 24))
        self.ui2.time_in.setFont(QFont('Arial', 24))
        self.ui2.date_in.setFont(QFont('Arial', 16))

        self.ui3.state_in.setFont(QFont('Arial', 24))
        self.ui3.score_in.setFont(QFont('Arial', 24))
        self.ui3.time_in.setFont(QFont('Arial', 24))
        self.ui3.date_in.setFont(QFont('Arial', 16))

        self.eff_btn.clicked.connect(self.start_eff)

        self.yolo_btn.clicked.connect(self.start_yolo)

        self.browser_main.clicked.connect(self.getfiles_main)
        self.browser_sp.clicked.connect(self.getfiles_sp)

        self.apply_model.clicked.connect(self.model_function)
        self.apply_date.clicked.connect(self.apply_date_func)
        self.search.clicked.connect(self.apply_time)

        self.system_dict = {"verbose": verbose, "local": {}}
        self.system_dict["local"]["common_size"] = 512
        self.system_dict["local"]["mean"] = np.array([[[0.485, 0.456, 0.406]]])
        self.system_dict["local"]["std"] = np.array([[[0.229, 0.224, 0.225]]])

        self.today = date.today()
        self.day = self.today.strftime("%d/%m/%Y")

    def getfiles_main(self):
        fileName_main, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Single File', os.getcwd(), '*.jpg')
        self.input_main.setText(fileName_main)

    def getfiles_sp(self):
        fileName_sp, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Single File', os.getcwd(), '*.jpg')
        self.input_sp.setText(fileName_sp)

    def start_eff(self):
        if self.input_sp.text().strip() == '':
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please fill the path to input image!")
            msg.setWindowTitle("Warning")
            msg.exec_()
        else:
            img_path_eff = self.input_sp.text().strip()
            efficientnet_detector.input_dir = img_path_eff
            efficientnet_detector.day = self.day

            efficientnet_detector.predict()

            self.sp_path = str(efficientnet_detector.eff_path)
            self.sp_code = str(efficientnet_detector.code)
            self.sp_state = str(efficientnet_detector.label)
            self.sp_score = str(efficientnet_detector.score)
            self.sp_speed = str(efficientnet_detector.detection_time)
            self.sp_now = efficientnet_detector.now

            model = "support"
            self.display(self.sp_path, self.sp_state, self.sp_score, self.sp_speed, self.day, model)

    def start_yolo(self):
        if self.input_main.text().strip() == '':
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please fill the path to input image!")
            msg.setWindowTitle("Warning")
            msg.exec_()
        else:
            img_path_yolo = self.input_main.text().strip()
            yolo_detector.image_path = img_path_yolo
            yolo_detector.day = self.day
            yolo_detector.predict()
            # self.open_yolo()
            self.main_path = str(yolo_detector.yolo_path)
            self.main_code = str(yolo_detector.code)
            self.main_state = str(yolo_detector.label)
            self.main_score = str(yolo_detector.score)
            self.main_speed = str(yolo_detector.detection_time)
            self.main_now = yolo_detector.now

            model = "main"
            self.display(self.main_path, self.main_state, self.main_score, self.main_speed, self.day, model)

    def display(self, path, state, score, speed, date, model):
        if model == "support":
            self.ui2.img_in.setPixmap(QtGui.QPixmap(path))

            self.ui2.score_in.clear()
            self.ui2.score_in.append(score)

            self.ui2.time_in.clear()
            self.ui2.time_in.append(speed)

            self.ui2.state_in.clear()
            self.ui2.state_in.append(state)

            self.ui2.date_in.clear()
            self.ui2.date_in.append(str(date))

            self.ui2.save_btn.clicked.connect(lambda: self.save(model))
            self.open_eff()
        if model == "main":
            self.ui3.img_in.setPixmap(QtGui.QPixmap(path))

            self.ui3.score_in.clear()
            self.ui3.score_in.append(score)

            self.ui3.time_in.clear()
            self.ui3.time_in.append(speed)

            self.ui3.state_in.clear()
            self.ui3.state_in.append(state)

            self.ui3.date_in.clear()
            self.ui3.date_in.append(str(date))

            self.ui3.save_btn.clicked.connect(lambda: self.save(model))
            self.open_yolo()

    def open_eff(self):
        self.eff_win.show()

    def open_yolo(self):
        self.yolo_win.show()

    def save(self, model):
        if model == "support":
            saved = save_data(self.sp_now, self.sp_code, self.sp_state, self.sp_score, self.sp_speed, model)
        if model == "main":
            saved = save_data(self.main_now, self.main_code, self.main_state, self.main_score, self.main_speed, model)
        message = saved.message
        if message is not None:
            print(message)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText(message)
            msg.setWindowTitle("Error")
            msg.exec_()

    def model_function(self):
        self.model_selected = self.choose_model.currentText()

    def apply_date_func(self):
        try:
            self.day_value = self.dateEdit.date().day()
            self.month_value = self.dateEdit.date().month()
            self.year_value = self.dateEdit.date().year()
            self.Path_to_find = "./output/Info/" + self.model_selected + "/" + self.model_selected + "_" \
                                + str(self.month_value) + str(self.year_value) + '.xlsx'
            print(self.Path_to_find)
        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("please choose the model and put Apply button!")
            msg.setWindowTitle("Error")
            msg.exec_()
            pass
        try:
            if not os.path.isfile(self.Path_to_find):
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("No data on " + str(self.month_value) + "/" + str(self.year_value))
                msg.setWindowTitle("Error")
                msg.exec_()
            book = xlrd.open_workbook(self.Path_to_find)
            self.choose_time.clear()
            self.available_list = []
            for sh in book.sheets():
                for row in range(sh.nrows):
                    myCell = sh.cell_value(row, 0)
                    print(" myCell: " + str(myCell))
                    print("book: " + str(book.datemode))
                    # print(date(year_value, month_value, day_value))
                    self.date_value = None
                    try:
                        self.date_value = datetime(*xlrd.xldate_as_tuple(myCell, book.datemode)).date()
                        print('datetime: %s' % self.date_value)
                    except:
                        pass

                    if self.date_value == date(self.year_value, self.month_value, self.day_value):
                        print("position: " + str(xl_rowcol_to_cell(row, 0)))
                        self.myCol = sh.cell_value(row, 1)
                        print("Available value: " + str(self.myCol))
                        print("Available: " + str(xl_rowcol_to_cell(row, 1)))
                        self.available_list.append(row)
                        self.choose_time.addItem(self.myCol)
        except Exception as e:
            s = str(e)
            print(s)
            pass

    def apply_time(self):
        try:
            time_value_selected = self.choose_time.currentText()
            print("type: " + str(type(time_value_selected)))
            print("time value seclected: " + str(time_value_selected))
        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("please choose the model and put Apply button!")
            msg.setWindowTitle("Error")
            msg.exec_()
            pass
        try:
            self.Path_to_search = './output/Info/' + self.model_selected + '/' + self.model_selected + '_' + str(
                self.month_value) + str(self.year_value) + '.xlsx'
            print(self.Path_to_search)
            if not os.path.isfile(self.Path_to_search):
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("No such file or directory: " + str(self.Path_to_search))
                msg.setWindowTitle("Error")
                msg.exec_()
            book = xlrd.open_workbook(self.Path_to_search)
            for sh in book.sheets():
                for row in self.available_list:
                    myCell = sh.cell_value(row, 1)
                    print(" myCell: " + str(myCell))
                    print("book: " + str(book.datemode))
                    # print(date(year_value, month_value, day_value))
                    self.time_value = None
                    try:
                        # self.time_value = datetime(*xlrd.xldate_as_tuple(myCell, book.datemode)).time()
                        # self.time_value = datetime(*xlrd.xldate_as_tuple(myCell, book.datemode)).time()
                        print('time_value: %s' % myCell)
                    except Exception as e:
                        s = str(e)
                        print(s)
                        pass

                    # if self.time_value == time(self.hour_value, self.min_value, self.sec_value):

                    if str(myCell) == time_value_selected:
                        print("yes!")
                        print("position: " + str(xl_rowcol_to_cell(row, 0)))
                        code_value = sh.cell_value(row, 2)
                        state_value = sh.cell_value(row, 3)
                        score_value = sh.cell_value(row, 4)
                        speed_value = sh.cell_value(row, 5)
                        path_value = "./output/Image/" + self.model_selected + "/" + str(
                            code_value) + ".jpg"
                        self.display(path_value, str(state_value), str(score_value), str(speed_value), self.date_value,
                                     self.model_selected)
        except Exception as e:
            s = str(e)
            print(s)
            pass


if __name__ == "__main__":
    import sys

    # ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 6)
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    pr = project()
    pr.show()
    sys.exit(app.exec_())
