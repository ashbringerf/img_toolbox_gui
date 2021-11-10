# -*- coding: utf-8 -*-
from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget)
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.Qt import QThread
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pandas as pd
import numpy as np
import os
import sys
import time
import cv2

## notes

class GUI(QDialog):
    def __init__(self, parent=None):
        ## layout related
        super(GUI, self).__init__(parent)
        
        self.create_img_groupbox()
        self.create_mainbut_groupbox()
        self.create_subbut_groupbox()
        self.create_thumbnail_groupbox()
        self.create_hist_groupbox()
        self.create_colormap_groupbox()
        
        
        self.mainLayout = QGridLayout()
        self.mainLayout.addWidget(self.img_groupbox, 0, 0, 3, 1)
        self.mainLayout.setColumnStretch(0, 4)
        self.mainLayout.addWidget(self.mainbut_groupbox, 0, 1)
        self.mainLayout.addWidget(self.subbut_groupbox, 1, 1)
        self.mainLayout.setColumnStretch(1, 0.5)
        self.mainLayout.addWidget(self.thumbnail_groupbox, 0, 2, 1, 1)
        self.mainLayout.addWidget(self.hist_groupbox, 1, 2, 1, 1)
        self.mainLayout.addWidget(self.colormap_groupbox, 2, 2, 1, 1)
        self.mainLayout.setColumnStretch(2, 1)
        
        self.setAcceptDrops(True)
        
        self.setGeometry(0, 0, 1200, 900)
        self.setLayout(self.mainLayout)
        self.setWindowTitle('img toolbox')
        self.move(100, 100)
        self.setStyleSheet("background-color: grey")
        
        ## class varibles
        self.path_img = r'example.jpg'
        self.roi_loc_x, self.roi_loc_y = 0, 0
        self.roi_bound_x, self.roi_bound_y = 0, 0
        self.roi_size = 100
        
    """ layouts """
    """                """
    def create_img_groupbox(self):
        self.img_groupbox = QScrollArea()
        self.imagelabel = QLabel('img')
        self.img_groupbox.setWidget(self.imagelabel)
    
    def create_thumbnail_groupbox(self):
        self.thumbnail_groupbox = QScrollArea()
        self.imagelabel_sub = QLabel('img thumbnail')
        self.thumbnail_groupbox.setWidget(self.imagelabel_sub)    
        
        #print(' width :', self.thumbnail_groupbox.frameGeometry().width())
        #print(' height :', self.thumbnail_groupbox.frameGeometry().height())
    
    def create_hist_groupbox(self):
        self.hist_groupbox = QScrollArea()
        self.imagelabel_hist = QLabel('roi hist')
        self.hist_groupbox.setWidget(self.imagelabel_hist)  

    def create_colormap_groupbox(self):
        self.colormap_groupbox = QScrollArea()
        self.imagelabel_colormap = QLabel('roi colormap')
        self.colormap_groupbox.setWidget(self.imagelabel_colormap) 
    
    def create_mainbut_groupbox(self):
        self.mainbut_groupbox = QGroupBox('mainfunc button')
        PushButton_exit = QPushButton('exit')
        PushButton_load_img = QPushButton('load_img')
        layout = QVBoxLayout()
        layout.setSpacing(5)
        layout.addWidget(PushButton_load_img)
        layout.addWidget(PushButton_exit)
        layout.addStretch(1)
        self.mainbut_groupbox.setLayout(layout)
        
        ## connect
        PushButton_exit.clicked.connect(self.close_GUI)
        PushButton_load_img.clicked.connect(self.select_img)


    def create_subbut_groupbox(self):
        self.subbut_groupbox = QGroupBox('subfunc button')
        PushButton_crop = QPushButton('roi crop')
        layout = QVBoxLayout()
        layout.setSpacing(5)
        layout.addWidget(PushButton_crop)
        layout.addStretch(1)
        self.subbut_groupbox.setLayout(layout)
        
        ## connect
        #PushButton_exit.clicked.connect(self.closeGUI)
        #PushButton_load_img.clicked.connect(self.load_img)
    
    
    """ calback events """
    """                """
    def dragEnterEvent(self, event):
        print('drag_event_enter')
        if event.mimeData().hasUrls():
            print('file_has_urls')
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        lines = []
        for url in event.mimeData().urls():
            lines.append('dropped: %r' % url.toLocalFile())
            print('url ', url.toLocalFile())
            self.path_img = url.toLocalFile()
            self.refresh_img()
        #self.setText('\n'.join(lines))
        
    def mousePressEvent(self, QMouseEvent):
        print('mouse press: ', QMouseEvent.pos())
        ## the bot right img thumbnail
        ## get the absolute pos in full gui
        ## set relative pos 
        roi_bound_x, roi_bound_y = self.thumbnail_groupbox.frameGeometry().width() * 0.9, \
                                   self.thumbnail_groupbox.frameGeometry().width() * 0.9 / 4 * 3
        roi_x, roi_y = QMouseEvent.pos().x() - self.thumbnail_groupbox.geometry().x(), \
                       QMouseEvent.pos().y() - self.thumbnail_groupbox.geometry().y()
        print('bot right Qlabel width: ', self.thumbnail_groupbox.frameGeometry().width())
        print('bot right Qlabel height: ', self.thumbnail_groupbox.frameGeometry().height())
        print('roi_x: ', roi_x)
        print('roi_y: ', roi_y)
        
        self.roi_loc_x, self.roi_loc_y = roi_x/roi_bound_x, \
                                         roi_y/roi_bound_y
        
        self.roi_bound_x, self.roi_bound_y = roi_bound_x, roi_bound_y
        print('roi_x_: ', self.roi_loc_x)
        print('roi_y_: ', self.roi_loc_y)
        
        self.refresh_img()
        
    def wheelEvent(self, QMouseEvent):
        angle = QMouseEvent.angleDelta()
        print('mouse scroll angle ', angle)
        if angle.y() > 0:
            self.roi_size = self.roi_size * 1.1
        else:
            self.roi_size = self.roi_size * 0.9
        self.refresh_img()
        
        
    """ subfunctions """
    """              """
    def select_img(self):
        self.path_img = QFileDialog.getOpenFileName(self, 'Open a file', '', 'All Files (*.*)')[0]
        self.refresh_img()
        
    def refresh_img(self):
        ## convert cv2 mat to qr pixelmap to bypass QPixmap::scaled: Pixmap is a null pixmap
        image = cv2.imread(self.path_img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ## draw roi
        cv2.rectangle(image, (int(self.roi_loc_x * image.shape[1]), int(self.roi_loc_y * image.shape[0])), \
                             (int(self.roi_loc_x * image.shape[1] + self.roi_size), int(self.roi_loc_y * image.shape[0] + self.roi_size)), \
                             color = (200, 0, 0), \
                             thickness = 5)
        
        image = QImage(image, image.shape[1], image.shape[0], image.shape[1] * 3, QImage.Format_RGB888)
        pixmap = QPixmap(image)
        self.imagelabel.setPixmap(pixmap)
        self.imagelabel.resize(self.width(), self.height())
        
        self.imagelabel_sub.setPixmap(pixmap)
        self.imagelabel_sub.setScaledContents(True)
        #self.imagelabel_sub.resize(200, 200)
        self.imagelabel_sub.resize(\
            self.thumbnail_groupbox.frameGeometry().width() * 0.9, \
            self.thumbnail_groupbox.frameGeometry().width() * 0.9 / 4 * 3 )
        print('self.width()',self.imagelabel_sub.width())
        
    def close_GUI(self): 
        sys.exit()    
    



if __name__ == '__main__':
     
    app = QApplication(sys.argv)
    ui = GUI()
    ui.show()
    sys.exit(app.exec_()) 
    
    
    
    