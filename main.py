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
import io
## 
import matplotlib.pyplot as plt
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
        self.mainLayout.setColumnStretch(2, 2)
        
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
        self.image = []
        self.image_roi = []
        self.fig_hist = []
        self.fig_cc = []
        
        
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
        PushButton_crop.clicked.connect(self.plot_hist)
        PushButton_crop.clicked.connect(self.drawFigure_ellipse)
    
    
    """ calback events """
    """                """
    def dragEnterEvent(self, event):
        #print('drag_event_enter')
        if event.mimeData().hasUrls():
            #print('file_has_urls')
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
        #print('mouse press: ', QMouseEvent.pos())
        ## the bot right img thumbnail
        ## get the absolute pos in full gui
        ## set relative pos 
        roi_bound_x, roi_bound_y = self.thumbnail_groupbox.frameGeometry().width() * 0.9, \
                                   self.thumbnail_groupbox.frameGeometry().width() * 0.9 / 4 * 3
        roi_x, roi_y = QMouseEvent.pos().x() - self.thumbnail_groupbox.geometry().x(), \
                       QMouseEvent.pos().y() - self.thumbnail_groupbox.geometry().y()
        #print('bot right Qlabel width: ', self.thumbnail_groupbox.frameGeometry().width())
        #print('bot right Qlabel height: ', self.thumbnail_groupbox.frameGeometry().height())
        #print('roi_x: ', roi_x)
        #print('roi_y: ', roi_y)
        
        self.roi_loc_x, self.roi_loc_y = roi_x/roi_bound_x, \
                                         roi_y/roi_bound_y
        
        self.roi_bound_x, self.roi_bound_y = roi_bound_x, roi_bound_y
        #print('roi_x_: ', self.roi_loc_x)
        #print('roi_y_: ', self.roi_loc_y)
        
        self.refresh_img()
        
    def wheelEvent(self, QMouseEvent):
        angle = QMouseEvent.angleDelta()
        #print('mouse scroll angle ', angle)
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
        if self.image == []:
            self.image = cv2.imread(self.path_img)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        
        self.image_roi = self.image.copy()
        cv2.rectangle(self.image_roi, (int(self.roi_loc_x * self.image_roi.shape[1]), int(self.roi_loc_y * self.image_roi.shape[0])), \
                             (int(self.roi_loc_x * self.image_roi.shape[1] + self.roi_size), int(self.roi_loc_y * self.image_roi.shape[0] + self.roi_size)), \
                             color = (200, 0, 0), \
                             thickness = 5)
        
        ## update main image by roi
        
        try:
        #self.image_cropped = self.image
            self.image_cropped = self.image[int(self.roi_loc_y * self.image_roi.shape[0]):\
                                        int(self.roi_loc_y * self.image_roi.shape[0] + self.roi_size), \
                                        int(self.roi_loc_x * self.image_roi.shape[1]):\
                                        int(self.roi_loc_x * self.image_roi.shape[1] + self.roi_size), \
                                        :]
            print(self.image_cropped.shape)                                
            #cv2.imshow('img ',self.image_cropped)
            #cv2.waitKey()
            print('before crop', self.image_cropped.shape[1], self.image_cropped.shape[0])
            self.image_cropped_ = QImage(self.image_cropped.data.tobytes(), self.image_cropped.shape[1], self.image_cropped.shape[0], \
                                self.image_cropped.shape[1] * 3, QImage.Format_RGB888)
            print('after crop')
            #self.imagelabel.resize(self.img_groupbox.frameGeometry().width(), \
            #            self.img_groupbox.frameGeometry().height())   
            pixmap = QPixmap(self.image_cropped_)
            self.imagelabel.setPixmap(pixmap)
            self.imagelabel.setScaledContents(True)
            self.imagelabel.resize(self.width(), self.height())            
        except:
            pass
        
        #print(self.image_cropped.shape[1])
        #print(self.image_cropped.shape[0])
        #self.image_cropped_ = QImage(self.image_cropped.data.tobytes(), self.image_cropped.shape[1], self.image_cropped.shape[0], \
        #                             self.image_cropped.shape[1] * 3, QImage.Format_RGB888)
        #pixmap = QPixmap(self.image_cropped_)
        #self.imagelabel.setPixmap(pixmap)
        #self.imagelabel.setScaledContents(True)
        #self.imagelabel.resize(self.width(), self.height())

        
        

        self.image_roi_ = QImage(self.image_roi, self.image_roi.shape[1], self.image_roi.shape[0], \
                                 self.image_roi.shape[1] * 3, QImage.Format_RGB888)
        pixmap = QPixmap(self.image_roi_)
        self.imagelabel_sub.setPixmap(pixmap)
        self.imagelabel_sub.setScaledContents(True)
        #self.imagelabel_sub.resize(200, 200)
        self.imagelabel_sub.resize(\
            self.thumbnail_groupbox.frameGeometry().width() * 0.9, \
            self.thumbnail_groupbox.frameGeometry().width() * 0.9 / 4 * 3 )
        #print('self.width()',self.imagelabel_sub.width())
        
        if self.fig_hist != []:
            #print('in hist loop')
            image = QImage(self.fig_hist, self.fig_hist.shape[1], self.fig_hist.shape[0], self.fig_hist.shape[1] * 3, QImage.Format_RGB888)
            pixmap = QPixmap(image)
            self.imagelabel_hist.setPixmap(pixmap)
            self.imagelabel_hist.setScaledContents(True)
            self.imagelabel_hist.resize(\
                self.hist_groupbox.frameGeometry().width() * 0.9, \
                self.hist_groupbox.frameGeometry().width() * 0.9 / 4 * 3 )

        if self.fig_cc != []:
            #print('in cc loop')
            image = QImage(self.fig_cc, self.fig_cc.shape[1], self.fig_cc.shape[0], self.fig_cc.shape[1] * 3, QImage.Format_RGB888)
            pixmap = QPixmap(image)
            self.imagelabel_colormap.setPixmap(pixmap)
            self.imagelabel_colormap.setScaledContents(True)
            self.imagelabel_colormap.resize(\
                self.colormap_groupbox.frameGeometry().width() * 0.9, \
                self.colormap_groupbox.frameGeometry().width() * 0.9 / 4 * 3 )

    def plot_hist(self):
    
        ## 
        gray_int = 0.5
        ##
        image = cv2.imread(self.path_img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ## plot histogram of ROIs
        fig = plt.figure()
        fig.set_facecolor([gray_int, gray_int, gray_int])
        n, bins, patches = plt.hist(image[:, 0], 50, density = True, facecolor = 'b', alpha = 0.3, label = 'blue')
        n, bins, patches = plt.hist(image[:, 1], 50, density = True, facecolor = 'g', alpha = 0.3, label = 'green')
        n, bins, patches = plt.hist(image[:, 2], 50, density = True, facecolor = 'r', alpha = 0.3, label = 'red')
        plt.xlabel('pixel value')
        plt.ylabel('Probability')
        plt.title('Hist of img')
        plt.xlim(0, 255)
        plt.ylim(0, 0.2)
        plt.legend(loc = 'upper right')
        ax = plt.axes()
        ax.set_facecolor([gray_int, gray_int, gray_int])
        
        ## dump matplotlib figure to numpy array
        io_buf = io.BytesIO()
        plt.savefig(io_buf, format='png', dpi = 300)
        io_buf.seek(0)
        fig_hist = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)
        fig_hist = cv2.imdecode(fig_hist, 1)  
        self.fig_hist = fig_hist       
        io_buf.close()
        plt.close()
        
        ## refresh
        self.refresh_img()
       
    
    def drawFigure_ellipse(self):
        """
        draw fiugre of ellipse in Lab space
        """
        gray_int = 0.5
        range_i = 100
        L_fix = 75
        #color_scale_ratio = 2
        map_Lab = np.ones((range_i * 2, range_i * 2, 3), dtype = np.float32)
        map_Lab[:, :, 0] = L_fix * np.ones((range_i * 2, range_i * 2))
        map_Lab[:, :, 2], map_Lab[:, :, 1] = np.mgrid[-range_i:range_i, -range_i:range_i]
        map_RGB = cv2.cvtColor(map_Lab, cv2.COLOR_Lab2RGB)
        map_RGB = np.flip(map_RGB, 0)
        map_RGB = cv2.cvtColor(map_RGB, cv2.COLOR_RGB2BGR)
        
        fig = plt.figure(dpi = 300)
        fig.set_facecolor([gray_int, gray_int, gray_int])
        plt.imshow(map_RGB, extent=[-range_i, range_i, -range_i, range_i])
        plt.xlabel('a*')
        plt.ylabel('b*')
        plt.xticks(np.arange(-range_i, range_i, int(range_i / 3)))
        plt.yticks(np.arange(-range_i, range_i, int(range_i / 3)))
        plt.title('CIELab coordinates')
        plt.ylim([-range_i, range_i])
        plt.xlim([-range_i, range_i])
        plt.legend(loc = 3, fontsize = 8, framealpha = 0.5)
        
        ## dump matplotlib figure to numpy array
        io_buf = io.BytesIO()
        plt.savefig(io_buf, format='png', dpi = 300)
        io_buf.seek(0)
        fig_cc = np.frombuffer(io_buf.getvalue(), dtype = np.uint8)
        fig_cc = cv2.imdecode(fig_cc, 1)  
        self.fig_cc = fig_cc       
        io_buf.close()
        plt.close()
        
        ## refresh
        self.refresh_img()
        #plt.savefig(os.path.join(img_path, img_name))
        #plt.close(fig)
    
    def close_GUI(self): 
        sys.exit()    
    



if __name__ == '__main__':
     
    app = QApplication(sys.argv)
    ui = GUI()
    ui.show()
    sys.exit(app.exec_()) 
    
    
    
    