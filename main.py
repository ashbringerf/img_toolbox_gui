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
import json
## 
import matplotlib.pyplot as plt
from color_CF_elli import chroma_shift, fit_elli_trace
import color_ulti as cu
import utility
import utility_signal_processing
import utility_img_processing
import pywt
import pywt.data
## notes

class GUI(QDialog):

    resized = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        ## layout related
        super(GUI, self).__init__(parent)
        
        self.create_img_groupbox()
        self.create_mainbut_groupbox()
        self.create_subbut_groupbox()
        self.create_thumbnail_groupbox()
        self.create_hist_groupbox()
        self.create_colormap_groupbox()
        self.create_fre_dft_groupbox()
        self.create_fre_dwt_groupbox()
        self.create_fre_hpf_groupbox()
        
        self.mainLayout = QGridLayout()
        self.mainLayout.addWidget(self.img_groupbox, 0, 0, 3, 1)
        self.mainLayout.setColumnStretch(0, 6)
        self.mainLayout.addWidget(self.mainbut_groupbox, 0, 1)
        self.mainLayout.addWidget(self.subbut_groupbox, 1, 1)
        self.mainLayout.setColumnStretch(1, 1)
        self.mainLayout.addWidget(self.thumbnail_groupbox, 0, 2, 1, 1)
        self.mainLayout.addWidget(self.hist_groupbox, 1, 2, 1, 1)
        self.mainLayout.addWidget(self.colormap_groupbox, 2, 2, 1, 1)
        self.mainLayout.setColumnStretch(2, 2)
        self.mainLayout.addWidget(self.fre_hpf_groupbox, 0, 3, 1, 1)
        self.mainLayout.addWidget(self.fre_dwt_groupbox, 1, 3, 1, 1)
        self.mainLayout.addWidget(self.fre_dft_groupbox, 2, 3, 1, 1)
        self.mainLayout.setColumnStretch(3, 2)
        
        
        self.setAcceptDrops(True)
        
        self.setGeometry(0, 0, 1600, 900)
        self.setLayout(self.mainLayout)
        self.setWindowTitle('img toolbox')
        self.move(100, 100)
        self.setStyleSheet('background-color: grey')
        
        ## callbacks
        self.resized.connect(self.refresh_img)
        
        ## class varibles
        self.path_img = r'example.jpg'
        self.roi_loc_x, self.roi_loc_y = 0, 0
        self.roi_bound_x, self.roi_bound_y = 0, 0
        self.roi_size = 100
        
        self.image = []
        self.image_roi = []
        self.image_roi_cal = []
        self.fig_hist = []
        self.fig_cc = []
        self.fig_fre_dft = []
        self.fig_fre_hpf = []
        self.fig_fre_dwt = []
        
        ## roi results
        self.roi_results_basic_sRGB = []
        self.roi_results_basic_Lab = []
        self.roi_results_basic_LCH = []
        
        self.roi_results_fre_hpf_sobel_abs_ave = []
        self.roi_results_fre_dwt_haar_dig_5 = []
        self.roi_results_fre_dft_psd = []
        self.roi_results_fre_dft_fre = []
        
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
        
    def create_fre_dft_groupbox(self):
        self.fre_dft_groupbox = QScrollArea()
        self.imagelabel_fre_dft = QLabel('roi fre dft')
        self.fre_dft_groupbox.setWidget(self.imagelabel_fre_dft) 
    
    def create_fre_dwt_groupbox(self):
        self.fre_dwt_groupbox = QScrollArea()
        self.imagelabel_fre_dwt = QLabel('roi fre dwt')
        self.fre_dwt_groupbox.setWidget(self.imagelabel_fre_dwt) 
    
    def create_fre_hpf_groupbox(self):
        self.fre_hpf_groupbox = QScrollArea()
        self.imagelabel_fre_hpf = QLabel('roi fre hpf')
        self.fre_hpf_groupbox.setWidget(self.imagelabel_fre_hpf) 
    
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
        PushButton_crop_cal = QPushButton('cal roi')
        PushButton_crop_homo = QPushButton('crop roi homo')
        PushButton_crop_dump = QPushButton('dump roi results')
        
        layout = QVBoxLayout()
        layout.setSpacing(5)
        layout.addWidget(PushButton_crop_cal)
        layout.addWidget(PushButton_crop_homo)
        layout.addWidget(PushButton_crop_dump)
        layout.addStretch(1)
        self.subbut_groupbox.setLayout(layout)
        
        ## connect
        PushButton_crop_cal.clicked.connect(self.update_roi)
        PushButton_crop_cal.clicked.connect(self.plot_hist)
        PushButton_crop_cal.clicked.connect(self.drawFigure_ellipse)
        PushButton_crop_cal.clicked.connect(self.fre_hpf_analysis)
        PushButton_crop_cal.clicked.connect(self.fre_dwt_analysis)
        PushButton_crop_cal.clicked.connect(self.fre_dft_analysis)
        
        PushButton_crop_dump.clicked.connect(self.dump_roi_results)
        
        
    """ calback events """
    """                """
    def resizeEvent(self, event):
        self.resized.emit()
        return super(GUI, self).resizeEvent(event)
        
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
            
            self.image = cv2.imread(self.path_img)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
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
        scroll_angle = QMouseEvent.angleDelta()
        #print('mouse scroll angle ', angle)
        if scroll_angle.y() > 0:
            self.roi_size = self.roi_size * 1.1
        else:
            self.roi_size = self.roi_size * 0.9
        self.refresh_img()
        
        
    """ subfunctions """
    """              """
    def update_roi(self):
        try: 
            self.image_roi_cal = self.image[int(self.roi_loc_y * self.image_roi.shape[0]):\
                                        int(self.roi_loc_y * self.image_roi.shape[0] + self.roi_size), \
                                        int(self.roi_loc_x * self.image_roi.shape[1]):\
                                        int(self.roi_loc_x * self.image_roi.shape[1] + self.roi_size), \
                                        :]
        except:
            pass
    
    def numpy_to_qt_img(self, mat):
        image = QImage(mat, mat.shape[1], mat.shape[0], mat.shape[1] * 3, QImage.Format_RGB888)
        pixmap = QPixmap(image)    
        return pixmap
        
    def select_img(self):
        self.path_img = QFileDialog.getOpenFileName(self, 'Open a file', '', 'All Files (*.*)')[0]
        ## update self image
        self.image = cv2.imread(self.path_img)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.refresh_img()
        
    def refresh_img(self):
        ## convert cv2 mat to qr pixelmap to bypass QPixmap::scaled: Pixmap is a null pixmap
        if len(self.image) == 0:
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
            #print(self.image_cropped.shape)                                
            #cv2.imshow('img ',self.image_cropped)
            #cv2.waitKey()
            #print('before crop', self.image_cropped.shape[1], self.image_cropped.shape[0])
            self.image_cropped_ = QImage(self.image_cropped.data.tobytes(), self.image_cropped.shape[1], self.image_cropped.shape[0], \
                                self.image_cropped.shape[1] * 3, QImage.Format_RGB888)
            #print('after crop')
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
        
        if len(self.fig_hist) != 0:
            #print('in hist loop')
            image = QImage(self.fig_hist, self.fig_hist.shape[1], self.fig_hist.shape[0], self.fig_hist.shape[1] * 3, QImage.Format_RGB888)
            pixmap = QPixmap(image)
            self.imagelabel_hist.setPixmap(pixmap)
            self.imagelabel_hist.setScaledContents(True)
            self.imagelabel_hist.resize(\
                self.hist_groupbox.frameGeometry().width() * 0.9, \
                self.hist_groupbox.frameGeometry().width() * 0.9 / 4 * 3 )

        if len(self.fig_cc) != 0:
            #print('in cc loop')
            image = QImage(self.fig_cc, self.fig_cc.shape[1], self.fig_cc.shape[0], self.fig_cc.shape[1] * 3, QImage.Format_RGB888)
            pixmap = QPixmap(image)
            self.imagelabel_colormap.setPixmap(pixmap)
            self.imagelabel_colormap.setScaledContents(True)
            self.imagelabel_colormap.resize(\
                self.colormap_groupbox.frameGeometry().width() * 0.9, \
                self.colormap_groupbox.frameGeometry().width() * 0.9 / 4 * 3 )
                
                
        if len(self.fig_fre_dft) != 0:
            #print('in cc loop')
            image = QImage(self.fig_fre_dft, self.fig_fre_dft.shape[1], self.fig_fre_dft.shape[0], self.fig_fre_dft.shape[1] * 3, QImage.Format_RGB888)
            pixmap = QPixmap(image)
            self.imagelabel_fre_dft.setPixmap(pixmap)
            self.imagelabel_fre_dft.setScaledContents(True)
            self.imagelabel_fre_dft.resize(\
                self.fre_dft_groupbox.frameGeometry().width() * 0.9, \
                self.fre_dft_groupbox.frameGeometry().width() * 0.9 / 4 * 3 ) 
                
        if len(self.fig_fre_dwt) != 0:
            #print('in cc loop')
            image = QImage(self.fig_fre_dwt, self.fig_fre_dwt.shape[1], self.fig_fre_dwt.shape[0], self.fig_fre_dwt.shape[1] * 3, QImage.Format_RGB888)
            pixmap = QPixmap(image)
            self.imagelabel_fre_dwt.setPixmap(pixmap)
            self.imagelabel_fre_dwt.setScaledContents(True)
            self.imagelabel_fre_dwt.resize(\
                self.fre_dwt_groupbox.frameGeometry().width() * 0.9, \
                self.fre_dwt_groupbox.frameGeometry().width() * 0.9 / 4 * 3 )        
                
        if len(self.fig_fre_hpf) != 0:
            #print('in cc loop')
            image = QImage(self.fig_fre_hpf, self.fig_fre_hpf.shape[1], self.fig_fre_hpf.shape[0], self.fig_fre_hpf.shape[1] * 1, QImage.Format_Indexed8)
            pixmap = QPixmap(image)
            self.imagelabel_fre_hpf.setPixmap(pixmap)
            self.imagelabel_fre_hpf.setScaledContents(True)
            self.imagelabel_fre_hpf.resize(\
                self.fre_hpf_groupbox.frameGeometry().width() * 0.9, \
                self.fre_hpf_groupbox.frameGeometry().width() * 0.9 / 4 * 3 ) 
        

    def plot_hist(self):
        bg_gray_level = 0.5
        ##
        ## image = cv2.cvtColor(self.image_roi_cal, cv2.COLOR_BGR2RGB)
        ## plot histogram of ROIs
        #fig = plt.figure()
        #fig.set_facecolor([bg_gray_level, bg_gray_level, bg_gray_level])
        #n, bins, patches = plt.hist(image[:, 0], 50, density = True, facecolor = 'b', alpha = 0.9, label = 'blue')
        #n, bins, patches = plt.hist(image[:, 1], 50, density = True, facecolor = 'g', alpha = 0.9, label = 'green')
        #n, bins, patches = plt.hist(image[:, 2], 50, density = True, facecolor = 'r', alpha = 0.9, label = 'red')
        #plt.xlabel('pixel value')
        #plt.ylabel('Probability')
        #plt.title('Hist of img')
        #plt.xlim(0, 255)
        #plt.ylim(0, 0.2)
        #plt.legend(loc = 'upper right')
        #ax = plt.axes()
        #ax.set_facecolor([bg_gray_level, bg_gray_level, bg_gray_level])
        
        ##
        fig, axs = plt.subplots(3, figsize=(4,3))
        fig.set_facecolor([bg_gray_level, bg_gray_level, bg_gray_level])
        
        n, bins, patches = axs[0].hist(self.image_roi_cal[:, :, 0].flatten(), 50, density = True, facecolor = 'r', alpha = 0.9, label = 'red')
        n, bins, patches = axs[1].hist(self.image_roi_cal[:, :, 1].flatten(), 50, density = True, facecolor = 'g', alpha = 0.9, label = 'green')
        n, bins, patches = axs[2].hist(self.image_roi_cal[:, :, 2].flatten(), 50, density = True, facecolor = 'b', alpha = 0.9, label = 'blue')
        axs[0].set_xlabel('pixel value')
        axs[0].set_ylabel('Probability')
        axs[0].set_title('Hist of img')
        axs[0].set_xlim(0, 255)
        axs[0].set_ylim(0, 0.2)
        axs[0].set_facecolor([bg_gray_level, bg_gray_level, bg_gray_level])
        
        axs[1].set_xlabel('pixel value')
        axs[1].set_ylabel('Probability')
        #axs[1].set_title('Hist of img')
        axs[1].set_xlim(0, 255)
        axs[1].set_ylim(0, 0.2)
        axs[1].set_facecolor([bg_gray_level, bg_gray_level, bg_gray_level])
        
        axs[2].set_xlabel('pixel value')
        axs[2].set_ylabel('Probability')
        #axs[2].set_title('Hist of img')
        axs[2].set_xlim(0, 255)
        axs[2].set_ylim(0, 0.2)        
        axs[2].set_facecolor([bg_gray_level, bg_gray_level, bg_gray_level])
        
        fig.subplots_adjust(hspace=0.5)
        #plt.legend(loc = 'upper right')
        #ax = plt.axes()
        #ax.set_facecolor([bg_gray_level, bg_gray_level, bg_gray_level])
        
        
        ## dump matplotlib figure to numpy array
        io_buf = io.BytesIO()
        plt.savefig(io_buf, format='png', dpi = 300, bbox_inches='tight')
        io_buf.seek(0)
        fig_hist = np.frombuffer(io_buf.getvalue(), dtype = np.uint8)
        fig_hist = cv2.imdecode(fig_hist, 1)
        fig_hist = cv2.cvtColor(fig_hist, cv2.COLOR_BGR2RGB)
        self.fig_hist = fig_hist       
        io_buf.close()
        plt.close()
        
        ## refresh
        self.refresh_img()
       
    
    def drawFigure_ellipse(self):
        """
        draw fiugre of ellipse in Lab space
        """
        bg_gray_level = 0.5
        range_i = 100
        L_fix = 75
        #color_scale_ratio = 2
        map_Lab = np.ones((range_i * 2, range_i * 2, 3), dtype = np.float32)
        map_Lab[:, :, 0] = L_fix * np.ones((range_i * 2, range_i * 2))
        map_Lab[:, :, 2], map_Lab[:, :, 1] = np.mgrid[-range_i:range_i, -range_i:range_i]
        map_RGB = cv2.cvtColor(map_Lab, cv2.COLOR_Lab2RGB)
        map_RGB = np.flip(map_RGB, 0)
        #map_RGB = cv2.cvtColor(map_RGB, cv2.COLOR_RGB2BGR)
        
        ##
        image = cv2.cvtColor(self.image_roi_cal, cv2.COLOR_BGR2RGB)
        roi_sRGB_mean = np.array([self.image_roi_cal[:, :, 0].mean(), \
                         self.image_roi_cal[:, :, 1].mean(), \
                         self.image_roi_cal[:, :, 2].mean()] )
                         
        print('roi_sRGB_mean: ', roi_sRGB_mean)
        roi_Lab_mean = cu.sRGB2Lab(roi_sRGB_mean)
        roi_LCH_mean = cu.Lab2LCh(roi_Lab_mean)
        
        print('roi_Lab_mean: ', roi_Lab_mean)
        
        fig = plt.figure(dpi = 300, figsize=(4,3))
        fig.set_facecolor([bg_gray_level, bg_gray_level, bg_gray_level])
        plt.imshow(map_RGB, extent = [-range_i, range_i, -range_i, range_i])
        
        ## mfc color should within 0~1
        plt.plot([], lw = 0, marker = 'o', ms = 4, mec = 'k', mfc = roi_sRGB_mean/256, label = 'Measured')
        plt.plot(roi_Lab_mean[1], roi_Lab_mean[2], marker = 'o', mec = 'k', mfc = roi_sRGB_mean/256, ms = 4.5)
         
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
        plt.savefig(io_buf, format = 'png', dpi = 300, bbox_inches = 'tight')
        io_buf.seek(0)
        fig_cc = np.frombuffer(io_buf.getvalue(), dtype = np.uint8)
        fig_cc = cv2.imdecode(fig_cc, 1)  
        fig_cc = cv2.cvtColor(fig_cc, cv2.COLOR_BGR2RGB)
        self.fig_cc = fig_cc       
        io_buf.close()
        plt.close()
        
        
        ##
        self.roi_results_basic_sRGB = roi_sRGB_mean
        self.roi_results_basic_Lab = roi_Lab_mean
        self.roi_results_basic_LCH = roi_LCH_mean
        
        
        ## refresh
        self.refresh_img()
        #plt.savefig(os.path.join(img_path, img_name))
        #plt.close(fig)
    
    def fre_dft_analysis(self):
    
        img_measured_gray = utility_img_processing.color2gray(self.image_roi_cal)
        ## crop image to square
        img_measured_gray = utility_img_processing.img_crop2square(img_measured_gray)
        ## apply a window before to minimize the miss alignment
        img_measured_gray = utility_img_processing.img_apply_weighted_mask(img_measured_gray)
        img_measured_dft = utility_signal_processing.transform_dft2(img_measured_gray, 'forward')
        psd_mea, frequency_mea = utility_signal_processing.cal_auto_power_spectral_density(img_measured_dft)
        
        start_index = 1
        np.seterr(divide = 'ignore', invalid = 'ignore')
        MTF_raw_vect = np.sqrt((psd_mea[start_index::]) / np.max(psd_mea[start_index::]))
        #MTF_raw_vect = (psd_mea[start_index::])
        
        bg_gray_level = 0.5
        fig = plt.figure(dpi = 300, figsize=(4,3))
        fig.set_facecolor([bg_gray_level, bg_gray_level, bg_gray_level])        
        plt.plot(frequency_mea[start_index::], MTF_raw_vect, 'r-', label = 'raw')
        plt.plot([0, 0.5], [1, 1], 'g-', alpha = 0.9, lw='1', label = 'ideal MTF')
        plt.ylim([-0.2, 2])
        plt.title(' normalized power spectrum density ' +'\n', wrap = True)
        plt.xlabel(' frequency ')
        plt.ylabel(' MTF direct method')
        ax = plt.axes()
        ax.set_facecolor([bg_gray_level, bg_gray_level, bg_gray_level])
        #plt.show()
    
        ## dump matplotlib figure to numpy array
        io_buf = io.BytesIO()
        plt.savefig(io_buf, format = 'png', dpi = 300, bbox_inches='tight')
        io_buf.seek(0)
        fig_fre_dft = np.frombuffer(io_buf.getvalue(), dtype = np.uint8)
        fig_fre_dft = cv2.imdecode(fig_fre_dft, 1)  
        fig_fre_dft = cv2.cvtColor(fig_fre_dft, cv2.COLOR_BGR2RGB)
        self.fig_fre_dft = fig_fre_dft       
        io_buf.close()
        plt.close()
        
        
        self.roi_results_fre_dft_psd = MTF_raw_vect
        self.roi_results_fre_dft_fre = frequency_mea
        ## refresh
        self.refresh_img()
    
    
    def fre_dwt_analysis(self):
        
        original_gray = cv2.cvtColor(self.image_roi_cal, cv2.COLOR_RGB2GRAY)
        
        ## DWT 
        dwt_levels = 5
        dwt_coeff_d = []
        for level in range(dwt_levels):
            if level == 0:
                coeffs = pywt.dwt2(original_gray, 'haar')
                cA, (cH, cV, cD) = coeffs
            else:
                coeffs = pywt.dwt2(cA / 2, 'haar')
                cA, (cH, cV, cD) = coeffs
            print(level)
            
            dwt_coeff_d.append(np.mean(np.abs(cD)))
            
            
        print('dwt_coeff_d ', dwt_coeff_d) 
        
        ##
        bg_gray_level = 0.5
        fig = plt.figure(dpi = 300, figsize=(4,3))
        fig.set_facecolor([bg_gray_level, bg_gray_level, bg_gray_level])        
        plt.plot(dwt_coeff_d, 'r-', label = 'dwt digonal ')
        plt.xlabel(' dwt levels ')
        plt.ylabel(' dwt coeff abs ave')
        plt.ylim([-1, 20])
        plt.title('dwt analysis')
        ax = plt.axes()
        ax.set_facecolor([bg_gray_level, bg_gray_level, bg_gray_level])
        
        ## dump matplotlib figure to numpy array
        io_buf = io.BytesIO()
        plt.savefig(io_buf, format = 'png', dpi = 300, bbox_inches='tight')
        io_buf.seek(0)
        fig_fre_dwt = np.frombuffer(io_buf.getvalue(), dtype = np.uint8)
        fig_fre_dwt = cv2.imdecode(fig_fre_dwt, 1)  
        fig_fre_dwt = cv2.cvtColor(fig_fre_dwt, cv2.COLOR_BGR2RGB)
        self.fig_fre_dwt = fig_fre_dwt       
        io_buf.close()
        plt.close()
        
        
        self.roi_results_fre_dwt_haar_dig_5 = dwt_coeff_d
        ## refresh
        self.refresh_img()
    
    def fre_hpf_analysis(self):
    
        ##
        gray = cv2.cvtColor(self.image_roi_cal, cv2.COLOR_RGB2GRAY)
        dx = cv2.Sobel(gray, cv2.CV_32F, 1,0, ksize = 3, scale=1)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0,1, ksize = 3, scale=1)
        absx= cv2.convertScaleAbs(dx)
        absy = cv2.convertScaleAbs(dy)
        edge = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
        
        ## put derivative sum to img
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (int(edge.shape[1]/10), int(edge.shape[0]/4*3))
        # fontScale
        fontScale = 0.2 * (self.roi_size / 50)
        # Blue color in BGR
        color = (255)
        # Line thickness of 2 px
        thickness = 1
        # Using cv2.putText() method
        edge = cv2.putText(edge, 'hpf_mu:' "{:.2f}".format(np.mean(edge)), org, font, fontScale, color, thickness, cv2.LINE_AA)
        
        self.fig_fre_hpf = edge       
        
        ## dump matplotlib figure to numpy array
        #io_buf = io.BytesIO()
        #plt.savefig(io_buf, format = 'png', dpi = 300, bbox_inches='tight')
        #io_buf.seek(0)
        #fig_fre_dft = np.frombuffer(io_buf.getvalue(), dtype = np.uint8)
        #fig_fre_dft = cv2.imdecode(fig_fre_dft, 1)  
        #fig_fre_dft = cv2.cvtColor(fig_fre_dft, cv2.COLOR_BGR2RGB)
        #self.fig_fre_dft = fig_fre_dft       
        #io_buf.close()
        #plt.close()
        
        
        self.roi_results_fre_hpf_sobel_abs_ave = np.mean(edge)
        ## refresh
        self.refresh_img()
    
    
    def dump_roi_results(self):
        
        roi_results_dict = {}
        roi_results_dict['img_name'] = 'example.png'
        roi_results_dict['timestamp'] = '2021 11 15'
        
        roi_results_dict['roi_results'] = {}
        
        roi_results_dict['roi_results']['basic_sRGB'] = self.roi_results_basic_sRGB.flatten().tolist()
        roi_results_dict['roi_results']['basic_Lab'] = self.roi_results_basic_Lab.flatten().tolist()
        roi_results_dict['roi_results']['basic_LCH'] = self.roi_results_basic_LCH.flatten().tolist()
        
        roi_results_dict['roi_results']['fre_hpf_sobel_abs_ave'] = self.roi_results_fre_hpf_sobel_abs_ave
        roi_results_dict['roi_results']['fre_dwt_haar_dig_5'] = self.roi_results_fre_dwt_haar_dig_5
        ## convert nd arrays to 1d
        roi_results_dict['roi_results']['fre_dft_psd'] = self.roi_results_fre_dft_psd.flatten().tolist()
        roi_results_dict['roi_results']['fre_dft_fre'] = self.roi_results_fre_dft_fre.flatten().tolist()
        
        with open("result.json", "w") as outfile:
            #roi_results_dict = json.dumps(roi_results_dict)
            json.dump(roi_results_dict, outfile, indent = 4)
    
    def close_GUI(self): 
        sys.exit()    
    



if __name__ == '__main__':
     
    app = QApplication(sys.argv)
    ui = GUI()
    ui.show()
    sys.exit(app.exec_()) 
    
    
    
    