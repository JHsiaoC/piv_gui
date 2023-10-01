# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:/Users/jhsia/Documents/GitHub/piv_gui/design.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(350, 570)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(350, 570))
        MainWindow.setMaximumSize(QtCore.QSize(350, 570))
        MainWindow.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_generate = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_generate.setGeometry(QtCore.QRect(20, 520, 151, 21))
        self.pushButton_generate.setObjectName("pushButton_generate")
        self.groupBox_particleGen = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_particleGen.setEnabled(True)
        self.groupBox_particleGen.setGeometry(QtCore.QRect(10, 70, 331, 191))
        font = QtGui.QFont()
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        font.setKerning(True)
        self.groupBox_particleGen.setFont(font)
        self.groupBox_particleGen.setAutoFillBackground(False)
        self.groupBox_particleGen.setFlat(False)
        self.groupBox_particleGen.setCheckable(False)
        self.groupBox_particleGen.setObjectName("groupBox_particleGen")
        self.checkBox_randomSeed = QtWidgets.QCheckBox(self.groupBox_particleGen)
        self.checkBox_randomSeed.setGeometry(QtCore.QRect(10, 130, 221, 21))
        self.checkBox_randomSeed.setChecked(True)
        self.checkBox_randomSeed.setObjectName("checkBox_randomSeed")
        self.formLayoutWidget_3 = QtWidgets.QWidget(self.groupBox_particleGen)
        self.formLayoutWidget_3.setGeometry(QtCore.QRect(10, 20, 311, 101))
        self.formLayoutWidget_3.setObjectName("formLayoutWidget_3")
        self.formLayout_particleGen = QtWidgets.QFormLayout(self.formLayoutWidget_3)
        self.formLayout_particleGen.setContentsMargins(0, 0, 0, 0)
        self.formLayout_particleGen.setObjectName("formLayout_particleGen")
        self.label_ppp = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_ppp.setObjectName("label_ppp")
        self.formLayout_particleGen.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_ppp)
        self.label_xdim = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_xdim.setObjectName("label_xdim")
        self.formLayout_particleGen.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_xdim)
        self.label_ydim = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_ydim.setObjectName("label_ydim")
        self.formLayout_particleGen.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_ydim)
        self.label_sigma = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_sigma.setObjectName("label_sigma")
        self.formLayout_particleGen.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_sigma)
        self.lineEdit_ppp = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.lineEdit_ppp.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.lineEdit_ppp.setObjectName("lineEdit_ppp")
        self.formLayout_particleGen.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_ppp)
        self.lineEdit_xdim = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.lineEdit_xdim.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.lineEdit_xdim.setObjectName("lineEdit_xdim")
        self.formLayout_particleGen.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_xdim)
        self.lineEdit_ydim = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.lineEdit_ydim.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.lineEdit_ydim.setObjectName("lineEdit_ydim")
        self.formLayout_particleGen.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.lineEdit_ydim)
        self.lineEdit_sigma = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.lineEdit_sigma.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.lineEdit_sigma.setObjectName("lineEdit_sigma")
        self.formLayout_particleGen.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lineEdit_sigma)
        self.formLayoutWidget_2 = QtWidgets.QWidget(self.groupBox_particleGen)
        self.formLayoutWidget_2.setGeometry(QtCore.QRect(10, 160, 311, 22))
        self.formLayoutWidget_2.setObjectName("formLayoutWidget_2")
        self.formLayout_seed = QtWidgets.QFormLayout(self.formLayoutWidget_2)
        self.formLayout_seed.setContentsMargins(0, 0, 0, 0)
        self.formLayout_seed.setObjectName("formLayout_seed")
        self.lineEdit_seed = QtWidgets.QLineEdit(self.formLayoutWidget_2)
        self.lineEdit_seed.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.lineEdit_seed.setObjectName("lineEdit_seed")
        self.formLayout_seed.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_seed)
        self.label_seed = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_seed.setObjectName("label_seed")
        self.formLayout_seed.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_seed)
        self.groupBox_flowType = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_flowType.setGeometry(QtCore.QRect(10, 10, 331, 51))
        self.groupBox_flowType.setObjectName("groupBox_flowType")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.groupBox_flowType)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 19, 311, 22))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_flowType = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_flowType.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_flowType.setObjectName("verticalLayout_flowType")
        self.comboBox_flowType = QtWidgets.QComboBox(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setItalic(False)
        self.comboBox_flowType.setFont(font)
        self.comboBox_flowType.setObjectName("comboBox_flowType")
        self.comboBox_flowType.addItem("")
        self.comboBox_flowType.addItem("")
        self.comboBox_flowType.addItem("")
        self.comboBox_flowType.addItem("")
        self.comboBox_flowType.addItem("")
        self.comboBox_flowType.addItem("")
        self.verticalLayout_flowType.addWidget(self.comboBox_flowType)
        self.pushButton_clear = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_clear.setGeometry(QtCore.QRect(260, 520, 71, 21))
        self.pushButton_clear.setObjectName("pushButton_clear")
        self.pushButton_save = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_save.setGeometry(QtCore.QRect(180, 520, 71, 21))
        self.pushButton_save.setObjectName("pushButton_save")
        self.groupBox_flowGen = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_flowGen.setGeometry(QtCore.QRect(10, 270, 331, 211))
        self.groupBox_flowGen.setCheckable(False)
        self.groupBox_flowGen.setObjectName("groupBox_flowGen")
        self.formLayoutWidget = QtWidgets.QWidget(self.groupBox_flowGen)
        self.formLayoutWidget.setGeometry(QtCore.QRect(10, 20, 311, 181))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout_flowGen = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout_flowGen.setContentsMargins(0, 0, 0, 0)
        self.formLayout_flowGen.setObjectName("formLayout_flowGen")
        self.label_Gamma = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_Gamma.setObjectName("label_Gamma")
        self.formLayout_flowGen.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_Gamma)
        self.lineEdit_Gamma = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit_Gamma.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.lineEdit_Gamma.setObjectName("lineEdit_Gamma")
        self.formLayout_flowGen.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.lineEdit_Gamma)
        self.label_nu = QtWidgets.QLabel(self.formLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_nu.sizePolicy().hasHeightForWidth())
        self.label_nu.setSizePolicy(sizePolicy)
        self.label_nu.setObjectName("label_nu")
        self.formLayout_flowGen.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_nu)
        self.lineEdit_nu = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit_nu.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.lineEdit_nu.setObjectName("lineEdit_nu")
        self.formLayout_flowGen.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lineEdit_nu)
        self.label_omega = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_omega.setObjectName("label_omega")
        self.formLayout_flowGen.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_omega)
        self.lineEdit_omega = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit_omega.setInputMethodHints(QtCore.Qt.ImhDigitsOnly|QtCore.Qt.ImhLatinOnly)
        self.lineEdit_omega.setObjectName("lineEdit_omega")
        self.formLayout_flowGen.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.lineEdit_omega)
        self.label_centerX = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_centerX.setEnabled(True)
        self.label_centerX.setObjectName("label_centerX")
        self.formLayout_flowGen.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_centerX)
        self.lineEdit_centerX = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit_centerX.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.lineEdit_centerX.setObjectName("lineEdit_centerX")
        self.formLayout_flowGen.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.lineEdit_centerX)
        self.label_centerY = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_centerY.setObjectName("label_centerY")
        self.formLayout_flowGen.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_centerY)
        self.lineEdit_centerY = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit_centerY.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.lineEdit_centerY.setObjectName("lineEdit_centerY")
        self.formLayout_flowGen.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.lineEdit_centerY)
        self.label_Vmax = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_Vmax.setObjectName("label_Vmax")
        self.formLayout_flowGen.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_Vmax)
        self.lineEdit_Vmax = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit_Vmax.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.lineEdit_Vmax.setObjectName("lineEdit_Vmax")
        self.formLayout_flowGen.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_Vmax)
        self.label_timestep = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_timestep.setObjectName("label_timestep")
        self.formLayout_flowGen.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_timestep)
        self.lineEdit_timestep = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit_timestep.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.lineEdit_timestep.setObjectName("lineEdit_timestep")
        self.formLayout_flowGen.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_timestep)
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(20, 490, 311, 20))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.checkBox_visualize = QtWidgets.QCheckBox(self.horizontalLayoutWidget)
        self.checkBox_visualize.setChecked(False)
        self.checkBox_visualize.setObjectName("checkBox_visualize")
        self.buttonGroup = QtWidgets.QButtonGroup(MainWindow)
        self.buttonGroup.setObjectName("buttonGroup")
        self.buttonGroup.setExclusive(False)
        self.buttonGroup.addButton(self.checkBox_visualize)
        self.horizontalLayout.addWidget(self.checkBox_visualize)
        self.label_OR = QtWidgets.QLabel(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_OR.sizePolicy().hasHeightForWidth())
        self.label_OR.setSizePolicy(sizePolicy)
        self.label_OR.setMinimumSize(QtCore.QSize(16, 0))
        self.label_OR.setMaximumSize(QtCore.QSize(16, 16777215))
        self.label_OR.setObjectName("label_OR")
        self.horizontalLayout.addWidget(self.label_OR)
        self.checkBox_plot = QtWidgets.QCheckBox(self.horizontalLayoutWidget)
        self.checkBox_plot.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.checkBox_plot.setChecked(True)
        self.checkBox_plot.setObjectName("checkBox_plot")
        self.buttonGroup.addButton(self.checkBox_plot)
        self.horizontalLayout.addWidget(self.checkBox_plot)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setEnabled(False)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 350, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.actionContact = QtWidgets.QAction(MainWindow)
        self.actionContact.setObjectName("actionContact")
        self.actionNew = QtWidgets.QAction(MainWindow)
        self.actionNew.setObjectName("actionNew")
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSave_As = QtWidgets.QAction(MainWindow)
        self.actionSave_As.setObjectName("actionSave_As")
        self.actionCtrl_S = QtWidgets.QAction(MainWindow)
        self.actionCtrl_S.setObjectName("actionCtrl_S")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.comboBox_flowType, self.lineEdit_ppp)
        MainWindow.setTabOrder(self.lineEdit_ppp, self.lineEdit_xdim)
        MainWindow.setTabOrder(self.lineEdit_xdim, self.lineEdit_ydim)
        MainWindow.setTabOrder(self.lineEdit_ydim, self.lineEdit_sigma)
        MainWindow.setTabOrder(self.lineEdit_sigma, self.checkBox_randomSeed)
        MainWindow.setTabOrder(self.checkBox_randomSeed, self.lineEdit_seed)
        MainWindow.setTabOrder(self.lineEdit_seed, self.lineEdit_Vmax)
        MainWindow.setTabOrder(self.lineEdit_Vmax, self.lineEdit_timestep)
        MainWindow.setTabOrder(self.lineEdit_timestep, self.lineEdit_Gamma)
        MainWindow.setTabOrder(self.lineEdit_Gamma, self.lineEdit_nu)
        MainWindow.setTabOrder(self.lineEdit_nu, self.lineEdit_omega)
        MainWindow.setTabOrder(self.lineEdit_omega, self.lineEdit_centerX)
        MainWindow.setTabOrder(self.lineEdit_centerX, self.lineEdit_centerY)
        MainWindow.setTabOrder(self.lineEdit_centerY, self.checkBox_visualize)
        MainWindow.setTabOrder(self.checkBox_visualize, self.checkBox_plot)
        MainWindow.setTabOrder(self.checkBox_plot, self.pushButton_generate)
        MainWindow.setTabOrder(self.pushButton_generate, self.pushButton_save)
        MainWindow.setTabOrder(self.pushButton_save, self.pushButton_clear)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Flow Visualization"))
        self.pushButton_generate.setText(_translate("MainWindow", "Generate Field"))
        self.groupBox_particleGen.setTitle(_translate("MainWindow", "Particle Generation"))
        self.checkBox_randomSeed.setText(_translate("MainWindow", "Randomize particle generation"))
        self.label_ppp.setText(_translate("MainWindow", "Particles per pixel (PPP)                 "))
        self.label_xdim.setText(_translate("MainWindow", "Horizontal window size [px]"))
        self.label_ydim.setText(_translate("MainWindow", "Vertical window size [px]"))
        self.label_sigma.setText(_translate("MainWindow", "Standard deviation(s)"))
        self.lineEdit_ppp.setText(_translate("MainWindow", "0.01"))
        self.lineEdit_xdim.setText(_translate("MainWindow", "256"))
        self.lineEdit_ydim.setText(_translate("MainWindow", "256"))
        self.lineEdit_sigma.setText(_translate("MainWindow", "0"))
        self.lineEdit_seed.setText(_translate("MainWindow", "0"))
        self.label_seed.setText(_translate("MainWindow", "Particle generation seed                "))
        self.groupBox_flowType.setTitle(_translate("MainWindow", "Flow Type"))
        self.comboBox_flowType.setItemText(0, _translate("MainWindow", "Uniform"))
        self.comboBox_flowType.setItemText(1, _translate("MainWindow", "Couette"))
        self.comboBox_flowType.setItemText(2, _translate("MainWindow", "Poiseuille"))
        self.comboBox_flowType.setItemText(3, _translate("MainWindow", "Lamb-Oseen"))
        self.comboBox_flowType.setItemText(4, _translate("MainWindow", "Rayleigh Problem"))
        self.comboBox_flowType.setItemText(5, _translate("MainWindow", "Stokes Problem"))
        self.pushButton_clear.setText(_translate("MainWindow", "Clear Data"))
        self.pushButton_save.setText(_translate("MainWindow", "Save Data"))
        self.groupBox_flowGen.setTitle(_translate("MainWindow", "Flow Generation"))
        self.label_Gamma.setText(_translate("MainWindow", "Circulation [px^2/s]"))
        self.lineEdit_Gamma.setText(_translate("MainWindow", "655360"))
        self.label_nu.setText(_translate("MainWindow", "Kinematic viscosity [px^2/s] "))
        self.lineEdit_nu.setText(_translate("MainWindow", "655.36"))
        self.label_omega.setText(_translate("MainWindow", "Oscillation frequency [cycles/sec]  "))
        self.lineEdit_omega.setText(_translate("MainWindow", "1"))
        self.label_centerX.setText(_translate("MainWindow", "Center X (px)"))
        self.lineEdit_centerX.setText(_translate("MainWindow", "128"))
        self.label_centerY.setText(_translate("MainWindow", "Center Y (px)"))
        self.lineEdit_centerY.setText(_translate("MainWindow", "128"))
        self.label_Vmax.setText(_translate("MainWindow", "Maximum particle velocity [px/sec]"))
        self.lineEdit_Vmax.setText(_translate("MainWindow", "4"))
        self.label_timestep.setText(_translate("MainWindow", "Sampling interval [sec]"))
        self.lineEdit_timestep.setText(_translate("MainWindow", "0.01"))
        self.checkBox_visualize.setText(_translate("MainWindow", "Visualize Field"))
        self.label_OR.setText(_translate("MainWindow", "OR"))
        self.checkBox_plot.setText(_translate("MainWindow", "Plot Field"))
        self.actionContact.setText(_translate("MainWindow", "Contact"))
        self.actionNew.setText(_translate("MainWindow", "New..."))
        self.actionOpen.setText(_translate("MainWindow", "Open..."))
        self.actionSave_As.setText(_translate("MainWindow", "Save As..."))
        self.actionCtrl_S.setText(_translate("MainWindow", "Ctrl + S"))
