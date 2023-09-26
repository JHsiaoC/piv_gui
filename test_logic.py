# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 02:52:15 2023

@author: jhsia
"""
from PyQt5 import QtCore, QtGui, QtWidgets
from test_GUI import Ui_MainWindow

class Logic(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        self.pushButton.clicked.connect(self.the_buttonn_was_clicked) # type: ignore
        
    def the_buttonn_was_clicked(self):
        print("Clicked!")
        
def main():
    if not QtWidgets.QApplication.instance():
        _ = QtWidgets.QApplication(sys.argv)
    else:
        _ = QtWidgets.QApplication.instance()
    main = Logic()
    main.show()
    return main

if __name__ == '__main__':
    import sys
    m = main()
