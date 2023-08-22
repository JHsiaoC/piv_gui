# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 02:32:04 2023

@author: jhsia
"""

import os
os.system('pyuic5 -o output.py input.ui')

#%% Place this in the new file

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(w)
    w.show()
    sys.exit(app.exec_())