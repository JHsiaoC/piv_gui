# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 02:50:28 2023

@author: jhsia
"""

import os

INPUT = 'C:/Users/jhsia/Documents/GitHub/piv_gui/test_GUI.ui'
OUTPUT = 'C:/Users/jhsia/Documents/GitHub/piv_gui/test_GUI.py'

# If file exists, delete it.
if os.path.isfile(OUTPUT):
    os.remove(OUTPUT)
else:
    # If it fails, inform the user.
    print("Error: %s file not found" % OUTPUT)

cmd = 'pyuic5 -o ' + OUTPUT + ' ' + INPUT
os.system(cmd)