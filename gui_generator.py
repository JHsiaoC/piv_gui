# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 02:32:04 2023

@author: jhsia
"""
#%% creates an PY file that will host the designed GUI class

# set up the input and output files (transform from UI to PY file)
import os

INPUT = 'C:/Users/jhsia/Documents/GitHub/piv_gui/design.ui'
OUTPUT = 'C:/Users/jhsia/Documents/GitHub/piv_gui/design.py'

# If file exists, delete it.
if os.path.isfile(OUTPUT):
    os.remove(OUTPUT)
else:
    # If it fails, inform the user.
    print("Error: %s file not found" % OUTPUT)

cmd = 'pyuic5 -o ' + OUTPUT + ' ' + INPUT
os.system(cmd)