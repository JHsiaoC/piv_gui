# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 02:32:04 2023

@author: jhsia
"""
#%% v2.0, creates an PY file that will host the designed GUI class

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

#%% v1.1, creates a PY file that can be run to launch the GUI without backend

# # set up the input and output files (transform from UI to PY file)
# import os

# INPUT = 'C:/Users/jhsia/Documents/GitHub/piv_gui/input.ui'
# OUTPUT = 'C:/Users/jhsia/Documents/GitHub/piv_gui/output.py'

# # If file exists, delete it.
# if os.path.isfile(OUTPUT):
#     os.remove(OUTPUT)
# else:
#     # If it fails, inform the user.
#     print("Error: %s file not found" % OUTPUT)

# cmd = 'pyuic5 -o ' + OUTPUT + ' ' + INPUT
# os.system(cmd)

# # Adds the needed lines to operate the UI as a PY file
# with open(OUTPUT, 'a+') as file:
#     file.write("\n")
#     file.write("def main():\n")
#     file.write("    if not QtWidgets.QApplication.instance():\n")
#     file.write("        app = QtWidgets.QApplication(sys.argv)\n")
#     file.write("    else:\n")
#     file.write("        app = QtWidgets.QApplication.instance()\n")
#     file.write("    main = QtWidgets.QMainWindow()\n")
#     file.write("    ui = Ui_MainWindow()\n")
#     file.write("    ui.setupUi(main)\n")
#     file.write("    main.show()\n")
#     file.write("    return main\n")
#     file.write("\n")
#     file.write("if __name__ == \"__main__\":\n")
#     file.write("    import sys\n")
#     file.write("    m = main()")
    
# # Runs the output file
# with open(OUTPUT) as f:
#     exec(f.read())

# # Place this in the new file if above doesn't work
# """
# def main():
#     if not QtWidgets.QApplication.instance():
#         app = QtWidgets.QApplication(sys.argv)
#     else:
#         app = QtWidgets.QApplication.instance()
#     main = QtWidgets.QMainWindow()
#     ui = Ui_MainWindow()
#     ui.setupUi(main)
#     main.show()
#     return main

# if __name__ == '__main__':
#     import sys
#     m = main()
# """

#%% v1.0, do not use (KERNEL WILL FREEZE ON CLOSE OF GUI)

# # set up the input and output files (transform from UI to PY file)
# import os

# INPUT = 'C:/Users/jhsia/Documents/GitHub/piv_gui/input.ui'
# OUTPUT = 'C:/Users/jhsia/Documents/GitHub/piv_gui/output.py'

# # If file exists, delete it.
# if os.path.isfile(OUTPUT):
#     os.remove(OUTPUT)
# else:
#     # If it fails, inform the user.
#     print("Error: %s file not found" % OUTPUT)

# cmd = 'pyuic5 -o ' + OUTPUT + ' ' + INPUT
# os.system(cmd)

# # Adds the needed lines to operate the UI as a PY file
# with open(OUTPUT, 'a+') as file:
#     file.write("\n")
#     file.write("if __name__ == \"__main__\":\n")
#     file.write("    import sys\n")
#     file.write("    app = QtWidgets.QApplication(sys.argv)\n")
#     file.write("    w = QtWidgets.QMainWindow()\n")
#     file.write("    ui = Ui_MainWindow()\n")
#     file.write("    ui.setupUi(w)\n")
#     file.write("    w.show()\n")
#     file.write("    sys.exit(app.exec_())")

# # Place this in the new file if above doesn't work
# """
# if __name__ == "__main__":
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     w = QtWidgets.QMainWindow()
#     ui = Ui_MainWindow()
#     ui.setupUi(w)
#     w.show()
#     sys.exit(app.exec_())
# """