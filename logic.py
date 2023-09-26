# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:16:57 2023

@author: jhsia
"""
import numpy as np
import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt

#%%

def pytgen(ppp, xdim = 256, ydim = 256, plot = False):
    number_pixels = int(xdim*ydim)  #counts number of pixels in the frame
    particles = int(number_pixels*ppp)  #counts number of particles that should be present
    x0 = xdim*torch.rand((particles,2),dtype=(torch.float32)) # make the "initial" data
    
    if plot == True:
        fig, ax = plt.subplots()
        ax.scatter(x0[:,0], x0[:,1], s = 3)
        ax.set_title('Pytorch')
    return x0

def npgen(ppp, xdim, ydim, plot = False):
    number_pixels = int(xdim*ydim)  #counts number of pixels in the frame
    particles = int(number_pixels*ppp)  #counts number of particles that should be present
    x0_num = xdim*np.random.rand(particles,2) # make the "initial" data
    x0 = torch.from_numpy(x0_num) #conversion from numpy to torch
    x0 = x0.type(torch.float32) #conversion from float64 to float32 (not sure why)
    
    if plot == True:
        fig, ax = plt.subplots()
        ax.scatter(x0[:,0], x0[:,1], s = 3)
        ax.set_title('X = '+ str(xdim) + ' Y = '+ str(ydim) + ' ppp = ' + str(ppp))
    return x0

#%% Flow Functions
"""
Nomenclature:
    INPUTS:
    x0 = initial vector of particle positions [pixels]
    Vmax = Maximum Velocity [pixels/second]
    timestep = time from initial to final positions [second]
    sigma = value for one standard deviation from mean (mean is assumed to be zero) [pixel]
    
    OUTPUTS:
    X = Final particle postion [pixels]
    V = Velocity vector [pixels/second]
    theta = matrix of velocity coefficient for the position vector
        (a.k.a. what number multiplied to position result in velocity)  
    phi = matrix of velocity coefficient for the position vector SQUARED
        (a.k.a. what number multiplied to position squared result in velocity)
"""

## Steady 2D flows
# Uniform flow
def uniform_flow(x0, Vmax, timestep, sigma = 0, plot = False):
    datapts = len(x0)
    
    # uniform flow field    
    V = torch.zeros((datapts,2))
    V[:,0] = Vmax # flow velocity in the x dir
    V[:,1] = 0    # flow velocity in the y dir

    X = x0 + V*timestep

    if plot == True:
        fig, ax = plt.subplots()
        ax.scatter(x0[:,0], x0[:, 1], s = 3)
        ax.scatter(X[:,0], X[:, 1], s = 3)
        ax.set_title('Uniform Flow')
    if sigma != 0:
        noise = sigma*torch.randn_like(X,dtype=torch.float64)
        X_noisy = X + noise
        fig, ax = plt.subplots()
        ax.scatter(x0[:, 0], x0[:, 1], s = 3)
        ax.scatter(X_noisy[:, 0], X_noisy[:, 1], s = 3)
        ax.set_title('Uniform Flow w/ Noise')
              
    # if plot == True:
    #     fig, ax = plt.subplots()
    #     ax.quiver(X[:,0], X[:,1], V[:,0], V[:,1])
    #     ax.set_title('Uniform Flow')
    # if  sigma > 0:
    #     noise = sigma*torch.randn_like(X,dtype=torch.float64)
    #     X_noisy = X + noise
    #     fig, ax = plt.subplots()
    #     ax.quiver(X_noisy[:,0], X_noisy[:,1], V[:,0], V[:,1])
    #     ax.set_title('Uniform Flow w/ Noise')
        
    return X,V

# General Couette flow
def couette_flow(x0, Vmax, timestep, ydim, sigma = 0, plot = False):
    datapts = len(x0)
    
    # Couette flow field
    theta = torch.tensor([[0,Vmax/ydim],[0,0]])

    # transpose init con for correct mat mul and then transpose back to vel
    V = torch.zeros((datapts,2))
    V = torch.matmul(theta,torch.transpose(x0,0,1))
    V = torch.transpose(V,0,1)

    X = x0 + V*timestep # generate new positions
    
    if plot == True:
        fig, ax = plt.subplots()
        ax.scatter(x0[:, 0], x0[:, 1], s = 3)
        ax.scatter(X[:, 0], X[:, 1], s = 3)
        ax.set_title('Couette Flow')
        
    if sigma != 0:
        noise = sigma*torch.randn_like(X,dtype=torch.float64)
        X_noisy = X + noise
        fig, ax = plt.subplots()
        ax.scatter(x0[:, 0], x0[:, 1], s = 3)
        ax.scatter(X_noisy[:, 0], X_noisy[:, 1], s = 3)
        ax.set_title('Couette Flow w/ Noise')
    
    # if plot == True:
    #     fig, ax = plt.subplots()
    #     ax.scatter(X[:,0], X[:,1], s = 3)
    #     ax.set_title('Couette Flow')
    #     fig, ax = plt.subplots()
    #     ax.quiver(X[:,0], X[:,1], V[:,0], V[:,1])
    #     ax.set_title('Couette Flow w/ Velocity Field')
    # if  sigma > 0:
    #     noise = sigma*torch.randn_like(X,dtype=torch.float64)
    #     X_noisy = X + noise
    #     fig, ax = plt.subplots()
    #     ax.scatter(X_noisy[:,0], X_noisy[:,1], s = 3)
    #     ax.set_title('Couette Flow w/ Noise')
    #     fig, ax = plt.subplots()
    #     ax.quiver(X[:,0], X[:,1], V[:,0], V[:,1])
    #     ax.set_title('Couette Flow w/ Velocity Field & Noise')
    return X,V,theta

# Plane Poiseuille flow
def poiseuille_flow(x0, Vmax, timestep, ydim, sigma = 0, plot = False):
    datapts = len(x0)
    
    # Poiseuille flow field
    R = ydim/2
    theta = torch.tensor([[0,2*Vmax/R],[0,0]])
    phi = torch.tensor([[0,-Vmax/R**2],[0,0]])
    
    # transpose init con for correct mat mul and then transpose back to vel
    V = torch.zeros((datapts,2))
    V = torch.matmul(theta,torch.transpose(x0,0,1)) + torch.matmul(phi,torch.transpose(x0**2,0,1))
    V = torch.transpose(V,0,1)
    
    X = x0 + V*timestep # generate new positions
    
    if plot == True:
        fig, ax = plt.subplots()
        ax.scatter(x0[:, 0], x0[:, 1], s = 3)
        ax.scatter(X[:, 0], X[:, 1], s = 3)
        ax.set_title('Poiseuille Flow')      

    if sigma > 0:
        noise = sigma*torch.randn_like(X,dtype=torch.float64)
        X_noisy = X + noise
        fig, ax = plt.subplots()
        ax.scatter(x0[:, 0], x0[:, 1], s = 3)
        ax.scatter(X_noisy[:, 0], X_noisy[:, 1], s = 3)
        ax.set_title('Poiseuille Flow w/ Noise') 
    
    # if plot == True:
    #     fig, ax = plt.subplots()
    #     ax.scatter(X[:,0], X[:,1], s = 3)
    #     ax.set_title('Poiseuille Flow')     
    #     fig, ax = plt.subplots()
    #     ax.quiver(X[:,0], X[:,1], V[:,0], V[:,1])
    #     ax.set_title('Poiseuille Flow w/ Velocity Field')
    # if sigma > 0:
    #     noise = sigma*torch.randn_like(X,dtype=torch.float64)
    #     X_noisy = X + noise
    #     fig, ax = plt.subplots()
    #     ax.scatter(X_noisy[:,0], X_noisy[:,1], s = 3)
    #     ax.set_title('Poiseuille Flow w/ Noise')
    #     fig, ax = plt.subplots()
    #     ax.quiver(X_noisy[:,0], X_noisy[:,1], V[:,0], V[:,1])
    #     ax.set_title('Poiseuille Flow w/ Velocity Field & Noise')
    return X,V,theta,phi

## "Steady" Unsteady 2D flows
# Lamb–Oseen Vortex at fixed time, assumes CCW circulation as positive
    # Gamma = circulation of vortex
    # nu = kinematic viscosity of working fluid
    # r = radial distance
def lamb_oseen_visualization(x0, Gamma, nu, t, center_X = 256/2, center_Y = 256/2):
    datapts = len(x0)
    center_X_vec = torch.full((datapts,),center_X).reshape(-1,1)
    center_Y_vec = torch.full((datapts,),center_Y).reshape(-1,1)
    x = x0[:,0].reshape(-1,1)-center_X_vec
    y = x0[:,1].reshape(-1,1)-center_Y_vec
    r = torch.empty((datapts,1),dtype=torch.float32).reshape(-1,1)
    for ii in range(datapts):
        r[ii,0] = torch.sqrt(x[ii,]**2+y[ii,]**2)
        
    radius = torch.linspace(0,torch.max(r),100)
        
    g = 1 - torch.exp(-radius**2/(4*torch.pi*nu*t))
    v_theta = Gamma/(2*torch.pi*radius)*g

    fig, ax = plt.subplots()
    ax.scatter(radius, v_theta, s = 3)
    ax.set_title('Lamb–Oseen Velocity Visualization @ %f s' %t)
    return

    
def lamb_oseen_vortex(x0, Gamma, nu, t, center_X = 256/2, center_Y = 256/2, sigma = 0, plot = False):
    datapts = len(x0)
    # vectors of location of vortex center w/ vector length of datapts
    center_X_vec = torch.full((datapts,),center_X).reshape(-1,1)
    center_Y_vec = torch.full((datapts,),center_Y).reshape(-1,1)

    def lamb_oseen_ODE(t,x0):
        x = x0[:,0].reshape(-1,1)-center_X_vec
        y = x0[:,1].reshape(-1,1)-center_Y_vec
        z_hat = torch.tensor([0,0,1],dtype=torch.float32).reshape(-1,3)
        
        r = torch.empty((datapts,1),dtype=torch.float32).reshape(-1,1)
        vel_dir = torch.empty((datapts,3),dtype=torch.float32).reshape(-1,3)
        for ii in range(datapts):
            r[ii,0] = torch.sqrt(x[ii,]**2+y[ii,]**2)
            radial_dir = torch.tensor([x[ii,],y[ii,],0]).reshape(-1,3)
            vel_dir[ii,:] = torch.linalg.cross(z_hat,radial_dir)
        vel_dir = vel_dir[:,0:2]/r # trim to 2D
            
        g = 1 - torch.exp(-r**2/(4*torch.pi*nu*t))
        v_theta = Gamma/(2*torch.pi*r)*g*vel_dir
        return v_theta
    
    time_LO = torch.linspace(0,t,100+int(t*25))
    history = odeint(lamb_oseen_ODE,x0,time_LO)
    X = history[-1,:,:]
    
    if plot == True:
        fig, ax = plt.subplots()
        ax.scatter(X[:,0], X[:,1], s = 3)
        ax.set_title('Lamb–Oseen Flow @ %f s' %t)
    if sigma > 0:
        noise = sigma*torch.randn_like(X,dtype=torch.float64)
        X_noisy = X + noise
        fig, ax = plt.subplots()
        ax.scatter(X_noisy[:,0], X_noisy[:,1], s = 3)
        ax.set_title('Lamb–Oseen Flow w/ Noise') 
        
    return X

# Rayleigh problem (or Stokes First Problem)
def rayleigh_problem(x0, Vmax, nu, t, sigma = 0, plot = False):
        
    def rayleigh_ODE(t,x0):
        y = x0[:,1].reshape(-1,1)
        x_vel = Vmax*torch.special.erfc(y/torch.sqrt(4*nu*t))
        y_vel = torch.zeros_like(x_vel)
        vel = torch.cat((x_vel,y_vel),1)
        return vel
    
    time_S1 = torch.linspace(0,t,100+int(t*25))
    history = odeint(rayleigh_ODE,x0,time_S1)
    X = history[-1,:,:]
    
    if plot == True:
        fig, ax = plt.subplots()
        ax.scatter(X[:,0], X[:,1], s = 3)
        ax.set_title('Stokes First Problem @ %f s' %t)
    if sigma > 0:
        noise = sigma*torch.randn_like(X,dtype=torch.float64)
        X_noisy = X + noise
        fig, ax = plt.subplots()
        ax.scatter(X_noisy[:,0], X_noisy[:,1], s = 3)
        ax.set_title('Stokes First Problem w/ Noise')
        
    # if plot == True:
    #     fig, ax = plt.subplots()
    #     ax.quiver(X[:,0], X[:,1], V[:,0], V[:,1])
    #     ax.set_title('Stokes First Problem @ %f s' %t)
    # if sigma > 0:
    #     noise = sigma*torch.randn_like(X,dtype=torch.float64)
    #     X_noisy = X + noise
    #     fig, ax = plt.subplots()
    #     ax.quiver(X_noisy[:,0], X_noisy[:,1], V[:,0], V[:,1])
    #     ax.set_title('Stokes First Problem w/ Noise')
    
    return X

# Stokes problem (or Stokes Second Problem)
    # moving plate's surface at y = 0
def stokes_problem(x0, Vmax, omega, nu, t, sigma = 0, plot = False):
    
    def stokes_ODE(t,x0):
        y = x0[:,1].reshape(-1,1)
        exponent = torch.exp(-omega*y/2/nu)
        trig_func = torch.cos(omega*t-torch.sqrt(omega/2/nu)*y)
        x_vel = Vmax*torch.mul(exponent,trig_func)
        y_vel = torch.zeros_like(x_vel)
        vel = torch.cat((x_vel,y_vel),1)
        return vel
    
    time_S2 = torch.linspace(0,t,100+int(t*25))
    history = odeint(stokes_ODE,x0,time_S2)
    X = history[-1,:,:]
    
    if plot == True:
        fig, ax = plt.subplots()
        ax.scatter(X[:,0], X[:,1], s = 3)
        ax.set_title('Stokes Second Problem @ %f s' %t)
    if sigma > 0:
        noise = sigma*torch.randn_like(X,dtype=torch.float64)
        X_noisy = X + noise
        fig, ax = plt.subplots()
        ax.scatter(X_noisy[:,0], X_noisy[:,1], s = 3)
        ax.set_title('Stokes Second Problem w/ Noise')
        
    # if plot == True:
    #     fig, ax = plt.subplots()
    #     ax.quiver(X[:,0], X[:,1], V[:,0], V[:,1])
    #     ax.set_title('Stokes Second Problem @ %f s' %t)
    # if sigma > 0:
    #     noise = sigma*torch.randn_like(X,dtype=torch.float64)
    #     X_noisy = X + noise
    #     fig, ax = plt.subplots()
    #     ax.quiver(X_noisy[:,0], X_noisy[:,1], V[:,0], V[:,1])
    #     ax.set_title('Stokes Second Problem w/ Noise')
    
    return X

#%%

from PyQt5 import QtWidgets
# from PyQt5 import QtCore, QtGui, QtWidgets
    # might need all three later on
from design import Ui_MainWindow

class Logic(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
    
    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        
        self.comboBox.currentTextChanged.connect(self.flow_selection)
        flow_name = self.comboBox.currentText()
        print(f'The flow type selected is {flow_name}. \n')
        
        self.pushButton.clicked.connect(self.generate_field)
        self.pushButton_2.clicked.connect(self.simulate_flows)
        self.pushButton_3.clicked.connect(self.save_data)
        self.pushButton_4.clicked.connect(self.clear_data)
        
        self.checkBox.toggled.connect(self.plot_field)
        plot = self.checkBox.isChecked()
        if plot == True:
            print('Plotting is enabled. \n')
        else:
            print('Plotting is disabled. \n')
        
    def flow_selection(self):
        flow_name = self.comboBox.currentText()
        print(f'The flow type selected is {flow_name}. \n')
        
    def generate_field(self):
        ppp = self.lineEdit.text()
        xdim = self.lineEdit_2.text()
        ydim = self.lineEdit_3.text()
        
        
        print(f'Particles per pixel = {ppp}')
        print(f'Window size is {xdim} x {ydim} pixels')
        print("\n Field Generated")

    def simulate_flows(self):
        print("Flow Simulated")  
        
    def save_data(self):
        print("Data Saved")
        
    def clear_data(self):
        print("Data Cleared")
    
    def plot_field(self):
        plot = self.checkBox.isChecked()
        if plot == True:
            print('Plotting enabled')
        else:
            print('Plotting disabled')
        
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