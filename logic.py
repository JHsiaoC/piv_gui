# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:16:57 2023

@author: jhsia
"""
import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt

#%%

def pytgen(ppp, xdim = 256, ydim = 256, random = True, seed = 'foo'):
    number_pixels = int(xdim*ydim)  #counts number of pixels in the frame
    particles = int(number_pixels*ppp)  #counts number of particles that should be present
    if random == True:
        x0 = xdim*torch.rand((particles,2),dtype=(torch.float32)) # make the "initial" data
    if random == False:
        torch.manual_seed(seed)
        x0 = xdim*torch.rand((particles,2),dtype=(torch.float32)) # make the data with seeded value

    # if plot == True:
    #     fig, ax = plt.subplots()
    #     ax.scatter(x0[:,0], x0[:,1], s = 3)
    #     ax.set_title('Pytorch')
    return x0

# def npgen(ppp, xdim, ydim, plot = False):
#     number_pixels = int(xdim*ydim)  #counts number of pixels in the frame
#     particles = int(number_pixels*ppp)  #counts number of particles that should be present
#     x0_num = xdim*np.random.rand(particles,2) # make the "initial" data
#     x0 = torch.from_numpy(x0_num) #conversion from numpy to torch
#     x0 = x0.type(torch.float32) #conversion from float64 to float32 (not sure why)
    
#     if plot == True:
#         fig, ax = plt.subplots()
#         ax.scatter(x0[:,0], x0[:,1], s = 3)
#         ax.set_title('X = '+ str(xdim) + ' Y = '+ str(ydim) + ' ppp = ' + str(ppp))
#     return x0

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

    # if plot == True:
    #     fig, ax = plt.subplots()
    #     ax.scatter(x0[:,0], x0[:, 1], s = 3)
    #     ax.scatter(X[:,0], X[:, 1], s = 3)
    #     ax.set_aspect('equal', adjustable='box')
    #     ax.set_title('Uniform Flow')
    # if sigma != 0:
    #     noise = sigma*torch.randn_like(X,dtype=torch.float64)
    #     X_noisy = X + noise
    #     fig, ax = plt.subplots()
    #     ax.scatter(x0[:, 0], x0[:, 1], s = 3)
    #     ax.scatter(X_noisy[:, 0], X_noisy[:, 1], s = 3)
    #     ax.set_aspect('equal', adjustable='box')
    #     ax.set_title('Uniform Flow w/ Noise')
              
    if plot == True:
        fig, ax = plt.subplots()
        ax.quiver(X[:,0], X[:,1], V[:,0], V[:,1])
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Uniform Flow')
    if  sigma > 0:
        noise = sigma*torch.randn_like(X,dtype=torch.float64)
        X_noisy = X + noise
        
        fig, ax = plt.subplots()
        ax.quiver(X_noisy[:,0], X_noisy[:,1], V[:,0], V[:,1])
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Uniform Flow w/ Noise')
        
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
    
    # if plot == True:
    #     fig, ax = plt.subplots()
    #     ax.scatter(x0[:, 0], x0[:, 1], s = 3)
    #     ax.scatter(X[:, 0], X[:, 1], s = 3)
    #     ax.set_aspect('equal', adjustable='box')
    #     ax.set_title('Couette Flow')
        
    # if sigma != 0:
    #     noise = sigma*torch.randn_like(X,dtype=torch.float64)
    #     X_noisy = X + noise
    #     fig, ax = plt.subplots()
    #     ax.scatter(x0[:, 0], x0[:, 1], s = 3)
    #     ax.scatter(X_noisy[:, 0], X_noisy[:, 1], s = 3)
    #     ax.set_aspect('equal', adjustable='box')
    #     ax.set_title('Couette Flow w/ Noise')
    
    if plot == True:
        fig, ax = plt.subplots()
        ax.quiver(X[:,0], X[:,1], V[:,0], V[:,1])
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Couette Flow')
    if  sigma > 0:
        noise = sigma*torch.randn_like(X,dtype=torch.float64)
        X_noisy = X + noise
        
        fig, ax = plt.subplots()
        ax.quiver(X_noisy[:,0], X_noisy[:,1], V[:,0], V[:,1])
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Couette Flow w/ Noise')
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
    
    # if plot == True:
    #     fig, ax = plt.subplots()
    #     ax.scatter(x0[:, 0], x0[:, 1], s = 3)
    #     ax.scatter(X[:, 0], X[:, 1], s = 3)
    #     ax.set_aspect('equal', adjustable='box')
    #     ax.set_title('Poiseuille Flow')      

    # if sigma > 0:
    #     noise = sigma*torch.randn_like(X,dtype=torch.float64)
    #     X_noisy = X + noise
    #     fig, ax = plt.subplots()
    #     ax.scatter(x0[:, 0], x0[:, 1], s = 3)
    #     ax.scatter(X_noisy[:, 0], X_noisy[:, 1], s = 3)
    #     ax.set_aspect('equal', adjustable='box')
    #     ax.set_title('Poiseuille Flow w/ Noise')
    
    if plot == True:     
        fig, ax = plt.subplots()
        ax.quiver(X[:,0], X[:,1], V[:,0], V[:,1])
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Poiseuille Flow')
    if sigma > 0:
        noise = sigma*torch.randn_like(X,dtype=torch.float64)
        X_noisy = X + noise
        
        fig, ax = plt.subplots()
        ax.quiver(X_noisy[:,0], X_noisy[:,1], V[:,0], V[:,1])
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Poiseuille Flow w/ Noise')
    return X,V,theta,phi

## "Steady" Unsteady 2D flows
# Lamb–Oseen Vortex at fixed time, assumes CCW circulation as positive
    # Gamma = circulation of vortex
    # nu = kinematic viscosity of working fluid
    # r = radial distance
    
def lamb_oseen_vortex(x0, Gamma, nu, t, centerX = 256/2, centerY = 256/2, sigma = 0, plot = False, visualize = False):
    datapts = len(x0)
    # vectors of location of vortex center w/ vector length of datapts
    centerX_vec = torch.full((datapts,),centerX).reshape(-1,1)
    centerY_vec = torch.full((datapts,),centerY).reshape(-1,1)
    
    if visualize == True:
        datapts = len(x0)
        centerX_vec = torch.full((datapts,),centerX).reshape(-1,1)
        centerY_vec = torch.full((datapts,),centerY).reshape(-1,1)
        x = x0[:,0].reshape(-1,1)-centerX_vec
        y = x0[:,1].reshape(-1,1)-centerY_vec
        r = torch.empty((datapts,1),dtype=torch.float32).reshape(-1,1)
        for ii in range(datapts):
            r[ii,0] = torch.sqrt(x[ii,]**2+y[ii,]**2)
            
        radius = torch.linspace(0,torch.max(r),1000)
            
        g = 1 - torch.exp(-radius**2/(4*torch.pi*nu*t))
        v_theta = Gamma/(2*torch.pi*radius)*g

        fig, ax = plt.subplots()
        ax.scatter(radius, v_theta, s = 3)
        ax.set_title('Lamb–Oseen Velocity Visualization @ %f s' %t)
        ax.set_xlabel('Radius from Vortex Center [px]')
        ax.set_ylabel('Particle Velocity [px/s]')
        
    else:
        def lamb_oseen_ODE(t,x0):
            x = x0[:,0].reshape(-1,1)-centerX_vec
            y = x0[:,1].reshape(-1,1)-centerY_vec
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
            ax.scatter(x0[:, 0], x0[:, 1], s = 3)
            ax.scatter(X[:,0], X[:,1], s = 3)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title('Lamb–Oseen Flow @ %f s' %t)
        if sigma > 0:
            noise = sigma*torch.randn_like(X,dtype=torch.float64)
            X_noisy = X + noise
            fig, ax = plt.subplots()
            ax.scatter(x0[:, 0], x0[:, 1], s = 3)
            ax.scatter(X_noisy[:,0], X_noisy[:,1], s = 3)
            ax.set_aspect('equal', adjustable='box')
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
        ax.scatter(x0[:, 0], x0[:, 1], s = 3)
        ax.scatter(X[:,0], X[:,1], s = 3)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Rayleigh Problem @ %f s' %t)
    if sigma > 0:
        noise = sigma*torch.randn_like(X,dtype=torch.float64)
        X_noisy = X + noise
        fig, ax = plt.subplots()
        ax.scatter(x0[:, 0], x0[:, 1], s = 3)
        ax.scatter(X_noisy[:,0], X_noisy[:,1], s = 3)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Rayleigh Problem w/ Noise')
        
    # if plot == True:
    #     fig, ax = plt.subplots()
    #     ax.quiver(X[:,0], X[:,1], V[:,0], V[:,1])
    #     ax.set_title('Stokes First Problem @ %f s' %t)
    #     ax.set_aspect('equal', adjustable='box')
    # if sigma > 0:
    #     noise = sigma*torch.randn_like(X,dtype=torch.float64)
    #     X_noisy = X + noise
    #     fig, ax = plt.subplots()
    #     ax.quiver(X_noisy[:,0], X_noisy[:,1], V[:,0], V[:,1])
    #     ax.set_aspect('equal', adjustable='box')
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
    
    # if plot == True:
    #     fig, ax = plt.subplots()
    #     ax.scatter(x0[:, 0], x0[:, 1], s = 3)
    #     ax.scatter(X[:,0], X[:,1], s = 3)
    #     ax.set_aspect('equal', adjustable='box')
    #     ax.set_title('Stokes Problem @ %f s' %t)
    # if sigma > 0:
    #     noise = sigma*torch.randn_like(X,dtype=torch.float64)
    #     X_noisy = X + noise
    #     fig, ax = plt.subplots()
    #     ax.scatter(x0[:, 0], x0[:, 1], s = 3)
    #     ax.scatter(X_noisy[:,0], X_noisy[:,1], s = 3)
    #     ax.set_aspect('equal', adjustable='box')
    #     ax.set_title('Stokes Problem w/ Noise')
        
    if plot == True:
        fig, ax = plt.subplots()
        ax.quiver(X[:,0], X[:,1], V[:,0], V[:,1])
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Stokes Second Problem @ %f s' %t)
    if sigma > 0:
        noise = sigma*torch.randn_like(X,dtype=torch.float64)
        X_noisy = X + noise
        fig, ax = plt.subplots()
        ax.quiver(X_noisy[:,0], X_noisy[:,1], V[:,0], V[:,1])
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Stokes Second Problem w/ Noise')
    
    return X

#%%

from PyQt5 import QtGui, QtWidgets
# from PyQt5 import QtCore, QtGui, QtWidgets
    # might need all three later on
from design import Ui_MainWindow

class Logic(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
    
    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        
        # initialize input validation
        onlyDbl = QtGui.QDoubleValidator()
        onlyInt = QtGui.QIntValidator()
        
        self.lineEdit_ppp.setValidator(onlyDbl)
        self.lineEdit_xdim.setValidator(onlyInt)
        self.lineEdit_ydim.setValidator(onlyInt)
        self.lineEdit_sigma.setValidator(onlyDbl)
        
        self.lineEdit_seed.setValidator(onlyInt)
        
        self.lineEdit_Vmax.setValidator(onlyDbl)
        self.lineEdit_timestep.setValidator(onlyDbl)
        self.lineEdit_Gamma.setValidator(onlyDbl)
        self.lineEdit_nu.setValidator(onlyDbl)
        self.lineEdit_omega.setValidator(onlyDbl)
        self.lineEdit_centerX.setValidator(onlyInt)
        self.lineEdit_centerY.setValidator(onlyInt)
        
        # initialize booleans
        self.flow_selection()
        self.random_seed()
        self.plot_field()
        
        ## signals to trigger slots
        # connect comboBox for flow type
        self.comboBox_flowType.currentIndexChanged.connect(self.flow_selection)
        
        # connect checkBox for toggling particle randomizing
        self.checkBox_randomSeed.toggled.connect(self.random_seed)

        # connect pushButtons
        self.pushButton_generate.clicked.connect(self.generate_field)            
        self.pushButton_save.clicked.connect(self.save_data)
        self.pushButton_clear.clicked.connect(self.clear_data)
        
        # connect checkBoxes
        self.checkBox_visualize.stateChanged.connect(self.visualize_field)
        self.checkBox_plot.stateChanged.connect(self.plot_field)

            
    ## slots that trigger from signals
    # update flow type
    def flow_selection(self):
        flow_type = self.comboBox_flowType.currentText()
        print(f'The flow type selected is {flow_type}')
        
        if self.comboBox_flowType.currentIndex() == 0 or 1 or 2:
            self.lineEdit_Gamma.setDisabled(True)
            self.lineEdit_nu.setDisabled(True)
            self.lineEdit_omega.setDisabled(True)
            self.lineEdit_centerX.setDisabled(True)
            self.lineEdit_centerY.setDisabled(True)
            
            self.checkBox_visualize.setDisabled(True)
            self.label_OR.setDisabled(True)
            
            self.checkBox_visualize.setChecked(False)
            self.checkBox_plot.setChecked(True)
            
        if self.comboBox_flowType.currentIndex() == 3:
            self.lineEdit_Gamma.setDisabled(False)
            self.lineEdit_nu.setDisabled(False)
            self.lineEdit_omega.setDisabled(True)
            self.lineEdit_centerX.setDisabled(False)
            self.lineEdit_centerY.setDisabled(False)
            
            self.checkBox_visualize.setDisabled(False)
            self.label_OR.setDisabled(False)
            
        if self.comboBox_flowType.currentIndex() == 4:
            self.lineEdit_Gamma.setDisabled(True)
            self.lineEdit_nu.setDisabled(False)
            self.lineEdit_omega.setDisabled(True)
            self.lineEdit_centerX.setDisabled(True)
            self.lineEdit_centerY.setDisabled(True)
            
            self.checkBox_visualize.setDisabled(False)
            self.label_OR.setDisabled(False)
        
        if self.comboBox_flowType.currentIndex() == 5:
            self.lineEdit_Gamma.setDisabled(True)
            self.lineEdit_nu.setDisabled(False)
            self.lineEdit_omega.setDisabled(False)
            self.lineEdit_centerX.setDisabled(True)
            self.lineEdit_centerY.setDisabled(True)
            
            self.checkBox_visualize.setDisabled(False)
            self.label_OR.setDisabled(False)

    def generate_field(self):
        
        # Update values
        ppp = float(self.lineEdit_ppp.text())
        xdim = float(self.lineEdit_xdim.text())
        ydim = float(self.lineEdit_ydim.text())
        sigma = float(self.lineEdit_sigma.text())
        
        random = self.checkBox_randomSeed.isChecked()
        seed = self.lineEdit_seed.text()
        
        Vmax = float(self.lineEdit_Vmax.text())
        timestep = float(self.lineEdit_timestep.text())
        Gamma = float(self.lineEdit_Gamma.text())
        nu = float(self.lineEdit_nu.text())
        omega = float(self.lineEdit_omega.text())
        centerX = float(self.lineEdit_centerX.text())
        centerY = float(self.lineEdit_centerY.text())
        
        t = timestep
        
        plot = self.checkBox_plot.isChecked()
        visualize = self.checkBox_visualize.isChecked()       

        x0 = pytgen(ppp, xdim, ydim, random, seed)
        
        # Generate flows        
        if self.comboBox_flowType.currentIndex() == 0:
            print("Uniform")
            X,V = uniform_flow(x0, Vmax, timestep, sigma, plot)
        
        if self.comboBox_flowType.currentIndex() == 1:
            print("Couette")
            X,V,_ = couette_flow(x0, Vmax, timestep, ydim, sigma, plot)
            
        if self.comboBox_flowType.currentIndex() == 2:
            print("Poiseuille")
            X,V,_,_ = poiseuille_flow(x0, Vmax, timestep, ydim, sigma, plot)
    
        if self.comboBox_flowType.currentIndex() == 3:
            print("Lamb-Oseen")
            X = lamb_oseen_vortex(x0, Gamma, nu, t, centerX, centerY, sigma, plot, visualize)
            
        if self.comboBox_flowType.currentIndex() == 4:
            print("Rayleigh Problem")
            X = rayleigh_problem(x0, Vmax, nu, t, sigma, plot)
            
        if self.comboBox_flowType.currentIndex() == 5:
            print("Stokes Problem")
            X = stokes_problem(x0, Vmax, omega, nu, t, sigma, plot)
            
    def save_data(self):
        print("Data Saved")
        
    def clear_data(self):
        print("Data Cleared")
    
    def random_seed(self):
        random = self.checkBox_randomSeed.isChecked()
        if random == True:
            print('Random generation enabled')
            self.lineEdit_seed.setDisabled(True)
        else:
            print('Random generation disabled')
            self.lineEdit_seed.setDisabled(False)
    
    def visualize_field(self):
        visualize = self.checkBox_visualize.isChecked()
        
        if visualize and self.checkBox_plot.isChecked() == True:
            self.checkBox_plot.setChecked(False)

        if visualize == True:
            print('Visualization enabled')
        elif visualize == False:
            print('Visualization disabled')
    
    def plot_field(self):
        plot = self.checkBox_plot.isChecked()
        
        if plot and self.checkBox_visualize.isChecked() == True:
            self.checkBox_visualize.setChecked(False)
            
        if plot == True:
            print('Plotting enabled')
        elif plot == False:
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