import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tool import DumpFig
from tool import Compute_Tangent
from geometry import Find_R, Find_Equilibrium_Length, Find_Equilibrium_Angles, Compute_Signed_Angle_Degrees

def Compute_Potential_Energy_Linear(r):
    U = 0.
    for ID in id_list[:-1]:
        ri,rp = r[ID], r[ID+1]
        #l0_ip = 0.5* (Lengths_0[ID] + Lengths_0[ID+1])
        l0_ip = l_eq[ID]
        [Delta_r_ip, Delta_s_ip, t_ip] = Compute_Tangent(ri,rp)
        U += 0.5 *  k_linear *  (Delta_s_ip - l0_ip)**2.0
    return U
#=====    
def Compute_Potential_Energy_Bending(r):
    U = 0.
    for ID in id_list[0:-2]:
        ri,rp,rp2 = r[ID], r[ID+1],r[ID+2]
        [Delta_r_ip, Delta_s_ip, t_ip] = Compute_Tangent(ri,rp)
        [Delta_r_pp2, Delta_s_pp2, t_pp2] = Compute_Tangent(rp,rp2)
        [theta_p,cos_theta_p, sin_theta_p] = Compute_Signed_Angle_Degrees(t_ip, -t_pp2)
        theta_p_0 = angles_0[ID+1]  
        #print (theta_p_0)
        #U += 0.5 * k_torsional * (theta_p - theta_p_0)**2.0
        #U += k_torsional * (1- np.cos(math.radians(theta_p) - math.radians(theta_p_0)))
        U += k_torsional * (1- np.cos(theta_p - theta_p_0))
    return U

#=====    
def Compute_Potential_Energy(r):
    U_linear = Compute_Potential_Energy_Linear(r)
    U_bending = Compute_Potential_Energy_Bending(r)
    U_total = U_linear + U_bending
    return U_total
#=====
def Compute_Force_Variational(r):
    fx = np.zeros(n_particles)
    fy = np.zeros(n_particles)
    x_virtual = np.zeros(n_particles)
    y_virtual = np.zeros(n_particles)
    
    U_ref = Compute_Potential_Energy(r)
    for ID in id_list:
        #r_virtual = np.copy(r)
        #r_virtual[ID][0] += TINY
        x_virtual[:] = x[:]
        y_virtual[:] = y[:]
        x_virtual[ID] += TINY
        r_virtual = Find_R(x_virtual,y_virtual)
        
        U_virtual = Compute_Potential_Energy(r_virtual)
        fx[ID] = -(U_virtual - U_ref)/TINY
        #===
        #r_virtual = np.copy(r)
        #r_virtual[ID][1] += TINY
        x_virtual[:] = x[:]
        y_virtual[:] = y[:]
        y_virtual[ID] += TINY
        r_virtual = Find_R(x_virtual,y_virtual)
        U_virtual = Compute_Potential_Energy(r_virtual)
        fy[ID] = -(U_virtual - U_ref)/TINY
    return [fx,fy]

SMALL = 1.E-8
TINY = 1.E-10
#=====INITIALIZATION=====
#x = np.array([0,1,1.5,1,1])#,3,4,5]) # six particles
#y = np.array([0,0,1,2,3])#,-1,0,0]) # six particles
#angles_0 = np.array([90.,90.,180.,180.,180.])#,170.,170.,170.])
#Lengths_0 = np.array([1.,1.,1.,1.,1.,])#,1.,1.,1.])

#x = np.array([-2.,-1./2.**0.5, 0.,1./2.**0.5, 2. ])
#y = np.array([0., 0., 1./2.**0.5, 0.,0.])
# move ID = 2 slightly from eq point

x = np.array([-2.,-1.2,0.,1.1,2.])
y = np.array([0.,0.3,0.1,-0.3,0.])
x_temp = np.copy(x)
y_temp = np.copy(y)
l_eq = Find_Equilibrium_Length(x,y) 
theta_eq = Find_Equilibrium_Angles(x,y)
x[2] = x[2] + 0.3
y[2] = y[2] + 0.2


angles_0 = theta_eq
Lengths_0 = l_eq


#delta_x = x[1:] - x[:-1]
#delta_y = y[1:] - y[:-1]
#angles_0 = np.array([np.nan, 180., 270., 180.,  np.nan])
#Lengths_0 = np.array([1.,1.,1.,1.,1.])
#print ("equilibrium length = ", 0.5*(Lengths_0[1:]+Lengths_0[:-1]))
k_linear = 0.5#1.0
k_torsional = 0.1
dt = 0.005
mass = 1.
beta = 0.5

#===== Setting =====
n_particles = len(x)
#r = Find_R(x,y)
vx = np.zeros(n_particles)
vy = np.zeros(n_particles)
n_particles = len(x)
id_list = range(0,n_particles)



plt.figure(figsize=(8, 6), dpi=80)

# main loop   
r = Find_R(x,y)
for n_iter in range(2001):
    
    #[fx,fy] = Find_Force_All_Particles() 
    [fx,fy] = Compute_Force_Variational(r)
    fx = fx - beta*vx # adding damping force
    fy = fy - beta*vy
    #x_temp = x[0:2]
    #y_temp = y[0:2]
    
    
    
    x = x + vx * dt + 0.5 * fx/mass * dt**2. # x_n1
    y = y + vy * dt + 0.5 * fy/mass * dt**2. # y_n1
    #===force fixed position
    x[0] = x_temp[0]
    x[1] = x_temp[1]
    x[-2] = x_temp[-2]
    x[-1] = x_temp[-1]
    y[0] = y_temp[0]
    y[1] = y_temp[1]
    y[-2] = y_temp[-2]
    y[-1] = y_temp[-1]
    
    #x[0:2] = x_temp
    #y[0:2] = y_temp
    r = Find_R(x,y)
    [fx_n1,fy_n1] = Compute_Force_Variational(r) 
    fx_n1 = fx_n1 - beta*vx
    fy_n1 = fy_n1 - beta*vy
    vx = vx + 0.5 * (fx+fx_n1)/mass*dt
    vy = vy + 0.5 * (fy+fy_n1)/mass*dt        
#====
    if (n_iter % 100 == 0):
        plt.clf()
        
        time = dt * n_iter
        print ("time = %5.3f"%time)
        fname = "image_%05d_variational.png"%(n_iter)
        
        #DumpFig(fname,x,y)
        plt.clf()
        #plt.contourf(phi_data,20)
        plt.plot(x,y,'k-o', markersize = 10, lw = 3)
        plt.plot(x_temp,y_temp,'r-o', markersize = 8, lw = 3)
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        #plt.colorbar()

        plt.xlim(-10,10)
        plt.ylim(-10,10)
        plt.tight_layout()
        plt.savefig(fname)
        plt.figure(figsize=(8, 6), dpi=80)
        #plt.plot(x,y,'k-o', markersize = 10, lw = 3)