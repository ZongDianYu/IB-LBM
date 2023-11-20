import math
import numpy as np   
import matplotlib.pyplot as plt
from tool import Compute_Tangent

def Compute_Signed_Angle_Degrees(tmc, tpc):
    angle = math.atan2(tmc[1],tmc[0]) - math.atan2(tpc[1],tpc[0])
    #angle = (angle % 2.*np.pi)
    angle = (angle + 2 * np.pi) % (2 * np.pi)
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    #angle = math.degrees(angle)
    #cos_angle = Dot(tmc,tpc)
    #sin_angle = (1.- cos_angle**2.0)**0.5
    return [angle,cos_angle, sin_angle]   




def Find_R(x,y):
    r = []
    n_particles = len(x)
    for ID in range(n_particles):
        tmp = np.array([x[ID],y[ID]])
        r.append(tmp)
    return r


def Find_Equilibrium_Length(x,y):
    n_particles = len(x)
    l_eq = np.zeros(n_particles)
    delta_x = x[1:] - x[:-1]
    delta_y = y[1:] - y[:-1]
    #print(delta_x)
    #print(delta_y)
    temp = (delta_x**2. + delta_y**2.)**0.5
    for ID in range(n_particles-1):
        l_eq[ID] = temp[ID]
    
    l_eq[n_particles-1] = np.nan
    #n_total = len(x)
    #l_eq = np.empty(n_total)
    return l_eq

    
def Find_Equilibrium_Angles(x,y):
    r = Find_R(x,y)    
    n_particles = len(x)
    theta_eq = np.zeros(n_particles)
    for ID in range(1,n_particles-1):
        rm = r[ID -1]
        ri = r[ID]
        rp = r[ID +1]
        [Delta_r_mi, Delta_s_mi, t_mi] = Compute_Tangent(rm,ri)
        [Delta_r_ip, Delta_s_ip, t_ip] = Compute_Tangent(ri,rp)
        [theta_i,cos_theta_i, sin_theta_i] = Compute_Signed_Angle_Degrees(t_mi, -t_ip)
        theta_eq[ID] = theta_i
    theta_eq[0] = np.nan    
    theta_eq[-1] = np.nan  
    return theta_eq

x = np.array([-2.,-1.2,0.,1.1,2.])
y = np.array([0.,0.3,0.1,-0.3,0.])    
plt.plot(x,y,'ro-')
l_eq = Find_Equilibrium_Length(x,y) 
print (l_eq)
theta_eq = Find_Equilibrium_Angles(x,y)
print (np.rad2deg(theta_eq))
plt.xlim(-5,5)
plt.ylim(-1,1)