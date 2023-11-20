import matplotlib.pyplot as plt
import math
import numpy as np
#=====
def Compute_Tangent(ra,rb):
    Delta_r_ab = ra - rb
    #print ("===", Delta_r_ab)
    Delta_s_ab = Norm(Delta_r_ab)
    t_ab = Delta_r_ab/Delta_s_ab
    return [Delta_r_ab, Delta_s_ab, t_ab]



def DumpFig(fname,data_x,data_y):
    #plt.figure()
    plt.clf()
    #plt.contourf(phi_data,20)
    plt.plot(data_x,data_y,'k-o', markersize = 10, lw = 3)
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    #plt.colorbar()
    
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.tight_layout()
    plt.savefig(fname)
    
#=====
def Norm(vec):
    ans = (vec[0]*vec[0] + vec[1]*vec[1])**0.5
    return ans

#=====
def Dot(vec1,vec2):
    ans = vec1[0]*vec2[0] + vec1[1]*vec2[1]  
    return ans