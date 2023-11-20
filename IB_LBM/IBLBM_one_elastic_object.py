import matplotlib.pyplot as plt
import numpy as np
#=====
data = np.loadtxt("/home/user/dell2/Handover/IBLBM_one_elastic_object/Data_all.txt")
datay = data[:,1]

t = [];
for i in np.arange (1,10002,1):
    t.append(i)

plt.clf()
fig = plt.figure(dpi = 100)
fig.set_size_inches(10,6)
plt.xlabel("time step", fontsize=20)
plt.ylabel("y position", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.plot(t,datay,'r-')
plt.grid(True)
