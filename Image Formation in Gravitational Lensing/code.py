###########################
### Importing Libraries ###
###########################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy import units as u
from astropy import constants as const
import pylab as py
from matplotlib import animation, rc
#from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
from matplotlib import gridspec
import sys

###########################
###### User Inputs ########
###########################

x0,x1,y0,y1 = float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4])
step = float(sys.argv[5])
savename= sys.argv[6]
scale = float(sys.argv[7]) # 0.0075 or 0.0025 for given set of paths

###########################
#######  Constants  #######
###########################

DL = 10*u.pc
DS = 500*u.pc
DLS = DS - DL
ML = 100*u.M_sun
Rs = (2*const.G*ML)/const.c**2
theta_E = np.sqrt((2*Rs*DLS)/(DL*DS))

###########################
#######  Variables  #######
###########################

x = []
y = []
if(x0<x1):
  x = np.arange(x0,x1+step,step)
elif(x0>x1):
  x = np.arange(x1,x0+step,step)[::-1]

if(y0<y1):
  y = np.arange(y0,y1,step)
elif(y0>y1):
  y = np.arange(y1,y0+step,step)[::-1]

if(len(x) == 0 and len(y) != 0):
  if(x0 == x1):
    x = x0*np.ones_like(y)
elif(len(y) == 0 and len(x) != 0):
  if(y0 == y1):
    y = y0*np.ones_like(x)

a = x[np.where(np.round(y,8)==0)]
b = y[np.where(np.round(x,8)==0)]

if(len(a)==0 and len(b)==0):
  exit()
elif(len(a)==0):
  a = 0
  b = b[0]
  u0 = np.abs(b)
elif(len(b)==0):
  b = 0
  a = a[0]
  u0 = np.abs(a)
else:
  a = a[0]
  b = b[0]
  u0 = np.abs(a*b)/np.sqrt(a**2 + b**2)

u_t = np.sqrt(x**2 + y**2) # u(t) wiht motion arbitrary direction
A_plus = (u_t**2+2)/(2*u_t*np.sqrt(u_t**2+4))+0.5
A_minus = (u_t**2+2)/(2*u_t*np.sqrt(u_t**2+4))-0.5
A = A_plus+A_minus
theta_S = u_t*theta_E
theta_I_plus = 0.5*(theta_S + np.sqrt(theta_S**2+4*theta_E**2))
theta_I_minus = 0.5*(theta_S - np.sqrt(theta_S**2+4*theta_E**2))
v_plus = theta_I_plus/theta_E
v_minus = theta_I_minus/theta_E
theta = np.arange(0,np.pi*2,0.01)
x_circ = np.cos(theta)
y_circ = np.sin(theta)
theta_S = u_t*theta_E
v_plus = theta_I_plus/theta_E 
v_minus = theta_I_minus/theta_E
x_plus, y_plus = v_plus.value*(x/u_t),v_plus.value*(y/u_t)
x_minus, y_minus = v_minus.value*(x/u_t),v_minus.value*(y/u_t)
t_arr = np.ones_like(A_plus+A_minus)
t_arr[:np.argmin(np.abs(u_t-u0))] *= -1
t_arr = t_arr*np.sqrt(u_t**2-u0**2)
eccs = np.abs(u0/u_t)

###########################
#######  Plotting  ########
###########################

eps = 0.1
ft = 20
plt.style.use('dark_background')
fig = plt.figure(figsize=(10, 15)) 
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
ax0 = plt.subplot(gs[0])

plt.scatter(x,y,s=100,label="Source",c='yellow')
ax0.plot(x_circ,y_circ,'--',c='greenyellow', label="Einstein Circle")
plt.scatter(0,0,s=100,label="Lens",c='orange')
idx = np.argmin(u_t)
plt.scatter(x_plus[idx],y_plus[idx],s=50,label="Major Image")
plt.scatter(x_minus[idx],y_minus[idx],s=50,label="Minor Image")
for i in range(len(eccs)):
    rot = np.arctan(b/a)
    rot -= (b*x[i]-a*y[i])/u_t[i]
    A = scale*A_plus[i]
    b_plus = np.sqrt(A/np.pi  * np.sqrt(1+eps-eccs[i]**2))
    a_plus = A / np.pi / b_plus
    A = scale*A_minus[i]
    b_minus = np.sqrt(A/np.pi  * np.sqrt(1+eps-eccs[i]**2))
    a_minus = A / np.pi / b_minus
    ellipse_plus = Ellipse(xy=(x_plus[i],y_plus[i]), width=2*b_plus, height=2*a_plus, angle=(180/np.pi)*rot,color="lightblue",label="Major Image")
    ellipse_minus = Ellipse(xy=(x_minus[i],y_minus[i]), width=2*b_minus, height=2*a_minus, angle=(180/np.pi)*rot,color="lightpink",label="Minor Image")
    ax0.add_artist(ellipse_plus)
    ax0.add_artist(ellipse_minus)

plt.legend(fontsize=ft,loc="lower left")
ax0.set_xlabel(r'$x/\theta_E$',fontsize=ft)
ax0.set_ylabel(r'$y/\theta_E$',fontsize=ft)
plt.xticks(fontsize=ft)
plt.yticks(fontsize=ft)
plt.grid()
lims = [0,0]
lims[0] = -step+np.amin([np.amin(y_plus),np.amin(y_minus),np.amin(x_plus),np.amin(x_minus)])
lims[1] = step+np.amax([np.amax(y_plus),np.amax(y_minus),np.amax(x_plus),np.amax(x_minus)])
ax0.set_ylim(lims[0],lims[1])
ax0.set_xlim(lims[0],lims[1])
ax0.set_aspect('equal')

ax1 = plt.subplot(gs[1])
spline_f = interp1d(x=t_arr,y= A_plus+A_minus, kind='cubic')
t_linspace = np.linspace(t_arr[0],t_arr[-1],1000)
y_linspace = spline_f(t_linspace)
plt.scatter(t_arr,A_plus+A_minus,c='white')
plt.plot(t_linspace,y_linspace,c='white')
plt.ylabel("Magnification",fontsize=ft)
plt.xlabel(r'$\frac{(t-t_0)}{t_E}$',fontsize=ft)
plt.xticks(fontsize=ft)
plt.yticks(fontsize=ft)
ax1.set_xlim(np.amin(t_arr)-0.1, np.amax(t_arr)+0.1)
plt.grid()
plt.savefig("{}1_black.png".format(savename),bbox_inches="tight")

###########################
#######  Animating  #######
###########################

def init():
    line0.set_data([], [])
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    return line1,line2,line3,

def update(i):
    plt.cla()
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    ax0 = plt.subplot(gs[0])
    ax0.axis('square')
    ax0.set_ylim(lims[0],lims[1])
    ax0.set_xlim(lims[0],lims[1])
    ax0.plot(0,0,'o',markersize = 9, markerfacecolor = "orange",markeredgecolor ="orange", label="Lens")
    ax0.plot(x_circ,y_circ,'--',c='greenyellow', label="Einstein Circle")
    ax0.plot(x[i], y[i], 'o',color = 'yellow', markersize=10, markevery=10000, markerfacecolor = 'yellow',lw=2) # source
    plt.ylabel(r'$y/\theta_E$',fontsize=ft)
    plt.xlabel(r'$x/\theta_E$',fontsize=ft)
    plt.xticks(fontsize=ft)
    plt.yticks(fontsize=ft)
    plt.grid()

    rot = np.arctan(b/a)
    rot -= (b*x[i]-a*y[i])/u_t[i]
    A = scale*A_plus[i]
    b_plus = np.sqrt(A/np.pi  * np.sqrt(1+eps-eccs[i]**2))
    a_plus = A / np.pi / b_plus
    A = scale*A_minus[i]
    b_minus = np.sqrt(A/np.pi  * np.sqrt(1+eps-eccs[i]**2))
    a_minus = A / np.pi / b_minus
    ellipse_plus = Ellipse(xy=(x_plus[i],y_plus[i]), width=2*b_plus, height=2*a_plus, angle=(180/np.pi)*rot,color="lightblue",label="Major Image")
    ellipse_minus = Ellipse(xy=(x_minus[i],y_minus[i]), width=2*b_minus, height=2*a_minus, angle=(180/np.pi)*rot,color="lightpink",label="Minor Image")
    ax0.add_artist(ellipse_plus)
    ax0.add_artist(ellipse_minus)

    line0.set_data(0,0)
    line1.set_data(x[i],y[i])
    line2.set_data(x_plus[i],y_plus[i])
    line3.set_data(x_minus[i],y_minus[i])
    line4.set_data(x_circ, y_circ)
    
    lgnd = ax0.legend((line4,line0,line1, line2, line3), ('Einstein Circle','Lens','Source', 'Major Image', 'Minor Image'), loc="best", fontsize=ft-2.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(6)
    lgnd.legendHandles[1]._legmarker.set_markersize(6)
    lgnd.legendHandles[2]._legmarker.set_markersize(6)
    lgnd.legendHandles[3]._legmarker.set_markersize(6)

    ax1 = plt.subplot(gs[1])
    idx = np.where(t_linspace<=t_arr[i])
    plt.scatter(t_arr[:i+1],(A_plus+A_minus)[:i+1],c='white')
    plt.plot(t_linspace[idx],y_linspace[idx],c='white')
    plt.ylabel("Magnification",fontsize=ft)
    plt.xlabel(r'$\frac{(t-t_0)}{t_E}$',fontsize=ft)
    plt.xticks(fontsize=ft)
    plt.yticks(fontsize=ft)
    ax1.set_xlim(np.amin(t_arr)-0.1, np.amax(t_arr)+0.1)
    ax1.set_ylim(np.amin(A_plus+A_minus)-0.1, np.amax(A_plus+A_minus)+0.1)
    plt.grid()
    return (line0,line1,line2,line3,line4)

fig = plt.figure(figsize=(10, 15)) 
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
ax0 = plt.subplot(gs[0])

plt.rcParams["figure.figsize"] = (10,10)
line0, = ax0.plot([], [], 'o',color = 'orange', markersize=10, markevery=10000, markerfacecolor = 'orange',lw=2) # lens
line1, = ax0.plot([], [], 'o',color = 'yellow', markersize=10, markevery=10000, markerfacecolor = 'yellow',lw=2) # source
line2, = ax0.plot([], [], 'o',color = 'lightblue', markersize=1, markevery=10000, markerfacecolor = 'lightblue',lw=2) # Major Image
line2.set_animated(True)
line3, = ax0.plot([], [], 'o',color = 'lightpink', markersize=1, markevery=10000, markerfacecolor = 'lightpink',lw=2) # minor image
line3.set_animated(True)
line4, = ax0.plot([], [], '--',color = 'greenyellow', markersize=10, markevery=10000, markerfacecolor = 'greenyellow',lw=2) # radius

anim = FuncAnimation(fig, update, init_func=init, frames=np.arange(len(x)), interval=100, blit=True, repeat=True)
#HTML(anim.to_html5_video())
anim.save('{}1.gif'.format(savename), writer='pillow', fps=10)

###########################
####### End of Code #######
###########################
