#!/usr/bin/env python
# coding: utf-8

# In[99]:


##################################
###--- Importing Libraries ---####
##################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import imageio
from astropy import units as u
from astropy import constants as const
from itertools import cycle


# In[100]:


##################################
###--- Function Defintion ---#####
##################################

def rv(theta, e=0, w=0, i=60, a=0.05*u.au, P=5*u.yr, theta_radians=True):  # angles -> degrees, theta -> radians
    #theta = np.radians(theta)
    w = np.radians(w)
    i = np.radians([i])
    k = ((2*np.pi*np.sin(i)*a.to(u.km))/(P.to(u.s)*np.sqrt(1-e**2))).value
    return k*( e*np.cos(w) + np.cos(w+theta) ).squeeze(), k*( e*np.cos(w)).squeeze()

def vrt(i,m,M):
    # Data for plotting
    th = np.arange(0,4*np.pi,0.01)
    es = orbit(P=5*u.yr,w=0.0)
    t = es.dt(es.w_v(th,i),dth)
    t = np.cumsum(t).to(u.yr).value
    t = np.append([0],t[:-1])
    v_r, v_r_avg = rv(th,i,P=es.P,w=es.w)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(t, v_r,'orangered',label="e = {}".format(np.round(i,2)))
    ax.plot(t, v_r_avg*np.ones_like(t),"g--")
    ax.grid()
    leg = plt.legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=25)
    for item in leg.legendHandles:
        item.set_visible(False)
    ax.set(xlabel='Time (in Earth years)', ylabel='Radial Velocity (km/s)', title="$\omega$ = ${}^o$, $a$ = 0.05 AU, P = {} Earth years".format(es.w, es.P))
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    ax.set_ylim(m-0.05, M+0.05)

    # Used to return the plot as an image rray
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image

##################################
#####--- Class Defintion ---######
##################################

class orbit:
    def __init__(self, M_star=1*const.M_sun, P=1*u.yr, w=0.0): # w in degrees
        self.M_star = M_star
        self.P = P
        self.w = np.radians(w)
        a = np.cbrt(const.G*M_star.to(u.kg)*(P.to(u.s)**2)*(1/(4*(np.pi**2))))
        self.a = a.to(u.m)

    def w_v(self,th,e=0.0):
        return ((2*np.pi)*(1+e*np.cos(th))**2)/(self.P*(1-e**2)**(1.5))

    def dt(self,wv,dth=0.1):
        return dth/wv


# In[ ]:



##################################
#####--- Input Variables ---######
##################################

dth = 0.01
th = np.arange(0.0, 4*np.pi,dth)
step = 0.01
e = np.arange(0,0.9+step,step)
M_star = 1*u.M_sun
m = np.inf
M = -np.inf
for i in e:
    tmp,_ = rv(th,i)
    if(np.amin(tmp) < m):
        m = np.amin(tmp)
    if(np.amax(tmp) > M):
        M = np.amax(tmp)

matplotlib.rc('text', usetex=True) #use latex for text

# add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

kwargs_write = {'fps':100.0, 'quantizer':'nq'}
imageio.mimsave('./variation_of_rv_curve_with_e.gif', [vrt(i,m,M) for i in e], fps=5)

##################################
#######--- End of Code ---########
##################################


# In[71]:


def vrt_collage(i,m,M):
    ft = 30
    # Data for plotting
    th = np.arange(0,4*np.pi,0.01)
    es0 = orbit(P=5*u.yr,w=0.0)
    es1 = orbit(P=5*u.yr,w=30.0)
    es2 = orbit(P=5*u.yr,w=60.0)
    es3 = orbit(P=5*u.yr,w=90.0)
    t = es0.dt(es0.w_v(th,i),dth)
    t = np.cumsum(t).to(u.yr).value
    t = np.append([0],t[:-1])
#    fig, ax = plt.subplots(2,2,figsize=(16,12))
    fig, ax = plt.subplots(2,2, sharex=True, sharey=True, figsize=(16,12))
# add a big axis, hide frame
#plt.rcParams["figure.figsize"] = (16, 12 * len(labels) / 10)
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Time (in Earth years)",fontsize=ft,labelpad=ft-5)
    plt.ylabel("Radial Velocity (km/s)",fontsize=ft,labelpad=ft-5)
    plt.rcParams['xtick.labelsize']=ft-5
    plt.rcParams['ytick.labelsize']=ft-5
    fig.suptitle("$a$ = 0.05 AU, P = 5 Earth years, e = {}".format(np.round(i,2)),fontsize=ft)
    v_r, v_r_avg = rv(th,i,P=es0.P,w=np.degrees(es0.w))
    ax[0,0].plot(t, v_r,'orangered')
    ax[0,0].plot(t, v_r_avg*np.ones_like(t),"k--")
    ax[0,0].grid()
    #ax[0,0].set_ylim(m[0]-0.05, M[0]+0.05)
    ax[0,0].set_ylim(m-0.05, M+0.05)
    ax[0,0].set_title('$\omega$ = ${}^o$'.format(np.round(np.degrees(es0.w),1)),fontsize=ft)
    v_r, v_r_avg = rv(th,i,P=es1.P,w=np.degrees(es1.w))
    ax[0,1].plot(t, v_r,'magenta')
    ax[0,1].plot(t, v_r_avg*np.ones_like(t),"k--")
    ax[0,1].grid()
    #ax[0,1].set_ylim(m[1]-0.05, M[1]+0.05)
    ax[0,0].set_ylim(m-0.05, M+0.05)
    ax[0,1].set_title('$\omega$ = ${}^o$'.format(np.round(np.degrees(es1.w),1)),fontsize=ft)
    v_r, v_r_avg = rv(th,i,P=es2.P,w=np.degrees(es2.w))
    ax[1,0].plot(t, v_r,'darkcyan')
    ax[1,0].plot(t, v_r_avg*np.ones_like(t),"k--")
    ax[1,0].grid()
    #ax[1,0].set_ylim(m[2]-0.05, M[2]+0.05)
    ax[0,0].set_ylim(m-0.05, M+0.05)
    ax[1,0].set_title('$\omega$ = ${}^o$'.format(np.round(np.degrees(es2.w),1)),fontsize=ft)
    v_r, v_r_avg = rv(th,i,P=es3.P,w=np.degrees(es3.w))
    ax[1,1].plot(t, v_r,'green')
    ax[1,1].plot(t, v_r_avg*np.ones_like(t),"k--")
    ax[1,1].grid()
    #ax[1,1].set_ylim(m[3]-0.05, M[3]+0.05)
    ax[0,0].set_ylim(m-0.05, M+0.05)
    ax[1,1].set_title('$\omega$ = ${}^o$'.format(np.round(np.degrees(es3.w),1)),fontsize=ft)

    #leg = plt.legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=25)
    #for item in leg.legendHandles:
    #    item.set_visible(False)
    #for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +  + ax.get_yticklabels()):
        #item.set_fontsize(20)
    
    # Used to return the plot as an image rray
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image


# In[72]:


plt.imshow(vrt_collage(0.1,m,M))


# In[73]:



##################################
#####--- Input Variables ---######
##################################

dth = 0.01
th = np.arange(0.0, 4*np.pi,dth)
step = 0.01
e = np.arange(0,0.9+step,step)
M_star = 1*u.M_sun
w = np.array([0.0,30.0,60.0,90.0])
'''m = np.ones(w.size)*np.inf
M = np.ones(w.size)*(-np.inf)
for j in range(len(w)):
    for i in e:
        tmp,_ = rv(th,i,w[j])
        if(np.amin(tmp) < m[j]):
            m[j] = np.amin(tmp)
        if(np.amax(tmp) > M[j]):
            M[j] = np.amax(tmp)'''
m = np.inf
M = -np.inf
for j in range(len(w)):
    for i in e:
        tmp,_ = rv(th,i,w[j])
        if(np.amin(tmp) < m):
            m = np.amin(tmp)
        if(np.amax(tmp) > M):
            M = np.amax(tmp)
print(M,m)
matplotlib.rc('text', usetex=True) #use latex for text

# add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

kwargs_write = {'fps':100.0, 'quantizer':'nq'}
imageio.mimsave('./variation_of_rv_curve_with_e_collage.gif', [vrt_collage(i,m,M) for i in e], fps=5)

##################################
#######--- End of Code ---########
##################################


# In[110]:



##################################
#####--- Input Variables ---######
##################################

dth = 0.01
th = np.arange(0.0, 4*np.pi,dth)
step = 0.01
e = np.arange(0,0.9+step,step)
M_star = 1*u.M_sun
m = np.inf
M = -np.inf
for j in range(len(w)):
    for i in e:
        tmp,_ = rv(th,i,w[j])
        if(np.amin(tmp) < m):
            m = np.amin(tmp)
        if(np.amax(tmp) > M):
            M = np.amax(tmp)

        
th = np.arange(0,4*np.pi,0.01)
es0 = orbit(P=5*u.yr,w=0.0)
es1 = orbit(P=5*u.yr,w=30.0)
es2 = orbit(P=5*u.yr,w=60.0)
es3 = orbit(P=5*u.yr,w=90.0)
t = es0.dt(es0.w_v(th,i),dth)
t = np.cumsum(t).to(u.yr).value
t = np.append([0],t[:-1])

import matplotlib.animation as animation

ft = 15
# create a figure with two subplots
#fig, (ax1, ax2, ax3, ax4) = plt.subplots(2,2)
fig, (ax1, ax2) = plt.subplots(2,2, figsize=(8,6), sharex=True, sharey=True)
# intialize two line objects (one in each axes)
line1, = ax1[0].plot([], [], lw=2,color='orangered')
line11, = ax1[0].plot([], [],'k--')
line2, = ax1[1].plot([], [], lw=2,color='magenta')
line22, = ax1[1].plot([], [],'k--')
line3, = ax2[0].plot([], [], lw=2,color='darkcyan')
line33, = ax2[0].plot([], [],'k--')
line4, = ax2[1].plot([], [], lw=2,color='green')
line44, = ax2[1].plot([], [],'k--')
line = [line1,line11, line2,line22, line3,line33, line4,line44]

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("Time (in Earth years)",fontsize=ft,labelpad=ft-5)
plt.ylabel("Radial Velocity (km/s)",fontsize=ft,labelpad=ft)
plt.rcParams['xtick.labelsize']=ft-5
plt.rcParams['ytick.labelsize']=ft-5
fig.suptitle("$a$ = 0.05 AU, P = 5 Earth years, e = {}".format(np.round(i,2)),fontsize=ft-5)
#fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.subplots_adjust(top=0.88)
#fig.tight_layout(rect=(0,0,0,4))
# the same axes initalizations as before (just now we do it for both of them)
for ax in [ax1, ax2]:
  for j in range(2):
    ax[j].set_ylim(m-0.05, M+0.05)
    ax[j].set_xlim(np.amin(t)-0.5, np.amax(t)+0.5)
    ax[j].grid()

ax1[0].set_title('$\omega$ = ${}^o$'.format(np.round(np.degrees(es0.w),1)),fontsize=ft)
ax1[1].set_title('$\omega$ = ${}^o$'.format(np.round(np.degrees(es1.w),1)),fontsize=ft)
ax2[0].set_title('$\omega$ = ${}^o$'.format(np.round(np.degrees(es2.w),1)),fontsize=ft)
ax2[1].set_title('$\omega$ = ${}^o$'.format(np.round(np.degrees(es3.w),1)),fontsize=ft)

def animate(i):
    ft = 30
    i = e[i]
    # Data for plotting
    th = np.arange(0,4*np.pi,0.01)
    es0 = orbit(P=5*u.yr,w=0.0)
    es1 = orbit(P=5*u.yr,w=30.0)
    es2 = orbit(P=5*u.yr,w=60.0)
    es3 = orbit(P=5*u.yr,w=90.0)
    t = es0.dt(es0.w_v(th,i),dth)
    t = np.cumsum(t).to(u.yr).value
    t = np.append([0],t[:-1])
    v_r, v_r_avg = rv(th,i,P=es0.P,w=np.degrees(es0.w))
    line[0].set_data(t, v_r)
    line[1].set_data(t, v_r_avg)
    v_r, v_r_avg = rv(th,i,P=es1.P,w=np.degrees(es1.w))
    line[2].set_data(t, v_r)
    line[3].set_data(t, v_r_avg)
    v_r, v_r_avg = rv(th,i,P=es2.P,w=np.degrees(es2.w))
    line[4].set_data(t, v_r)
    line[5].set_data(t, v_r_avg)
    v_r, v_r_avg = rv(th,i,P=es3.P,w=np.degrees(es3.w))
    line[6].set_data(t, v_r)
    line[7].set_data(t, v_r_avg)
    fig.suptitle("$a$ = 0.05 AU, P = 5 Earth years, e = {}".format(np.round(i,2)),fontsize=ft-5)
    # axis limits checking. Same as before, just for both axes

    return line

anim = FuncAnimation(fig, animate, frames=len(e), interval=100, blit=True)


# In[111]:


#plt.rcParams['savefig.bbox'] = 'tight' 
#fig.tight_layout()
anim.save('sine_wave.gif', writer='imagemagick')


# In[109]:


m

