##################################
###--- Importing Libraries ---####
##################################

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy import units as u
from matplotlib.pyplot import cm

##################################
###--- Function Defintion ---####
##################################

def rv(theta, e=0, w=0, i=60, a=0.05, P=5):  # angles -> degrees
    theta = np.radians(theta)
    w = np.radians(w)
    i = np.radians([i])
    k = ((2*np.pi*np.sin(i)*a.to(u.km))/(P.to(u.s)*np.sqrt(1-e**2))).value
    return k*( e*np.cos(w) + np.cos(w+theta) ).squeeze()

##################################
#####--- Input Variables ---######
##################################

a = 0.05 * u.au
P = 5 * u.yr
i = 60
w = np.array([0,30,60,90])
e= np.array([0,0.7])
theta = np.arange(0,720,0.01)

n_row, n_col = len(w),len(e)

matplotlib.rc('text', usetex=True) #use latex for text

# add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

##################################
#########--- Grid Plot---#########
##################################

plt.figure(figsize=(8 * n_col, 6 * n_row))
plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)

colors_list = ["red","green","blue","magenta","orangered","darkcyan","purple","darkolivegreen"]
color=iter(colors_list)
ctr = 1
for k in range(len(w)):
    for j in range(len(e)):
        c = next(color)
        vr = rv(theta=theta,e=e[j],w=w[k],i=i,a=a,P=P)
        t='e = ' +str(e[j])+'\n $\omega$ = {}$^o$'.format(w[k])
        plt.subplot(n_row, n_col, ctr)
        plt.xlabel("Phase of orbit (in degrees)",fontsize = 25)
        plt.ylabel("Radial Velocity (in km/s)",fontsize = 25)
        plt.rcParams['xtick.labelsize']=25
        plt.rcParams['ytick.labelsize']=25
        plt.ylim(np.amin(vr)-0.05, np.amax(vr)+0.05)
        plt.plot(theta, vr, c=c,label=t)
        plt.grid()
        leg = plt.legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=25,loc="lower right")
        for item in leg.legendHandles:
            item.set_visible(False)
        ctr += 1

plt.savefig("rvc.png", bbox_inches='tight')
plt.close()

###################################
#####--- Plots with fixed ---######
##--- angle of periastron and ---##
###--- variable eccentricity ---###
###################################

plt.figure(figsize=(8 * 1, 6 * n_row))
plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)

colors_list = ["red","green","blue","magenta","orangered","darkcyan","purple","darkolivegreen"]
color=iter(colors_list)
ctr = 1
for k in range(len(w)):
    plt.subplot(n_row, 1, ctr)
    min_lim = np.inf
    max_lim = -np.inf
    for j in range(len(e)):
        c = next(color)
        vr = rv(theta=theta,e=e[j],w=w[k],i=i,a=a,P=P)
        t='e = ' +str(e[j])
        plt.plot(theta, vr, c=c,label=t)
        plt.legend(fontsize=25)
        if(np.amin(vr)-0.05 < min_lim):
            min_lim = np.amin(vr)-0.05
        if(np.amax(vr)+0.05 > max_lim):
            max_lim = np.amax(vr)+0.05
        plt.legend(fontsize = 25)
    plt.text(450,max_lim-0.1,r"$\omega = {}$".format(w[k]),fontsize=25, bbox=dict(facecolor='none', edgecolor='black'))
    plt.xlabel("Phase of orbit (in degrees)",fontsize = 25)
    plt.ylabel("Radial Velocity (in km/s)",fontsize = 25)
    plt.grid()     
    plt.rcParams['xtick.labelsize']=25
    plt.rcParams['ytick.labelsize']=25
    plt.ylim(min_lim,max_lim)
    ctr += 1

plt.savefig("rvc_e.png", bbox_inches='tight')
plt.close()

###################################
#####--- Plots with fixed ---######
#--- eccentricity and variable ---#
####--- angle of periastron ---####
###################################

plt.figure(figsize=(8 * n_col, 6 * 1))
plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)

colors_list = ["red","green","blue","magenta","orangered","darkcyan","purple","darkolivegreen"]
color=iter(colors_list)
ctr = 1
for j in range(len(e)):
    plt.subplot(1, n_col, ctr)
    min_lim = np.inf
    max_lim = -np.inf
    for k in range(len(w)):
        c = next(color)
        vr = rv(theta=theta,e=e[j],w=w[k],i=i,a=a,P=P)
        t='$\omega$ = {}$^o$'.format(w[k])
        plt.plot(theta, vr, c=c,label=t)
        plt.grid()
        plt.legend(fontsize=25)
        if(np.amin(vr)-0.05 < min_lim):
            min_lim = np.amin(vr)-0.05
        if(np.amax(vr)+0.05 > max_lim):
            max_lim = np.amax(vr)+0.05
        plt.legend(fontsize=20)
    plt.text(450,max_lim-0.1,r"e = {}".format(e[j]),fontsize=25, bbox=dict(facecolor='none', edgecolor='black'))
    plt.xlabel("Phase of orbit (in degrees)",fontsize = 25)
    plt.ylabel("Radial Velocity (in km/s)",fontsize = 25)
    plt.rcParams['xtick.labelsize']=25
    plt.rcParams['ytick.labelsize']=25
    plt.ylim(min_lim,max_lim)
    plt.grid()
    ctr += 1

plt.savefig("rvc_w.png", bbox_inches='tight')
plt.close()

###################################
####--- Plotting variation ---#####
####--- of degree of offset ---####
#####--- with eccentricity ---#####
###################################

e = np.arange(0,1,0.01)
plt.plot(e,e/np.sqrt(1-e**2))
plt.xlabel("Eccentricity (e)",fontsize = 15)
plt.ylabel(r"$f(e) = \frac{e}{\sqrt{1-e^2}}$",fontsize = 15)
plt.title("Variation of offset in RV curve with eccentricity", fontsize=25)
plt.grid()
plt.savefig("offset.png", bbox_inches='tight')
plt.close()

##################################
#######--- End of Code ---########
##################################
