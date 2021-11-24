##################################
###--- Importing Libraries ---####
##################################

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy import constants as const
from itertools import cycle

##################################
###--- Function Defintion ---####
##################################

def rv(theta, e=0, w=0, i=60, a=0.05*u.au, P=5*u.yr, theta_radians=True):  # angles -> degrees, theta -> radians
    #theta = np.radians(theta)
    w = np.radians(w)
    i = np.radians([i])
    k = ((2*np.pi*np.sin(i)*a.to(u.km))/(P.to(u.s)*np.sqrt(1-e**2))).value
    return k*( e*np.cos(w) + np.cos(w+theta) ).squeeze(), k*( e*np.cos(w)).squeeze()

##################################
#####--- Class Defintion ---######
##################################

class transit:
    
    def __init__(self,M = 1*const.M_sun,R = 1*const.R_sun,m = 1*const.M_earth,r = 1*const.R_earth,inc = [90],radians=False):
        self.M = M.to(u.kg)
        self.R = R.to(u.au)
        self.m = m.to(u.kg)
        self.r = r.to(u.au)
        if(radians==True):
            self.inc = np.array(inc)    
            self.inc_deg = self.inc * (180/np.pi)
        else:
            self.inc_deg = np.array(inc)
            self.inc = self.inc_deg*(np.pi/180)
        
    def get_t(self,i,N=1000,max_max_a = 10*u.au,yscale="day"):
        min_a = (self.R+self.r)
        max_a = np.abs((self.R+self.r)/np.cos(i))
        if(max_a > max_max_a):
            max_a = max_max_a
        a = np.linspace(min_a,max_a,N)
        P = np.sqrt(((a**3)*(4*np.pi**2))/(const.G*(self.m+self.M)))
        if(yscale=="day"):
            P = P.to(u.day)
        if(yscale=="hr"):
            P = P.to(u.hr)
        t = ((P.value/np.pi)*np.arcsin( (1/np.sin(i))*np.sqrt(((self.r+self.R)/a)**2 - np.cos(i)**2)) )
        return a,P,t
    
    def plot_and_save(self,savename,N=1000,max_max_a = 10*u.au,ft = 20,log=False,yscale="day",p_ref="earth"):
        lines = ["-","--","-.",":"]
        linecycler = cycle(lines)
        plt.figure(figsize=(8,6))
        for i,j in zip(self.inc,self.inc_deg):
            a,_,t = self.get_t(i,N,max_max_a,yscale)
            plt.plot(a,t,label="i = ${}^o$".format(j),ls=next(linecycler))
        plt.grid()
        plt.legend(fontsize=ft)
        plt.xlabel("Orbital Separation (in AU)",fontsize=ft)
        if(yscale=="day"):
            plt.ylabel("Transit Duration (in Earth days)",fontsize=ft)
        elif(yscale=="hr"):
            plt.ylabel("Transit Duration (in Earth hours)",fontsize=ft)
        if(log==True):
            plt.gca().set_xscale("log")
        plt.rcParams['xtick.labelsize']=ft+5
        plt.rcParams['ytick.labelsize']=ft+5
        if(p_ref=="earth"):
            text = "Transit duration curve for Earth-Sun like system \n"
            plt.title(text+"$R_S$ = {}$R_\odot$, $R_P$ = {}$R_\oplus$".format(np.round(self.R.to(u.R_sun).value,2),np.round(self.r.to(u.R_earth).value,2)), fontsize=ft)
        elif(p_ref == "jupiter"):
            text = "Transit duration curves for Jupiter-Sun like system \n for different orbital inclinations \n"
            plt.title(text+"\n $R_S$ = {}$R_\odot$, $R_P$ = {}$R_J$".format(np.round(self.R.to(u.R_sun).value,2),np.round(self.r.to(u.R_jup).value,2)), fontsize=ft)
        
        plt.savefig(savename,bbox_inches="tight")

#####################################
#--- Transit duration vs Orbital ---#
#####--- Separation Plot for ---#####
####--- Earth-Sun like system ---####
#####################################

es = transit()

# Plotting approximate curve

a,P,t = es.get_t(es.inc)
ax = plt.gca()
ax.plot(a,t,'-', label=r"$t_T = \frac{P}{\pi}sin^{-1}(\frac{R_S + R_P}{a})$")
tmp = (P*(es.r+es.R))/(np.pi*a)
ax.plot(a,tmp,'--',label=r"$t_T = \frac{(R_S+R_P)P}{a \pi} \propto \sqrt{a}$")#"")
plt.legend(fontsize=15)
plt.xlabel("Orbital Separation (in AU)",fontsize=15)
plt.ylabel("Transit Duration (in Earth days)",fontsize=15)
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20
plt.grid()
plt.title("Comparison of actual curve with approximate \n", fontsize=20)
plt.savefig("sqrt_a.png", bbox_inches="tight")

# plotting accurate curve
 
es.plot_and_save("3_a.png",10000,100*u.au)

# Verification

a = 1*u.au
R = 1*const.R_sun
r = 1*const.R_earth
P = 1*u.yr
inc = np.pi/2.
t = (P.to(u.day)/np.pi)*np.arcsin( (1/np.sin(inc))*np.sqrt( ((R+r)/a)**2 - np.cos(inc)**2 ) )
print("Transit Duration for Earth-Sun system in edge on configuration and circular orbit: {} Earth days".format(np.round(t.value,3)))

#####################################
#--- Transit duration vs Orbital ---#
#####--- Separation Plot for ---#####
###--- Jupiter-Sun like system ---###
#####################################

js = transit(m=const.M_jup,r=const.R_jup,inc=[90,89.9,89.5,89])
js.plot_and_save("3_b.png",max_max_a=10*u.au,log=True,yscale="hr",p_ref="jupiter")

##################################
#######--- End of Code ---########
##################################
