##################################
###--- Importing Libraries ---####
##################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import lombscargle
from astropy.stats import LombScargle
from astropy import timeseries
from scipy.optimize import curve_fit,fsolve
from astropy import constants as const
from astropy import units as u
from scipy import stats

##################################
####--- Function and Class ---####
####---     Defintion     ---#####
##################################

def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

def stopping_criteria(x1,x2,stop_type):
	if(stop_type == "abs"):
		return np.abs(x2 - x1)
	if(stop_type == "rel"):
		return np.abs(x2/x1 - 1.0)

def bisection(f,x1=0.0,x2=2*np.pi,max_iters=1000,xtol=1e-6,ftol=1e-4,*args):
    if(f(x1,*args)*f(x2,*args)>0):
        print("Both f(x1) and f(x2) have same sign! There possibly does not exist a unique root in the given interval. ")
        return
    elif(f(x1,*args) == 0):
        return x1, 0
    elif(f(x2,*args) == 0):
        return x2, 0
    tol = stopping_criteria(x1,x2,"abs")
    x0 = (x1+x2)/2.0;
    ctr = 0
    while(stopping_criteria(x1,x2,"abs")>xtol and f(x0,*args) > ftol and ctr <max_iters):
        x1,x2 = np.where(f(x1,*args)*f(x0,*args) < 0,(x1,x0),(x0,x2))
        ctr += 1
    return x0,ctr

def get_vr(t,K,P,e,w,tau,vr0,return_theta=False):
  E = get_E((t-tau)*2*np.pi/P,e)
  theta = 2*np.arctan2(np.sqrt(1.+e)*np.sin(0.5*E),
                       np.sqrt(1.-e)*np.cos(0.5*E))
  if(not return_theta):
    return K*(np.cos(theta+w)+e*np.cos(w))+vr0
  else:
    return K*(np.cos(theta+w)+e*np.cos(w))+vr0, theta


def get_E(M,e):
  E = []
  for i,mi in enumerate(M):
    x1 = mi - e
    x2 = mi + e
    E_tmp = bisection(lambda E: E-e*np.sin(E)-mi,x1,x2,1000,1e-6,1e-4)[0]
    E.append(E_tmp)
  return np.array(E)

def reduced_chi_square(x,y,s,m): # ddof = v
  v = x.size - m
  chi2 = (np.sum((x-y)**2/s**2))/v
  p = 1 - stats.chi2.cdf(chi2, v)
  return chi2,p

##################################
###---     Data loading    ---####
###---  and Visualization  ---####
##################################

data = np.loadtxt("./RV_51Pegasi_Data.txt",dtype = [("time","f8"),("vr","f8"),("dvr","f8")])
time = data["time"].astype(float) # time in days
vr = data["vr"].astype(float) # vr in m/s
dvr = data["dvr"].astype(float) # dvr in m/s
JD_ref = 2450000
time_jd = time+JD_ref
time0 = time-time[0]
ft = 17.5
fig = plt.figure(figsize=(10,6))
plt.errorbar(time+JD_ref,vr,yerr=dvr,fmt='o',ls='-',lw=0.3, capsize=4,ecolor='k',elinewidth=2)
plt.xlabel("Time in Barycentric Julian Date",fontsize=ft)
plt.ylabel("Radial Velocity (in m/s)",fontsize=ft)
plt.title("Radial Velocity Data vs Time",fontsize=ft)
plt.xticks(fontsize=ft)
plt.yticks(fontsize=ft)
plt.grid()
plt.savefig("rvdata_wline.png",bbox_inches="tight")
plt.close()

##################################
###--- Periodogram Analysis ---###
##################################

frequency,power = timeseries.LombScargle(time_jd,vr,dvr).autopower()
ft = 20
fig = plt.figure(figsize=(8,6))
plt.semilogx(1/frequency,power)
plt.title('Lomb-Scargle Periodogram for 51 Pegasi b',fontsize=ft)
plt.xlabel('Period (in days)',fontsize=ft)
plt.ylabel('Power Spectral Density',fontsize=ft)
plt.xticks(fontsize=ft)
plt.yticks(fontsize=ft)
plt.grid()
plt.savefig("periodogram_astropy.png",bbox_inches="tight")
plt.close()

tp = 1/frequency
idx = np.argmax(power)-1
delta = []
for i in range(2):
  delta.append(tp[idx-i]-tp[idx-i+1])
dP = np.amax(delta)
dP = dP*(u.day)

# find period of strongest signal
period = 1/frequency[np.argmax(power)]
P = period*u.day
print("Possible Period of given time-series radial velocity data is : {} +/- {} Earth Days".format(np.round(P.value,4),np.round(dP.value,4)))

##################################
###---    Analysis with     ---### 
###---     astropy.stats    ---###
###---     LombScargle      ---###
##################################

print("Initial Analysis with astropy.stats's LombScargle \n")
tfit = np.linspace(0,period,1000)
rvfit = LombScargle(time0,vr,dvr).model(tfit,1/period)
semi_amplitude = 0.5*(np.max(rvfit)-np.min(rvfit))
print("The fit semi-amplitude is %10.5f m/s" % semi_amplitude)
phase = (time0 % period)
voffset = np.mean(rvfit)
print("The velocity offset is %10.5f m/s" % voffset)

##################################
###--- Plotting Time folded ---###
##--- Radial Velocity Curves ---##
##################################

ft = 17.5
fig = plt.figure(figsize=(8,6))
plt.errorbar((time-time[0])%period,vr,dvr,fmt='.b',capsize=4,ecolor='k',elinewidth=1.3)
plt.title(r'Time-Folded Radial Velocity Curve',fontsize=ft)
ax = plt.gca()
ax.grid(True)
ax.axhline(0, color='black', lw=2)
ax.axvline(0, color='black', lw=2)
plt.xticks(fontsize=ft)
plt.yticks(fontsize=ft)
plt.xlabel("Time (in days)",fontsize=ft)
plt.ylabel("Radial Velocity (in m/s)",fontsize=ft)
plt.savefig("time_folded_rv_curve_astropy.png",bbox_inches="tight")
plt.close()

##################################
###---   Synthetic Radial   ---###
##--- Velocity Curve Fitting ---##
##################################

# Using Scipy's Curve Fit

print("RVC fitting using Scipy's Curve Fit \n")

K = semi_amplitude
P = period
e = 0.
w = 0.
tau = time0[0]
vr0 = voffset
guess = (K,e,w,vr0)
modified_get_vr = lambda jd,K,e,w,vr0: get_vr(jd,K,P,e,w,tau,vr0)
rvfit = modified_get_vr(time,K,e,w,vr0)
chisq = np.sum(((vr-rvfit)/dvr)**2)
print("Chi-squared of initial guess is %10.5f" % chisq)

popt,pcov = curve_fit(modified_get_vr,time0,vr,sigma=dvr,absolute_sigma=True,p0=guess)

(K,e,w,vr0) = popt
rvfit_discrete = get_vr(time0,K,P,e,w,tau,vr0)
red_chisq,p = reduced_chi_square(rvfit_discrete,vr,dvr,len(popt))
print("Reduced Chi-squared of least-squares fit is %10.5f" % red_chisq)

ft=17.5
fig1 = plt.figure(1,figsize=(8,6))
#Plot Data-model
frame1 = fig1.add_axes((.1,.3,.8,.6))
plt.errorbar(phase,vr,dvr,fmt='.b',capsize=4,label="Data",ecolor='k',elinewidth=1.3)
plt.scatter(phase,rvfit_discrete,c='r',label="Best fit model from Lombscargle fitting")
frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
plt.ylabel("Radial Velocity (in m/s)",fontsize=ft)
plt.xticks(fontsize=ft)
plt.yticks(fontsize=ft)
plt.grid()
plt.legend(fontsize=ft-5)
plt.title("Folded RV curve with the best-fit model superimposed",fontsize=ft)
#Residual plot
difference = rvfit_discrete - vr
frame2=fig1.add_axes((.1,.1,.8,.2))        
plt.errorbar(phase,difference,yerr=dvr,fmt='r.',capsize=4,ecolor='b',elinewidth=1.3)
plt.ylabel("Residue",fontsize=ft)
plt.xlabel("Time (in days)",fontsize=ft)
plt.xticks(fontsize=ft)
plt.yticks(fontsize=ft)
plt.grid()
plt.savefig("astropy_curve_fit_scatter.png",bbox_inches="tight")
plt.close()

tfit = np.linspace(0,P,1000)
rvfit_continuous = get_vr(tfit,K,P,e,w,tau,vr0)

ft=17.5
fig1 = plt.figure(1,figsize=(8,6))
#Plot Data-model
frame1 = fig1.add_axes((.1,.3,.8,.6))
plt.errorbar(phase,vr,dvr,fmt='.b',capsize=4,label="Data",ecolor='k',elinewidth=1.3)
plt.plot(tfit,rvfit_continuous,'-r',label="Best fit model from Lombscargle fitting")
frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
plt.ylabel("Radial Velocity (in m/s)",fontsize=ft)
plt.xticks(fontsize=ft)
plt.yticks(fontsize=ft)
plt.grid()
plt.legend(fontsize=ft-5)
plt.title("Folded RV curve with the best-fit model superimposed",fontsize=ft)
#Residual plot
difference = rvfit_discrete - vr
frame2=fig1.add_axes((.1,.1,.8,.2))        
plt.errorbar(phase,difference,yerr=dvr,fmt='b.',capsize=4,ecolor='k',elinewidth=1.3)
plt.ylabel("Residue",fontsize=ft)
plt.xlabel("Time (in days)",fontsize=ft)
plt.xticks(fontsize=ft)
plt.yticks(fontsize=ft)
plt.grid()
plt.savefig("astropy_curve_fit.png",bbox_inches="tight")
plt.close()

if e<0:
  w -= np.pi
  e *= -1

if K<0:
  K *= -1
  w += np.pi

dK   = np.sqrt(pcov[0,0])
de   = np.sqrt(pcov[1,1])
dw   = np.sqrt(pcov[2,2])
dvr0   = np.sqrt(pcov[3,3])
P = period*u.day
P_error = dP.value
M =  1.11*u.M_sun 
dM = 0.06*u.M_sun
G = const.G
a = (((P**2)*G*M)/(2*np.pi)**2)**(1/3) 
a = a.to(u.au)
da = a*np.sqrt( ((2./3)*(dP/P))**2 + ((1./3)*(dM/M))**2)
da = da.to(u.au)
mpsini = np.cbrt(((M.to(u.kg)**2)*P.to(u.s))/(2*np.pi*G))*K*(u.m/u.s)
mpsini = mpsini.to(u.Mjup)
d_mpsini = (mpsini)*np.sqrt(((2./3)*(dM/M))**2 + ((1./3)*(dP/P))**2)
asini = (K*(u.m/u.s)*P*np.sqrt(1-e**2))/(2*np.pi)
dasini = asini*np.sqrt((np.abs(dK/K))**2 + np.abs(dP/P)**2)
w_deg = w*180/np.pi
dw_deg = dw*180/np.pi

final_param_values = {"vr0":"{} +/= {} m/s".format(np.round(vr0,3),np.round(dvr0,3)), "mpsini": "{} +/- {} M_Jup".format(np.round(mpsini.value,3),np.round(d_mpsini.value,3)), "Orbital separation": "{} +/- {} AU".format(np.round(a.value,4),np.round(da.value,4)), "Period": "{} +/- {} Earth Days".format(np.round(P.value,4),np.round(dP.value,4)), "asini (a = a_star)": "{} +/- {} m".format(np.round(asini.to(u.m).value,3),np.round(dasini.to(u.m).value,3)), "radial velocity semi-amplitude": "{} +/- {} m/s".format(np.round(K,2),np.round(dK,2)), "angle of periastron": "{} +/- {} ".format(np.round(w_deg,2),np.round(dw_deg,2)), "eccentricity": "{} +/- {}".format(np.round(e,3),np.round(de,3)) }

print(final_param_values)

# phase folded rv curve
E = get_E(phase/((2*np.pi)/P.to(u.day).value),e)
_,theta = get_vr(phase,K,P.value,e,w,tau,vr0,True)
ft = 17.5
fig = plt.figure(figsize=(8,6))
plt.errorbar(theta,vr,yerr=dvr,fmt='.b',capsize=4,ecolor='k',elinewidth=1.3)
plt.title(r'Phase-Folded Radial Velocity Curve',fontsize=ft)
ax = plt.gca()
ax.grid(True)
ax.axhline(0, color='black', lw=2)
ax.axvline(0, color='black', lw=2)
ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
plt.xticks(fontsize=ft)
plt.yticks(fontsize=ft)
plt.xlabel("Phase (in radians)",fontsize=ft)
plt.ylabel("Radial Velocity (in m/s)",fontsize=ft)
plt.savefig("phase_folded_rv_curve_astropy.png",bbox_inches="tight")
plt.close()

##################################
#######--- End of Code ---########
##################################
