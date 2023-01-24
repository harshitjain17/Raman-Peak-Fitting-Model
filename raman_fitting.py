import time

import numpy as np # For data manipulation
import scipy # For data manipulation
import random
import matplotlib.pyplot as plt # For doing the plots
import lmfit
from lmfit.models import GaussianModel
import rampy as rp # Charles' libraries and functions


# IMPORTING AND LOOKING AT THE DATA
# get the spectrum to deconvolute, with skipping header and footer comment lines from the spectrometer
inputsp = np.genfromtxt("./data/LS4.txt",skip_header=20, skip_footer=43) 

# CREATE A NEW PLOT FOR SHOWING THE SPECTRUM
plt.figure(figsize=(5,5))
plt.plot(inputsp[:,0],inputsp[:,1],'k.',markersize=1)
plt.xlabel("Raman shift, cm$^{-1}$", fontsize = 12)
plt.ylabel("Normalized intensity, a. u.", fontsize = 12)
plt.title("Fig. 1: the raw data",fontsize = 12,fontweight="bold")

# BASELINE REMOVAL
bir = np.array([(860,874),(1300,1330)]) # The regions where the baseline will be fitted
y_corr, y_base = rp.baseline(inputsp[:,0],inputsp[:,1],bir,'poly',polynomial_order=3)# We fit a polynomial background.

# SIGNAL SELECTION
lb = 867 # The lower boundary of interest
hb = 1300 # The upper boundary of interest
x = inputsp[:,0]
x_fit = x[np.where((x > lb)&(x < hb))]
y_fit = y_corr[np.where((x > lb)&(x < hb))]
ese0 = np.sqrt(abs(y_fit[:,0]))/abs(y_fit[:,0]) # the relative errors after baseline subtraction
y_fit[:,0] = y_fit[:,0]/np.amax(y_fit[:,0])*10 # normalise spectra to maximum intensity, easier to handle 
sigma = abs(ese0*y_fit[:,0]) #calculate good ese

# CREATE A NEW PLOT FOR SHOWING THE SPECTRUM
plt.figure()
plt.subplot(1,2,1)
inp = plt.plot(x,inputsp[:,1],'k-',label='Original')
corr = plt.plot(x,y_corr,'b-',label='Corrected') #we use the sample variable because it is not already normalized...
bas = plt.plot(x,y_base,'r-',label='Baseline')
plt.xlim(lb,1300)
plt.ylim(0,40000)
plt.xlabel("Raman shift, cm$^{-1}$", fontsize = 14)
plt.ylabel("Normalized intensity, a. u.", fontsize = 14)
plt.legend()
plt.title('A) Baseline removal')
plt.subplot(1,2,2)
plt.plot(x_fit,y_fit,'k.')
plt.xlabel("Raman shift, cm$^{-1}$", fontsize = 14)
plt.title('B) signal to fit')
#plt.tight_layout()
plt.suptitle('Figure 2', fontsize = 14,fontweight = 'bold')


def residual(pars, x, data=None, eps=None): #Function definition
    # unpack parameters, extract .value attribute for each parameter
    a1 = pars['a1'].value
    a2 = pars['a2'].value
    a3 = pars['a3'].value
    a4 = pars['a4'].value
    a5 = pars['a5'].value
    
    f1 = pars['f1'].value
    f2 = pars['f2'].value
    f3 = pars['f3'].value
    f4 = pars['f4'].value
    f5 = pars['f5'].value 
    
    l1 = pars['l1'].value
    l2 = pars['l2'].value
    l3 = pars['l3'].value
    l4 = pars['l4'].value
    l5 = pars['l5'].value
    
    # Using the Gaussian model function from rampy
    peak1 = rp.gaussian(x,a1,f1,l1)
    peak2 = rp.gaussian(x,a2,f2,l2)
    peak3 = rp.gaussian(x,a3,f3,l3)
    peak4 = rp.gaussian(x,a4,f4,l4)
    peak5 = rp.gaussian(x,a5,f5,l5)
    
    model = peak1 + peak2 + peak3 + peak4 + peak5 # The global model is the sum of the Gaussian peaks
    
    if data is None: # if we don't have data, the function only returns the direct calculation
        return model, peak1, peak2, peak3, peak4, peak5
    if eps is None: # without errors, no ponderation
        return (model - data)
    return (model - data)/eps # with errors, the difference is ponderated


# FITTING THE SPECTRUM
params = lmfit.Parameters()
#               (Name,  Value,  Vary,   Min,  Max,  Expr)
params.add_many(('a1',   2.4,   True,  0,      None,  None),
                ('f1',   946,   True, 910,    970,  None),
                ('l1',   26,   True,  20,      50,  None),
                ('a2',   3.5,   True,  0,      None,  None),
                ('f2',   1026,  True, 990,   1070,  None),
                ('l2',   39,   True,  20,   55,  None),  
                ('a3',   8.5,    True,    7,      None,  None),
                ('f3',   1082,  True, 1070,   1110,  None),
                ('l3',   31,   True,  25,   35,  None),  
                ('a4',   2.2,   True,  0,      None,  None),
                ('f4',   1140,  True, 1110,    1160,  None),
                ('l4',   35,   True,  20,   50,  None),  
                ('a5',   2.,   True,  0,      None,  None),
                ('f5',   1211,  True, 1180,   1220,  None),
                ('l5',   28,   True,  20,   45,  None))

# we constrain the positions
params['f1'].vary = False
params['f2'].vary = False
params['f3'].vary = False
params['f4'].vary = False
params['f5'].vary = False

algo = 'nelder'  
    
result = lmfit.minimize(residual, params, method = algo, args=(x_fit, y_fit[:,0])) # fit data with  nelder model from scipy

# we release the positions but contrain the FWMH and amplitude of all peaks 
params['f1'].vary = True
params['f2'].vary = True
params['f3'].vary = True
params['f4'].vary = True
params['f5'].vary = True

#we fit twice
result2 = lmfit.minimize(residual, params,method = algo, args=(x_fit, y_fit[:,0])) # fit data with leastsq model from scipy

model = lmfit.fit_report(result2.params)
yout, peak1,peak2,peak3,peak4,peak5 = residual(result2.params,x_fit) # the different peaks
rchi2 = (1/(float(len(y_fit))-15-1))*np.sum((y_fit - yout)**2/sigma**2) # calculation of the reduced chi-square


##### WE DO A NICE FIGURE THAT CAN BE IMPROVED FOR PUBLICATION
plt.plot(x_fit,y_fit,'k-')
plt.plot(x_fit,yout,'r-')
plt.plot(x_fit,peak1,'b-')
plt.plot(x_fit,peak2,'b-')
plt.plot(x_fit,peak3,'b-')
plt.plot(x_fit,peak4,'b-')
plt.plot(x_fit,peak5,'b-')
    
plt.xlim(lb,hb)
plt.ylim(-0.5,10.5)
plt.xlabel("Raman shift, cm$^{-1}$", fontsize = 14)
plt.ylabel("Normalized intensity, a. u.", fontsize = 14)
plt.title("Fig. 3: Fit of the Si-O stretch vibrations\n in LS4 with \nthe Nelder Mead algorithm ",fontsize = 14,fontweight = "bold")
print("rchi-2 = \n"+str(rchi2))