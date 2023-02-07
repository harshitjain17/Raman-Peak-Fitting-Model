# # Define initial parameters for the peaks
# # amplitude, center, fwhm, x-min, x-max
# peak_params = [
#     [1000, 200, 10, 150, 250],
#     [500, 300, 15, 250, 350]
# ]

# # # Define the parameters for the peaks
# # peaks = [
# #     {'type': 'Lorentzian', 'amplitude': 1000, 'x': 500, 'FWHM': 10},
# #     {'type': 'Gaussian', 'amplitude': 2000, 'x': 700, 'FWHM': 20},
# #     {'type': 'Voigt', 'amplitude': 1500, 'x': 600, 'FWHM_L': 15, 'FWHM_G': 25}
# # ]

# # # Fit the data with the defined peaks
# # fit = ramanfitter.fit(x, y_corr, peaks)


import os
import numpy as np
import pandas as pd
from ramanfitter import RamanFitter
from ramanfitter.mapper import Mapper

filename = os.path.join('GC_532nm.txt' ) # Get File
data = np.genfromtxt(filename) # Open File

# Extract the Raman shift and intensity values
raman_shift = data[:, 0] # Parse x-values - typically cm^-1 or nm values
intensity   = data[:, 1] # Parse y-values - typically intensity or counts

raman_fitter = RamanFitter(
        x            = raman_shift, # a 1D array of the x-axis values
        y            = intensity,   # a 1D array of the y-axis values
        autorun      = False,       # attempt to calculate all steps and fit a model
        threshold    = 12.,         # helps the code determine what is a worthy peak and what isn't
        PercentRange = 0.2,         # determines what percent error the fit model can have in regards to the amplitude and position of fit curves under found peaks
        Sigma        = 15,          # the expected width of fit curves, in terms of data points
        SigmaMin     = 3,           # the expected minimum width allowed of fit curves, in terms of data points
        SigmaMax     = 100          # the expected maximum width allowed of fit curves, in terms of data points
    )


''' Each step ran when autorun = False '''

# Normalizes `y` data to 1
raman_fitter.NormalizeData()

# Removes noise from input data
raman_fitter.Denoise(
        UseFFT          = True,     # a Fast Fourier Transform will be used to cutoff higher frequency noise and transformed back
        FFT_PS_Cutoff   = 1./1200., # this value is used to differentiate between noise and data
        UseSavgol       = True,     # this function will implement a Savitzky Golay filter to remove noise
        SavgolWindow    = 25,       # how many datapoints to iterate over in the Savitzky Golay filter
        SavgolOrder     = 3,        # what order of polynomial to use for the designated Savitzky Golay filter window
        ShowPlot        = True      # this will show a plot of the smoothed data
    )

# Find the peaks in the data
raman_fitter.FindPeaks(
        DistBetweenPeaks = 50,  # minimum distance between peaks, in terms of data points
        showPlot         = True # this will show a plot of the found peaks
    )


''' 
If you want to update the parameters for all curves,
you will need to loop through raman_fitter.params and update each curve's parameters individually.
'''
# for param_name, param in raman_fitter.params.items():
#     # update each curve's parameter individually
#     param.value = updated_value

# Fits the data with associated curve types
raman_fitter.FitData(
        type     = 'Lorentzian', # which type of curve to use for peak - options are 'Lorentzian', 'Gaussian', and 'Voigt'
        showPlot = True          # this will show a plot of the fit data
    )


components  = raman_fitter.comps      # Returns a dictionary of each curve plot
curveParams = raman_fitter.params     # Returns a dictionary of the parameters of each Lorentzian, Gaussian, or Voigt curve
bestFitLine = raman_fitter.fit_line   # Returns the plot data of the model

'''
This will create an .xlsx file named fitted_data.xlsx in the current working directory with three columns:
Raman Shift, Intensity, and Best Fit Line.
'''
# Create a DataFrame from the data
df = pd.DataFrame({'Raman Shift': raman_shift, 'Intensity': intensity, 'Best Fit Line': bestFitLine})

# Export the DataFrame to an .xlsx file
df.to_excel('fitted_data.xlsx', index=False)