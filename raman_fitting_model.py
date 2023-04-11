import os
import numpy as np
import pandas as pd
from ramanfitter import RamanFitter
from ramanfitter.mapper import Mapper
import matplotlib.pyplot as plt

filename = 'GC_532nm'
file = os.path.join('results', filename, f'{filename}.txt') # Get File
curves_data = np.genfromtxt(file) # Open File

# Extract the Raman shift and intensity values
raman_shift = curves_data[:, 0] # Parse x-values - typically cm^-1 or nm values
intensity   = curves_data[:, 1] # Parse y-values - typically intensity or counts

# using zip() to create a dictionary of (raman_shift: intensity pairs)
txt_file_dictionary = dict(zip(raman_shift, intensity))

# Load the user input for the bounds of the center of the peak from an Excel file
center_bounds_path = f'results/{filename}/center_bounds_{filename}.xlsx'
bounds_df = pd.read_excel(center_bounds_path)

# Create a dictionary to store the bounds for each peak
center_bounds = {}

# Loop through the rows of the input file and extract the bounds for each peak
for i, row in bounds_df.iterrows():
    peak_index = row['Peak Index']
    center_min = row['Center Min']
    center_max = row['Center Max']
    type       = row['Type']
    center_bounds[peak_index] = [center_min, center_max, type]

raman_fitter = RamanFitter(
        x            = raman_shift, # a 1D array of the x-axis values
        y            = intensity,   # a 1D array of the y-axis values
        autorun      = False,       # attempt to calculate all steps and fit a model
        threshold    = 12,          # helps the code determine what is a worthy peak and what isn't
        PercentRange = 0.11,        # determines what percent error the fit model can have in regards to the amplitude and position of fit curves under found peaks
        Sigma        = 15,          # the expected width of fit curves, in terms of data points
        SigmaMin     = 3,           # the expected minimum width allowed of fit curves, in terms of data points
        SigmaMax     = 50           # the expected maximum width allowed of fit curves, in terms of data points
    )


# Normalizes `y` data to 1
raman_fitter.NormalizeData()

# # Removes noise from input data
# raman_fitter.Denoise(
#         UseFFT          = True,     # a Fast Fourier Transform will be used to cutoff higher frequency noise and transformed back
#         FFT_PS_Cutoff   = 1./1200., # this value is used to differentiate between noise and data
#         UseSavgol       = True,     # this function will implement a Savitzky Golay filter to remove noise
#         SavgolWindow    = 25,       # how many datapoints to iterate over in the Savitzky Golay filter
#         SavgolOrder     = 3,        # what order of polynomial to use for the designated Savitzky Golay filter window
#         ShowPlot        = True      # this will show a plot of the smoothed data
#     )

# Find the peaks in the data
raman_fitter.FindPeaks(
        txt_file_dictionary = txt_file_dictionary,  # a dictionary of (raman_shift: intensity pairs)
        center_bounds       = center_bounds,        # a dict center_bounds[peak_index] = [center_min, center_max, type] for bounds to find peak
        DistBetweenPeaks    = 1,                    # minimum distance between peaks, in terms of data points
        showPlot            = True                  # this will show a plot of the found peaks
    )


# Fits the data with associated curve types
raman_fitter.FitData(
        txt_file_dictionary = txt_file_dictionary,  # a dictionary of (raman_shift: intensity pairs)
        center_bounds = center_bounds,              # a dict center_bounds[peak_index] = [center_min, center_max, type] for bounds to find peak
        showPlot      = True                        # this will show a plot of the fit data
    )


components  = raman_fitter.comps      # Returns a dictionary of each curve plot
curveParams = raman_fitter.params     # Returns a dictionary of the parameters of each Lorentzian, Gaussian, or Voigt curve
bestFitLine = raman_fitter.fit_line   # Returns the plot data of the model



'''--------------------------------------Results--------------------------------------'''
# DataFrame for fitted_data from the data
df_fitted_data = pd.DataFrame({'Raman Shift': raman_shift, 'Intensity': intensity, 'Best Fit Line': bestFitLine, 'Residual': (bestFitLine - intensity)})
df_fitted_data.to_excel(f'results/{filename}/fitted_data_{filename}.xlsx', index=False)

# get the (x,y) values for each curve
curves_data = {}
for key in components.keys():
    curves_data[key] = components[key]
curves_data['x'] = raman_shift

# DataFrame for curves_data from the data
df_curves_data = pd.DataFrame(curves_data)
df_curves_data.to_excel(f'results/{filename}/curves_data_{filename}.xlsx', index=False)