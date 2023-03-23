import os
import numpy as np
import pandas as pd
from ramanfitter import RamanFitter
from ramanfitter.mapper import Mapper
import matplotlib.pyplot as plt

filename = os.path.join('GC_633nm.txt') # Get File
data = np.genfromtxt(filename) # Open File

# Extract the Raman shift and intensity values
raman_shift = data[:, 0] # Parse x-values - typically cm^-1 or nm values
intensity   = data[:, 1] # Parse y-values - typically intensity or counts

# using zip() to create a dictionary of (raman_shift: intensity pairs)
txt_file_dictionary = dict(zip(raman_shift, intensity))

# Load the user input for the bounds of the center of the peak from an Excel file
bounds_df = pd.read_excel('centre_bounds2.xlsx')

# Create a dictionary to store the bounds for each peak
centre_bounds = {}

# Loop through the rows of the input file and extract the bounds for each peak
for i, row in bounds_df.iterrows():
    peak_index = row['Peak Index']
    center_min = row['Center Min']
    center_max = row['Center Max']
    type       = row['Type']
    centre_bounds[peak_index] = [center_min, center_max, type]

raman_fitter = RamanFitter(
        x            = raman_shift, # a 1D array of the x-axis values
        y            = intensity,   # a 1D array of the y-axis values
        autorun      = False,       # attempt to calculate all steps and fit a model
        threshold    = 12,          # helps the code determine what is a worthy peak and what isn't
        PercentRange = 0.11,        # determines what percent error the fit model can have in regards to the amplitude and position of fit curves under found peaks
        Sigma        = 15,          # the expected width of fit curves, in terms of data points
        SigmaMin     = 3,           # the expected minimum width allowed of fit curves, in terms of data points
        SigmaMax     = 300          # the expected maximum width allowed of fit curves, in terms of data points
    )


''' Each step ran when autorun = False '''

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
        centre_bounds       = centre_bounds,        # a dict centre_bounds[peak_index] = [center_min, center_max, type] for bounds to find peak
        DistBetweenPeaks    = 1,                    # minimum distance between peaks, in terms of data points
        showPlot            = True                  # this will show a plot of the found peaks
    )


# Fits the data with associated curve types
raman_fitter.FitData(
        # type          = 'Voigt',       # which type of curve to use for peak - options are 'Lorentzian', 'Gaussian', and 'Voigt'
        centre_bounds = centre_bounds, # a dict centre_bounds[peak_index] = [center_min, center_max, type] for bounds to find peak
        showPlot      = True           # this will show a plot of the fit data
    )


components  = raman_fitter.comps      # Returns a dictionary of each curve plot
curveParams = raman_fitter.params     # Returns a dictionary of the parameters of each Lorentzian, Gaussian, or Voigt curve
bestFitLine = raman_fitter.fit_line   # Returns the plot data of the model


'''
This will create an .xlsx file named fitted_data.xlsx in the current working directory with three columns:
Raman Shift, Intensity, and Best Fit Line.
'''
# Create a DataFrame from the data
df = pd.DataFrame({'Raman Shift': raman_shift, 'Intensity': intensity, 'Best Fit Line': bestFitLine, 'Residual': (bestFitLine - intensity)})

# Export the DataFrame to an .xlsx file
df.to_excel('fitted_data.xlsx', index=False)




'''
ABOUT THRESHOLD:

The threshold parameter is used to specify the minimum intensity threshold that a Raman peak must meet in order to be
detected and fitted.

When fitting Raman spectra, it is often necessary to set a threshold to filter out noise or unwanted signals that may be
present in the data. The threshold parameter allows you to specify a minimum intensity value that a peak must have in order
to be considered significant enough to be fitted.

Any peaks in the Raman spectrum that do not meet this threshold will be ignored and not included in the final fit.
Setting an appropriate threshold can help to improve the accuracy and reliability of Raman peak fitting by filtering out
low-intensity or noise-related peaks that may not be relevant to the sample being analyzed.

The threshold parameter is typically specified as a numerical value, representing the minimum intensity threshold required
for peak detection and fitting. The specific value used may depend on the characteristics of the Raman spectrum and the
specific analysis being performed.


ABOUT GAMMA:

The gamma parameter in the RamanFitter library in Python is a parameter that is used to model the linewidth of a Raman peak.
The gamma parameter is typically used in conjunction with other parameters, such as the peak position, intensity, and shape,
to model the Raman spectrum of a sample. The gamma parameter determines the width of the Raman peak in the fitted spectrum,
with larger values of gamma resulting in broader peaks. The gamma parameter can be set manually in the RamanFitter library or
can be automatically determined by the fitting algorithm.
'''