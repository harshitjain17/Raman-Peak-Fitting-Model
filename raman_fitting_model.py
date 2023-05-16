import os
import numpy as np
import pandas as pd
from ramanfitter_stage2 import RamanFitter_stage_2
from ramanfitter_stage1 import RamanFitter_stage_1

# Filename of the spectral file 1
filename_spectral_1 = 'Fluid_Coke_as-is_532nm'

# Parse the .txt file of spectral file 1
file_spectral_1 = os.path.join('results', filename_spectral_1, f'{filename_spectral_1}.txt') # Get File
curves_data_spectral_1 = np.genfromtxt(file_spectral_1)                                      # Open File

# Extract the Raman shift and intensity values of spectral file 1
raman_shift_spectral_1 = curves_data_spectral_1[:, 0] # Parse x-values - typically cm^-1 or nm values
intensity_spectral_1   = curves_data_spectral_1[:, 1] # Parse y-values - typically intensity or counts

# Using zip() to create a dictionary of (raman_shift: intensity pairs) of spectral file 1
txt_file_dictionary_spectral_1 = dict(zip(raman_shift_spectral_1, intensity_spectral_1))

# Load the user input for the bounds of the center of the peak from an Excel file of spectral file 1
center_bounds_path_spectral_1 = f'results/{filename_spectral_1}/center_bounds_{filename_spectral_1}.xlsx'
bounds_df_spectral_1 = pd.read_excel(center_bounds_path_spectral_1)

# Create a dictionary to store the bounds for each peak of spectral file 1
center_bounds_spectral_1 = {}

# Loop through the rows of the input file and extract the bounds for each peak of spectral file 1
for i, row in bounds_df_spectral_1.iterrows():
    peak_index = row['Peak Index']
    center_min = row['Center Min']
    center_max = row['Center Max']
    type       = row['Type']
    center_bounds_spectral_1[peak_index] = [center_min, center_max, type]

# handling the spectral file 2, if exists
spectral_2 = False
try:
    filename_spectral_2 = ''
    file_spectral_2 = os.path.join('results', filename_spectral_2, f'{filename_spectral_2}.txt')
    curves_data_spectral_2 = np.genfromtxt(file_spectral_2)
    raman_shift_spectral_2 = curves_data_spectral_2[:, 0] 
    intensity_spectral_2   = curves_data_spectral_2[:, 1] 
    txt_file_dictionary_spectral_2 = dict(zip(raman_shift_spectral_2, intensity_spectral_2))
    center_bounds_path_spectral_2 = f'results/{filename_spectral_2}/center_bounds_{filename_spectral_2}.xlsx'
    bounds_df_spectral_2 = pd.read_excel(center_bounds_path_spectral_2)
    center_bounds_spectral_2 = {}
    for i, row in bounds_df_spectral_2.iterrows():
        peak_index = row['Peak Index']
        center_min = row['Center Min']
        center_max = row['Center Max']
        type       = row['Type']
        center_bounds_spectral_2[peak_index] = [center_min, center_max, type]
    spectral_2 = True
except:
    print("2nd spectral file not found!")

# Run Stage 2 code on Spectral 1 
raman_fitter_spectral_1 = RamanFitter_stage_2(
        x                   = raman_shift_spectral_1,        # a 1D array of the x-axis values
        y                   = intensity_spectral_1,          # a 1D array of the y-axis values
        autorun             = False,                         # attempt to calculate all steps and fit a model
        threshold           = 12,                            # helps the code determine what is a worthy peak and what isn't
        PercentRange        = 0.11,                          # determines what percent error the fit model can have in regards to the amplitude and position of fit curves under found peaks
        Sigma               = 15,                            # the expected width of fit curves, in terms of data points
        SigmaMin            = 3,                             # the expected minimum width allowed of fit curves, in terms of data points
        SigmaMax            = 50,                            # the expected maximum width allowed of fit curves, in terms of data points
        txt_file_dictionary = txt_file_dictionary_spectral_1,# a dictionary of (raman_shift: intensity pairs)
        center_bounds       = center_bounds_spectral_1,      # a dictionary version of peak_index, bounds, and type_of_curve (which type of curve to use for peaks. Options are 'Lorentzian', 'Gaussian', and 'Voigt')
        filename            = filename_spectral_1            # name of the current file being examined
)


# Normalizes `y` data to 1
raman_fitter_spectral_1.NormalizeData()

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
raman_fitter_spectral_1.FindPeaks(
        DistBetweenPeaks    = 1,                    # minimum distance between peaks, in terms of data points
        showPlot            = True                  # this will show a plot of the found peaks
    )


# Fits the data with associated curve types
raman_fitter_spectral_1.FitData(
        showPlot      = True                        # this will show a plot of the fit data
    )

g_peak_of_spectral_1 = raman_fitter_spectral_1.get_g_peak()
intensity_y_values_spectral_1 = raman_fitter_spectral_1.get_intensity_y_values()
raman_shift_x_values_spectral_1 = raman_fitter_spectral_1.get_raman_shift_x_values()

if (spectral_2):
    raman_fitter_spectral_2 = RamanFitter_stage_2(
            x                   = raman_shift_spectral_2,        # a 1D array of the x-axis values
            y                   = intensity_spectral_2,          # a 1D array of the y-axis values
            autorun             = False,                         # attempt to calculate all steps and fit a model
            threshold           = 12,                            # helps the code determine what is a worthy peak and what isn't
            PercentRange        = 0.11,                          # determines what percent error the fit model can have in regards to the amplitude and position of fit curves under found peaks
            Sigma               = 15,                            # the expected width of fit curves, in terms of data points
            SigmaMin            = 3,                             # the expected minimum width allowed of fit curves, in terms of data points
            SigmaMax            = 50,                            # the expected maximum width allowed of fit curves, in terms of data points
            txt_file_dictionary = txt_file_dictionary_spectral_2,# a dictionary of (raman_shift: intensity pairs)
            center_bounds       = center_bounds_spectral_2,      # a dictionary version of peak_index, bounds, and type_of_curve (which type of curve to use for peaks. Options are 'Lorentzian', 'Gaussian', and 'Voigt')
            filename            = filename_spectral_2            # name of the current file being examined
    )

    # Normalizes `y` data to 1
    raman_fitter_spectral_2.NormalizeData()

    # Find the peaks in the data
    raman_fitter_spectral_2.FindPeaks(
            DistBetweenPeaks    = 1,                    # minimum distance between peaks, in terms of data points
            showPlot            = True                  # this will show a plot of the found peaks
        )

    # Fits the data with associated curve types
    raman_fitter_spectral_2.FitData(
            showPlot      = True                        # this will show a plot of the fit data
        )

    g_peak_of_spectral_2 = raman_fitter_spectral_2.get_g_peak()
    intensity_y_values_spectral_2 = raman_fitter_spectral_2.get_intensity_y_values()
    raman_shift_x_values_spectral_2 = raman_fitter_spectral_2.get_raman_shift_x_values()

    if abs(g_peak_of_spectral_1 - g_peak_of_spectral_2) < 2:

        # Run Stage 1 code on Spectral 1
        raman_fitter_spectral_1 = RamanFitter_stage_1(
            x                   = raman_shift_spectral_1,        # a 1D array of the x-axis values
            y                   = intensity_spectral_1,          # a 1D array of the y-axis values
            autorun             = False,                         # attempt to calculate all steps and fit a model
            threshold           = 12,                            # helps the code determine what is a worthy peak and what isn't
            PercentRange        = 0.11,                          # determines what percent error the fit model can have in regards to the amplitude and position of fit curves under found peaks
            Sigma               = 15,                            # the expected width of fit curves, in terms of data points
            SigmaMin            = 3,                             # the expected minimum width allowed of fit curves, in terms of data points
            SigmaMax            = 50,                            # the expected maximum width allowed of fit curves, in terms of data points
            txt_file_dictionary = txt_file_dictionary_spectral_1,# a dictionary of (raman_shift: intensity pairs)
            center_bounds       = center_bounds_spectral_1,      # a dictionary version of peak_index, bounds, and type_of_curve (which type of curve to use for peaks. Options are 'Lorentzian', 'Gaussian', and 'Voigt')
            filename            = filename_spectral_1            # name of the current file being examined
        )

        # Normalizes `y` data to 1
        raman_fitter_spectral_1.NormalizeData()

        # Find the peaks in the data
        raman_fitter_spectral_1.FindPeaks(
                DistBetweenPeaks    = 1,                    # minimum distance between peaks, in terms of data points
                showPlot            = True                  # this will show a plot of the found peaks
        )

        # Fits the data with associated curve types
        raman_fitter_spectral_1.FitData(
                showPlot      = True                        # this will show a plot of the fit data
        )

        intensity_y_values_spectral_1 = raman_fitter_spectral_1.get_intensity_y_values()
        raman_shift_x_values_spectral_1 = raman_fitter_spectral_1.get_raman_shift_x_values()

        # Run Stage 1 code on Spectral 2
        raman_fitter_spectral_2 = RamanFitter_stage_1(
            x                   = raman_shift_spectral_2,        # a 1D array of the x-axis values
            y                   = intensity_spectral_2,          # a 1D array of the y-axis values
            autorun             = False,                         # attempt to calculate all steps and fit a model
            threshold           = 12,                            # helps the code determine what is a worthy peak and what isn't
            PercentRange        = 0.11,                          # determines what percent error the fit model can have in regards to the amplitude and position of fit curves under found peaks
            Sigma               = 15,                            # the expected width of fit curves, in terms of data points
            SigmaMin            = 3,                             # the expected minimum width allowed of fit curves, in terms of data points
            SigmaMax            = 50,                            # the expected maximum width allowed of fit curves, in terms of data points
            txt_file_dictionary = txt_file_dictionary_spectral_2,# a dictionary of (raman_shift: intensity pairs)
            center_bounds       = center_bounds_spectral_2,      # a dictionary version of peak_index, bounds, and type_of_curve (which type of curve to use for peaks. Options are 'Lorentzian', 'Gaussian', and 'Voigt')
            filename            = filename_spectral_2            # name of the current file being examined
        )

        # Normalizes `y` data to 1
        raman_fitter_spectral_2.NormalizeData()

        # Find the peaks in the data
        raman_fitter_spectral_2.FindPeaks(
                DistBetweenPeaks    = 1,                    # minimum distance between peaks, in terms of data points
                showPlot            = True                  # this will show a plot of the found peaks
        )

        # Fits the data with associated curve types
        raman_fitter_spectral_2.FitData(
                showPlot      = True                        # this will show a plot of the fit data
        )

        intensity_y_values_spectral_2 = raman_fitter_spectral_2.get_intensity_y_values()
        raman_shift_x_values_spectral_2 = raman_fitter_spectral_2.get_raman_shift_x_values()

components_spectral_1  = raman_fitter_spectral_1.comps      # Returns a dictionary of each curve plot
curveParams_spectral_1 = raman_fitter_spectral_1.params     # Returns a dictionary of the parameters of each Lorentzian, Gaussian, or Voigt curve
bestFitLine_spectral_1 = raman_fitter_spectral_1.fit_line   # Returns the plot data of the model

if (spectral_2):
    components_spectral_2  = raman_fitter_spectral_2.comps      # Returns a dictionary of each curve plot
    curveParams_spectral_2 = raman_fitter_spectral_2.params     # Returns a dictionary of the parameters of each Lorentzian, Gaussian, or Voigt curve
    bestFitLine_spectral_2 = raman_fitter_spectral_2.fit_line   # Returns the plot data of the model


'''--------------------------------------Results--------------------------------------'''
# DataFrame for fitted_data from the Spectral 1
df_fitted_data_spectral_1 = pd.DataFrame({'Raman Shift': raman_shift_x_values_spectral_1, 'Intensity': intensity_y_values_spectral_1, 'Best Fit Line': bestFitLine_spectral_1, 'Residual': (bestFitLine_spectral_1 - intensity_y_values_spectral_1)})
df_fitted_data_spectral_1.to_excel(f'results/{filename_spectral_1}/fitted_data_{filename_spectral_1}.xlsx', index=False)

if (spectral_2):
    # DataFrame for fitted_data from the Spectral 2
    df_fitted_data_spectral_2 = pd.DataFrame({'Raman Shift': raman_shift_x_values_spectral_2, 'Intensity': intensity_y_values_spectral_2, 'Best Fit Line': bestFitLine_spectral_2, 'Residual': (bestFitLine_spectral_2 - intensity_y_values_spectral_2)})
    df_fitted_data_spectral_2.to_excel(f'results/{filename_spectral_2}/fitted_data_{filename_spectral_2}.xlsx', index=False)


# get the (x,y) values for each curve in Spectral 1
curves_data_spectral_1 = {}
for key in components_spectral_1.keys():
    curves_data_spectral_1[key] = components_spectral_1[key]
curves_data_spectral_1['x'] = raman_shift_x_values_spectral_1

# DataFrame for curves_data from the Spectral 1
df_curves_data_spectral_1 = pd.DataFrame(curves_data_spectral_1)
df_curves_data_spectral_1.to_excel(f'results/{filename_spectral_1}/curves_data_{filename_spectral_1}.xlsx', index=False)

if (spectral_2):
    # get the (x,y) values for each curve in Spectral 2
    curves_data_spectral_2 = {}
    for key in components_spectral_2.keys():
        curves_data_spectral_2[key] = components_spectral_2[key]
    curves_data_spectral_2['x'] = raman_shift_x_values_spectral_2

    # DataFrame for curves_data from the Spectral 2
    df_curves_data_spectral_2 = pd.DataFrame(curves_data_spectral_2)
    df_curves_data_spectral_2.to_excel(f'results/{filename_spectral_2}/curves_data_{filename_spectral_2}.xlsx', index=False)