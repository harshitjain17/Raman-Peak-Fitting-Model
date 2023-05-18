import os
import numpy as np
import pandas as pd
from ramanfitter_stage2 import RamanFitter_stage_2
from ramanfitter_stage1 import RamanFitter_stage_1

def spectral_files_handler(spectral_file_1, spectral_file_2 = None):
    """
            spectral_files_handler(spectral_file_1, spectral_file_2 = None)

            Handles the spectral files ( File 1 and File 2 ). It loads the .txt file given by the user and
            extracts the Raman shift and intensity values of spectral file 1 and 2. It then loads .xlsx file
            which is the bounds of the center of the peak from an Excel file "center_bounds" of spectral file 1 and 2
            and creates a dictionary to store the center bounds for each peak of spectral file 1 and 2.

            Parameters
            ----------
            spectral_file_1 : str
                File name of the spectral file 1
            spectral_file_2 : str, optional, default: None
                File name of the spectral file 2
    """

    # Filename of the spectral file 1
    global filename_spectral_1
    filename_spectral_1 = spectral_file_1

    # Parse the .txt file of spectral file 1
    file_spectral_1 = os.path.join('results', filename_spectral_1, f'{filename_spectral_1}.txt') # Get File
    curve_data_spectral_1 = np.genfromtxt(file_spectral_1)                                       # Open File

    # Extract the Raman shift and intensity values of spectral file 1
    global raman_shift_spectral_1
    raman_shift_spectral_1 = curve_data_spectral_1[:, 0] # Parse x-values - typically cm^-1 or nm values
    global intensity_spectral_1
    intensity_spectral_1 = curve_data_spectral_1[:, 1] # Parse y-values - typically intensity or counts

    # Using zip() to create a dictionary of (raman_shift: intensity pairs) of spectral file 1
    global txt_file_dictionary_spectral_1
    txt_file_dictionary_spectral_1 = dict(zip(raman_shift_spectral_1, intensity_spectral_1))

    # Load the bounds of the center of the peak from an Excel file "center_bounds" of spectral file 1
    center_bounds_path_spectral_1 = f'results/{filename_spectral_1}/center_bounds_{filename_spectral_1}.xlsx'
    bounds_df_spectral_1 = pd.read_excel(center_bounds_path_spectral_1)

    # Create a dictionary to store the bounds for each peak of spectral file 1
    global center_bounds_spectral_1
    center_bounds_spectral_1 = {}

    # Loop through the rows of the input file and extract the bounds for each peak of spectral file 1
    for i, row in bounds_df_spectral_1.iterrows():
        peak_index = row['Peak Index']
        center_min = row['Center Min']
        center_max = row['Center Max']
        type       = row['Type']
        center_bounds_spectral_1[peak_index] = [center_min, center_max, type]

    # Handling the spectral file 2, if exists
    global spectral_2
    spectral_2 = False

    # Trying to search for Spectral 2 file
    try:
        global filename_spectral_2
        filename_spectral_2 = spectral_file_2                                                                     # Filename of the spectral file 2
        
        file_spectral_2 = os.path.join('results', filename_spectral_2, f'{filename_spectral_2}.txt')              # Parse the .txt file of spectral file 2
        curve_data_spectral_2 = np.genfromtxt(file_spectral_2)
        
        global raman_shift_spectral_2
        raman_shift_spectral_2 = curve_data_spectral_2[:, 0]                                                      # Extract the Raman shift of spectral file 2
        
        global intensity_spectral_2
        intensity_spectral_2   = curve_data_spectral_2[:, 1]                                                      # Extract the Intensity values of spectral file 2
        
        global txt_file_dictionary_spectral_2
        txt_file_dictionary_spectral_2 = dict(zip(raman_shift_spectral_2, intensity_spectral_2))                  # Using zip() to create a dictionary of (raman_shift: intensity pairs) of spectral file 2

        center_bounds_path_spectral_2 = f'results/{filename_spectral_2}/center_bounds_{filename_spectral_2}.xlsx' # Load the bounds of the center of the peak from an Excel file "center_bounds" of spectral file 2
        bounds_df_spectral_2 = pd.read_excel(center_bounds_path_spectral_2)
        
        global center_bounds_spectral_2
        center_bounds_spectral_2 = {}                                                                             # Create a dictionary to store the bounds for each peak of spectral file 2
        for i, row in bounds_df_spectral_2.iterrows():                                                            # Loop through the rows of the input file and extract the bounds for each peak of spectral file 2
            peak_index = row['Peak Index']
            center_min = row['Center Min']
            center_max = row['Center Max']
            type       = row['Type']
            center_bounds_spectral_2[peak_index] = [center_min, center_max, type]
        spectral_2 = True
    except:
        print("2nd spectral file not found!")



def stage_2_with_stage_1_code_runner():
    """
            stage_2_with_stage_1_code_runner()

            Runs the Stage 2 code on both the spectral files. Later, if the condition satisfies that
            there is no change in the G-Peak position ( < 2 cm-1 ) between both the given spectral files,
            then the Stage 1 code runs.

    """

    # Running Stage 2 code on Spectral file 1
    global raman_fitter_spectral_1
    raman_fitter_spectral_1 = RamanFitter_stage_2(
            x                   = raman_shift_spectral_1,        # The 1D array of the x-axis values (Raman Shift)
            y                   = intensity_spectral_1,          # The 1D array of the y-axis values (Intensity)
            autorun             = False,                         # The attempt to calculate all steps and fit a model; by default it should be False
            threshold           = 12,                            # It helps the code determine what is a worthy peak and what isn't
            PercentRange        = 0.11,                          # Determines what percent error the fit model can have in regards to the amplitude and position of fit curves under found peaks
            Sigma               = 15,                            # The expected width of fit curves, in terms of data points
            SigmaMin            = 3,                             # The expected minimum width allowed of fit curves, in terms of data points
            SigmaMax            = 50,                            # The expected maximum width allowed of fit curves, in terms of data points
            txt_file_dictionary = txt_file_dictionary_spectral_1,# The dictionary of (raman_shift: intensity pairs)
            center_bounds       = center_bounds_spectral_1,      # The dictionary version of peak_index, bounds, and type_of_curve (which type of curve to use for peaks. Options are 'Lorentzian', 'Gaussian', and 'Voigt')
            filename            = filename_spectral_1            # The name of the current file being examined
    )


    # Normalizes `y` data to 1
    raman_fitter_spectral_1.NormalizeData()

    # # Removes noise from input data
    # raman_fitter.Denoise(
    #         UseFFT          = True,     # A Fast Fourier Transform will be used to cutoff higher frequency noise and transformed back
    #         FFT_PS_Cutoff   = 1./1200., # This value is used to differentiate between noise and data
    #         UseSavgol       = True,     # This function will implement a Savitzky Golay filter to remove noise
    #         SavgolWindow    = 25,       # How many datapoints to iterate over in the Savitzky Golay filter
    #         SavgolOrder     = 3,        # What order of polynomial to use for the designated Savitzky Golay filter window
    #         ShowPlot        = True      # This will show a plot of the smoothed data
    #     )

    # Find the peaks in the data
    raman_fitter_spectral_1.FindPeaks(
            DistBetweenPeaks    = 1,      # Minimum distance between peaks, in terms of data points, keep it at 1
            showPlot            = True    # This will show a plot of the found peaks, always remain at True
        )


    # Fits the data with associated curve types
    raman_fitter_spectral_1.FitData(
            showPlot      = True          # This will show a plot of the fit data, always remain at True
        )

    # If you are running Stage 1 code directly, then G-Peak will not be returned, otherwise, G-Peak from Stage 2 code returned
    try:
        global g_peak_of_spectral_1
        g_peak_of_spectral_1 = raman_fitter_spectral_1.get_g_peak()
    except:
        pass

    # Raman Shift and Intensity values for Spectral file 2 returned from Stage 1 or Stage 2 code because Stage 2 code will crop the data
    # which will change the Raman Shift and Intensity values
    global raman_shift_x_values_spectral_1, intensity_y_values_spectral_1
    raman_shift_x_values_spectral_1 = raman_fitter_spectral_1.get_raman_shift_x_values()
    intensity_y_values_spectral_1 = raman_fitter_spectral_1.get_intensity_y_values()

    if (spectral_2 == True):                                         # It will run only if the valid Spectral File 2 is provided

        # Running Stage 2 code on Spectral file 2
        global raman_fitter_spectral_2
        raman_fitter_spectral_2 = RamanFitter_stage_2(
                x                   = raman_shift_spectral_2,        # The 1D array of the x-axis values (Raman Shift)
                y                   = intensity_spectral_2,          # The 1D array of the y-axis values (Intensity)
                autorun             = False,                         # The attempt to calculate all steps and fit a model; by default it should be False
                threshold           = 12,                            # It helps the code determine what is a worthy peak and what isn't
                PercentRange        = 0.11,                          # Determines what percent error the fit model can have in regards to the amplitude and position of fit curves under found peaks
                Sigma               = 15,                            # The expected width of fit curves, in terms of data points
                SigmaMin            = 3,                             # The expected minimum width allowed of fit curves, in terms of data points
                SigmaMax            = 50,                            # The expected maximum width allowed of fit curves, in terms of data points
                txt_file_dictionary = txt_file_dictionary_spectral_2,# The dictionary of (raman_shift: intensity pairs)
                center_bounds       = center_bounds_spectral_2,      # The dictionary version of peak_index, bounds, and type_of_curve (which type of curve to use for peaks. Options are 'Lorentzian', 'Gaussian', and 'Voigt')
                filename            = filename_spectral_2            # The name of the current file being examined
        )

        # Normalizes `y` data to 1
        raman_fitter_spectral_2.NormalizeData()

        # Find the peaks in the data
        raman_fitter_spectral_2.FindPeaks(
                DistBetweenPeaks    = 1,                             # minimum distance between peaks, in terms of data points
                showPlot            = True                           # this will show a plot of the found peaks
            )

        # Fits the data with associated curve types
        raman_fitter_spectral_2.FitData(
                showPlot      = True                                 # this will show a plot of the fit data
            )

        # G-Peak x-value from Spectral File 2 is returned
        global g_peak_of_spectral_2
        g_peak_of_spectral_2 = raman_fitter_spectral_2.get_g_peak()

        # Raman Shift and Intensity values for Spectral file 2 returned from Stage 2 code because Stage 2 code will crop the data
        # which will change the Raman Shift and Intensity values
        global raman_shift_x_values_spectral_2, intensity_y_values_spectral_2
        raman_shift_x_values_spectral_2 = raman_fitter_spectral_2.get_raman_shift_x_values()
        intensity_y_values_spectral_2 = raman_fitter_spectral_2.get_intensity_y_values()

        # Condition: if there is no change in the G-Peak position ( < 2 cm-1 ) between both the given spectral files
        if abs(g_peak_of_spectral_1 - g_peak_of_spectral_2) < 2:
            
            print('Stage 1 code is running now because there is no change in the G-Peak position ( < 2 cm-1 ) between both spectral files.')
            
            # Running Stage 1 code on Spectral file 1
            raman_fitter_spectral_1 = RamanFitter_stage_1(
                    x                   = raman_shift_spectral_1,        # The 1D array of the x-axis values (Raman Shift)
                    y                   = intensity_spectral_1,          # The 1D array of the y-axis values (Intensity)
                    autorun             = False,                         # The attempt to calculate all steps and fit a model; by default it should be False
                    threshold           = 12,                            # It helps the code determine what is a worthy peak and what isn't
                    PercentRange        = 0.11,                          # Determines what percent error the fit model can have in regards to the amplitude and position of fit curves under found peaks
                    Sigma               = 15,                            # The expected width of fit curves, in terms of data points
                    SigmaMin            = 3,                             # The expected minimum width allowed of fit curves, in terms of data points
                    SigmaMax            = 50,                            # The expected maximum width allowed of fit curves, in terms of data points
                    txt_file_dictionary = txt_file_dictionary_spectral_1,# The dictionary of (raman_shift: intensity pairs)
                    center_bounds       = center_bounds_spectral_1,      # The dictionary version of peak_index, bounds, and type_of_curve (which type of curve to use for peaks. Options are 'Lorentzian', 'Gaussian', and 'Voigt')
                    filename            = filename_spectral_1            # The name of the current file being examined
            )

            # Normalizes `y` data to 1
            raman_fitter_spectral_1.NormalizeData()

            # Find the peaks in the data
            raman_fitter_spectral_1.FindPeaks(
                    DistBetweenPeaks    = 1,                             # minimum distance between peaks, in terms of data points
                    showPlot            = True                           # this will show a plot of the found peaks
            )

            # Fits the data with associated curve types
            raman_fitter_spectral_1.FitData(
                    showPlot      = True                                 # this will show a plot of the fit data
            )

            # Raman Shift and Intensity values for Spectral file 1 returned from Stage 1 code
            raman_shift_x_values_spectral_1 = raman_fitter_spectral_1.get_raman_shift_x_values()
            intensity_y_values_spectral_1 = raman_fitter_spectral_1.get_intensity_y_values()

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
                    DistBetweenPeaks    = 1,                             # minimum distance between peaks, in terms of data points
                    showPlot            = True                           # this will show a plot of the found peaks
            )

            # Fits the data with associated curve types
            raman_fitter_spectral_2.FitData(
                    showPlot      = True                                 # this will show a plot of the fit data
            )

            # Raman Shift and Intensity values for Spectral file 2 returned from Stage 1 code
            raman_shift_x_values_spectral_2 = raman_fitter_spectral_2.get_raman_shift_x_values()
            intensity_y_values_spectral_2 = raman_fitter_spectral_2.get_intensity_y_values()

    global components_spectral_1, curveParams_spectral_1, bestFitLine_spectral_1
    components_spectral_1  = raman_fitter_spectral_1.comps               # Returns a dictionary of each curve plot
    curveParams_spectral_1 = raman_fitter_spectral_1.params              # Returns a dictionary of the parameters of each Lorentzian, Gaussian, or Voigt curve
    bestFitLine_spectral_1 = raman_fitter_spectral_1.fit_line            # Returns the plot data of the model

    if (spectral_2 == True):
        global components_spectral_2, curveParams_spectral_2, bestFitLine_spectral_2
        components_spectral_2  = raman_fitter_spectral_2.comps           # Returns a dictionary of each curve plot
        curveParams_spectral_2 = raman_fitter_spectral_2.params          # Returns a dictionary of the parameters of each Lorentzian, Gaussian, or Voigt curve
        bestFitLine_spectral_2 = raman_fitter_spectral_2.fit_line        # Returns the plot data of the model




'''---------------------------------------------------------Results---------------------------------------------------------'''

'''----------------------Results of Spectral File 1----------------------'''

def spectral_1_results_generator():
    """
            spectral_1_results_generator()

            Generates the results of the Spectral File 1 and 2 in the Excel files.
            Results include the following:
                1. Curve Parameters: Parameters for each fit curve in the model
                2. Fitted Data: Fitted Data includes complete fit line of the model along with the residual
                3. Curves Data: Data of the individual models of each curve - (x,y) values

    """

    print(f'Generating Excel files for allocating the results of Spectral File 1 ({filename_spectral_1})...' )

    ''' Curve Paramters '''
    # Contructing a curve parameters dictionary for Spectral file 1 of (Parameter : Value) pairs of the curves fitted in the model
    curve_parameters_spectral_1 = {}
    for key in curveParams_spectral_1:
        curve_parameters_spectral_1[key] = curveParams_spectral_1[key].value

    # Creating a DataFrame of curve parameters dictionary to export the data to excel
    df = pd.DataFrame(data = curve_parameters_spectral_1.items(), columns=['Parameter', 'Value'])
    df.to_excel(f'results/{filename_spectral_1}/curves_parameters_{filename_spectral_1}.xlsx', index=False)

    ''' Fitted Data '''
    # DataFrame for fitted_data from the Spectral 1
    df_fitted_data_spectral_1 = pd.DataFrame({'Raman Shift': raman_shift_x_values_spectral_1, 'Intensity': intensity_y_values_spectral_1, 'Best Fit Line': bestFitLine_spectral_1, 'Residual': (bestFitLine_spectral_1 - intensity_y_values_spectral_1)})
    df_fitted_data_spectral_1.to_excel(f'results/{filename_spectral_1}/fitted_data_{filename_spectral_1}.xlsx', index=False)

    ''' Curves Data '''
    # get the (x,y) values for each curve in Spectral 1
    curves_data_spectral_1 = {}
    for key in components_spectral_1.keys():
        curves_data_spectral_1[key] = components_spectral_1[key]
    curves_data_spectral_1['x'] = raman_shift_x_values_spectral_1

    # DataFrame for curves_data from the Spectral 1
    df_curves_data_spectral_1 = pd.DataFrame(curves_data_spectral_1)
    df_curves_data_spectral_1.to_excel(f'results/{filename_spectral_1}/curves_data_{filename_spectral_1}.xlsx', index=False)


'''----------------------Results of Spectral File 2----------------------'''

def spectral_2_results_generator():
    """
            spectral_2_results_generator()

            Generates the results of the Spectral File 1 and 2 in the Excel files.
            Results include the following:
                1. Curve Parameters: Parameters for each fit curve in the model
                2. Fitted Data: Fitted Data includes complete fit line of the model along with the residual
                3. Curves Data: Data of the individual models of each curve - (x,y) values

    """

    print(f'Generating Excel files for allocating the results of Spectral File 2 ({filename_spectral_2})...' )

    ''' Curve Paramters '''
    # Contructing a curve parameters dictionary for Spectral file 2 of (Parameter : Value) pairs of the curves fitted in the model
    curve_parameters_spectral_2 = {}
    for key in curveParams_spectral_2:
        curve_parameters_spectral_2[key] = curveParams_spectral_2[key].value

    # Creating a DataFrame of curve parameters dictionary to export the data to excel
    df = pd.DataFrame(data = curve_parameters_spectral_2.items(), columns=['Parameter', 'Value'])
    df.to_excel(f'results/{filename_spectral_2}/curves_parameters_{filename_spectral_2}.xlsx', index=False)
    
    ''' Fitted Data '''
    # DataFrame for fitted_data from the Spectral 2
    df_fitted_data_spectral_2 = pd.DataFrame({'Raman Shift': raman_shift_x_values_spectral_2, 'Intensity': intensity_y_values_spectral_2, 'Best Fit Line': bestFitLine_spectral_2, 'Residual': (bestFitLine_spectral_2 - intensity_y_values_spectral_2)})
    df_fitted_data_spectral_2.to_excel(f'results/{filename_spectral_2}/fitted_data_{filename_spectral_2}.xlsx', index=False)

    ''' Curves Data '''
    # get the (x,y) values for each curve in Spectral 2
    curves_data_spectral_2 = {}
    for key in components_spectral_2.keys():
        curves_data_spectral_2[key] = components_spectral_2[key]
    curves_data_spectral_2['x'] = raman_shift_x_values_spectral_2

    # DataFrame for curves_data from the Spectral 2
    df_curves_data_spectral_2 = pd.DataFrame(curves_data_spectral_2)
    df_curves_data_spectral_2.to_excel(f'results/{filename_spectral_2}/curves_data_{filename_spectral_2}.xlsx', index=False)




# MAIN FUNCTION
if __name__ == "__main__":
    spectral_files_handler('GC_532nm', 'GC_633nm') # Enter both the spectral file name here (2nd spectral file is optional)
    stage_2_with_stage_1_code_runner()             # Runs Stage 2 by default on both Spectral files and then Stage 1 code on a condition
    spectral_1_results_generator()                 # Generates the results of Spectral File 1 in excel files
    if (spectral_2 == True):
        spectral_2_results_generator()             # Generates the results of Spectral File 1 in excel files