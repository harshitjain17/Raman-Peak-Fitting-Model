'''
This is a Python script for processing Raman spectral data.
The script does the following operations on each .xlsx file in a specified directory:

1. Extracts the sample ID and position information from the file name.
2. Preprocesses the raw spectral data by removing NaN values, applying Savitzky-Golay smoothing, and normalizing the data to values between 0 and 1.
3. Fits the preprocessed spectral data to a specified model.
4. Exports the fit parameters as an Excel file and saves a plot of the original spectral data and the fit results as a PNG image.
'''


import os
import pandas as pd
import numpy as np
from raman_fitting import Fitting
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


'''
This function first splits the filename by the - character, and then concatenates the first two parts of the resulting list to form the sample_id
The position information is extracted from the last part of the list by splitting by the . character and taking the first part of the resulting list.
It takes a filename as input and returns the sample ID and position extracted from the filename
'''
# The function is designed to extract the sample ID and position information from the filename
# Following format: <Sample ID>-Analysis-GC-and-PC-<Position>.xlsx
def extract_sample_id_and_position(filename): # filename: a string representing the name of a file
    parts = filename.split('-')
    sample_id = '-'.join(parts[:2])
    position = parts[-1].split('.')[0]
    return sample_id, position




# This function is responsible for preprocessing the raw spectral data in preparation for fitting
# It takes the spectral data as input and returns the preprocessed spectral data
def preprocess_spectral_data(spectral_data):
    
    # Remove any NaN values
    spectral_data = spectral_data[~np.isnan(spectral_data[:, 1]), :]

    # Apply Savitzky-Golay smoothing to the data to reduce noise
    smoothed_data = savgol_filter(spectral_data, window_length=51, polyorder=3)
    
    # Normalize the data to have values between 0 and 1
    min_value = np.min(smoothed_data)
    max_value = np.max(smoothed_data)
    normalized_data = (smoothed_data - min_value) / (max_value - min_value)
    
    return normalized_data



# This function performs a fitting on the spectral data with the specified model
# It takes the spectral data and the model name as input and returns the fit parameters
def perform_fitting(spectral_data, model):
    fitter = Fitting(model=model)
    fitter.fit(spectral_data[0], spectral_data[1]) # passing the x-values (spectral_data[0]) and y-values (spectral_data[1]) of the spectral data as arguments
    return fitter.fit_params




# This function takes the sample ID, position, spectral data, and fit parameters as input and,
# exports the results as an excel file and a plot
def export_results(sample_id, position, spectral_data, fit_params):
    
    # creates a new figure object and an axis object to create a plot of the spectral data and the fit results
    figure, axis = plt.subplots()
    
    # plots the original spectral data on the axis
    axis.plot(spectral_data[0], spectral_data[1], label='data')

    # plots the fit results on the axis
    axis.plot(spectral_data[0], fit_params.best_fit, label='fit')
    
    # adds a legend to the plot to distinguish between the original spectral data and the fit results
    axis.legend()
    
    # sets the label of the x-axis to 'Raman shift (cm-1)'
    plt.xlabel('Raman shift (cm-1)')

    # sets the label of the y-axis to 'Intensity (a.u.)'
    plt.ylabel('Intensity (a.u.)')

    # sets the title of the plot to 'Sample <Sample ID>, position <Position>'
    plt.title(f'Sample {sample_id}, position {position}')

    # saves the plot as a PNG image, with the file name '<Sample ID>_<Position>.png'
    plt.savefig(f'{sample_id}_{position}.png')

    # exports the fit parameters as an Excel file, with the file name '<Sample ID>_<Position>_fit_params.xlsx'
    fit_params.to_excel(f'{sample_id}_{position}_fit_params.xlsx')





'''
It loops through all the .xlsx files in the specified data_folder,
extracts the sample ID and position,
preprocesses the spectral data,
performs the fitting,
and exports the results
'''
# Main function
if __name__ == '__main__':
    
    data_folder = 'Examples_raman_of_carbon' # contains the Excel files containing the spectral data
    model = 'model_1' # the name of the model to be used for the fitting
    data_index = [] 
    
    # iterates over the names of all the files in the data_folder directory
    for filename in os.listdir(data_folder):
        if filename.endswith('.xlsx'):
            (sample_id, position) = extract_sample_id_and_position(filename) # extracts sample_id, position from the filename
            data_index.append((sample_id, position))
    
    # converts the data_index list into a Pandas DataFrame and sets the column names to 'Sample ID' and 'Position'
    data_index = pd.DataFrame(data_index, columns=['Sample ID', 'Position'])
    
    # iterates over the rows of the data_index DataFrame
    # The 'i' variable holds the index of the current row
    # the 'row' variable holds the current row as a Series
    for i, row in data_index.iterrows():

        # spectral data is read from the file, the file path is constructed using 'os.path.join', the result is assigned to 'spectral_data'
        spectral_data = preprocess_spectral_data(pd.read_excel(os.path.join(data_folder, f'{row["Sample ID"]}-Analysis-GC-and-PC.xlsx')))
        
        fit_params = perform_fitting(spectral_data, model)
        
        export_results(row['Sample ID'], row['Position'], spectral_data, fit_params)
