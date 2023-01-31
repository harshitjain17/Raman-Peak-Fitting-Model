# import numpy as np
# from scipy.signal import find_peaks
# from scipy.optimize import curve_fit
# import os

# # Gaussian Curve - defining peak shapes
# def gaussian(x, A, mu, sigma):
#     return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

# # Lorentzian Curve - defining peak shapes
# def lorentzian(x, A, mu, gamma):
#     return A * gamma / ((x - mu) ** 2 + gamma ** 2)


# def composite_peak(x, A1, mu1, sigma1, A2, mu2, sigma2):
#     return gaussian(x, A1, mu1, sigma1) + gaussian(x, A2, mu2, sigma2)

# # Fits the spectrum to a model function with the given initial guess
# def fit_spectrum(x, y, model_func, initial_guess):
#     popt, pcov = curve_fit(model_func, x, y, p0=initial_guess)
#     return popt


# def deconvolve_spectrum(x, y):
#     peaks, _ = find_peaks(y)
#     popt_list = []
#     for peak in peaks:
#         peak_x = x[peak]
#         initial_guess = [y[peak], peak_x, 1]
#         popt = fit_spectrum(x, y, gaussian, initial_guess)
#         popt_list.append(popt)
#     return popt_list

# # Processing the spectrum
# def process_spectrum(spectrum_path):
#     x, y = np.loadtxt(spectrum_path, delimiter=',', unpack=True)
#     popt_list = deconvolve_spectrum(x, y)
#     return popt_list

# # Data File Processing
# def process_folder(folder_path):
#     file_list = os.listdir(folder_path)
#     result = []
#     for file_name in file_list:
#         if file_name.endswith('.csv'):
#             file_path = os.path.join(folder_path, file_name)
#             result.append(process_spectrum(file_path))
#     return result

# # Main Function
# folder_path = '/path/to/spectrum/folder'
# result = process_folder(folder_path)
# print(result)


def main():
    
    # Create the models
    models = [
        models.LorentzianModel(),
        models.GaussianModel(),
        models.VoigtModel(),
        ...
    ]
    
    # Create the index
    index = []
    folder = 'data'
    for file in os.listdir(folder):
        sample_id, position = extract_sample_id_and_position(file)
        index.append((sample_id, position, os.path.join(folder, file)))
    index = pd.DataFrame(index, columns=['Sample ID', 'Position', 'File'])
    
    # Loop over the index
    for i, row in index.iterrows():
        x, y = load_data(row['File'])
        x, y = preprocess(x, y)
        for model in models:
            result = model.fit(x, y)
            # Plot the results
            result.plot()
            plt.show()
            # Save the results to an Excel file
            result.save('results.xlsx')

if __name__ == '__main__':
    main()





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



# It loops through all the .xlsx files in the specified data_folder,
# extracts the sample ID and position,
# preprocesses the spectral data,
# performs the fitting,
# and exports the results
if __name__ == '__main__':
    
    data_folder = 'Examples_raman_of_carbon'
    model = 'model_1'
    data_index = []
    
    for filename in os.listdir(data_folder):
        if filename.endswith('.xlsx'):
            sample_id, position = extract_sample_id_and_position(filename)
            data_index.append((sample_id, position))
    
    data_index = pd.DataFrame(data_index, columns=['Sample ID', 'Position'])
    
    for i, row in data_index.iterrows():
        spectral_data = preprocess_spectral_data(pd.read_excel(os.path.join(data_folder, f'{row["Sample ID"]}-Analysis-GC-and-PC.xlsx')))
        fit_params = perform_fitting(spectral_data, model)
        export_results(row['Sample ID'], row['Position'], spectral_data, fit_params)
