# **Raman Fitting Model**

# About the Project: 📚
The Raman Fitting Library is a comprehensive software package designed to facilitate the analysis and fitting of Raman spectra. Raman spectroscopy is a powerful technique used in various scientific and industrial applications to study molecular vibrations and obtain structural information about materials.

The goal of the Raman Fitting Library is to provide researchers, scientists, and spectroscopy enthusiasts with a versatile and user-friendly toolset for analyzing Raman spectra. It offers a range of advanced algorithms and methods for spectral fitting, enabling accurate determination of peak positions, peak intensities, and other relevant parameters.

# Key Features
- ### Spectral Fitting:
    The library provides robust algorithms for fitting Raman spectra to various functional forms, such as Lorentzian, Gaussian, Voigt, and more. These functions can be combined to model complex spectral features accurately.

- ### Baseline Correction:
    Raman spectra often exhibit a baseline that can obscure the underlying peaks. The library includes sophisticated algorithms for baseline correction, allowing users to remove unwanted variations and enhance the visibility of spectral features.

- ### Peak Analysis:
    Accurate determination of peak positions and intensities is crucial in Raman spectroscopy. The library provides tools for peak identification, peak integration, and extraction of peak parameters, enabling comprehensive analysis of Raman spectra.

- ### Visualization:
    Visual representation of the fitted spectra and associated parameters greatly aids in the interpretation of Raman data. The library offers plotting functions for displaying the original spectra, fitted curves, residuals, and other relevant information.

# Setup / Installation: 💻

To install the dependencies and devDependencies for this project, you can run the following command in the terminal:

    pip install numpy pandas scipy lmfit matplotlib

# Usage:

To use this project, download or clone the repository. Then, you can run the following command in the terminal:
    
    python raman_fitting_model.py
    
This will start running the `raman_fitting_model.py` file.

You can go down to the main function in `raman_fitting_model.py` file and change the spectral file 1 and spectral file 2 name in the `spectral_files_handler()` function paramaters. Note that, spectral file 2 name is optional.