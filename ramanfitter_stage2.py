'''
    Name:           raman_fitting_model

    Description:    RamanFitter takes the input of a single raman measurement and applies common fitting techniques to solve for Lorentzians, Gaussian, or Voigt curves.

'''

import os
import numpy as np
import pandas as pd
from scipy import signal
from lmfit.models import ExponentialModel, LorentzianModel, GaussianModel, VoigtModel
import matplotlib.pyplot as plt

class RamanFitter_stage_2:

    """
    A class that fits Raman data with Lorentzian, Gaussian, or Voigt curves

    ...

    Attributes
    ----------
    exposure : float
        Exposure in seconds.

    x : numpy.array
        1D array of x-axis values
    y : numpy.array
        1D array of y-axis values
    y_old : numpy.array
        1D array of original y-axis values
    threshold : float
        value that helps the code determine what is a worthy peak and what isn't.
    perc_range : float
        Must be between 0 and 1. The `PercentRange` determines what percent error the fit model can have in regards to the amplitude and position of fit curves under found peaks.
    sigmas : list 
        List of [ Sigma, SigmaMin, SigmaMax ]
    scale : float
        Scale used for normalization
    fit_line : numpy.array
        1D array of y-values associated with the fit model
    params : dict
        Dictionary of the fit curve parameters
    comps : dict
        Dictionary of 1D numpy.arrays of y-values associated with the individual fit curves

    Methods
    -------
    NormalizeData()
        Normalizes `y` data to 1
    Denoise( x, y, UseFFT = True, FFT_PS_Cutoff = 1./1200., UseSavgol = True, SavgolWindow = 25, SavgolOrder = 3, ShowPlot = False )
        Removes noise from input data.
    FindPeaks( DistBetweenPeaks = 50, Threshold = 12., showPlot = False )
            Finds peaks in the `self.y` numpy array given during the initialization of the class.
    FitData( type = 'Lorentzian', showPlot = True )
        Fits the data with associated curve types. The function `FindPeaks()` must be run before this step.
    get( index )
        Returns the model curve associated with the given index.
    show( index )
        Plots the model curve associated with the given index.
    """

    # Initializes class
    def __init__( self, x, y, autorun = True, threshold = 12., PercentRange = 0.2, Sigma = 15, SigmaMin = 3, SigmaMax = 100, txt_file_dictionary = {}, center_bounds = {}, filename = '' ):

        """
            __init__( self, x, y, autorun = True, threshold = 12., PercentRange = 0.2, Sigma = 15, SigmaMin = 3, SigmaMax = 100 )

            Initializes the RamanFitter Class

            Parameters
            ----------
            x : numpy.array
                `x` is a 1D array of the x-axis values, typically cm^-1 or nm values
            y : numpy.array
                `y` is a 1D array of the y-axis values, typically intensity or counts
            autorun : bool, optional, default: True
                If set to `True`, the class will automatically attempt to calculate all steps and fit a model
            threshold : float, optional, default: 12
                `threshold` helps the code determine what is a worthy peak and what isn't. If `autorun` is set to `False`, this value must be explicitly set each time that the function `FindPeaks` is called.
            PercentRange : float, optional, default: 0.2
                Must be between 0 and 1. The `PercentRange` determines what percent error the fit model can have in regards to the amplitude and position of fit curves under found peaks.
            Sigma : int, optional, default: 15
                The expected width of fit curves, in terms of data points
            SigmaMin : int, optional, default: 3
                The expected minimum width allowed of fit curves, in terms of data points
            SigmaMax : int, optional, default: 100
                The expected maximum width allowed of fit curves, in terms of data points
            txt_file_dictionary : dictionary
                Dictionary version of the raw data provided by the user
            center_bounds : dictionary
                Dictionary version of peak_index, bounds, and type_of_curve (which type of curve to use for peaks. Options are 'Lorentzian', 'Gaussian', and 'Voigt')
            filename : string
                name of the current file being examined
        """

        # Set values
        self.threshold           = threshold
        self.perc_range          = PercentRange
        self.sigmas              = [ Sigma, SigmaMin, SigmaMax ]
        self.scale               = 1.
        self.txt_file_dictionary = txt_file_dictionary
        self.filename            = filename
        self.g_peak              = None
        
        # updating center_bounds dictionary to include ONLY D-Peaks and G-Peaks
        center_bounds_d_g_peaks = {}
        for peak_index in center_bounds:
            if (1300 <= peak_index <= 1370) or (1540 <= peak_index <= 1610):
                center_bounds_d_g_peaks[peak_index] = center_bounds[peak_index]
        
        # finding D-Peak's x-values and y-values
        d_peak_lower_index = self.find_closest_key_index(self.txt_file_dictionary, 1300)
        d_peak_upper_index = self.find_closest_key_index(self.txt_file_dictionary, 1370)
        d_peak_x = list(x)[d_peak_lower_index : d_peak_upper_index]
        d_peak_y = list(y)[d_peak_lower_index : d_peak_upper_index]
        
        # finding G-Peak's x-values and y-values
        g_peak_lower_index = self.find_closest_key_index(self.txt_file_dictionary, 1540)
        g_peak_upper_index = self.find_closest_key_index(self.txt_file_dictionary, 1630)
        g_peak_x = list(x)[g_peak_lower_index : g_peak_upper_index]
        g_peak_y = list(y)[g_peak_lower_index : g_peak_upper_index]
        
        x_values = d_peak_x + g_peak_x
        y_values = d_peak_y + g_peak_y

        self.x                   = np.array(x_values)
        self.y                   = np.array(y_values)
        self.y_old               = np.array(y_values)
        self.center_bounds       = center_bounds_d_g_peaks
        
        if autorun:

            # Normalize the Data
            self.NormalizeData()

            # Denoise Data
            self.Denoise()

            # Find Peaks
            self.FindPeaks()

            # Fit Data
            self.FitData()
    
    # Normalize the data to 1
    def NormalizeData( self ):
        """
            NormalizeData( self )

            Normalizes `y` data to 1

        """
        self.min_y      = np.min( self.y )
        self.max_y      = np.max( self.y )
        self.scale      = 1./( self.max_y - self.min_y )
        self.y          = ( self.y - self.min_y )*self.scale
        self.threshold  *= self.scale

    def get_raman_shift_x_values(self):
        
        """
            get_raman_shift_x_values(self)

            Get the new x-values after cropping the raw data and defining D and G_peaks.

            Parameters
            ----------
            None

            Returns
            -------
            array
                The array of x-values of the cropped raw data.

        """
        return self.x

    def get_intensity_y_values(self):
        
        """
            get_intensity_y_values(self)

            Get the new y-values after cropping the raw data and defining D and G_peaks.

            Parameters
            ----------
            None

            Returns
            -------
            array
                The array of y-values of the cropped raw data.

        """
        return self.y

    # Denoise the data
    def Denoise( self,
                UseFFT          = True,             # (bool)        Use FFT to smooth
                FFT_PS_Cutoff   = 1./1200.,         # (float)       FFT Power Spectrum cutoff
                UseSavgol       = True,
                SavgolWindow    = 25,
                SavgolOrder     = 3,
                ShowPlot        = False ):          # (bool)        Show plots of original vs cleaned data

        """
            Denoise( self, UseFFT = True, FFT_PS_Cutoff = 1./1200., UseSavgol = True, SavgolWindow = 25, SavgolOrder = 3, ShowPlot = False )

            Removes noise from input data.

            It is recommended that you run `NormalizeData()` before attempting a denoise.

            Parameters
            ----------
            UseFFT : bool, optional, default: True
                If set to `True`, a Fast Fourier Transform will be used to cutoff higher frequency noise and transformed back.
            FFT_PS_Cutoff : float, optional, default: 1/1200
                `FFT_PS_Cutoff` is the Fast Fourier Transform Power Spectrum cutoff value. This value is used to differentiate between noise and data.
            UseSavgol : bool, optional, default: True
                If set to `True`, this function will implement a Savitzky Golay filter to remove noise
            SavgolWindow : int, optional, default: 25
                How many datapoints to iterate over in the Savitzky Golay filter.
            SavgolOrder : int, optional, default: 3
                What order of polynomial to use for the designated Savitzky Golay filter window
            ShowPlot : bool, optional, default: False
                If set to `True` this will show a plot of the smoothed data

        """

        print( "Removing Noise..." )

        # If UseFFT, FFT filtering will be used
        if UseFFT:
            # Get points and dx
            x_pnts      = len( self.x )                                             # Number of points
            self.Δx     = self.x[1] - self.x[0]                                     # Δx


            fhat        =  np.fft.fft( self.y, x_pnts )                             # Computes the fft
            psd         = fhat * np.conj( fhat )/x_pnts                             # Compute Power Spectrum
            L           = np.arange( 1, np.floor( x_pnts/2. ), dtype = np.int32 )   # Only look at real components

            
            cutoff      = np.max( psd[L] )*FFT_PS_Cutoff                            # Calculate cutoff of power spectrum
            indices     = psd > cutoff                                              # Get indices where PSD is above cutoff
            psd_clean   = psd * indices                                             # Remove cutoff indices from PSD
            fhat        = fhat * indices                                            # Remove cutoff indices from FHat
            self.y      = np.fft.ifft( fhat )                                       # Calculate Inverse FFT

        # If
        if UseSavgol:
            self.y   = signal.savgol_filter( self.y, SavgolWindow, SavgolOrder )

        # If ShowPlot, build plots with matplotlib
        if ShowPlot:
            plt.plot( self.x, self.y_old, color = 'c', label = 'Noisy' )
            plt.plot( self.x, self.y/self.scale+self.min_y, color = 'k', label = 'Filtered' )
            if len(self.peaks_x) > 0:
                x_min = min(self.peaks_x) - 150                                      # defining the lower x-limit 
                x_max = max(self.peaks_x) + 150                                      # defining the upper x-limit
                if (self.x[-1] < x_max):                                             # handling out-of-bounds-error
                    x_max = self.x[-1]
                if (self.x[0] > x_min):                                              # handling out-of-bounds-error
                    x_min = self.x[0]
            plt.xlim(x_min, x_max)                                                   # set the x-axis limits
            plt.legend()
            plt.show()



    def find_closest_key_index(self, dictionary, value):
        """
            find_closest_key_index(self, dictionary, value)

            Find the index of the closest key in the dictionary to the given number
            
            Parameters
            ----------
            dictionary : A dictionary object to search for the closest key.
            value : A numerical value for which we want to find the closest key index.
            
            Returns:
            The index of the closest key in the dictionary to the given number.
        """
        closest_key = None
        min_diff = None
        for i, key in enumerate(dictionary.keys()):
            diff = abs(value - key)
            if min_diff is None or diff < min_diff:
                min_diff = diff
                closest_key = key
        return list(dictionary.keys()).index(closest_key)+1

    # Find the peaks in the data
    def FindPeaks( self, DistBetweenPeaks = 50, showPlot = False ):

        """
            FindPeaks( self, DistBetweenPeaks = 50, showPlot = False )

            Finds peaks in the `self.y` numpy array given during the initialization of the class.

            It is recommended that you run the `Denoise()` function before attempted to find peaks. This will reduce computation.

            Parameters
            ----------
            DistBetweenPeaks : int, optional, default: 50
                Minimum distance between peaks, in terms of data points
            ShowPlot : bool, optional, default: False
                If set to `True` this will show a plot of the found peaks

        """

        print( "Finding Peaks..." )
        
        # Find Peaks
        peaks, _  = signal.find_peaks( self.y, distance = DistBetweenPeaks )
        
        # Filter peaks by x-axis bounds
        peak_x = np.arange(len(self.y))[peaks]  # x-coordinates of peaks

        # Determine prominence of peaks, as compared to surrounding data
        prominence = signal.peak_prominences( self.y, peaks, wlen = len( self.x )-1 )[0]
        rem_ind = [ i for ( i, p ) in enumerate( prominence ) if p < self.threshold ]
        self.npeaks = np.delete( peaks, rem_ind ) # stores the indices of the peaks found        
        self.peaks_y = np.array( [ self.y[i] for i in self.npeaks ] )
        self.peaks_x = np.array( [ self.x[i] for i in self.npeaks ] )

        self.filtered_peaks_x = []
        self.filtered_peaks_y = []
        self.filtered_npeaks = []

        # dictionary - (type_of_peak(D/G) : number_of_peaks_in_that_region) pairs
        counter_for_peaks = {'D-Peak' : 0, 'G-Peak' : 0}

        # loop through all the peaks software found
        if len(self.peaks_x) > 0:

            for i in range(len(self.peaks_x)):
                
                # condition for D-peak existence
                if (1300 <= self.peaks_x[i] <= 1400):
                    
                    # D-Peak operation starts
                    if (counter_for_peaks['D-Peak'] == 0):
                        
                        self.filtered_peaks_x.append(self.peaks_x[i])    # take the x-value of the peak
                        self.filtered_peaks_y.append(self.peaks_y[i])    # take the y-value of the peak
                        peak_index = list(self.x).index(self.peaks_x[i]) # finding the data point for the peak
                        self.filtered_npeaks.append(peak_index)          # filling the data point in self.npeaks
                        counter_for_peaks['D-Peak'] += 1                 # increase the counter of peaks in that specific region by 1 (final) - it should not be more than 1
                
                # condition for G-peak existence
                if (1540 <= self.peaks_x[i] <= 1620):
                    
                    # G-Peak operation starts
                    if (counter_for_peaks['G-Peak'] == 0):
                        self.filtered_peaks_x.append(self.peaks_x[i])    # take the x-value of the peak
                        self.filtered_peaks_y.append(self.peaks_y[i])    # take the y-value of the peak
                        peak_index = list(self.x).index(self.peaks_x[i]) # finding the data point for the peak
                        self.filtered_npeaks.append(peak_index)          # filling the data point in self.npeaks
                        counter_for_peaks['G-Peak'] += 1                 # increase the counter of peaks in that specific region by 1 (final) - it should not be more than 1
        
        self.npeaks = self.filtered_npeaks
        self.peaks_x = self.filtered_peaks_x
        self.peaks_y = self.filtered_peaks_y
        
        # setting the G-Peak
        try:
            self.set_g_peak(self.peaks_x[1]) # 0th index is D-Peak, 1st index is G-Peak
        except:
            print("G-Peak not found!")

        # fetching the data point for plotting of raw data
        for i in range(len(self.x)):
            if (self.x[i] > 1400):
                data_point = i
                break

        # showing the plot
        if showPlot:
            plt.plot( self.x[:data_point], self.y[:data_point], color = 'c', label = 'D-peak' )     # plot the curve D-Peak
            plt.plot( self.x[data_point+1:], self.y[data_point+1:], color = 'c', label = 'G-Peak' ) # plot the curve G-Peak
            plt.scatter( self.peaks_x, self.peaks_y, color = 'k', label = 'Peaks' )                 # scatter the peaks
            if (len(self.peaks_x) > 0):
                x_min = min(self.peaks_x) - 200                                                     # defining the lower x-limit 
                x_max = max(self.peaks_x) + 200                                                     # defining the upper x-limit
                if (self.x[-1] < x_max):                                                            # handling out-of-bounds-error
                    x_max = self.x[-1]
                if (self.x[0] > x_min):                                                             # handling out-of-bounds-error
                    x_min = self.x[0]
            plt.xlim(x_min, x_max)                                                                  # set the x-axis limits
            plt.legend()                                                                            # set the legends for graph
            plt.gcf().suptitle(f'Peaks of {self.filename} - Stage 2')                               # set the title of the figure
            plt.show()                                                                              # show the graph
        

    # Fit the Data to a Lorentzian, Gaussian, or Voigt model
    def FitData( self, showPlot = False ):
        
        """
            FitData( self, type = 'Lorentzian', showPlot = True )

            Fits the data with associated curve types. The function `FindPeaks()` must be run before this step.

            This function will set `self.fit_line`, `self.params`, and `self.comps`.

            `self.fit_line` is the complete fit line of the model.
            `self.params` are the parameters for each fit curve in the model.
            `self.comps` are the individual models of each curve.

            Parameters
            ----------
            ShowPlot : bool, optional, default: False
                If set to `True` this will show a plot of the fit data

        """
        print( "Fitting Model..." )

        # Fit exponential to remove background
        if len( self.peaks_y ) > 0:

            exp_mod     = ExponentialModel()
            pars        = exp_mod.guess( self.y_old, x = self.x )
            mod         = None
            Model_list  = []

            # Cycle through each peak to fit the required type
            i = 0
            MARGIN = 5

            assert len(self.center_bounds) == 2, "Center bounds has more than 2 peaks. It should only have 2 peaks in it. Try print the self.center_bounds to see which peak(s) is/are extra."
            
            for key in self.center_bounds:
                curve_type = self.center_bounds[key][2]
                pref = 'Curve_'+str(i+1)+'_'
                if (curve_type == 'Lorentzian'):
                    Model_list.append( LorentzianModel( prefix = pref ) )
                elif (curve_type == 'Gaussian'):
                    Model_list.append( GaussianModel( prefix = pref ) )
                elif (curve_type == 'Voigt'):
                    Model_list.append( VoigtModel( prefix = pref ) )

                pars.update( Model_list[i].make_params() )
                pars[pref+'center'].set(value = (self.peaks_x[i]), min = (self.peaks_x[i] - MARGIN),  max = (self.peaks_x[i] + MARGIN))
                pars[pref+'sigma'].set(value = (self.sigmas[0]), min = (self.sigmas[1]), max = (self.sigmas[2]))
                pars[pref+'amplitude'].set(value = (self.y_old[self.npeaks[i]]), min = (self.y_old[self.npeaks[i]]*(1. - self.perc_range)))
                mod = (mod + Model_list[i]) if mod else (Model_list[i])
                i+=1

            out             = mod.fit( self.y_old, pars, x = self.x )   # out.best_fit = total of fit values
            self.fit_line   = out.best_fit
            self.params     = out.params
            self.comps      = out.eval_components( x = self.x )         # comps[pref]  = specific lorentz curve from best_fit
            
            for i in range(len(self.x)):
                if (self.x[i] > 1400):
                    data_point = i
                    break

            if showPlot:
                plt.plot( self.x[:data_point], self.y_old[:data_point], color = 'c', label = 'D-peak' )                # plot the curve D-Peak
                plt.plot( self.x[data_point+1:], self.y_old[data_point+1:], color = 'c', label = 'G-Peak' )            # plot the curve G-Peak
                plt.plot( self.x, self.fit_line, label = "Fit Model", linewidth = 1 )
                for i, l in enumerate( self.comps.items() ):
                    plt.plot( self.x, l[1], label = f"Curve {i+1}", lw = 1, linestyle = 'dotted' )
                plt.xlabel( 'cm^-1' )
                plt.ylabel( 'Intensity' )
                plt.grid(True)
                plt.title( "Fit Model" )
                if len(self.peaks_x) > 0:
                    x_min = min(self.peaks_x) - 150                                      # defining the lower x-limit 
                    x_max = max(self.peaks_x) + 150                                      # defining the upper x-limit
                    if (self.x[-1] < x_max):                                             # handling out-of-bounds-error
                        x_max = self.x[-1]
                    if (self.x[0] > x_min):                                              # handling out-of-bounds-error
                        x_min = self.x[0]
                plt.xlim(x_min, x_max)                                                   # set the x-axis limits
                plt.legend()
                plt.gcf().suptitle(f'Curves of {self.filename} - Stage 2')               # set the title of the figure
                plt.show()

                plt.plot( self.x, self.fit_line, label = "Background Removed" )
                plt.xlabel( 'cm^-1' )
                plt.ylabel( 'Intensity' )
                plt.grid(True)
                plt.title( "Fit Model with background removed" )
                if len(self.peaks_x) > 0:
                    x_min = min(self.peaks_x) - 150                                      # defining the lower x-limit 
                    x_max = max(self.peaks_x) + 150                                      # defining the upper x-limit
                    if (self.x[-1] < x_max):                                             # handling out-of-bounds-error
                        x_max = self.x[-1]
                    if (self.x[0] > x_min):                                              # handling out-of-bounds-error
                        x_min = self.x[0]
                plt.xlim(x_min, x_max)                                                   # set the x-axis limits
                plt.legend()
                plt.show()

        # Fit curves at peaks
        else:
            print( "No peaks found!" )

    # Setter of G-Peak
    def set_g_peak(self, x_value):
        
        """
            set_g_peak(self, x_value)

            Set the x-value of the G peak.

            Parameters
            ----------
            x_value : float
                The x-value of the G peak to set.

            Returns
            -------
            None

        """
        self.g_peak = x_value
    
    # Getter of G-Peak
    def get_g_peak(self):
        
        """
            get_g_peak(self)
            
            Get the x-value of the G peak.

            Parameters
            ----------
            None

            Returns
            -------
            float
                The x-value of the G peak.

        """
        return self.g_peak

    # Returns component of fit peak curves
    def get( self, index ):

        """
            get( self, index )

            Returns the model curve associated with the given index.

            Parameters
            ----------
            index : int
                Index of the fit curve in the model. This value must be between 0 and the number of peaks found

            Returns
            -------
            curve : numpy.array
                1D array of the y-values associated with the requested peak curve

        """

        #print( self.comps )

        # Check if index is within range
        if index > len( self.comps ) or index < -1:
            print( "Index out of range!" )
            return None

        else:
            return list( self.comps.values() )[index]

    # Plots the component of the fit peak curves
    def show( self, index ):

        """
            show( self, index )

            Plots the model curve associated with the given index.

            Parameters
            ----------
            index : int
                Index of the fit curve in the model. This value must be between 0 and the number of peaks found

        """

        ret     = self.get( index )

        if not ret is None:
            plt.plot( self.x, ret )
            plt.show()

if __name__ == "__main__":

    here    = os.path.abspath( os.path.dirname( __file__ ) )
    fname   = os.path.join( here, 'sample_raman_data', 'sample_raman.csv' )
    data    = np.genfromtxt( fname, delimiter = ',' )                           # Loads data with Numpy into numpy arrays

    x       = data[ :, 0 ]                                                      # Slice data to get x-values (cm^-1)
    y       = data[ :, 1 ]                                                      # Slice data to get y-values (intensity)

    if x[0] > x[1]:                                                             # Sort lowest to highest
        x = x[::-1]
        y = y[::-1]

    RF      = RamanFitter_stage_2( x = x, y = y, autorun = True )                       # Call class

    RF.NormalizeData()                                                                                 # Normalize data to 1 (Good for comparisons of other plots and machine learning)
    RF.Denoise( ShowPlot = True )                                                                      # Remove noise from data
    RF.FindPeaks( RF.txt_file_dictionary, RF.center_bounds, DistBetweenPeaks = 50, showPlot = False )  # Find the peaks in the cleaned data
    RF.FitData( RF.center_bounds, showPlot = True )                                                    # Fit the original data utilizing the found peaks