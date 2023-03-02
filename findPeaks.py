def find_closest_key_index(self, dict_obj, num):
    """
        Find the index of the closest key in the dictionary to the given number
    """
    closest_key = None
    min_diff = None
    for i, key in enumerate(dict_obj.keys()):
        diff = abs(num - key)
        
        if min_diff is None or diff < min_diff:
            min_diff = diff
            closest_key = key

    return list(dict_obj.keys()).index(closest_key)+1

# Find the peaks in the data
def FindPeaks( self, txt_file_dictionary, center_bounds, DistBetweenPeaks = 50, showPlot = False ):

    """
        FindPeaks( self, center_bounds, DistBetweenPeaks = 50, showPlot = False )

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

    self.peaks_x = []   # stores the x_values of peaks
    self.npeaks = []    # stores the indices of the peaks found

    for x_value in center_bounds:
        self.peaks_x.append(x_value)
        closest_x_value_key = self.find_closest_key_index(txt_file_dictionary, x_value)
        self.npeaks.append(closest_x_value_key)
    # self.peaks_y = [ self.y[i] for i in self.npeaks ]

    # showing the plot
    if showPlot:
        plt.plot( self.x, self.y, color = 'c', label = 'Data' )                  # plot the curve
        line = plt.gca().get_lines()[0]                                          # get the first Line2D object
        self.peaks_y = line.get_ydata()[np.searchsorted(self.x, self.peaks_x)]   # get the y-values
        plt.scatter( self.peaks_x, self.peaks_y, color = 'k', label = 'Peaks' )  # scatter the peaks
        plt.xlim( self.x[0], self.x[-1] )                                        # set the x-axis limits
        plt.legend()                                                             # set the legends for graph
        plt.show()                                                               # show the graph


    # print( "Finding Peaks..." )
    # #Find Peaks
    # peaks, _  = signal.find_peaks( self.y, distance = DistBetweenPeaks )
    
    # # Filter peaks by x-axis bounds
    # peak_x = np.arange(len(self.y))[peaks]  # x-coordinates of peaks

    # # Determine prominence of peaks, as compared to surrounding data
    # prominence = signal.peak_prominences( self.y, peaks, wlen = len( self.x )-1 )[0]
    # rem_ind = [ i for ( i, p ) in enumerate( prominence ) if p < self.threshold]
    
    # self.npeaks = np.delete( peaks, rem_ind )
    # # print(f'npeaks: {self.npeaks} \n')
    
    # self.peaks_y    = np.array( [ self.y[i] for i in self.npeaks ] )
    # # print(f'peak_y: {self.peaks_y} \n')
    
    # self.peaks_x    = np.array( [ self.x[i] for i in self.npeaks ] )
    # # print(f'peak_x: {self.peaks_x} \n')

    # filtered_peaks_x = []
    # filtered_peaks_y = []
    # for i in range(len(self.peaks_x)):
    #     for key in center_bounds:
    #         if (self.peaks_x[i] >= center_bounds[key][0]) and (self.peaks_x[i] <= center_bounds[key][1]):
    #             filtered_peaks_x.append(self.peaks_x[i])
    #             filtered_peaks_y.append(self.peaks_y[i])

    # if showPlot:
    #     plt.plot( self.x, self.y, color = 'c', label = 'Data' )
    #     plt.scatter( filtered_peaks_x, filtered_peaks_y, color = 'k', label = 'Peaks' )        
    #     plt.xlim( self.x[0], self.x[-1] )
    #     plt.legend()
    #     plt.show()
