import argparse
import numpy as np
import matplotlib.pyplot as plt

from pythonosc import dispatcher
from pythonosc import osc_server


data = np.zeros([1,5]) 


 
def BCIw_feature_names(ch_names):
    """
    Generate the name of the features
        
    Arguments
    ch_names: List with Electrode names
    """
    bands = ['pwr-delta', 'pwr-theta', 'pwr-alpha' ,'pwr-beta']

    feat_names = []
    for band in bands:
        for ch in range(1,len(ch_names)):
        #Last column is ommited because it is the Status Channel
            feat_names.append(band + '-' + ch_names[ch])
            
    return feat_names 
            
            
def BCIw_compute_feature_vector(eegdata, Fs):
    """
    Extract the features from the EEG
    
    Arguments:
    eegdata: array of dimension [number of samples, number of channels]
    Fs: sampling frequency of eegdata
    
    Outputs:
    feature_vector: np.array of shape [number of feature points; number of different features]

    """
    #Delete first column (Status)
    eegdata = np.delete(eegdata, 0 , 1)    
        
    # 1. Compute the PSD
    winSampleLength, nbCh = eegdata.shape
    
    # Apply Hamming window
    w = np.hamming(winSampleLength)
    dataWinCentered = eegdata - np.mean(eegdata, axis=0) # Remove offset
    dataWinCenteredHam = (dataWinCentered.T*w).T

    NFFT = nextpow2(winSampleLength)
    Y = np.fft.fft(dataWinCenteredHam, n=NFFT, axis=0)/winSampleLength
    PSD = 2*np.abs(Y[0:NFFT/2,:])
    f = Fs/2*np.linspace(0,1,NFFT/2)     
            
    # SPECTRAL FEATURES
    # Average of band powers
    # Delta <4
    ind_delta, = np.where(f<4)
    meanDelta = np.mean(PSD[ind_delta,:],axis=0)
    # Theta 4-8
    ind_theta, = np.where((f>=4) & (f<=8))
    meanTheta = np.mean(PSD[ind_theta,:],axis=0)
    # Alpha 8-12
    ind_alpha, = np.where((f>=8) & (f<=12)) 
    meanAlpha = np.mean(PSD[ind_alpha,:],axis=0)
    # Beta 12-30
    ind_beta, = np.where((f>=12) & (f<30))
    meanBeta = np.mean(PSD[ind_beta,:],axis=0)
    
    feature_vector = np.concatenate((meanDelta, meanTheta, meanAlpha, meanBeta),
                                    axis=0)
    
    feature_vector = np.log10(feature_vector)   
       
    return feature_vector

        
def nextpow2(i):
    """ 
    Find the next power of 2 for number i
    
    """
    n = 1
    while n < i: 
        n *= 2
    return n
       

    
def BCIw_updatebuffer(data_buffer, new_data):
    """
    Concatenates "new_data" into "buffer_array", and returns an array with 
    the same size than "buffer_array" 
    """    
    
    new_samples = new_data.shape[0]
    new_buffer = np.concatenate((data_buffer, new_data), axis =0)
    new_buffer = np.delete(new_buffer, np.s_[0:new_samples], 0)
    
    return new_buffer
    
    
    
def BCIw_getlastdata(data_buffer, newest_samples):
    """
    Obtains from "buffer_array" the "newest samples" (N rows from the bottom of the buffer)
    """
    new_buffer = data_buffer[(data_buffer.shape[0] - newest_samples)::,::]  

    return new_buffer  
    
    
class BCIw_dataPlotter():
    """ 
    Class for creating and updating a line plot
    """
    
    def __init__(self, nbPoints, chNames, fs=None, title=None):
        """Initialize the figure"""
        
        self.nbPoints = nbPoints
        self.chNames = chNames
        self.nbCh = len(self.chNames)
                                
        if fs is None:   # Verify Sampling frequency
            self.fs = 1
        else:
            self.fs = fs
            
        if title is None:
            self.figTitle = ''
        else:
            self.figTitle = title
                    
                    
        data = np.empty((self.nbPoints,1))*np.nan
        self.t = np.arange(data.shape[0])/float(self.fs)
        
        # Create offset parameters for plotting multiple signals
        self.yAxisRange = 100
        self.chRange = self.yAxisRange/float(self.nbCh)
        self.offsets = np.round((np.arange(self.nbCh)+0.5)*(self.chRange))
        
        # Create the figure and axis
        plt.ion()
        self.fig = plt.figure()
        self.ax =  plt.subplot()
        #self.ax.set_xticks([])
        self.ax.set_yticks(self.offsets)
        self.ax.set_yticklabels(self.chNames)
        #self.ax.yaxis.set_ticks(self.chNames)
        
        # Initialize the figure
        plt.title(self.figTitle)
        
        self.chLinesDict = {}
        for i, chName in enumerate(self.chNames):
            self.chLinesDict[chName], = plt.plot(self.t, data+self.offsets[i], label=chName)
            
        #plt.legend()
        plt.xlabel('Time')
        plt.ylim([0, self.yAxisRange])
        plt.xlim([np.min(self.t), np.max(self.t)])
        
        plt.show()
        plt.pause(0.1)
    
    def updatePlot(self, data):
        """ Update the plot """
        
        plt.figure(self.fig.number)  
        #assert (data.shape[1] == self.nbCh), 'new data does not have the same number of channels'
        #assert (data.shape[0] == self.nbPoints), 'new data does not have the same number of points'

        data = data - np.mean(data,axis=0)
        std_data = np.std(data,axis=0)
        std_data[np.where(std_data == 0)] = 1
        data = data/std_data*self.chRange/5.0     
        
        for i, chName in enumerate(self.chNames):
            self.chLinesDict[chName].set_ydata(data[:,i]+self.offsets[i])
        
        plt.draw()
    
    def clear(self):
        """ Clear the figure """
        
        blankData = np.empty((self.nbPoints,1))*np.nan
        
        for i, chName in enumerate(self.chNames):
            self.chLinesDict[chName].set_ydata(blankData)
        
        plt.draw()
    
    def close(self):
        """ Close the figure """
        
        plt.close(self.fig)
        
        
'''
BCIw  tool  END ////////////////////////////////////////////////
'''
 
    
def eeg_handler(unused_addr, args, TP9, AF7, AF8, TP10, Status):

    global data
    temp = np.array([[int(Status), int(TP10), int(AF8), int(AF7), int(TP9)]])
    data = np.concatenate((data, temp), axis=0)


def getdata(seconds, params):
    
    global data
    # Size of data requested
    n_samples = int(round(seconds * params['sampling frequency']))
    n_columns = len(params['data format'])
    data_buffer = -1 * np.ones((n_samples, n_columns)) 
 

    while (data_buffer[0, n_columns - 1]) < 0 : #While the first row has not been rewriten
        server.handle_request()
        new_samples = data.shape[0]
        data_buffer = np.concatenate((data_buffer, data), axis =0)
        data_buffer = np.delete(data_buffer, np.s_[0:new_samples], 0)
        data = np.delete(data, np.s_[0:n_samples], 0)

    return data_buffer
        
     
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--ip", default="0.0.0.0", help="The ip to listen on")

    parser.add_argument("--port", type=int, default=5049, help="The port to listen on")

    args = parser.parse_args()


    #%% Set the experiment parameters
    params = {'names of channels':['Status', 'TP10', 'AF8', 'AF7', 'TP9'], 'data format':[0,0,0,0,0], 'sampling frequency':256}

    eeg_buffer_secs = 5  # Size of the EEG data buffer used for plotting the 
                          # signal (in seconds) 
    win_test_secs = 1     # Length of the window used for computing the features 
                          # (in seconds)
    overlap_secs = 0.5    # Overlap between two consecutive windows (in seconds)
    shift_secs = win_test_secs - overlap_secs
  
    # Get name of features
    names_of_features = BCIw_feature_names(params['names of channels'])
    
    
    #%% Initialize the buffers for storing raw EEG and features

    # Initialize raw EEG data buffer (for plotting)
    eeg_buffer = np.zeros((params['sampling frequency']*eeg_buffer_secs, 
                           len(params['names of channels']))) 
    
    # Compute the number of windows in "eeg_buffer_secs" (used for plotting)
    n_win_test = int(np.floor((eeg_buffer_secs - win_test_secs) / float(shift_secs) + 1))
    
    # Initialize the feature data buffer (for plotting)
    feat_buffer = np.zeros((n_win_test, len(names_of_features)))
        
    # Initialize the plots
    plotter_eeg = BCIw_dataPlotter(params['sampling frequency']*eeg_buffer_secs, params['names of channels'],
                                   params['sampling frequency'])
    
    plotter_feat = BCIw_dataPlotter(n_win_test,
                                    names_of_features,
                                    1/float(shift_secs))


    dispatcher = dispatcher.Dispatcher()

    dispatcher.map("/debug", print)

    dispatcher.map("/muse/eeg", eeg_handler, "EEG")



    server = osc_server.ThreadingOSCUDPServer((args.ip, args.port), dispatcher)

    print("Serving on {}".format(server.server_address))
    
    try:
        while 1: 
            eeg_data = getdata(shift_secs, params) # Obtain EEG data from MuLES  
            eeg_buffer = BCIw_updatebuffer(eeg_buffer, eeg_data) # Update EEG buffer
            
            """ 2- COMPUTE FEATURES """
            # Get newest samples from the buffer 
            data_window = BCIw_getlastdata(eeg_buffer, win_test_secs * params['sampling frequency'])
            # Compute features on "data_window" 
            feat_vector = BCIw_compute_feature_vector(data_window, params['sampling frequency'])
            feat_buffer = BCIw_updatebuffer(feat_buffer, np.asarray([feat_vector])) # Update the feature buffer

            
            """ 3- VISUALIZE THE RAW EEG AND THE FEATURES """       
            plotter_eeg.updatePlot(eeg_buffer) # Plot EEG buffer     
            plotter_feat.updatePlot((feat_buffer)) # Plot the feature buffer 
            
            plt.pause(0.001)
                       
    except KeyboardInterrupt:
        server.shutdown()
   
    finally:
        server.shutdown()


        
        
        
     
    
