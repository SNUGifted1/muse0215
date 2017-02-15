import argparse
import numpy as np
import matplotlib.pyplot as plt
import bci_workshop_tools as BCIw

from pythonosc import dispatcher
from pythonosc import osc_server

data = np.zeros([1,5]) 

def eeg_handler(unused_addr, args, TP9, AF7, AF8, TP10, Status):
    # OSC가 /muse/eeg 주소의 데이터를 찾으면 실행되는 함수이다. 
    
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

    # cmd에서 파일명 뒤에 입력된 ip, port를 파악하여 서버 주소로 넘겨준다.

    parser = argparse.ArgumentParser()

    parser.add_argument("--ip", default="0.0.0.0", help="The ip to listen on")

    parser.add_argument("--port", type=int, default=0000, help="The port to listen on")

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
    names_of_features = BCIw.feature_names(params['names of channels'])
    
    
    #%% Initialize the buffers for storing raw EEG and features

    # Initialize raw EEG data buffer (for plotting)
    eeg_buffer = np.zeros((params['sampling frequency']*eeg_buffer_secs, 
                           len(params['names of channels']))) 
    
    # Compute the number of windows in "eeg_buffer_secs" (used for plotting)
    n_win_test = int(np.floor((eeg_buffer_secs - win_test_secs) / float(shift_secs) + 1))
    
    # Initialize the feature data buffer (for plotting)
    feat_buffer = np.zeros((n_win_test, len(names_of_features)))
        
    # Initialize the plots
    plotter_eeg = BCIw.dataPlotter(params['sampling frequency']*eeg_buffer_secs, params['names of channels'],
                                   params['sampling frequency'])
    
    plotter_feat = BCIw.dataPlotter(n_win_test,
                                    names_of_features,
                                    1/float(shift_secs))

    # 디스패쳐를 정의하고 OSC주소를 함수에 매칭시켜 놓는다.

    dispatcher = dispatcher.Dispatcher()

    dispatcher.map("/debug", print)

    dispatcher.map("/muse/eeg", eeg_handler, "EEG")

    # 서버를 정의하고 서버를 실행한다. 

    server = osc_server.ThreadingOSCUDPServer((args.ip, args.port), dispatcher)

    print("Serving on {}".format(server.server_address))
    
    try:
        while 1: 
            eeg_data = getdata(shift_secs, params) # Obtain EEG data from MuLES  
            eeg_buffer = BCIw.updatebuffer(eeg_buffer, eeg_data) # Update EEG buffer
            
            """ 2- COMPUTE FEATURES """
            # Get newest samples from the buffer 
            data_window = BCIw.getlastdata(eeg_buffer, win_test_secs * params['sampling frequency'])
            # Compute features on "data_window" 
            feat_vector = BCIw.compute_feature_vector(data_window, params['sampling frequency'])
            feat_buffer = BCIw.updatebuffer(feat_buffer, np.asarray([feat_vector])) # Update the feature buffer

            
            """ 3- VISUALIZE THE RAW EEG AND THE FEATURES """       
            plotter_eeg.updatePlot(eeg_buffer) # Plot EEG buffer     
            plotter_feat.updatePlot((feat_buffer)) # Plot the feature buffer 
            
            plt.pause(0.001)
                       
    except KeyboardInterrupt:
        server.shutdown()
   
    finally:
        server.shutdown()


