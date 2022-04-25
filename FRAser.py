#! /usr/bin/python
# FRAser - A command line frequency response analyser, written in Python
# Copyright (C) 2022  Simon Williams

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import configparser
import logging as lg
import os, sys, getopt, wave, time
import sounddevice as sd
import soundfile as sf
import numpy as np
import scipy.fft as fft
from numpy.random import default_rng
from scipy.signal import butter,sosfilt
from scipy.signal.windows import blackman
from scipy.interpolate import interp1d
from tabulate import tabulate

def callback(outdata, frames, time, status):
    try:
        lg.info("Callback called..")
        if status:
            print(status)
        for x in range(0, args.channels):     
            outdata[:,x] = noise_filtered
        lg.info("Callback done..")
    except KeyboardInterrupt:
        sys.exit(2)

##Function to compute the frequency dependent rolling average of the audio data as a smoothing function.
def rolling_average(input_array, output_array, smoothing_factor):
    for x in range(len(input_array)):
        low_index = x - int(x/smoothing_factor)
        if low_index < 0: low_index = 0
        high_index = x + int(x/smoothing_factor)
        if high_index > len(input_array)-1: high_index = len(input_array)-1
        output_array[x] = sum(input_array[low_index : high_index+1]) / ((high_index+1) - low_index)
        
def main(argv):
    global noise
    global noise_filtered
    global args
    input_type = 'int16'
    #output_type = 'float32'
    output_type = 'int16'
    smoothing_factor = 100
   
    #Parse arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-v", "--verbose",help="Be verbose",action='store_true')
    parser.add_argument("-q", "--quiet",help="Be quiet",action='store_true')    
    parser.add_argument("-p", "--plots",help="Show plots",action='store_true')
    parser.add_argument("-l", "--list-devices",help="List the available devices",action='store_true')
    parser.add_argument("-i", "--input",help="The input device",default=-1,type=int)
    parser.add_argument("-o", "--output",help="The output device",default=-1,type=int)
    parser.add_argument("-c", "--channels",help="The number of channels",default=1,type=int)
    parser.add_argument("-r", "--rate",help="The sample rate",default=-1,type=int)
    parser.add_argument("-t", "--time",help="Sampling time",default=5,type=int)
    #parser.add_argument("--report",help="Sampling time.",nargs='*',type=str)
    parser.add_argument("--min-freq",help="The minimum FFT frequency",default=10,type=int)
    parser.add_argument("--max-freq",help="The maximum FFT frequency",default=22000,type=int)
    parser.add_argument("--fft-window",help="The FFT window size",default=8192,type=int)
    parser.add_argument("--fft-threads",help="The number of threads to give the FFT argorithm. Default '-1' takes cpu_count",default=-1,type=int)
    parser.add_argument("-s", "--save-input",help="Save the recorded input to a .wav file",action='store_true')
    parser.add_argument("--save-filename",help="Filename to save input data",default="./FRAser_input.wav",type=str)
    parser.add_argument("--log",help="The level of logging to use (debug, info, warning, error, critical)",default="WARNING",type=str)
    parser.add_argument("--logfile",help="Where to log",default="./FRAser.log",type=str)
    
    args = parser.parse_args()
    
    ##Set up logging
    FORMAT = '%(asctime)-15s:%(levelname)s:%(message)s'
    numeric_level = getattr(lg, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log)
    logger = lg.getLogger()
    lg.basicConfig(filename = args.logfile, level=numeric_level, encoding='utf-8', format=FORMAT)
    logger.addHandler(lg.StreamHandler())
    lg.info("---FRAser.py starting---")
    #if not args.quiet: print("---FRAser.py starting---")
    
    if(args.plots):
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            lg.error("Matplotlib is required for plotting")
            args.plots = 0
            
    if args.input == -1 or args.output == -1:
        lg.error("Channel error, exiting")
        print("  You must specify channels for input and output, see usage:\n")
        parser.print_help()
        sys.exit(2)
    
    if args.list_devices == True:
        print(sd.query_devices())
        sys.exit()
            
    if args.rate == -1:
        args.rate = int(sd.query_devices(args.input)['default_samplerate'])
                
    
    
    ##Compute nyquist freqs
    nyquist = int(args.rate/2)
    if args.max_freq > nyquist:
        args.max_freq = nyquist-1
        
    ##Print useful information.
    lg.info("Input device: %s", sd.query_devices(args.input)["name"])
    lg.info("Output device: %s", sd.query_devices(args.output)["name"])
    lg.info("Sample rate: %skHz", args.rate/1000)
    lg.info("Nyquist frequency: %skHz", nyquist/1000)
    lg.info("Channels: %s", args.channels)
    lg.info("Sampling time: %ss", args.time)
    lg.info("FFT window: %s", args.fft_window)
    lg.info("Min frequency: %sHz", args.min_freq)
    lg.info("Max frequency: %skHz", args.max_freq/1000)
    
    ##Generate white noise
    frames = args.time * args.rate                              #Calculate number of frames to use
    lg.info("Calculated frames: %d", frames)
    #rng = default_rg()
    noise = np.random.uniform(-2**14,2**14,frames)          #Generate some white noise     
    lg.debug("Noise data: %r", noise)
    
    ##Filter the noise
    sos = butter(10,                                        #Generate filter params for SOS 10th order
        (args.min_freq,args.max_freq),
        'bandpass',
        fs=args.rate,
        output='sos')
    noise_filtered = np.int16(sosfilt(sos,noise))        #Pass the white noise through the bandpass filter
    
    ##FFT the noise
    window = blackman(args.fft_window*2)                    #The window function
    number_chunks = int(frames/(args.fft_window*2))         #The number of chunks we need

    if not number_chunks:
        lg.error("The sample size is too short for the chosen FFT window size.")
        sys.exit(2)
    
    lg.info("Number of chunks: %s", number_chunks)
    fft_mags = np.empty([int(args.fft_window), number_chunks])   #Make an empty array for the FFT magnitude data
    for x in range(number_chunks):
        chunk = noise_filtered[x*args.fft_window*2 : (x*args.fft_window*2)+args.fft_window*2]   #Split the data into a chunk
        fft_raw = fft.rfft(chunk*window, n=args.fft_window*2, workers=args.fft_threads)          #Do the actual FFT, applying the window function
        fft_mags[:,x] = abs(fft_raw[0:int(args.fft_window)])        #Convert the complex values to magnitude
    
    fft_mags_mean = np.average(fft_mags,1)                       #Average the FFT data.
    lg.debug("Averaged noise FFTs: %s", fft_mags_mean)
    
    ##Scale the FFT result to the nyquist frequency
    scale_factor = nyquist/(args.fft_window-1)                      #Calculate scaling factor.
    lg.debug("FFT frequency scaling factor: %s", scale_factor)      
    f_interp = interp1d(range(args.fft_window), fft_mags_mean)      #Get an interpolation function.
    fft_final = np.empty([nyquist])                              #Create an empty array for the final FFT result.
    for x in range(nyquist):
        fft_final[x] = f_interp(x/scale_factor)                     #Fill the output array with the interpolated results.
        
    fft_normal = 20*np.log10(fft_final) - (20 * np.log10(args.fft_window * pow(2,14)))
        
    if args.plots:
        plt.subplot(1+args.channels,1,1)
        plt.title("Output spectrum")
        plt.grid(True)
        plt.semilogx(fft_normal)
        fft_smooth = np.empty([nyquist])
        rolling_average(fft_normal, fft_smooth, smoothing_factor)
        plt.semilogx(fft_smooth)
        plt.ylim(bottom=fft_smooth[0]-10)
    
    #Check that the formats are supported.
    try:
        sd.check_input_settings(
            device = args.input,
            samplerate = args.rate,
            channels = args.channels,
            dtype = input_type)
    except sd.PortAudioError as e:
        lg.error("Input format not supported, exiting: %s", str(e))
        sys.exit(2)
        
    try:
        sd.check_output_settings(
            device = args.output,
            samplerate = args.rate,
            channels = args.channels,
            dtype = output_type)
    except sd.PortAudioError as e:
        lg.error("Output format not supported, exiting: %s", str(e))
        sys.exit(2)
                    
    #Open the streams
 
    #stream_in = sd.InputStream(
    #    device = args.input,
    #    samplerate = args.rate, 
    #    channels = args.channels,
    #    dtype = input_type,
    #    blocksize = frames)

    #stream_out = sd.OutputStream(
    #    device = args.output,
    #    samplerate = args.rate, 
    #    channels = args.channels,
    #    dtype = output_type,
    #   blocksize = frames,
    #    callback = callback)
    
    #record_data = np.empty([frames, args.channels], dtype='int16')
        
    try:
        lg.info("Starting output stream")
        #stream_out.start()
        lg.info("Starting input stream")
        #stream_in.start()
        
        record_data = sd.playrec(
            noise_filtered,
            #out = record_data,
            device = (args.input, args.output),
            channels = args.channels,
            samplerate = args.rate,
            blocksize = args.rate,
            dtype = (input_type, output_type)
            )
        
        stream = sd.get_stream()
        counter = float(args.time);
        while stream.active:
            print("Time left: %1.1f" %counter, end='\r')
            if counter > 0: counter -= 0.1
            time.sleep(0.1)
    except KeyboardInterrupt:
        sys.exit(2)
    
    lg.info("Stopping input stream")
    lg.info("Stopping output stream")

    lg.info("Sampling done")
    lg.debug("Recorded data: %r", record_data)
    
    if args.save_input:
        sf.write(args.save_filename, record_data, args.rate)
    
    ##Filter the noise, using the existing params
    #sos = signal.butter(10,(args.min_freq,args.max_freq),'bandpass',fs=args.rate,output='sos')
    lg.info("Filtering input")
    filtered = np.empty([frames, args.channels])
    for x in range(args.channels):
        filtered[:,x] = sosfilt(sos,record_data[:,x])
        #print(filtered[:,x])
    
    ##FFT the recorded data
    window = blackman(args.fft_window*2)                #The window function
    number_chunks = int(frames/(args.fft_window*2))     #The number of chunks we need

    if not number_chunks:
        lg.error("The sample size is too short for the chosen FFT window size.")
        sys.exit(2)
    
    lg.info("Number of chunks: %s", number_chunks)
    
    fft_normal = np.empty([nyquist, args.channels])
    fft_smooth = np.empty([nyquist, args.channels])
    
    ##Do the FFT for each channel, average the chunks, 
    for channel in range(args.channels):
        
        lg.info("FFT channel %s", str(channel))
        fft_mags = np.empty([int(args.fft_window), number_chunks])   #Make an empty array for the FFT magnitude data
        fft_mags_mean = []                                              #Make an empty array for the averaged FFTs
        for x in range(number_chunks):
            chunk = filtered[x*args.fft_window*2 : (x*args.fft_window*2)+args.fft_window*2, channel]   #Split the data into a chunk
            fft_raw = fft.rfft(chunk*window, n=args.fft_window*2, workers=args.fft_threads)    #Do the actual fft
            fft_mags[:,x] = abs(fft_raw[0:int(args.fft_window)])    #Convert the complex values to magnitude
        
        fft_mags_mean = np.average(fft_mags,1)                       #Average the FFT data
        lg.debug("Averaged data FFTs: %s", fft_mags_mean)
        
        ##Scale the FFT result to the nyquist frequency     
        f_interp = interp1d(range(args.fft_window), fft_mags_mean)      #Get an interpolation function.
        fft_final = np.empty([nyquist])                              #Create an empty array for the final FFT result.
        for x in range(nyquist):
            fft_final[x] = f_interp(x/scale_factor)                     #Fill the output array with the interpolated results.
            
        fft_normal[:,channel] = 20*np.log10(fft_final) - (20*np.log10(args.fft_window * pow(2, 14)))
        
        if args.plots:
            plt.subplot(1+args.channels,1,2+channel)
            plt.title("Recorded spectrum channel " + str(channel))
            plt.grid(True)
            plt.semilogx(fft_normal[:,channel])
            fft_smooth[:,channel] = np.empty([nyquist])
            rolling_average(fft_normal[:,channel], fft_smooth[:,channel], smoothing_factor)
            plt.semilogx(fft_smooth[:,channel])
            plt.ylim(bottom=fft_smooth[0,channel]-10)
        
    if args.plots:            
        plt.xlabel("Frequency (Hz)")       
        plt.show()
        
    ##Print data table
    table_data = [ ['20Hz', str(fft_smooth[20,0]) + 'dB'],
                  ['1kHz', str(fft_smooth[1000,0]) + 'dB'],
                  ['20kHz', str(fft_smooth[20000,0]) + 'dB'] ]
    print(tabulate(table_data, headers=("Freq", "Channel 0")))
    
    #for x in np.argsort(fft_mag)[::-1][:10]:
    #    print(x, fft_mag[x])
    
    #for x in args.report:
    #    if x.upper() == 'MAX':
    #        max_x = np.argmax(fft_mags_mean)
    #        print(max_x, fft_mags_mean[max_x])
    #    else:
    #        print(int(x), fft_scaled[int(x)])
    
    sys.exit()
    
if __name__ == "__main__":
    main(sys.argv[1:])
