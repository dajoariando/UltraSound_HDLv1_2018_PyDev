'''25% overlap python code optimized by vida
Created on Apr 10, 2018

@author: vxp126
'''

import time
from datetime import timedelta
import numpy as np  # To define array and some basic math on arrays
import os  # To find and change directories
import matplotlib.pyplot as plt  # To plot the data
from scipy.signal import hilbert  # To do hilbert transform
from subprocess import Popen, PIPE
import subprocess
from time import sleep
import pydevd

# settings
en_remote_dbg = 0  # enable remote debugging. Enable debug server first!
en_echo_fig = 1

# remote debug setup
server_ip = '129.22.143.84'
client_ip = '129.22.143.39'
if en_remote_dbg:
    from pydevd_file_utils import setup_client_server_paths
    server_path = '/root/ultrasound_python/'
    # client_path = 'D:\\GDrive\\WORKSPACES\\Eclipse_Python_2018\\RemoteSystemsTempFiles\\' + \
    # server_ip + '\\root\\nmr_pcb20_hdl10_2018\\MAIN_nmr_code\\' # client
    # path with remote system
    client_path = 'V:\\ultrasound_python\\'  # client path with samba
    PATH_TRANSLATION = [(client_path, server_path)]
    setup_client_server_paths(PATH_TRANSLATION)
    pydevd.settrace(client_ip)


time_sample = 8000  # number of time samples coming from sampling frequency
channel = 88  # number of total initial channels including overlapping

'''
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
b = np.array([10, 20, 50])

a = np.reshape(a, (3, 3))
b = np.reshape(b, (3, 1))
c = a * b

# test reshaping
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
              14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
a = np.reshape(a, (3, 8))

c = np.transpose(a)
d = np.reshape(c, (6, 4))
e = np.reshape(d, (8, 3))
f = np.transpose(e)
'''

while True:
    start_time = time.monotonic()
    
    
    # DATA FROM SOC
    I = np.zeros(shape=(channel, time_sample))  # define an array to keep
    process = Popen(['../c_exec/ultrasound_oldMux'],
                    stdout=PIPE, stderr=PIPE, shell=True)
    stdout, stderr = process.communicate()
    stdchar = stdout.split()
    I = [int(x) for x in stdchar]
    I = np.array(I)
    I = np.reshape(I, (channel, time_sample))
    # DATA FROM SOC
    
    '''
    # DATA FROM TEXTFILE
    I = np.loadtxt('D:/10_11_18_TX_ON_Probe_on_Jello_Tube_with_Water2.txt',
                   delimiter=' ', usecols=range(8000))  # in matlab
    # DATA FROM TEXTFILE
    '''
    '''
    # DATA FROM TEXTFILE
    with open(os.devnull, "w") as f:
        subprocess.call(['../c_exec/de10-standard_test'], stdout=f)
    sleep(0.1)
    I = np.loadtxt('databank.txt',
                   delimiter=' ', usecols=range(8000))  # in matlab
    # DATA FROM TEXTFILE
    '''

    # plot echo fig
    if (en_echo_fig):
                # plot many figures
        plt.figure(1)
        for i in range(0, 8, 1):
            plt.subplot(8, 1, i+1)
            plt.plot(I[i, :])
        echofig = plt.gcf()
        echofig.show()
        echofig.canvas.draw()
        echofig.clf()

    for i in range(0, channel):
        I[i, :] = I[i, :] - np.average(I[i, :])

    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))

    I = hilbert(I)
    N = 8

    I = np.transpose(I)
    I = np.reshape(I, (88000, 8))
    I = np.fft.fft(I, axis=1)

    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))

    window = np.array(
        [0.3, 0.7, 1, 1, 1, 1, 0.7, 0.3])  # define a triangular window

    I = I * np.transpose(window)
    I = np.reshape(I, (8000, 88))
    I = np.transpose(I)

    #S = np.zeros(shape=(8, time_sample))
    #S_tri = np.zeros(shape=(8, time_sample)).astype(complex)
    I_tri = np.zeros(shape=(channel, time_sample)).astype(complex)
    I_windowed = np.zeros(shape=(65, time_sample)).astype(complex)

    I_tri = I

    # for j in range(0, 81, 8):
    #    S = I[range(j, j + 8), :]
    #    S_tri = S * window
    #    I_tri[range(j, j + 8), :] = S_tri

    #I = np.transpose(I)

    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))

    # print(I_tri[10:15,:])
    # print(I_tri.shape)

    I_windowed[range(0, 2), :] = I_tri[range(0, 2), :]
    I_windowed[range(63, 65), :] = I_tri[range(83, 85), :]

    for i in range(4, 81, 8):
        I_windowed[range(int(3 * i / 4), int(3 * i / 4) + 4),
                   :] = I_tri[range(i - 1, i + 3), :]

    for i in range(8, 81, 8):
        I_windowed[range(int(3 * i / 4) + 1, int(3 * i / 4) + 3),
                   :] = I_tri[range(i - 1, i + 1), :] + I_tri[range(i + 1, i + 3), :]

    # print(I_windowed[0:5,:])
    # print(I_windowed.shape)

    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))

    P_magnitude = np.abs(I_windowed)
    P_dimension = P_magnitude.shape

    P_scaled = np.zeros(shape=(P_dimension[0], P_dimension[1] // 8))

    comp_factor = 10
    # for i in range(0, 64):
    for k in range(0, P_dimension[1] // comp_factor):
        #P_scaled[:, k] = (1 / 10) * (P_magnitude[:, (range(10 * k, 10 * (k + 1)))].sum())
        P_scaled[:, k] = np.mean(
            P_magnitude[:, (range(comp_factor * k, comp_factor * (k + 1)))], axis=1)

    #    P_scaled = zeros(size(P_magnitude, 1), size(P_magnitude, 2) / 16);

    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))

    # plt.figure()
    # im = plt.imshow(P_red, cmap = 'gray', interpolation='nearest')  # use glumpy for real time display, it is faster
    # im = plt.imshow(P_magnitude[:, :1000], cmap='gray')
    plt.figure(2)
    im = plt.imshow(P_scaled[:, 50:300], cmap='gray')

    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))
    print('\n')
    plt.colorbar(im, orientation='horizontal')

    fig = plt.gcf()
    fig.show()

    fig.canvas.draw()
    plt.pause(0.01)
    fig.clf()
