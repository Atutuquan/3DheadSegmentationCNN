#!/usr/bin/python

# given input train log file, extract loss function on validation and training. Feed into python script for plotting

import sys, getopt, os
import matplotlib.pyplot as plt
from subprocess import PIPE, Popen
import numpy as np


def movingAverageConv(a, window_size=1) :
    if not a : return a
    window = np.ones(int(window_size))
    result = np.convolve(a, window, 'full')[ : len(a)] # Convolve full returns array of shape ( M + N - 1 ).
    slotsWithIncompleteConvolution = min(len(a), window_size-1)
    result[slotsWithIncompleteConvolution:] = result[slotsWithIncompleteConvolution:]/float(window_size)
    if slotsWithIncompleteConvolution > 1 :
        divisorArr = np.asarray(range(1, slotsWithIncompleteConvolution+1, 1), dtype=float)
        result[ : slotsWithIncompleteConvolution] = result[ : slotsWithIncompleteConvolution] / divisorArr
    return result

def makeFiles(argv):
    file = ''
    w = 1
    try:
		opts, args = getopt.getopt(argv,"hf:m:",["file=", "movingAverage="])
    except getopt.GetoptError:
		print('plotLossFunctionKeras.py -f <train log full file address: /home/...> -m <moving average>')
		sys.exit(2)
    for opt, arg in opts:
		if opt == '-h':
			print('plotLossFunctionKeras.py -f <train log full file address : /home/...> -m <moving average>')
			sys.exit()
		elif opt in ("-f","--file"):
			file = str(arg)
		elif opt in ("-m","--movingAverage"):
			w = int(arg)   

    print(w)

    bashCommand_getTrain = "grep 'Train cost and accuracy' " +  file + " | awk '{print $5}' | grep -o '[0-9].*' | sed 's/,//' "
    bashCommand_getAccTrain = "grep 'Train cost and accuracy' " + file +  " | awk '{print $6}' | grep -o '[0-9].*' | sed 's/]//' "

    bashCommand_getVal = "grep 'Validation cost and accuracy' " +  file + " | awk '{print $5}' | grep -o '[0-9].*' | sed 's/,//' "
    bashCommand_getAccVal = "grep 'Validation cost and accuracy' " + file +  " | awk '{print $6}' | grep -o '[0-9].*' | sed 's/]//' "


    p = Popen(bashCommand_getTrain, stdout=PIPE, shell=True)
    output = p.communicate()
    train = output[0].split()

    p = Popen(bashCommand_getAccTrain, stdout=PIPE, shell=True)
    output = p.communicate()
    Tacc = output[0].split()
    	
    p = Popen(bashCommand_getVal, stdout=PIPE, shell=True)
    output = p.communicate()
    val = output[0].split()

    p = Popen(bashCommand_getAccVal, stdout=PIPE, shell=True)
    output = p.communicate()
    Vacc = output[0].split()


    for i in range(0,len(train)-1):
        train[i] = float(train[i])
    train = train[:-1]
    
   
    for i in range(0,len(val)-1):
        val[i] = float(val[i])
    val = val[:-1]
    
    for i in range(0, len(Tacc)-1):
        Tacc[i] = float(Tacc[i])
    Tacc = Tacc[:-1]

    for i in range(0, len(Vacc)-1):
    	Vacc[i] = float(Vacc[i])
    Vacc = Vacc[:-1]

    plt.clf()
    plt.subplot(211)
    ax = plt.gca()

    train = movingAverageConv(train, window_size = w)
    val = movingAverageConv(val, window_size = w)
    
    plt.plot(range(len(train)),train,'b-')
    plt.plot(range(0,len(train),(len(train)/len(val))),val,'r-')
    plt.show()

    plt.xlabel('weight updates')
    plt.ylabel('loss')
    plt.axis('tight')
    plt.legend(('train set', 'validation set','uniform sample validation set'))
    plt.subplot(212)
    ax = plt.gca()

    Tacc = movingAverageConv(Tacc, window_size = w)
    Vacc = movingAverageConv(Vacc, window_size = w)
    
    plt.plot(range(len(Tacc)),Tacc,'b-')
    plt.plot(range(0,len(Tacc),(len(Tacc)/len(Vacc))),Vacc,'r-')
    plt.show()

    #plt.plot(train)
    
    #plt.plot(val)

    #plt.plot(acc)

    #plt.plot(Tacc)

    #plt.show()




    '''plt.clf()
    plt.subplot(211)
    ax = plt.gca()
    ax.plot(train)
    ax.plot(val)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.axis('tight')
    plt.legend(('train set', 'validation set'))
    plt.subplot(212)
    ax = plt.gca()
    ax.plot(Tacc)
    ax.plot(Vacc)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.axis('tight')
    plt.legend(('train set', 'validation set'))
    plt.show()'''


    
    
if __name__ == "__main__":
	makeFiles(sys.argv[1:])
    
    
    
