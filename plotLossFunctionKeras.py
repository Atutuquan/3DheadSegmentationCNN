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

    bashCommand_getSpec = "grep -A 7 'VALIDATION EVALUATION' " + file + "  | grep 'Specificity' | awk '{print $11}'| sed 's/]//' "
    bashCommand_getSpec_Test = "grep -A 7 'SUMMARY STATIS' " + file + " | grep 'Specificity' | awk '{print $9}'| sed 's/]//' "

    bashCommand_getSens = "grep -A 7 'VALIDATION EVALUATION' " + file + "  | grep 'Sensitivity' | awk '{print $11}'| sed 's/]//' "
    bashCommand_getSens_Test = "grep -A 7 'SUMMARY STATIS' " + file + " | grep 'Sensitivity' | awk '{print $9}'| sed 's/]//' "

    bashCommand_getPrec = "grep -A 7 'VALIDATION EVALUATION' " + file + "  | grep 'Precision' | awk '{print $11}'| sed 's/]//' "
    bashCommand_getPrec_Test = "grep -A 7 'SUMMARY STATIS' " + file + " | grep 'Precision' | awk '{print $9}'| sed 's/]//' "

    bashCommand_getDSC = "grep -A 7 'VALIDATION EVALUATION' " + file + "  | grep 'DCS' | awk '{print $11}'| sed 's/]//' "
    bashCommand_getDSC_Test = "grep -A 7 'SUMMARY STATIS' " + file + " | grep 'DCS' | awk '{print $9}'| sed 's/]//' "

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

    p = Popen(bashCommand_getSens, stdout=PIPE, shell=True)
    output = p.communicate()
    sens = output[0].split()

    p = Popen(bashCommand_getSens_Test, stdout=PIPE, shell=True)
    output = p.communicate()
    sens_test = output[0].split()

    p = Popen(bashCommand_getDSC, stdout=PIPE, shell=True)
    output = p.communicate()
    DSC = output[0].split()

    p = Popen(bashCommand_getDSC_Test, stdout=PIPE, shell=True)
    output = p.communicate()
    DSC_test = output[0].split()

    p = Popen(bashCommand_getPrec, stdout=PIPE, shell=True)
    output = p.communicate()
    prec = output[0].split()

    p = Popen(bashCommand_getPrec_Test, stdout=PIPE, shell=True)
    output = p.communicate()
    prec_test = output[0].split()

    p = Popen(bashCommand_getSpec, stdout=PIPE, shell=True)
    output = p.communicate()
    spec = output[0].split()

    p = Popen(bashCommand_getSpec_Test, stdout=PIPE, shell=True)
    output = p.communicate()
    spec_test = output[0].split()



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

    for i in range(0, len(spec)):
    	spec[i] = float(spec[i])


    for i in range(0, len(spec_test)):
    	spec_test[i] = float(spec_test[i])


    for i in range(0, len(sens)):
    	sens[i] = float(sens[i])

    for i in range(0, len(sens_test)):
    	sens_test[i] = float(sens_test[i])


    for i in range(0, len(DSC)):
    	DSC[i] = float(DSC[i])


    for i in range(0, len(DSC_test)):
    	DSC_test[i] = float(DSC_test[i])


    for i in range(0, len(prec)):
    	prec[i] = float(prec[i])


    for i in range(0, len(prec_test)):
    	prec_test[i] = float(prec_test[i])


    train = movingAverageConv(train, window_size = w)
    Tacc = movingAverageConv(Tacc, window_size = w)
    val = movingAverageConv(val, window_size = w)
    Vacc = movingAverageConv(Vacc, window_size = w)
    spec = movingAverageConv(spec, window_size = w)
    #spec_test = movingAverageConv(spec_test, window_size = w)
    sens = movingAverageConv(sens, window_size = w)
    #sens_test = movingAverageConv(sens_test, window_size = w)
    prec = movingAverageConv(prec, window_size = w)
    #prec_test = movingAverageConv(prec_test, window_size = w)
    DSC = movingAverageConv(DSC, window_size = w)
    #DSC_test = movingAverageConv(DSC_test, window_size = w)


    plt.clf()
    plt.subplot(351)
    ax = plt.gca()
    #print(str(file)[])
    plt.plot(range(len(train)),train,'b-')
    plt.plot(range(len(Tacc)),Tacc,'r-')
    #plt.xlabel('weight updates')
    plt.title('Training')
    plt.axis('tight')
    plt.legend(('Loss', 'Accuracy'))

    plt.subplot(356)
    plt.plot(range(0,len(val)),val,'b-')
    plt.plot(range(0,len(Vacc)),Vacc,'r-')
    plt.legend(('Loss', 'Accuracy'))
    plt.title('Test')
    plt.ylabel('Test set during training')

    plt.subplot(357)
    plt.plot(range(len(spec)),spec,'b-')
    plt.title(('Specificity'))

    plt.subplot(358)
    plt.plot(range(len(sens)),sens,'b-')
    plt.title(('Sensitivity'))

    plt.subplot(359)
    plt.plot(range(len(prec)),prec,'b-')
    plt.title(('Precision'))

    plt.subplot(3,5,10)
    plt.plot(range(len(DSC)),DSC,'k-')
    plt.title(('DCS'))
    
    plt.subplot(3,5,12)
    plt.plot(range(len(spec_test)),spec_test,'b-')
    plt.title(('Specificity'))
    plt.ylabel('Full head segmentation')

    plt.subplot(3,5,13)    
    plt.plot(range(len(sens_test)),sens_test,'b-')
    plt.title(('Sensitivity'))


    plt.subplot(3,5,14)    
    plt.plot(range(len(prec_test)),prec_test,'b-')
    plt.title(('Precision'))


    plt.subplot(3,5,15)    
    plt.plot(range(len(DSC_test)),DSC_test,'k-')
    plt.title(('DCS'))
    


    

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
    
    
    
