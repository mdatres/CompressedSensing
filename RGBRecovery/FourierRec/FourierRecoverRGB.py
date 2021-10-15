from typing import Tuple
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 
import csv
import cmath as math
import cvxpy as cvx
from scipy.fft import fft, ifft
import argparse
import sys
sys.path.append("../..")
# Import from my script 

from utils.MeasurementsConstruction.FourierRandomMatrix.RGBFourierRandomMatrix import RGBFourierRandomMatrix
from utils.scripts.save_rec_as_txt import save_rec_as_txt
from utils.MeasurementsConstruction.FourierRandomMatrix.FourierRandomMatrix import FourierRandomMatrix
from utils.optimizers.optimizersLI import optimizerLI



def FourierRecoverRGB(imagepath, c, lamdathr, fft,  varepsilon=0.01, pathtosavetxt='', alg="ECOS", complex=True): 

    
    x = Image.open(imagepath).convert('RGB')
    width = x.size[0]
    height = x.size[1]
    n = width*height

    print("The dimesion of the signal is:   " + str(n))
    xarr = np.array(x)
    red = xarr[:,:,0].flatten(order='C')
    green = xarr[:,:,1].flatten(order='C')
    blue = xarr[:,:,2].flatten(order='C')


    if fft:
        lamdared = 0
        yred = fft(red)
        lamdared =(yred > lamdathr).sum()
        lamdagreen=0
        ygreen = fft(green)
        lamdagreen =(ygreen > lamdathr).sum()
        lamdablue=0
        yblue = fft(blue)
        lamdablue =(yblue > lamdathr).sum()
        FourRed = FourierRandomMatrix(n, lamdared, c, varepsilon)
        FourGreen = FourierRandomMatrix(n, lamdagreen, c, varepsilon)
        FourBlue = FourierRandomMatrix(n, lamdablue, c, varepsilon)
    else:
        Lamda = (x > lamdathr).sum()
        Fou = RGBFourierRandomMatrix(n, Lamda, c, varepsilon)
        FourRed = Fou[0]
        FourGreen = Fou[1]
        FourBlue = Fou[2]
    
    if fft:  
        bred = FourRed.dot(yred)
        bgreen = FourGreen.dot(ygreen)
        bblue = FourBlue.dot(yblue)
    else: 
        bred = FourRed.dot(red)
        bgreen = FourGreen.dot(green)
        bblue = FourBlue.dot(blue)

    signalBlue = optimizerLI(n, FourBlue, bblue,complex = complex, alg=alg)
    signalRed = optimizerLI(n, FourRed, bred,complex = complex, alg=alg)
    signalGreen = optimizerLI(n, FourGreen, bgreen,complex = complex, alg=alg)

    if pathtosavetxt != '':
        save_rec_as_txt(pathtosavetxt + 'ImmFouRed.txt', signalRed)
        save_rec_as_txt(pathtosavetxt + 'ImmFouGreen.txt', signalGreen)
        save_rec_as_txt(pathtosavetxt+ 'ImmFouBlue.txt', signalBlue)

    if fft: 
        imageGreen = ifft(signalGreen).real
        imageGreen = np.reshape(imageGreen, (width,height))

        imageRed = ifft(signalRed).real
        imageRed = np.reshape(imageRed, (width,height))

        imageBlue = ifft(signalBlue).real
        imageBlue = np.reshape(imageBlue, (width,height))
        recImage = np.stack((imageRed.astype('uint8'), imageGreen.astype('uint8'), imageBlue.astype('uint8')), axis=2)
    else:
        imageGreen = np.reshape(signalGreen, (width,height))
        imageRed = np.reshape(signalRed, (width,height))
        imageBlue = np.reshape(signalBlue, (width,height))
        recImage = np.stack((imageRed.astype('uint8'), imageGreen.astype('uint8'), imageBlue.astype('uint8')), axis=2)
        
    plt.imsave(pathtosavetxt, recImage)
    print('Done')

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-image", type=str,  help="path to image")
    parser.add_argument("--c", type=float, help="constant for measurements")
    parser.add_argument("--lamdathr", type=int, help="level below which we consider zero")
    parser.add_argument("--Fou", type=bool, default = True, help="apply Fast Fourier transform and recover in Fourier domain")
    parser.add_argument("--path-to-txt", type = str, default = '', help="path to save the reconstructed image")
    parser.add_argument("--varepsilon", type=float, default = 0.01, help="accuracy 1 - varepsilon")
    parser.add_argument("--alg", type=bool, default= "ECOS", help="algorithm for l1 minimization")
    parser.add_argument("--complex", type=bool, default=True, help="work in complex vector space")

    args = parser.parse_args()

    FourierRecoverRGB(imagepath = args.path_to_image, c = args.c, lamda = args.lamdathr, Fou = args.Fou, pathtosavetxt = args.path-to-txt, varepsilon= args.varepsilon, alg = args.alg, complex = args.complex)

main()