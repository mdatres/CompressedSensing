from PIL import Image
import os
import matplotlib
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np 
import csv
import cmath as math
import cvxpy as cvx
from scipy.fft import fft, ifft
import argparse
import sys
from multiprocessing import Pool
sys.path.append("../..")
# Import from my script 

from utils.MeasurementsConstruction.GaussianRandomMatrix.RGBGaussianRandomMatrix import RGBGaussianRandomMatrix
from utils.scripts.save_rec_as_txt import save_rec_as_txt
from utils.MeasurementsConstruction.GaussianRandomMatrix.GaussianRandomMatrix import GaussianRandomMatrix
from utils.optimizers.optimizersLI import optimizerLI

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

def GaussianRecoverRGB(imagepath, c, lamdathr, Fou,  ncore=1, varepsilon=0.01, pathtosavetxt='', alg="ECOS_BB", complex=True): 

    
    x = Image.open(imagepath).convert('RGB')
    width = x.size[0]
    height = x.size[1]
    n = width*height

    print("The dimesion of the signal is:   " + str(n))
    xarr = np.array(x)
    red = xarr[:,:,0].flatten(order='C')
    green = xarr[:,:,1].flatten(order='C')
    blue = xarr[:,:,2].flatten(order='C')


    if Fou:
        lamdared = 0
        yred = fft(red)
        lamdared =(yred > lamdathr).sum()
        lamdagreen=0
        ygreen = fft(green)
        lamdagreen =(ygreen > lamdathr).sum()
        lamdablue=0
        yblue = fft(blue)
        lamdablue =(yblue > lamdathr).sum()
        GaussRed = GaussianRandomMatrix(n, lamdared, c, varepsilon)
        GaussGreen = GaussianRandomMatrix(n, lamdagreen, c, varepsilon)
        GaussBlue = GaussianRandomMatrix(n, lamdablue, c, varepsilon)
    else:
        Lamda = (x > lamdathr).sum()
        Gaussian = RGBGaussianRandomMatrix(n, Lamda, c, varepsilon)
        GaussRed = Gaussian[0]
        GaussGreen = Gaussian[1]
        GaussBlue = Gaussian[2]

    if Fou:    
        bred = GaussRed.dot(yred)
        bgreen = GaussGreen.dot(ygreen)
        bblue = GaussBlue.dot(yblue)
    else: 
        bred = GaussRed.dot(red)
        bgreen = GaussGreen.dot(green)
        bblue = GaussBlue.dot(blue)

    param = [(n, GaussBlue, bblue, complex, alg), (n, GaussRed, bred, complex, alg), (n, GaussGreen, bgreen, complex, alg) ]
    pool = Pool(ncore)
    result = pool.starmap(optimizerLI, param)
    signalBlue = result[0]
    signalRed = result[1]
    signalGreen = result[2]
    #signalBlue = optimizerLI(n, GaussBlue, bblue,complex = complex, alg=alg)
    #signalRed = optimizerLI(n, GaussRed, bred,complex = complex, alg=alg)
    #signalGreen = optimizerLI(n, GaussGreen, bgreen,complex = complex, alg=alg)

    if pathtosavetxt != '':
        save_rec_as_txt(pathtosavetxt + 'ImmFouRed.txt', signalRed)
        save_rec_as_txt(pathtosavetxt + 'ImmFouGreen.txt', signalGreen)
        save_rec_as_txt(pathtosavetxt+ 'ImmFouBlue.txt', signalBlue)

    if Fou: 
        imageGreen = ifft(signalGreen).real
        imageGreen = np.reshape(imageGreen, (height,width))

        imageRed = ifft(signalRed).real
        imageRed = np.reshape(imageRed, (height,width))

        imageBlue = ifft(signalBlue).real
        imageBlue = np.reshape(imageBlue, (height,width))
        recImage = np.stack((imageRed.astype('uint8'), imageGreen.astype('uint8'), imageBlue.astype('uint8')), axis=2)
    else:
        imageGreen = np.reshape(signalGreen, (height, width))
        imageRed = np.reshape(signalRed, (height, width))
        imageBlue = np.reshape(signalBlue, (height, width))
        recImage = np.stack((imageRed.astype('uint8'), imageGreen.astype('uint8'), imageBlue.astype('uint8')), axis=2)
    
    plt.imsave(pathtosavetxt + 'rec' + str(c) + '.jpg', recImage)
    print('Done')

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str,  help="path to image")
    parser.add_argument("--lamdathr", type=int, help="level below which we consider zero")
    parser.add_argument("--Fou", type=bool, default = True, help="apply Fast Fourier transform and recover in Fourier domain")
    parser.add_argument("--pathtotxt", type = str, default = '', help="path to save the reconstructed image")
    parser.add_argument("--ncore", type=int, defualt=1, help="constant for measurements")
    parser.add_argument("--varepsilon", type=float, default = 0.01, help="accuracy 1 - varepsilon")
    parser.add_argument("--alg", type=str, default= "ECOS_BB", help="algorithm for l1 minimization")
    parser.add_argument("--complex", type=bool, default=True, help="work in complex vector space")

    args = parser.parse_args()

    GaussianRecoverRGB(imagepath = args.path, c = args.c, lamdathr = args.lamdathr, Fou = args.Fou, pathtosavetxt = args.pathtotxt, varepsilon= args.varepsilon, alg = args.alg, complex = args.complex)

if __name__ == '__main__':
    main()
