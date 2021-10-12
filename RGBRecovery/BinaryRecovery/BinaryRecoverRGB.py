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

from utils.MeasurementsConstruction.BinaryRandomMatrix.RGBBinaryRandomMatrix import RGBBinaryRandomMatrix
from utils.scripts.save_rec_as_txt import save_rec_as_txt
from utils.MeasurementsConstruction.BinaryRandomMatrix.BinaryRandomMatrix import BinaryRandomMatrix
from utils.optimizers.optimizersLI import optimizerLI


def BinaryRecoverRGB(imagepath, c, lamda, Fou, d, p=0.5, varepsilon=0.01, pathtosavetxt='', pathtosave= '.', alg="ECOS", complex=True):

    x = Image.open(imagepath).convert('RGB')
    width = x.size[0]
    height = x.size[1]
    n = width*height

    print("The dimesion of the signal is:   " + str(n))
    xarr = np.array(x)
    red = xarr[:, :, 0].flatten(order='C')
    green = xarr[:, :, 1].flatten(order='C')
    blue = xarr[:, :, 2].flatten(order='C')

    if Fou:
        lamdared = 0
        yred = fft(red)
        lamdared = (yred > lamda).sum()
        lamdagreen = 0
        ygreen = fft(green)
        lamdagreen = (ygreen > lamda).sum()
        lamdablue = 0
        yblue = fft(blue)
        lamdablue = (yblue > lamda).sum()
        BinaryRed = BinaryRandomMatrix(n, p, lamdared, c, varepsilon)
        BinaryGreen = BinaryRandomMatrix(n, p, lamdagreen, c, varepsilon)
        BinaryBlue = BinaryRandomMatrix(n, p, lamdablue, c, varepsilon)
    else:
        Lamda = (x > lamda).sum()
        Binary = RGBBinaryRandomMatrix(n, p,  Lamda, c, varepsilon)
        BinaryRed = Binary[0]
        BinaryGreen = Binary[1]
        BinaryBlue = Binary[2]

    if Fou:
        bred = BinaryRed.dot(yred)
        bgreen = BinaryGreen.dot(ygreen)
        bblue = BinaryBlue.dot(yblue)
    else:
        bred = BinaryRed.dot(red)
        bgreen = BinaryGreen.dot(green)
        bblue = BinaryBlue.dot(blue)

    signalBlue = optimizerLI(
        n, BinaryBlue, bblue, complex=complex, alg=alg)
    signalRed = optimizerLI(n, BinaryRed, bred, complex=complex, alg=alg)
    signalGreen = optimizerLI(
        n, BinaryGreen, bgreen, complex=complex, alg=alg)

    if pathtosavetxt != '':
        save_rec_as_txt(pathtosavetxt + 'ImmFouRed.txt', signalRed)
        save_rec_as_txt(pathtosavetxt + 'ImmFouGreen.txt', signalGreen)
        save_rec_as_txt(pathtosavetxt + 'ImmFouBlue.txt', signalBlue)

    if Fou:
        imageGreen = ifft(signalGreen).real
        imageGreen = np.reshape(imageGreen, (width, height))

        imageRed = ifft(signalRed).real
        imageRed = np.reshape(imageRed, (width, height))

        imageBlue = ifft(signalBlue).real
        imageBlue = np.reshape(imageBlue, (width, height))
        recImage = np.stack((imageRed.astype('uint8'), imageGreen.astype(
            'uint8'), imageBlue.astype('uint8')), axis=2)
    else:
        imageGreen = np.reshape(signalGreen, (width, height))
        imageRed = np.reshape(signalRed, (width, height))
        imageBlue = np.reshape(signalBlue, (width, height))
        recImage = np.stack((imageRed.astype('uint8'), imageGreen.astype(
            'uint8'), imageBlue.astype('uint8')), axis=2)

    plt.imsave(pathtosave, recImage)
    print('Done')

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-image", type=str,  help="path to image")
    parser.add_argument("--p", type=float, help="probability of success of Bernoulli")
    parser.add_argument("--c", type=float, help="constant for measurements")
    parser.add_argument("--lamda", type=int, help="level below which we consider zero")
    parser.add_argument("--Fou", type=bool, default = True, help="apply Fast Fourier transform and recover in Fourier domain")
    parser.add_argument("--path-to-save", type = str, default='.', help="path to save the reconstructed image")
    parser.add_argument("--path-to-txt", type = str, default = '', help="path to save the reconstructed image")
    parser.add_argument("--varepsilon", type=float, default = 0.01, help="accuracy 1 - varepsilon")
    parser.add_argument("--alg", type=bool, default= "ECOS", help="algorithm for l1 minimization")
    parser.add_argument("--complex", type=bool, default=True, help="work in complex vector space")

    args = parser.parse_args()

    BinaryRecoverRGB(imagepath = args.path_to_image, p = args.p, c = args.c, lamda = args.lamda, Fou = args.Fou, pathtosavetxt = args.path_to_txt, pathtosave= args.path_to_save, varepsilon= args.varepsilon, alg = args.val, complex = args.complex)

main()
