from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 
import csv
import cmath as math
import cvxpy as cvx
from scipy.fft import fft, ifft
import argparse
from multiprocessing import Pool
import sys
sys.path.append("../..")
# Import from my script 

from utils.MeasurementsConstruction.BinaryRandomMatrix.RGBBinaryRandomMatrix import RGBBinaryRandomMatrix
from utils.scripts.save_rec_as_txt import save_rec_as_txt
from utils.MeasurementsConstruction.BinaryRandomMatrix.BinaryRandomMatrix import BinaryRandomMatrix
from utils.optimizers.optimizersLI import optimizerLI


def BinaryRecoverRGB(imagepath, c, lamdathr, ncore = 1, Fou=True, p=0.5, varepsilon=0.01, pathtosavetxt='', alg="ECOS", complex=True):

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
        lamdared = (yred > lamdathr).sum()
        lamdagreen = 0
        ygreen = fft(green)
        lamdagreen = (ygreen > lamdathr).sum()
        lamdablue = 0
        yblue = fft(blue)
        lamdablue = (yblue > lamdathr).sum()
        BinaryRed = BinaryRandomMatrix(n, p, lamdared, c, varepsilon)
        BinaryGreen = BinaryRandomMatrix(n, p, lamdagreen, c, varepsilon)
        BinaryBlue = BinaryRandomMatrix(n, p, lamdablue, c, varepsilon)
    else:
        Lamda = (x > lamdathr).sum()
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


    param = [(n, BinaryBlue, bblue, complex, alg), (n, BinaryRed, bred, complex, alg), (n, BinaryGreen, bgreen, complex, alg) ]
    pool = Pool(ncore)
    result = pool.starmap(optimizerLI, param)
    signalBlue = result[0]
    signalRed = result[1]
    signalGreen = result[2]
    #signalBlue = optimizerLI( n, BinaryBlue, bblue, complex=complex, alg=alg)
    #signalRed = optimizerLI(n, BinaryRed, bred, complex=complex, alg=alg)
    #signalGreen = optimizerLI(n, BinaryGreen, bgreen, complex=complex, alg=alg)

    if pathtosavetxt != '':
        save_rec_as_txt(pathtosavetxt + 'ImmFouRed.txt', signalRed)
        save_rec_as_txt(pathtosavetxt + 'ImmFouGreen.txt', signalGreen)
        save_rec_as_txt(pathtosavetxt + 'ImmFouBlue.txt', signalBlue)

    if Fou:
        imageGreen = ifft(signalGreen).real
        imageGreen = np.reshape(imageGreen, (height,width))

        imageRed = ifft(signalRed).real
        imageRed = np.reshape(imageRed, (height,width))

        imageBlue = ifft(signalBlue).real
        imageBlue = np.reshape(imageBlue, (height,width))
        recImage = np.stack((imageRed.astype('uint8'), imageGreen.astype(
            'uint8'), imageBlue.astype('uint8')), axis=2)
    else:
        imageGreen = np.reshape(signalGreen, (height,width))
        imageRed = np.reshape(signalRed, (height,width))
        imageBlue = np.reshape(signalBlue, (height,width))
        recImage = np.stack((imageRed.astype('uint8'), imageGreen.astype(
            'uint8'), imageBlue.astype('uint8')), axis=2)

    plt.imsave(pathtosavetxt + 'rec' + str(c) + '.jpg', recImage)
    print('Done')

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str,  help="path to image")
    parser.add_argument("--c", type=float, help="constant for measurements")
    parser.add_argument("--lamdathr", type=int, help="level below which we consider zero")
    parser.add_argument("--Fou", type=bool, default = True, help="apply Fast Fourier transform and recover in Fourier domain")
    parser.add_argument("--ncore", type=int, default=1, help="constant for measurements")
    parser.add_argument("--p", type=float, default=0.5, help="probability of success of Bernoulli")
    parser.add_argument("--pathtotxt", type = str, default = '', help="path to save the reconstructed image")
    parser.add_argument("--varepsilon", type=float, default = 0.01, help="accuracy 1 - varepsilon")
    parser.add_argument("--alg", type=bool, default= "ECOS", help="algorithm for l1 minimization")
    parser.add_argument("--complex", type=bool, default=True, help="work in complex vector space")

    args = parser.parse_args()

    BinaryRecoverRGB(imagepath = args.path, p = args.p, ncore = args.ncore, c = args.c, lamdathr = args.lamdathr, Fou = args.Fou, pathtosavetxt = args.pathtotxt, varepsilon= args.varepsilon, alg = args.alg, complex = args.complex)

if __name__ == '__main__':
    main()

