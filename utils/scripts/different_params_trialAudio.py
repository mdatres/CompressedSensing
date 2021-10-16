from os import altsep
from PIL import Image
import os
import matplotlib
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
from AudioRecovery.BinaryAudioRec.BinaryAudioRecover import BinaryAudioRecover
from AudioRecovery.FourierAudioRecover.FourierAudioRecover import FourierAudioRecover
from AudioRecovery.GaussianAudioRec.GaussianAudioRecover import GaussianAudioRecover
from utils.scripts.save_rec_as_txt import save_rec_as_txt
from utils.MeasurementsConstruction.GaussianRandomMatrix.GaussianRandomMatrix import GaussianRandomMatrix
from utils.MeasurementsConstruction.BinaryRandomMatrix.BinaryRandomMatrix import BinaryRandomMatrix
from utils.MeasurementsConstruction.FourierRandomMatrix.FourierRandomMatrix import FourierRandomMatrix
from utils.optimizers.optimizersLI import optimizerLI

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

def different_trialsAudio(path, type, stop, cvalues, sr, lamdathr, Fou=True, ncore=1, varepsilon=0.01, pathtosavetxt='', alg="ECOS_BB", complex=True): 

    cs = np.linspace(0.5, stop, cvalues)
    params = []
    for c in cs:
        params.append((path, sr, c, lamdathr, Fou, varepsilon, pathtosavetxt, alg, complex))

    if type == 'Fourier':
        pool = Pool(ncore)
        pool.starmap(FourierAudioRecover, params)
    
    if type == 'Binary':
        pool = Pool(ncore)
        pool.starmap(BinaryAudioRecover, params)

    if type == 'Gaussian':
        pool = Pool(ncore)
        pool.starmap(GaussianAudioRecover, params)

        

def main(): 
    parser1 = argparse.ArgumentParser()
    parser1.add_argument("--path", type=str,  help="path to sound")
    parser1.add_argument("--type", type=str, help="type of measurement matrix.(Choose between Fourier, Binary or Gaussian)")
    parser1.add_argument("--stop", type=int, help="maximum value of c")
    parser1.add_argument("--cvalues", type=int, help="number of trials to prove")
    parser1.add_argument("--lamdathr", type=int, help="level below which we consider zero")
    parser1.add_argument("--sr", type=int, default = 800, help="sampling rate of the audio")
    parser1.add_argument("--Fou", type=bool, default = True, help="apply Fast Fourier transform and recover in Fourier domain")
    parser1.add_argument("--ncore", type=int, default=1, help="number of cores")
    parser1.add_argument("--pathtotxt", type = str, default = '', help="path to save the reconstructed image")
    parser1.add_argument("--varepsilon", type=float, default = 0.01, help="accuracy 1 - varepsilon")
    parser1.add_argument("--alg", type=str, default= "ECOS_BB", help="algorithm for l1 minimization")
    parser1.add_argument("--complex", type=bool, default=True, help="work in complex vector space")

    args1 = parser1.parse_args()

    different_trialsAudio(path = args1.path, sr = args1.sr, type = args1.type, stop = args1.stop, cvalues = args1.cvalues, ncore = args1.ncore, lamdathr = args1.lamdathr, Fou = args1.Fou, pathtosavetxt = args1.pathtotxt, varepsilon= args1.varepsilon, alg = args1.alg, complex = args1.complex)

main()
