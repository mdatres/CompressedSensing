from os import altsep
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 
import csv
import cmath as math
import cvxpy as cvx
from scipy.fft import fft, ifft
import argparse
import sys
from multiprocessing import Pool
from multiprocessing import Pool
sys.path.append("../..")
# Import from my script 

from utils.MeasurementsConstruction.GaussianRandomMatrix.RGBGaussianRandomMatrix import RGBGaussianRandomMatrix
from utils.MeasurementsConstruction.FourierRandomMatrix.RGBFourierRandomMatrix import RGBFourierRandomMatrix
from utils.MeasurementsConstruction.BinaryRandomMatrix.RGBBinaryRandomMatrix import RGBBinaryRandomMatrix
from RGBRecovery.BinaryRecovery.BinaryRecoverRGB import BinaryRecoverRGB
from RGBRecovery.FourierRec.FourierRecoverRGB import FourierRecoverRGB
from RGBRecovery.GaussianRecovery.GaussianRecoverRGB import GaussianRecoverRGB
from utils.scripts.save_rec_as_txt import save_rec_as_txt
from utils.MeasurementsConstruction.GaussianRandomMatrix.GaussianRandomMatrix import GaussianRandomMatrix
from utils.MeasurementsConstruction.BinaryRandomMatrix.BinaryRandomMatrix import BinaryRandomMatrix
from utils.MeasurementsConstruction.FourierRandomMatrix.FourierRandomMatrix import FourierRandomMatrix
from utils.optimizers.optimizersLI import optimizerLI

def different_trialsRGB(path, type, stop, cvalues, lamdathr, Fou=True,ncorechannels=1, ncore=1, varepsilon=0.01, pathtosavetxt='', alg="ECOS", complex=True): 

    cs = np.linspace(0.5, stop, cvalues)
    params = []
    for c in cs:
        params.append((path, c, lamdathr, Fou, ncorechannels, varepsilon, pathtosavetxt, alg, complex))

    if type == 'Fourier':
        pool = Pool(ncore)
        pool.starmap(FourierRecoverRGB, params)
    
    if type == 'Binary':
        pool = Pool(ncore)
        pool.starmap(BinaryRecoverRGB, params)

    if type == 'Gaussian':
        pool = Pool(ncore)
        pool.starmap(GaussianRecoverRGB, params)

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str,  help="path to sound")
    parser.add_argument("--type", type=str, help="type of measurement matrix.(Choose between Fourier, Binary or Gaussian)")
    parser.add_argument("--stop", type=float, help="maximum value of c")
    parser.add_argument("--cvalues", type=float, help="number of trials to prove")
    parser.add_argument("--lamdathr", type=int, help="level below which we consider zero")
    parser.add_argument("--Fou", type=bool, default = True, help="apply Fast Fourier transform and recover in Fourier domain")
    parser.add_argument("--ncorechannels", type=int, default=1, help="number of cores inside a type of reconstruction")
    parser.add_argument("--ncore", type=int, default=1, help="number of cores")
    parser.add_argument("--pathtotxt", type = str, default = '', help="path to save the reconstructed image")
    parser.add_argument("--varepsilon", type=float, default = 0.01, help="accuracy 1 - varepsilon")
    parser.add_argument("--alg", type=bool, default= "ECOS", help="algorithm for l1 minimization")
    parser.add_argument("--complex", type=bool, default=True, help="work in complex vector space")

    args = parser.parse_args()

    different_trialsRGB(path = args.path, type = args.type, stop = args.stop, cvalues = args.cvalues, ncore = args.ncore, c = args.c, lamdathr = args.lamdathr, ncorechannels = args.ncorechannels, Fou = args.Fou, pathtosavetxt = args.pathtotxt, varepsilon= args.varepsilon, alg = args.alg, complex = args.complex)

main()
