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


from utils.scripts.save_rec_as_txt import save_rec_as_txt
from utils.MeasurementsConstruction.GaussianRandomMatrix.GaussianRandomMatrix import GaussianRandomMatrix
from utils.optimizers.optimizersLI import optimizerLI

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

def GaussianRecoverBW(imagepath, c, lamdathr, Fou, varepsilon=0.01, pathtosavetxt='', alg="ECOS_BB", complex=True): 

    
    x = Image.open(imagepath).convert('RGB')
    width = x.size[0]
    height = x.size[1]
    n = width*height

    print("The dimesion of the signal is:   " + str(n))
    xarr = np.array(x)
    xarr = xarr.flatten(order='C')
    
    if Fou:
        lamdaarr = 0
        yarr = fft(xarr)
        lamdaarr =(yarr > lamdathr).sum()
    
        Gaussarr = GaussianRandomMatrix(n, lamdaarr, c, varepsilon)
        
    else:
        Lamda = (x > lamdathr).sum()
        Gaussarr = GaussianRandomMatrix(n, Lamda, c, varepsilon)
        
    if Fou:    
        barr = Gaussarr.dot(yarr)
    else: 
        barr = Gaussarr.dot(arr)
        

    signal = optimizerLI(n, Fourarr, barr, complex, alg)
    
    if pathtosavetxt != '':
        save_rec_as_txt(pathtosavetxt + 'ImmFouarr.txt', signal)
        
    if Fou: 
    
        imagearr = ifft(signal).real
        imagearr = np.reshape(imagearr, (height,width))

    
    else:
        
        imagearr = np.reshape(signal, (height, width))
        
    
    plt.imsave(pathtosavetxt + 'rec' + str(c) + '.jpg', imagearr, cmap='gray')
    print('Done')

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str,  help="path to image")
    parser.add_argument("--lamdathr", type=int, help="level below which we consider zero")
    parser.add_argument("--Fou", type=bool, default = True, help="apply Fast Fourier transform and recover in Fourier domain")
    parser.add_argument("--pathtotxt", type = str, default = '', help="path to save the reconstructed image")
    parser.add_argument("--varepsilon", type=float, default = 0.01, help="accuracy 1 - varepsilon")
    parser.add_argument("--alg", type=str, default= "ECOS_BB", help="algorithm for l1 minimization")
    parser.add_argument("--complex", type=bool, default=True, help="work in complex vector space")

    args = parser.parse_args()

    GaussianRecoverRGB(imagepath = args.path, c = args.c, lamdathr = args.lamdathr, Fou = args.Fou, pathtosavetxt = args.pathtotxt, varepsilon= args.varepsilon, alg = args.alg, complex = args.complex)

if __name__ == '__main__':
    main()
