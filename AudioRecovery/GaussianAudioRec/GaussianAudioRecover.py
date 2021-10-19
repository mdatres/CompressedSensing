import librosa
import soundfile as sf
import numpy as np
import os
import cmath as math
from scipy.fft import fft, ifft
import matplotlib
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cvxpy as cvx
import argparse
import scipy.io.wavfile
import sys
sys.path.append("../..")
from utils.MeasurementsConstruction.GaussianRandomMatrix.GaussianRandomMatrix import GaussianRandomMatrix
from utils.optimizers.optimizersLI import optimizerLI
from utils.scripts.save_rec_as_txt import save_rec_as_txt
from utils.scripts.audio_plots import pretty_plot, plot_signals




def GaussianAudioRecover(path, sr,  c, lamdathr, Fou, varepsilon=0.01, pathtosavetxt='', alg="ECOS_BB", complex=True):
    x, sr = librosa.load(path, sr= sr)
    n = len(x)
    print('The lenght of the signal is:     ' + str(n))
    
    if Fou:
        lamda=0
        y = fft(x)
        lamda =(y > lamdathr).sum()
        print('The sparsity level is:   '+ str(lamda))
    else:
        lamda=0 
        lamda =(x > lamdathr).sum()
        print('The sparsity level is:   '+ str(lamda))
    
    A =  GaussianRandomMatrix(n, lamda=lamda, c=c, varepsilon=varepsilon)
    
    if Fou:
        b = A.dot(y)
        bsave = ifft(b)
        if pathtosavetxt != '': 
            scipy.io.wavfile.write(pathtosavetxt +'signalmeas' + str(c) + '.wav', sr, bsave.real*100)
    else: 
        b = A.dot(x)
        if pathtosavetxt != '': 
            scipy.io.wavfile.write(pathtosavetxt +'signalmeas'+ str(c) + '.wav', sr, b.real*100)

    signal = optimizerLI(n, A, b,complex = complex, alg=alg)

    sign = ifft(signal)

    if pathtosavetxt != '':
        save_rec_as_txt(pathtosavetxt + 'audioBinRec.txt', sign)
        scipy.io.wavfile.write(pathtosavetxt +'rec'+ str(c) + '.wav', sr, np.array(sign).real*100)
        pretty_plot(x, title = 'Original Signal', path=pathtosavetxt + 'Original' + str(c) + '.jpg')
        pretty_plot(y, title = 'Original Signal in Fourier Domain', path=pathtosavetxt + 'OriginalInFourier'+ str(c) + '.jpg')
        pretty_plot(signal, title = 'Reconstructed Signal in Fourier Domain', path=pathtosavetxt + 'RecInFourier.'+ str(c) + '.jpg')
        pretty_plot(sign , title = 'Reconstructed Signal', path=pathtosavetxt + 'Rec'+ str(c) + '.jpg')
        plot_signals(x, sign, labelx='Original', labely='Recostructed', path=pathtosavetxt + 'RecvsOr'+ str(c) + '.jpg')

    print('Done!')

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str,  help="path to image")
    parser.add_argument("--c", type=float, help="constant for measurements")
    parser.add_argument("--sr", type=int, help="sampling rate")
    parser.add_argument("--lamdathr", type=int, help="level below which we consider zero")
    parser.add_argument("--Fou", type=bool, default = True, help="apply Fast Fourier transform and recover in Fourier domain")
    parser.add_argument("--path-to-txt", type = str, default = '', help="path to save the reconstructed image")
    parser.add_argument("--varepsilon", type=float, default = 0.01, help="accuracy 1 - varepsilon")
    parser.add_argument("--alg", type=str, default= "ECOS_BB", help="algorithm for l1 minimization")
    parser.add_argument("--complex", type=bool, default=True, help="work in complex vector space")

    args = parser.parse_args()

    GaussianAudioRecover(path = args.path, sr=args.sr, c = args.c, lamdathr = args.lamdathr, Fou = args.Fou, pathtosavetxt = args.path_to_txt, varepsilon= args.varepsilon, alg = args.alg, complex = args.complex)

if __name__ == '__main__':
    main()
