import librosa
import soundfile as sf
import numpy as np
import cmath as math
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import cvxpy as cvx
import argparse
import sys

sys.path.append("../..")
from utils.MeasurementsConstruction.BinaryRandomMatrix.BinaryRandomMatrix import BinaryRandomMatrix
from utils.optimizers.optimizersLI import optimizerLI
from utils.scripts.save_rec_as_txt import save_rec_as_txt
from utils.scripts.audio_plots import pretty_plot, plot_signals


def BinaryAudioRecover(path, sr, p, c, lamdathr, Fou, varepsilon=0.01, pathtosavetxt='', alg="ECOS", complex=True):
    x, sr = librosa.load(path, sr= sr)
    n = len(x)
    print('The lenght of the signal is:     ' + str(n))
    
    if Fou:
        lamda=0
        y = fft(x)
        plt.plot(y)
        lamda =(y > lamdathr).sum()
        if pathtosavetxt != '': 
            librosa.output.write_wav(pathtosavetxt + 'signalmeas.wav', y, sr)
        print('The sparsity level is:   '+ str(lamda))
    else:
        lamda=0 
        lamda =(x > lamdathr).sum()
        print('The sparsity level is:   '+ str(lamda))
    
    A =  BinaryRandomMatrix(n, p=p, lamda=lamda, c=c, varepsilon=varepsilon)
    
    if Fou:
        b = A.dot(y)
        if pathtosavetxt != '': 
            sf.write(pathtosavetxt + 'signalmeas.wav', b.real, sr)
    else: 
        b = A.dot(x)
        if pathtosavetxt != '': 
            sf.write(pathtosavetxt + 'signalmeas.wav', b.real, sr)

    signal = optimizerLI(n, A, b,complex = complex, alg=alg)
    sign = ifft(signal)

    if pathtosavetxt != '':
        save_rec_as_txt(pathtosavetxt + 'audioBinRec.txt', sign)
        librosa.output.write_wav(pathtosavetxt + 'rec.wav', np.array(sign).real, sr)
        pretty_plot(x, title = 'Original Signal', path=pathtosavetxt + 'Original.jpg')
        pretty_plot(y, title = 'Original Signal in Fourier Domain', path=pathtosavetxt + 'OriginalInFourier.jpg')
        pretty_plot(signal, title = 'Reconstructed Signal in Fourier Domain', path=pathtosavetxt + 'RecInFourier.jpg')
        pretty_plot(sign , title = 'Reconstructed Signal', path=pathtosavetxt + 'Rec.jpg')
        plot_signals(x, sign, labelx='Original', labely='Recostructed', path=pathtosavetxt + 'RecvsOr.jpg')

    print('Done!')

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str,  help="path to image")
    parser.add_argument("--p", type=float, help="probability of success of Bernoulli")
    parser.add_argument("--sr", type=int, help="probability of success of Bernoulli")
    parser.add_argument("--c", type=float, help="constant for measurements")
    parser.add_argument("--lamda", type=int, help="level below which we consider zero")
    parser.add_argument("--Fou", type=bool, default = True, help="apply Fast Fourier transform and recover in Fourier domain")
    parser.add_argument("--path-to-txt", type = str, default = '', help="path to save the reconstructed image")
    parser.add_argument("--varepsilon", type=float, default = 0.01, help="accuracy 1 - varepsilon")
    parser.add_argument("--alg", type=bool, default= "ECOS", help="algorithm for l1 minimization")
    parser.add_argument("--complex", type=bool, default=True, help="work in complex vector space")

    args = parser.parse_args()

    BinaryAudioRecover(path = args.path, sr = args.sr, p = args.p, c = args.c, lamda = args.lamda, fft = args.fft, pathtosavetxt = args.path_to_txt, varepsilon= args.varepsilon, alg = args.val, complex = args.complex)

main()