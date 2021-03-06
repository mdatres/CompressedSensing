# Compressed Sensing
This repository contains some python script useful to perform compressed sensing reconstruction. These script are basics, and useful only for small singals dimension.

# Quick Compressed Sensing Presentation  
A common goal of the engineering field of signal processing is to reconstruct a signalfrom a series of sampling measurements.  Compressed Sensing is a signal processing technique for efficiently acquiring and reconstructing a signal, by finding solutions to underdetermined linear systems. The problem reduces to solving a linear system of equations. In this repositor, different three different types of measurement matrices are available: 
1. *Gassian Random Matrices*, which are matrices which entries are sampled independently from a Normal distribution, see script [GaussianRandomMatrix.py](utils/MeasurementsConstruction/GaussianRandomMatrix/GaussianRandomMatrix.py);
2. *Binary Random Matrices*, which are matrices which entries are sampled independently from a Rademacher distribution, see script [BinaryRandomMatrix.py](utils/MeasurementsConstruction/BinaryRandomMatrix/BinaryRandomMatrix.py);
3. *Fourier Random Matrices*, which are matrices composed with $m$ random sampled rows of the discrete Fourier transform matrix, see script [FourierRandomMatrix.py](utils/MeasurementsConstruction/FourierRandomMatrix/FourierRandomMatrix.py);


This is done with an l1 norm optimization problem. With this repository, it is possible also to reconstruct multidimesional signals such as RGB images. This is perform by the recovery of all the slices, see [RGBRecovery](RGBRecovery). The l1 recovery algorithm is taken by the library cvxpy, see  [https://github.com/cvxpy/cvxpy](cvxpy). 

# Real Application of the Compressed Sensing Theory
For instance, examples of signal recovery practical applications are in the surface wave tomography and in the magnetic resonance imaging (MRI) fields. The surface wave tomography uses seismic waves produced by earthquakes or explosions for imaging the subsurface of the earth. Seismic data from array sensors are discrete samples of spatially and temporally continuous wavefield. Compressed sensing technique can be applied to reconstruct the original seismic wave from the measurments collected from the sensors. 

Another important application is in the MRI field. MRI is one of the most dynamic and safe imaging techniques available in the clinic today. However, MRI acquisitions tend to be slow, limiting patient throughput and limiting potential indications for use while driving up costs. Compressed sensing is a method for accelerating MRI acquisition by acquiring less data and then reconstruct the imaging in a higher dimensional space.
