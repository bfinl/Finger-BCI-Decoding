#   main_online_processing.py
#   
#   Modified version of a demo script originally developed by Jeremy Hill
#   for real-time EEG signal processing and decoding. This script has been 
#   tested with Python 3.8.4 and BCPy2000 2021.1.0, but compatibility with 
#   other versions has not been verified.
#
#   Original copyright (C) 2007–2010 Jeremy Hill
#   Contact: bcpy2000@bci2000.org
#
#   This modified version includes additional processing and decoding functionality
#   and is intended for research and educational purposes.
#
#   Copyright (C) Yidan Ding 2025
#
import numpy as np
import scipy
from scipy.signal import resample
from BCPy2000.GenericSignalProcessing import BciGenericSignalProcessing

# EEGNet-specific imports
from EEGModels_tf import EEGNet

#################################################################
#################################################################

class BciSignalProcessing(BciGenericSignalProcessing):	
	
	#############################################################
	
	def Construct(self):
		parameters = [
			"PythonSig:Processing	int	DownsampleRate=	100	100	0	1024	// downsampling rate",
			"PythonSig:Processing	int	WindowLength=	1000	1000	0	5000	// window length in ms",
			"PythonSig:Processing	string ModelPath= PathToTheTrainedModel % % %	// path to the trained model",

		]
		states = [
			"FeedbackProc	1 0 0 0",

		]
		return (parameters, states)
		
	#############################################################
	
	def Preflight(self, sigprops):
		self.out_signal_dim = (4,1) # send the prob 
		
		pass
		
	#############################################################
	
	def Initialize(self, indim, outdim):

		self.FeefbackOn = 0
		self.newsig = []
		self.chans = self.in_signal_dim[0]
		self.samplingRate = int(self.params['SamplingRate'].replace("Hz", ""))
		self.newsamplingRate = int(self.params['DownsampleRate'])
		self.DesiredLen = int(int(self.params['WindowLength'])/1000*self.newsamplingRate)

		self.nclasses = len(self.params['ClassList'])
		self.classlist = list(map(int, self.params['ClassList']))
		self.model = EEGNet(nb_classes = self.nclasses, Chans = self.chans, Samples = self.DesiredLen, 
						dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
						dropoutType = 'Dropout')
		self.model.load_weights(self.params['ModelPath'])
		pass
		
	#############################################################
	
	def Process(self, sig):
		
		kernels = 1

		chans, samples = np.shape(sig)
		sig = sig-sig.mean(axis=0)

		newSamples = int(samples/self.samplingRate*self.newsamplingRate)
		
		if not len(self.newsig):
			self.newsig = resample(sig, newSamples, t=None, axis=1, window=None, domain='time')
		else:
			self.newsig = np.concatenate((self.newsig, resample(sig, newSamples, t=None, axis=1, window=None, domain='time')),axis=1)
		

		if np.size(self.newsig,1) >= self.DesiredLen:
			self.FeefbackOn = 1
			self.newsig = self.newsig[:,-self.DesiredLen:]

		# feed into EEGNet
		if self.FeefbackOn:

			# bandpass filtering
			padding_length = 100  # Number of zeros to pad
			padded_sig = np.pad(self.newsig, ((0,0),(padding_length,padding_length)), 'constant', constant_values=0)

			b, a = scipy.signal.butter(4, [4, 40], btype='bandpass', fs=self.newsamplingRate)
			padded_sig = scipy.signal.lfilter(b, a, padded_sig, axis=-1)
			insig = padded_sig[:,padding_length:-padding_length]

			insig = scipy.stats.zscore(insig, axis=1, nan_policy='omit')
			insig = insig.reshape(1,self.chans,self.DesiredLen,kernels)
			output			 = self.model.predict(insig)
			output			 = output.flatten()
			self.probs       = np.zeros((4,))
			for i, j in enumerate(self.classlist):
				self.probs[j-1]		 = output[i]
		else:
			self.probs       = np.zeros((4,))

		self.states['FeedbackProc'] = self.FeefbackOn
		
		return self.probs.reshape(4,1)
		
#################################################################
#################################################################
