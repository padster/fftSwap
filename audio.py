"""
Audio processing code for playing with frequencies.
See post: https://padsterprogramming.blogspot.ca/2017/03/perceiving-frequencies.html
"""

import matplotlib
import matplotlib.pyplot as plt
from numpy.linalg import pinv as pInverse
from PIL import Image
import scipy.io.wavfile as wavfile

import math
import random
import numpy as np
import viz

# Read a .wav file, return sample rate and numpy array, normalized to [-1, 1]
def audioToNp(path):
    print ("Reading from %s..." % path)
    rate, samples = wavfile.read(path)
    return rate, samples / 32768.0

# Save a numpy array of samples back out to a wav file.
def npToAudio(path, rate, samples):
    print ("Writing to %s..." % path)
    samples = samples * 32768.0
    samples = samples.astype(np.int16)
    wavfile.write(path, rate, samples)

# Given a spectrogram, convert back to audio, normalize, and save
def saveAudioIFFT(path, specOut, nSamples, rate):
    sOut = np.fft.irfft(specOut, n=nSamples)
    maxV = np.max(np.abs(sOut)) # Normalize first :/
    sOut = sOut / maxV
    assert len(sOut) == nSamples
    npToAudio(path, rate, sOut)

# Utility for chunked short-time fourier transform, given input and chunk size
def stft(x, sz):
    assert sz % 2 == 0
    assert len(x) % sz == 0
    # Return has usual formatting: rows = frequency (high to low), columns = time
    return np.flipud(np.array([np.fft.rfft(x[i:i + sz]) for i in range(0, len(x), sz)]).T)

# Utility for reversing a chunked spectrogram back into samples.
def rstft(s, sz):
    # required: x == rstft(stft(x, sz), sz)
    assert sz % 2 == 0
    s = np.flipud(s).T
    return np.array([np.fft.irfft(s[i, :]) for i in range(0, s.shape[0])]).flatten()

# Run pairwise swaps and write results to file.
def runFull():
    r1, s1 = audioToNp('bush.wav')
    r2, s2 = audioToNp('churchill.wav')
    assert len(s1) == len(s2) and r1 == r2
    rate, nSamples = r1, len(s1)
    print "%d samples read at %d hz" % (nSamples, rate)

    spec1 = np.fft.rfft(s1)
    spec2 = np.fft.rfft(s2)
    assert len(spec1) == len(spec2)
    pow1, phase1 = np.abs(spec1), np.angle(spec1)
    pow2, phase2 = np.abs(spec2), np.angle(spec2)
    saveAudioIFFT('bushPowChurchPhase.wav', pow1 * np.exp(1j * phase2), nSamples, rate)
    saveAudioIFFT('churchPowBushPhase.wav', pow2 * np.exp(1j * phase1), nSamples, rate)

# Just experimenting, merge the sounds with random noise.
def runRandom():
    r1, s1 = audioToNp('bush.wav')
    r2, s2 = audioToNp('churchill.wav')
    sR = (np.random.random(s1.shape)) * 2.0 - 1.0
    assert len(s1) == len(s2) and r1 == r2
    rate, nSamples = r1, len(s1)
    print "%d samples read at %d hz" % (nSamples, rate)

    spec1 = np.fft.rfft(s1)
    spec2 = np.fft.rfft(s2)
    specR = np.fft.rfft(sR)
    assert len(spec1) == len(spec2)
    assert len(spec1) == len(specR)

    pow1, phase1 = np.abs(spec1), np.angle(spec1)
    pow2, phase2 = np.abs(spec2), np.angle(spec2)
    powR, phaseR = np.abs(specR), np.angle(specR)
    saveAudioIFFT('bushPowRandPhase.wav', pow1 * np.exp(1j * phaseR), nSamples, rate)
    saveAudioIFFT('randPowBushPhase.wav', powR * np.exp(1j * phase1), nSamples, rate)
    saveAudioIFFT('churchPowRandPhase.wav', pow2 * np.exp(1j * phaseR), nSamples, rate)
    saveAudioIFFT('randPowChurchPhase.wav', powR * np.exp(1j * phase2), nSamples, rate)


# Demo: Draw the 1D (unchunked) spectrogram for the bush audio.
def bushSpectrogramDemo():
    rate, samples = audioToNp('bush.wav')
    spec = np.fft.rfft(samples)
    spec = spec[::20]

    ax = viz.cleanSubplots(3, 1, pad=0.1)
    ax[0].set_title('sound')
    ax[0].plot(samples[::20])
    ax[0].set_xlim((0, len(samples[::20])))
    ax[1].set_title('power')
    ax[1].plot(np.log(np.abs(spec) + 1e-12), 'black')
    ax[1].set_xlim((0, len(spec)))
    ax[2].set_title('phase')
    ax[2].scatter(np.linspace(0,len(spec),len(spec)), np.angle(spec), c=np.angle(spec))
    ax[2].set_xlim((0, len(spec)))
    plt.show()

# Demo: Draw the normal 2D chunked STFT spectrogram for the bush audio
def bushChunked(chunkMs):
    r1, s1 = audioToNp('bush.wav')
    rate, nSamples = r1, len(s1)
    spec = stft(s1, int(chunkMs / 1000.0 * rate))
    ax = viz.cleanSubplots(2, 1, pad=0.1)
    ax[0].set_title('chunked power')
    ax[0].imshow(np.log(np.abs(spec) + 1e-12), cmap='gray')
    ax[1].set_title('chunked phase')
    ax[1].imshow(np.angle(spec), cmap='gist_rainbow')
    plt.show()

# Given two files, try STFT merging with different chunk sizes
def chunkReversalMerge():
    r1, s1 = audioToNp('bush.wav')
    r2, s2 = audioToNp('churchill.wav')
    assert r1 == r2 and len(s1) == len(s2)

    chunkSize = len(s1)
    while chunkSize > 2:
        if chunkSize % 2 == 1:
            chunkSize = chunkSize + 1
        print "Chunk size: %d" % chunkSize
        toAdd = chunkSize - len(s1) % chunkSize
        padded1 = np.concatenate([s1, np.zeros(toAdd)])
        padded2 = np.concatenate([s2, np.zeros(toAdd)])
        spec1 = stft(padded1, chunkSize)
        spec2 = stft(padded2, chunkSize)
        merged = np.abs(spec2) * np.exp(1j * np.angle(spec1))
        sOut = rstft(merged, chunkSize)
        path = "audioMerge/churchPowerBushPhaseChunk%d.wav" % (chunkSize)
        npToAudio(path, r1, sOut / np.max(np.abs(sOut)))
        chunkSize = chunkSize / 2


if __name__ == '__main__':
    random.seed(4321)
    np.random.seed(4321)

    bushSpectrogramDemo()
    # bushChunked(16)
    # chunkReversalMerge()
