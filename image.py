"""
Image processing code for playing with frequencies.
See post: https://padsterprogramming.blogspot.ca/2017/03/perceiving-frequencies.html
"""

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io.wavfile as wavfile

import math
import random
import numpy as np
import viz

# Load .png from file, drop alpha channel, normalize to [0.0, 1.0)
def imageToNp(path):
    print ("Reading from %s..." % path)
    return np.array(Image.open(path))[:, :, 0:3] / 256.0

# Save image back out as 8-bit int [0, 256)
def npToImage(path, image):
    print ("Writing to %s..." % path)
    image = image * 256.0
    image = image.astype(np.uint8)
    im = Image.fromarray(image)
    im.save(path)

# Normalize to use full spectrum (can overflow with merged examples)
def normalizeImg(data):
    minV, maxV = np.min(data), np.max(data)
    data = (data - minV) / (maxV - minV)
    return data

# Helper to inverse a spectrogram and save to file.
def saveImageIFFT(path, specOut):
    imOut = np.fft.irfft2(specOut, axes=(0, 1))
    # Normalize each channel separately first...
    for i in range(imOut.shape[2]):
        minV, maxV = np.min(imOut[:, :, i]), np.max(imOut[:, :, i])
        imOut[:, :, i] = (imOut[:, :, i] - minV) / (maxV - minV)
    npToImage(path, imOut)

# Utility to convert [A B; C D] to [D C; B A], to put the 0 frequency in the middle of fft2
def center0(data):
    (r, c) = data.shape
    assert r % 2 == 0 and c % 2 == 0
    hR, hC = r / 2, c / 2
    res = np.zeros(data.shape, dtype=complex)
    res[0:hR, 0:hC] = data[hR:r, hC:c]
    res[0:hR, hC:c] = data[hR:r, 0:hC]
    res[hR:r, 0:hC] = data[0:hR, hC:c]
    res[hR:r, hC:c] = data[0:hR, 0:hC]
    return res

# Make a small adjustment to properly center the 0 frequencies of the chunked fft2
def centerInterleave(data, sz):
    (r, c) = data.shape
    res = np.zeros(data.shape, dtype=complex)
    assert sz % 2 == 0
    sz = sz / 2
    res[0:r-sz, 0:c-sz] = data[sz:r, sz:c]
    res[r-sz:r, 0:c-sz] = data[0:sz, sz:c]
    res[0:r-sz, c-sz:c] = data[sz:r, 0:sz]
    res[r-sz:r, c-sz:c] = data[0:sz, 0:sz]
    return res

# Run pairwise swaps and write results to file.
def runFull():
    # 256 x 256 x 4
    i1 = imageToNp('bush.png')
    i2 = imageToNp('church.png')
    assert i1.shape == i2.shape
    (r, c, d) = i1.shape
    assert d == 3 # rgb
    spec1 = np.fft.rfft2(i1, axes=(0, 1))
    spec2 = np.fft.rfft2(i2, axes=(0, 1))
    assert spec1.shape == spec2.shape
    pow1, phase1 = np.abs(spec1), np.angle(spec1)
    pow2, phase2 = np.abs(spec2), np.angle(spec2)
    saveImageIFFT('bushPowChurchPhase.png', pow1 * np.exp(1j * phase2))
    saveImageIFFT('churchPowBushPhase.png', pow2 * np.exp(1j * phase1))

# Just experimenting, merge the images with random noise.
def runFullRandom():
    i1 = imageToNp('bush.png')
    i2 = imageToNp('church.png')
    iR = np.random.random(i1.shape)
    assert i1.shape == i2.shape
    (r, c, d) = i1.shape
    assert d == 3 # rgb

    spec1 = np.fft.rfft2(i1, axes=(0, 1))
    spec2 = np.fft.rfft2(i2, axes=(0, 1))
    specR = np.fft.rfft2(iR, axes=(0, 1))
    assert spec1.shape == spec2.shape
    assert spec1.shape == specR.shape

    pow1, phase1 = np.abs(spec1), np.angle(spec1)
    pow2, phase2 = np.abs(spec2), np.angle(spec2)
    powR, phaseR = np.abs(specR), np.angle(specR)
    saveImageIFFT('bushPowRandPhase.png', pow1 * np.exp(1j * phaseR))
    saveImageIFFT('randPowBushPhase.png', powR * np.exp(1j * phase1))
    saveImageIFFT('churchPowRandPhase.png', pow2 * np.exp(1j * phaseR))
    saveImageIFFT('randPowChurchPhase.png', powR * np.exp(1j * phase2))

# Chunked 2D FFT, assuming input is real (use rfft2) so result only fills half the values
def oneImageRealSpectrogram(sz, img):
    (r, c, d) = img.shape
    assert d == 3
    assert r % sz == 0 and c % sz == 0 and sz % 2 == 0
    spec = np.zeros(img.shape, dtype=complex)
    for i in range(0, r, sz):
        for j in range(0, c, sz):
            for k in range(0, d):
                # sz/2+1 as that's all that's needed for rfft2
                spec[i:i+sz,j:j+sz/2+1,k] = np.fft.rfft2(img[i:i+sz,j:j+sz,k])
    return spec

# Applies full (complex) 2D FFT then centers the result, optionally interleaving.
def oneImageSpectrogram(sz, interleave, img):
    (r, c) = img.shape
    assert r % sz == 0 and c % sz == 0
    rSz, cSz = r / sz, c / sz

    spec = np.zeros(img.shape, dtype=complex)
    for i in range(0, r, sz):
        for j in range(0, c, sz):
            fft = center0(np.fft.fft2(img[i:i+sz,j:j+sz]))
            if interleave:
                spec[i/sz::rSz, j/sz::cSz] = fft
            else:
                spec[i:i+sz,j:j+sz] = fft

    if interleave and (r / sz) > 1:
        spec = centerInterleave(spec, r / sz)
    return spec

# Show a possibly interleaved spectrogram for different chunk sizes.
def showSpectrogram(interleave, path='bush.png'):
    img = imageToNp(path)
    img = np.mean(img, axis=2) # greyscale

    size = 4
    steps = 4
    ax = viz.cleanSubplots(2, steps)
    for r in range(0, steps):
        spec = oneImageSpectrogram(size, interleave, img)
        power, phase = np.abs(spec), np.angle(spec)
        ax[0, r].set_title("%d x %d" % (size, size))
        ax[0, r].imshow(np.log(power+1e-12), cmap='gray')
        ax[1, r].imshow(phase, cmap='gist_rainbow')
        size = size * 4
    plt.show()

# Merge a power and phase source, using a given chunk size.
def imageMerge(power, phase, sz):
    assert power.shape == phase.shape
    (r, c, d) = power.shape
    assert d == 3
    assert r % sz == 0 and c % sz == 0

    img = np.zeros(power.shape, dtype=complex)
    for i in range(0, r, sz):
        for j in range(0, c, sz):
            for k in range(0, d):
                # Pick out the parts we need from each source
                fft = power[i:i+sz,j:j+sz/2+1,k] * np.exp(1j * phase[i:i+sz,j:j+sz/2+1,k])
                img[i:i+sz,j:j+sz,k] = np.fft.irfft2(fft)

    # Normalize to an 8-bit int image [0, 256)
    img = normalizeImg(img)
    img = img * 256.0
    img = img.astype(np.uint8)
    return img

# Show the result of merging two images at different chunk sizes.
def showMerged(powerPath, phasePath):
    powerImg = imageToNp(powerPath)
    phaseImg = imageToNp(phasePath)
    # powerImg = np.random.random(phaseImg.shape) # testing merging with noise.
    size = 2
    steps = 4
    ax = viz.cleanSubplots(1, steps)
    for r in range(0, steps):
        spec1 = oneImageRealSpectrogram(size, powerImg)
        spec2 = oneImageRealSpectrogram(size, phaseImg)
        power1, phase1 = np.abs(spec1), np.angle(spec1)
        power2, phase2 = np.abs(spec2), np.angle(spec2)
        result = imageMerge(power1, phase2, size)
        ax[r].set_title("%d x %d" % (size, size))
        ax[r].imshow(result)
        size = size * 4
    plt.show()

# Demo: Show the initial bush image, plus 2D power and phase spectrogram.
def bushSpectrogramDemo():
    img = imageToNp('bush.png')
    img = np.mean(img, axis=2) # greyscale
    spec = center0(np.fft.fft2(img, axes=(0, 1)))
    power, phase = np.abs(spec), np.angle(spec)

    ax = viz.cleanSubplots(2, 2)
    ax[0, 0].imshow(img, cmap='gray')
    ax[0, 1].imshow(np.log(power + 1e-12), cmap='gray')
    ax[1, 1].imshow(phase, cmap='gist_rainbow')
    ax[1, 0].axis('off')
    plt.show()

# Demo: Show the result of merging the power and phase of two sources, both ways.
def bushChurchMix():
    imgBush = imageToNp('bush.png')
    imgChur = imageToNp('church.png')
    specBush = center0(np.fft.fft2(np.mean(imgBush, axis=2)))
    specChur = center0(np.fft.fft2(np.mean(imgChur, axis=2)))

    # Ugly code that does the merging. Ew, sorry!
    bushPowChurPhase = (normalizeImg(np.fft.irfft2(
        np.abs(np.fft.rfft2(imgBush, axes=(0,1))) *
            np.exp(1j * np.angle(np.fft.rfft2(imgChur, axes=(0,1)))), axes=(0, 1)
    )) * 256).astype(np.uint8)
    churPowBushPhase = (normalizeImg(np.fft.irfft2(
        np.abs(np.fft.rfft2(imgChur, axes=(0,1))) *
            np.exp(1j * np.angle(np.fft.rfft2(imgBush, axes=(0,1)))), axes=(0, 1)
    )) * 256).astype(np.uint8)

    # Just displaying the result in an interpretably way:
    ax = viz.cleanSubplots(2, 3, pad=0.1)
    ax[0, 0].set_title('Bush power')
    ax[0, 0].imshow(np.log(np.abs(specBush) + 1e-12), cmap='gray')
    ax[0, 1].set_title('+ Churchill phase')
    ax[0, 1].imshow(np.angle(specChur), cmap='gist_rainbow')
    ax[0, 2].set_title(' = Combined')
    ax[0, 2].imshow(bushPowChurPhase)
    ax[1, 0].set_title('Churchill power')
    ax[1, 0].imshow(np.log(np.abs(specChur) + 1e-12), cmap='gray')
    ax[1, 1].set_title('+ Bush phase')
    ax[1, 1].imshow(np.angle(specBush), cmap='gist_rainbow')
    ax[1, 2].set_title(' = Combined')
    ax[1, 2].imshow(churPowBushPhase)
    plt.show()

# Demo to show the recombined STFT result for chunk sizes, with and without interleaving.
def bushSTFTDemo():
    showSpectrogram(False)
    showSpectrogram(True)

# Show the merged result of chunking, given two sources.
def mergedSTFT():
    showMerged('bush.png', 'church.png')


if __name__ == '__main__':
    random.seed(4321)
    np.random.seed(4321)
    # bushSpectrogramDemo()
    # bushChurchMix()
    # bushSTFTDemo()
    mergedSTFT()
