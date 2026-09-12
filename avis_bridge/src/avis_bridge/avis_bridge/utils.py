'''
@ 2025, Copyright AVIS Engine
'''
# Utils for AVISEngine Class
import base64
import cv2
import numpy as np
from PIL import Image
import io


__author__ = "Amirmohammad Zarif"
__email__ = "amirmohammadzarif@avisengine.com"

def stringToImage(base64_string):
    '''
    Converts Base64 String to Image

    Parameters
    ----------
    base64_string : str
        base64 image data to be converted
    '''
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

def BGRtoRGB(image):
    '''
    Converts PIL Image to an RGB image(technically a numpy array) that's compatible with opencv
    '''
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

def KMPSearch(pat, txt):
    '''
    Knuth-Morris-Pratt(KMP) Algorithm
    '''
    M = len(pat)
    N = len(txt)
    lps = [0] * M
    j = 0
    computeLPS(pat, M, lps)
    i = 0
    while i < N:
        if pat[j] == txt[i]:
            i += 1
            j += 1
        if j == M:
            return (i-j)
        elif i < N and pat[j] != txt[i]:
            if j != 0:
                j = lps[j-1]
            else:
                i += 1
    return -1

def computeLPS(pat, M, lps):
    '''
    Computing the LPS
    '''
    len_ = 0
    lps[0]
    i = 1
    while i < M:
        if pat[i] == pat[len_]:
            len_ += 1
            lps[i] = len_
            i += 1
        else:
            if len_ != 0:
                len_ = lps[len_-1]
            else:
                lps[i] = 0
                i += 1
