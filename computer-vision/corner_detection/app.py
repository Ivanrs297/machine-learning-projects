import cv2
from scipy import signal as sig
import numpy as np
from scipy.ndimage.filters import convolve
import gradio as gr

def gradient_x(imggray):
    ##Sobel operator kernels.
    kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    return sig.convolve2d(imggray, kernel_x, mode='same')

def gradient_y(imggray):
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return sig.convolve2d(imggray, kernel_y, mode='same')

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def harris(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    I_x = gradient_x(img_gray)p
    I_y = gradient_y(img_gray)
    Ixx = convolve(I_x**2, gaussian_kernel(3, 1))
    Ixy = convolve(I_y*I_x, gaussian_kernel(3, 1))
    Iyy = convolve(I_y**2, gaussian_kernel(3, 1))


    k = 0.05

    # determinant
    detA = Ixx * Iyy - Ixy ** 2
    # trace
    traceA = Ixx + Iyy
        
    harris_response = detA - k * traceA ** 2

    window_size = 3
    offset = window_size//2
    width, height = img_gray.shape

    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            Sxx = np.sum(Ixx[y-offset:y+1+offset, x-offset:x+1+offset])
            Syy = np.sum(Iyy[y-offset:y+1+offset, x-offset:x+1+offset])
            Sxy = np.sum(Ixy[y-offset:y+1+offset, x-offset:x+1+offset])

    det = (Sxx * Syy) - (Sxy**2)
    trace = Sxx + Syy
    r = det - k*(trace**2)


    img_copy_for_corners = np.copy(img)
    img_copy_for_edges = np.copy(img)

    for rowindex, response in enumerate(harris_response):
        for colindex, r in enumerate(response):
            if r > 0:
                # this is a corner
                img_copy_for_corners[rowindex, colindex] = [255,0,0]
            elif r < 0:
                # this is an edge
                img_copy_for_edges[rowindex, colindex] = [0,255,0]

    return img_copy_for_corners


interface = gr.Interface(
title = "Harris Corner Detector 🤖",
description = "<h3>The idea is to locate interest points where the surrounding neighbourhood shows edges in more than one direction.</h3> <br> <b>Select an image 🖼</b>",
article='Step-by-step on GitHub <a href="https://github.com/Ivanrs297/machine-learning-projects/blob/main/computer-vision/corner_detection/main.ipynb"> notebook </a> <br> ~ Ivanrs',
allow_flagging = "never",
fn = harris, 
inputs = [
    gr.Image(),
],
outputs = "image"
)

interface.launch(share = False)