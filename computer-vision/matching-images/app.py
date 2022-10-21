from cProfile import label
import gradio as gr
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage.filters import convolve
import numpy as np


def sift(img1, img2):
    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1,descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)

    img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
    return img3

def orb(img1, img2):
    orb = cv2.ORB_create()

    keypoints_1, descriptors_1 = orb.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1,descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)

    img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
    return img3

def match(img1, img2):

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    sift_res = sift(img1, img2)
    orb_res = orb(img1, img2)

    return [sift_res, orb_res]


interface = gr.Interface(
    title = "SIFT and ORB Image Matching 🖼 👉 🖼",
    description = "<h3>Scale Invariant Feature Transform (SIFT) & Oriented FAST and Rotated BRIEF (ORB) </h3> <br> <b>Select training and query images 🖼</b>",
    article='~ Ivanrs',
    allow_flagging = "never",
    fn = match, 
    inputs = [
        gr.Image(label = "Train Image", shape = [300, 200]),
        gr.Image(label = "Query Image", shape = [300, 200]),
    ],
    outputs = [
        gr.Image(label = "SIFT Output"),
        gr.Image(label = "ORB Output"),
    ],
    examples = [
        ["images/img1.jpg", "images/img2.jpg"],
        ["images/img3.jpg", "images/img4.jpg"],
        ["images/img5.jpg", "images/img6.png"],
        ["images/img7.jpeg", "images/img8.jpeg"]
    ]
)

interface.launch(share = False)