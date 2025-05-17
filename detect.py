import cv2
import numpy as np
import random as rd
import os

valid_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
jpg_files = [files for files in os.listdir("sample_imgs") if files.endswith(valid_formats)]

print("files loaded:", jpg_files[:5], "...")

if jpg_files:
    selected_file = rd.choice(jpg_files)
    img = cv2.imread(os.path.join('sample_imgs', selected_file))
    h, w, c = img.shape 
    print("height:", h)
    print("width:", w)
    print("channels:", c)
    cv2.imshow('image', img)
    # wait for 5 seconds
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
else:
    print("No jpg files found in the directory.") 