import cv2
import numpy as np 
import random as rd
import os


valid_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
jpg_files = [files for files in os.listdir("../sample_imgs") if files.endswith(valid_formats)]


def track_color():
    img = np.zeros((480, 640, 3), np.uint8)
    cv2.namedWindow('image')
    
    cv2.createTrackbar('R', 'image', 0, 255, lambda x: None)
    cv2.createTrackbar('G', 'image', 0, 255, lambda x: None)
    cv2.createTrackbar('B', 'image', 0, 255, lambda x: None)

    cv2.createTrackbar('Switch', 'image', 0, 1, lambda x: None)

    while True:
        cv2.imshow('image', img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
        r = cv2.getTrackbarPos('R', 'image')
        g = cv2.getTrackbarPos('G', 'image')
        b = cv2.getTrackbarPos('B', 'image')
        # Switch enabled for color change
        s = cv2.getTrackbarPos('Switch', 'image')

        if s == 0:
            pass
        else:
            img[:] = [b, g, r]

    cv2.destroyAllWindows() 



def isolate_color():
    # comment out the line below to use the webcam
    # cap = cv2.VideoCapture(0)

    sample_img = cv2.imread(os.path.join('../sample_imgs', rd.choice(jpg_files)))
    # change from BGR to HSV
    hsv = cv2.cvtColor(sample_img, cv2.COLOR_BGR2HSV)
    cv2.imshow('sample_img', sample_img)

    # creating trackbar to toggle HSV values
    cv2.namedWindow('Trackbars')
    # Takes in trackbar name, window name, initial value, max value, callback function
    cv2.createTrackbar('LH', 'Trackbars', 0, 179, lambda x: None)
    cv2.createTrackbar('LS', 'Trackbars', 0, 255, lambda x: None)
    cv2.createTrackbar('LV', 'Trackbars', 0, 255, lambda x: None)
    cv2.createTrackbar('HH', 'Trackbars', 179, 179, lambda x: None)
    cv2.createTrackbar('HS', 'Trackbars', 255, 255, lambda x: None)
    cv2.createTrackbar('HV', 'Trackbars', 255, 255, lambda x: None)

    while True:
        # Uncomment the line below to use the webcam
        # _, frame = cap.read()  # read frame from the webcam
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # convert to HSV

        L_H = cv2.getTrackbarPos('LH', 'Trackbars')
        L_S = cv2.getTrackbarPos('LS', 'Trackbars')
        L_V = cv2.getTrackbarPos('LV', 'Trackbars')
        H_H = cv2.getTrackbarPos('HH', 'Trackbars')
        H_S = cv2.getTrackbarPos('HS', 'Trackbars')
        H_V = cv2.getTrackbarPos('HV', 'Trackbars')
        # define the range of color to isolate 
        lower_bound = np.array([L_H, L_S, L_V])
        upper_bound = np.array([H_H, H_S, H_V])
        # Checks each pixel if within the range of H, S, V values. If True, mask reflects 255(white), else 0(black)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        cv2.imshow('mask', mask)
        # bitwise_and to isolate the color, where the oordinates of non-black pixels on the mask are used to extract pixels from the original image 
        # isolated_color = cv2.bitwise_and(sample_img, sample_img, mask=mask)
        isolated_color = cv2.bitwise_and(sample_img, sample_img, mask=mask)
        cv2.imshow('isolated_color', isolated_color)

        # bitwise_and between keyboard ASCII and 0xFF
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    print("Exiting")


def thresholding():
    # read the image
    img = cv2.imread(os.path.join('../sample_imgs', rd.choice(jpg_files)))
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # apply binary thresholding where any pixel value above 127 is set to 255 and below is set to 0
    # _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # apply adaptive thresholding where the mean of the neighbor pixels is used to determine the threshold value
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2) # 11 is the neighbouring block size considered and 2 is the constant subtracted from the mean
    thresh3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2) 
    
    # Utilizing dilations and erosion to reduce noise in thresholded image
    kernel = np.ones((10, 10), np.uint8)
    # cv2.dilate(thresh, kernel, iterations=1)
    # cv2.dilate(thresh2, kernel, iterations=1)
    # cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
    cv2.morphologyEx(thresh3, cv2.MORPH_CLOSE, kernel)


    cv2.imshow('original', img)
    # cv2.imshow('gray', gray)
    # cv2.imshow('thresholded', thresh)
    cv2.imshow('adaptive thresholded', thresh2)
    cv2.imshow('adaptive thresholded 2', thresh3)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
        print("Exiting")


if __name__ == "__main__":
    # track_color()
    # isolate_color()
    thresholding()