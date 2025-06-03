import cv2 
import numpy as np
import random as rd
import os
from sklearn.neighbors import NearestNeighbors


# Calculate median RGB of object selected
def find_item_RGB(object_path):
    # use opencv to return the matrix of the image
    item = cv2.imread(object_path)

    # scans the image and returns the median value of each channel
    median_b = int(np.median(item[:, :, 0]))
    median_g = int(np.median(item[:, :, 1]))
    median_r = int(np.median(item[:, :, 2]))
    median_rgb = (median_b, median_g, median_r)  # openCV uses BGR order

    print("Median RGB:", median_rgb)
    return median_rgb


# use k-nearest neighbors to find the closest 3 pixels in the base image
def find_closest_pixels(img, target_bgr, k=3):
    height, width, _ = img.shape
    # flatten the image to 2D array of pixels so e.g. (480x640x3) becomes (307200, 3)
    pixels = img.reshape(-1, 3)
    # create an array of coordinates for each pixel for (height, width)
    coordinates = np.array([(y, x) for y in range(height) for x in range(width)])

    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(pixels)
    distances, indices = knn.kneighbors([target_bgr])

    # get the coordinates of the k nearest neighbors
    matched_coordinates = coordinates[indices[0]]
    return matched_coordinates.tolist()


# randomly pick a coordinate from the list of closest pixels generated from the function above
def coordinate_to_add(path, base_img):
    # obtain median RGB of object
    median_BGR = find_item_RGB(path)
    # find the 3 closest pixels in base image
    list_of_coordinates = find_closest_pixels(base_img, median_BGR)
    return rd.choice(list_of_coordinates)


# paste png object whilst ensuring background noise is removed
def paste_object(base_img, object_path, target_coordinate):
    # read the object image with alpha channel 
    object_img = cv2.imread(object_path, cv2.IMREAD_UNCHANGED)
    h, w, c = object_img.shape
    new_h = int(h * 30 / w)  # resize the object image to a fixed height of 30 pixels
    object_img = cv2.resize(object_img, (30, new_h))  # resize the object image to a fixed size

    # split alpha channel as we are taking in png files
    b,g,r,a = cv2.split(object_img) 
    object_img = cv2.merge((b, g, r))
    mask = a

    # get the dimensions of the object image
    height, width, c = object_img.shape
    # unpack target coordinates to place the object in
    y, x = target_coordinate

    # isolate the region of interest (ROI) in the base image
    roi = base_img[y:y+height, x:x+width]

    # perform bitwise operations to get rid of the background
    mask_inverse = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(roi, roi, mask=mask_inverse) # based on masked image, only display the pixels that are not masked
    foreground = cv2.bitwise_and(object_img, object_img, mask=a) # based on masked image, only display the pixels that are masked
    blended_img = cv2.add(background, foreground) # add the two images together

    base_img[y:y+height, x:x+width] = blended_img # paste the blended image back to the base image
    return base_img


if __name__ == "__main__":

    base_img = cv2.imread("lena.jpg")

    items_list = [files for files in os.listdir("objects") if files.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]

    # randomly select an object to add from the list of object files
    selected_file = rd.choice(items_list)
    print("Selected object:", selected_file)
    path = os.path.join('objects', selected_file)

    # this coordinate will be added to the spot the difference coordinates
    target_coordinate = coordinate_to_add(path, base_img)
    print("Target coordinate:", target_coordinate)
    
    # paste the object on the base image
    result_img = paste_object(base_img, path, target_coordinate)
    # show the result
    cv2.imshow("Result", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
