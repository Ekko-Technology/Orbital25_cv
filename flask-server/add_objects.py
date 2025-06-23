import cv2 
import numpy as np
import random as rd
import os
from sklearn.neighbors import NearestNeighbors


coordinates = [] # list to store coordinates of the object

# Calculate median RGB of object selected
def find_item_RGB(object_path):
    # use opencv to return the matrix of the image
    item = cv2.imread(object_path)

    # scans the image and returns the median value of each channel
    median_b = int(np.median(item[:, :, 0]))
    median_g = int(np.median(item[:, :, 1]))
    median_r = int(np.median(item[:, :, 2]))
    median_bgr = (median_b, median_g, median_r)  # openCV uses BGR order

    # print("Median RGB:", median_bgr)
    return median_bgr


# use k-nearest neighbors to find the closest 25 pixels in the base image
def find_closest_pixels(img, target_bgr, k=25):
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
    global coordinates

    # obtain median RGB of object
    median_BGR = find_item_RGB(path)
    # find the 25 closest pixels in base image
    list_of_coordinates = find_closest_pixels(base_img, median_BGR)
    # print(list_of_coordinates)

    # randomly shuffle the list of coordinates presented and picked the first that is not within 35 pixels of the previous coordinates in x and y coorinates
    rd.shuffle(list_of_coordinates)
    for a, b in list_of_coordinates:
        for x, y in coordinates:
            if abs(x - a) > 50 and abs(y - b) > 50:
                return [a, b]
    
    # if all the coordinates are too close, simple return a random coordinate from the list then
    return rd.choice(list_of_coordinates)



def color_adjust_object(object_img, base_img, target_coordinate, alpha): # alpha determines the degree to color blend
    
    # adjust the colors of object_img so that its mean color matches the mean color of the base image region at target_coordinate.

    y, x = target_coordinate
    h, w, _ = object_img.shape

    # Crop corresponding region from base image
    base_patch = base_img[y:y+h, x:x+w]

    # Create a new blank image to store the adjusted pixels
    adjusted_img = np.zeros_like(object_img)
    # iterate through each pixel in base img and object and reduce their difference to blend them better
    for i in range(h):
        for j in range(w):
            base_pixel = base_patch[i, j].astype(np.float32)
            object_pixel = object_img[i, j].astype(np.float32)
            adjusted_pixel = object_pixel + (base_pixel - object_pixel) * alpha
            adjusted_img[i, j] = np.clip(adjusted_pixel, 0, 255) # ensure pixel values stays between 0 and 255

    # We realized that the function below existed, essentially doing the same as the code above but decided to keep it for reference and stick to our custom built version
    # adjusted_img = cv2.addWeighted(object_img, max(1-alpha, 0.3), base_patch, alpha, 0) 

    return adjusted_img


# paste png object whilst ensuring background noise is removed
def paste_object(base_img, object_path, target_coordinate, alpha=0.5, intended_width=30):
    # read the object image with alpha channel 
    object_img = cv2.imread(object_path, cv2.IMREAD_UNCHANGED)
    h, w, c = object_img.shape

    # resize the object image to ur intended width for the game
    intended_height = int(h * intended_width/ w)  # resize the object image height in proportion to intended width
    object_img = cv2.resize(object_img, (intended_width,intended_height))  # resize the object image to a fixed size

    # split alpha channel as we are taking in png files
    b,g,r,a = cv2.split(object_img) 
    object_img = cv2.merge((b, g, r))
    mask = a

    # object_rgb = color_adjust_object(object_img, base_img, target_coordinate)

    # get the dimensions of the object image
    height, width, c = object_img.shape
    # unpack target coordinates to place the object in
    y, x = target_coordinate

    base_h, base_w, _ = base_img.shape
    # Adjust if object goes outside the base image boundaries
    if y + height > base_h:
        y = base_h - height
    if x + width > base_w:
        x = base_w - width

    # ensure that coordinates are do not go to negative
    y = max(0, y)
    x = max(0, x)

    # blend the image with the colors of its surroundings
    object_img = color_adjust_object(object_img, base_img, (y, x), alpha) # adjust the color of the blended image

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
    # adjust the two parameters below to change the difficulty level of the game
    alpha = 0.3 # degree to color blend and its transparency with surroundings (0: no blend and 1: full blend)
    intended_width = 20 # intended width of the object to be pasted
    # adjust this to display number of objects you want to add
    num_objects = 3

    base_img = cv2.imread("lena.jpg")
    cv2.imshow("Base Image", base_img)

    items_list = [files for files in os.listdir("objects") if files.endswith(('.png'))]

    # randomly select intended number of objects to add into our spot the diff image from the list of object files
    selected_files = rd.sample(items_list, num_objects)

    for file in selected_files:
        print("Selected object:", file)
        path = os.path.join('objects', file)

        # this coordinate will be added to the spot the difference coordinates
        smart_coordinate = coordinate_to_add(path, base_img)
        coordinates.append(smart_coordinate)
        print(f"coordinates from top left corner:\nHeight: {smart_coordinate[0]}\nWidth: {smart_coordinate[1]}")
        
        # paste the object on the base image
        result_img = paste_object(base_img, path, smart_coordinate, alpha, intended_width)

    # show the result
    cv2.imshow("Result", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  


    