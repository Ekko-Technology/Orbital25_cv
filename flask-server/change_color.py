import cv2
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import os

"""
The code below takes in an image and performs image pre-processing in order to detect contours and objects within the image. Upon doing so, it randomly selects a contour of good area and applies two different transformations to the contour: changing the color of the contour to blend the contour better to its surrounding pixels, or expanding the contour by a certain factor. The transformations are applied a specified number of times, and the results are displayed using matplotlib.
"""

dict_transforms = {
    1: "change_color",
    2: "expand_contour"
}

def find_median_RGB(img):
    # scans the image and returns the median value of each channel
    median_b = int(np.median(img[:, :, 0]))
    median_g = int(np.median(img[:, :, 1]))
    median_r = int(np.median(img[:, :, 2]))
    median_bgr = (median_b, median_g, median_r)  # openCV uses BGR order

    # print("Median RGB:", median_bgr)
    return median_bgr



def find_suitable_contours(good_contours, contours_indices_picked):
    """
    Randomly selects a contour from the good contours that has not been picked before or not too close to those already picked.
    """
    threshold_distance = 100  # Minimum distance to consider a contour as "too close"
    too_close = False
    
    while True:
        contour_chosen = rd.choice(good_contours)
        M1 = cv2.moments(contour_chosen)

        centroid_chosen = [M1['m10']//M1['m00'], M1['m01']//M1['m00']]
        too_close = False

        #  iterate through all chosen contours to check if distance between the centroids is less than the threshold distance
        for picked_contour in contours_indices_picked:
            M2 = cv2.moments(picked_contour)
            picked_centroid = [M2['m10']//M2['m00'], M2['m01']//M2['m00']]

            # Calculate the distance between the centroids of the chosen contour and the picked contour through pythagoras
            x_diff = centroid_chosen[0] - picked_centroid[0]
            y_diff = centroid_chosen[1] - picked_centroid[1]
            distance = (x_diff**2 + y_diff**2) ** 0.5
            # print(distance)
            if distance < threshold_distance:
                too_close = True
                break
        
        if not too_close:
            return contour_chosen



def change_color(img, contour):
    # create a black mask of the contour
    mask_img = np.zeros(img.shape[:2], dtype=np.uint8)
    # print("Contour shape:", contour.shape)
    cv2.drawContours(mask_img, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)  # Fill the contour with white color

    # dilate that masked contour outwards in lights of capturing our surroundings RGB values
    kernel = np.ones((25, 25), np.uint8)
    dilated_mask = cv2.dilate(mask_img, kernel, iterations=1)
    # print("Dilated mask shape:", dilated_mask.shape)

    # bitwise Exlusive OR both masks such that only the surrounding pixels that are dillated are left
    ROI_mask = cv2.bitwise_xor(dilated_mask, mask_img)

    # extract the rgb values of the pixels of the dilated mask
    ROI_pixels = cv2.bitwise_and(img, img, mask=ROI_mask)  # Isolate the region of interest (ROI) in the original image

    # calculate the median color of the surrounding pixels in ROI_pixels
    median_surrounding_color = np.median(ROI_pixels[ROI_mask == 255], axis=0).astype(np.uint8)

    #replace the color of the contour with the median color
    img[mask_img == 255] = median_surrounding_color

    return img



def expand_contour(img, contour, expansion_factor):

    # Create a black mask of the same size as the image
    mask_img = np.zeros(img.shape[:2], dtype=np.uint8) 
    # Fill the contour with white color 
    mask_img = cv2.drawContours(mask_img, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)  
    # use bitwise and operator to isolate the region of interest (ROI) object from the original image
    ROI_masked = cv2.bitwise_and(img, img, mask=mask_img)

    # cv2.imshow("ROI Masked", ROI_masked)  # Display the masked image for debugging
    # cv2.waitKey(0)  # Wait for a key press to close the window

    # get boundaries of the contour
    x, y, width, height = cv2.boundingRect(contour) # x, y = top-left coordinates

    cropped = ROI_masked[y:y+height, x:x+width]  # Crop the masked image to the bounding rectangle of the contour
    
    expanded_crop = cv2.resize(cropped, (int(width * expansion_factor), int(height * expansion_factor)))  # Resize the cropped image to the new size

    # creates a boolean mask of whether each pixel is 0
    black_mask = np.all(expanded_crop == 0, axis=-1)  # Create a mask of black pixels in the expanded crop
    enlarge_original_crop = img[y:y+expanded_crop.shape[0], x:x+expanded_crop.shape[1]]  # Replace black pixels from black mask with original image pixels

    expanded_crop[black_mask] = enlarge_original_crop[black_mask]  # Replace black pixels in the expanded crop with the corresponding pixels from the original image

    # paste the crop into the original image at the same position
    img[y:y+expanded_crop.shape[0], x:x+expanded_crop.shape[1]] = expanded_crop  # Paste the expanded crop back into the original image

    return img



def contours(num_of_changes):

    # img_path = rd.choice(list_of_images)
    # img = cv2.imread(os.path.join("../sample_imgs", img_path))
    img = cv2.imread("family_animated.jpg")  # Load a sample image for contour detection

    # resize image to intended size for consistency
    img = cv2.resize(img, (640, 640))

    img_area = img.copy()
    img_ignored_area = img.copy()
    img_modified = img.copy()

    # convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # h,w = gray_img.shape

    # blur out the image to reduce noise, using bilateral filter for better preserving of edges
    blurred_img = cv2.bilateralFilter(gray_img, 3, 50, 50)  # image, diameter of the pixel neighborhood,sigma in color space(the greater the value, the colors farther to each other will start to get mixed), sigma in coordinate space (The greater its value, the more further pixels will mix together, given that their colors lie within the sigmaColor range).

    # Adaptive thresholding where the mean of the neighbor pixels is used to determine the threshold value
    # NOTE: experiment between ADAPTIVE_THRESH_MEAN_C and ADAPTIVE_THRESH_GAUSSIAN_C
    thresh = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2) # 11 is the neighbouring block size considered and 2 is the constant subtracted from the mean

    # Utilizing dilations and erosion to reduce noise in thresholded image, helps to join small gaps in the contours
    # NOTE: experiment between cv2.MORPH_CLOSE and cv2.MORPH_OPEN
    kernel = np.ones((5, 5), np.uint8)
    final_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # extracts contours in the form of numpy arrays of coordinates(x, y) representing a contour
    # RETR_TREE retrieves all contours and also returns a full hierarchy of nested contours, RETR_LIST retrieves all contours without returning hierarchical relationships and RETR_EXTERNAL retrieves only the external contours. CHAIN_APPROX_SIMPLE removes all redundant points and compresses the contour, thereby saving memory, while CHAIN_APPROX_NONE stores all the points of the contour.
    contours, _ = cv2.findContours(final_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 

    # filter the contours to only consider areas of sufficient area
    good_contours = []
    ignored_contours = []

    min_area_for_objects = 500  # Minimum area threshold for contours to be considered valid
    min_area = 500  # Minimum area threshold for contours to be considered valid
    max_area = 1500
    
    for contour in contours:
        area = cv2.contourArea(contour)
        # perimeter = cv2.arcLength(contour, True)  # Calculate the perimeter of the contour

        if area > min_area and area < max_area:
            good_contours.append(contour)
        elif area > min_area_for_objects:
            ignored_contours.append(contour)
    
    # Draw all contours on the original image
    cv2.drawContours(img_area, good_contours, -1, (0, 255, 0), 3) 
    cv2.drawContours(img_ignored_area, ignored_contours, -1, (255, 0, 0), 3) 

    # smaller objects in the image. can be used for expanding larger or changing color of objects
    centroids = []
    for contour in good_contours:
        # cv2.moments returns a dictionary of containing the spatial moments of the contour, like the area of the contour, values to calculate the centroid, etc.   
        M = cv2.moments(contour)
        # use the keys 'm10', 'm01', and 'm00' to calculate the centroid of the contour which are the sum of x-coordinates, sum of y-coordinates, and the area of the contour respectively
        Cx = int(M['m10'] / M['m00']) # x-coordinate of centroid
        Cy = int(M['m01'] / M['m00']) # y-coordinate of centroid
        centroids.append((Cx, Cy))

        # Draw the centroid on the original image
        cv2.circle(img_area, (Cx, Cy), 5, (255, 0, 255), -1)
        cv2.putText(img_area, f"Area:{M['m00']}", (Cx + 10, Cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)

    print("Centroids of contours:", centroids)

    for contour in ignored_contours:
        M = cv2.moments(contour)
        Cx = int(M['m10'] / M['m00']) # x-coordinate of centroid
        Cy = int(M['m01'] / M['m00']) # y-coordinate of centroid

        cv2.circle(img_ignored_area, (Cx, Cy), 5, (20, 60, 100), -1)  # Draw a circle at the random point
        cv2.putText(img_ignored_area, f"Area: {cv2.contourArea(contour)}", (Cx + 10, Cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)

    
    # Visualization tools of processing steps
    # plt.figure(figsize=(12, 8))

    # plt.subplot(3, 3, 1)
    # plt.title("Original Image")
    # plt.imshow(img)

    # plt.subplot(3, 3, 2)
    # plt.title("blurred Image")
    # plt.imshow(blurred_img, cmap="gray")

    # plt.subplot(3, 3, 3)
    # plt.title("Thresholded Gaussian")
    # plt.imshow(thresh, cmap="gray")

    # plt.subplot(3, 3, 4)
    # plt.title("Morphed Image")
    # plt.imshow(final_img, cmap="gray")

    # plt.subplot(3, 3, 5)
    # plt.title("Contours with good area")
    # plt.imshow(img_area)

    # plt.subplot(3, 3, 6)
    # plt.title("Contours ignored")
    # plt.imshow(img_ignored_area)

    # plt.subplot(3, 3, 7)
    # plt.title("Contours with good area")
    # plt.imshow(cv2.drawContours(np.zeros_like(img.copy()), good_contours, -1, (0, 255, 0), 3))

    # plt.subplot(3, 3, 8)
    # plt.title("Contours ignored")
    # plt.imshow(cv2.drawContours(np.zeros_like(img.copy()), ignored_contours, -1, (255, 0, 0), 3))

    contours_indices_picked = []

    for i in range(num_of_changes):
        # Comparison operators doesn't work with numpy arrays so we convert the numpy array to a list of indices
        # available_indices = [idx for idx in range(len(good_contours)) if idx not in contours_indices_picked]
        # idx_chosen = rd.choice(available_indices)
        # contour = good_contours[idx_chosen]
        # contours_indices_picked.append(idx_chosen)  # Remove the selected contour to avoid repeating the same contour

        contour_chosen = find_suitable_contours(good_contours, contours_indices_picked)
        contours_indices_picked.append(contour_chosen)

        transform_type = rd.choice(list(dict_transforms.keys()))
        if transform_type == 1:
            print(f"Color change transformation selected. Changing object color on contour coordinate {contour_chosen[0][0]}")
            change_color(img_modified, contour_chosen)
        elif transform_type == 2:
            expansion_factor = rd.uniform(1.4, 1.5)
            print(f"Expansion transformation selected. Changing object size on contour coordinate {contour_chosen[0][0]}")
            expand_contour(img_modified, contour_chosen, expansion_factor)
    
    plt.subplot(3, 3, 9)
    plt.title("Modified Image")
    plt.imshow(img_modified)
    plt.imsave("modified_image.jpg", img_modified)  # Save the modified image
        
    plt.tight_layout()
    plt.show()


    return (img_modified, contours_indices_picked)

                

if __name__ == "__main__":
    num_changes = 3  # Number of times to change color or expand contour
    # list_of_images = [files for files in os.listdir("../sample_imgs")]
    contours(num_changes)