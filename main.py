import cv2
import numpy as np
import random as rd
import os
from sklearn.neighbors import NearestNeighbors

#Create/Ensure a temporary objects directory exists
OBJECTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'objects')

# For contour manipulation
DICT_TRANSFORMS = {
    1: "change_color",
    2: "expand_contour"
}
MIN_AREA_FOR_CONTOURS = 500
MAX_AREA_FOR_CONTOURS = 1500
THRESHOLD_CONTOUR_DISTANCE = 100 # Minimum distance between chosen contour centroids

# For object addition
THRESHOLD_OBJECT_COORDINATE_DISTANCE = 50 # Minimum distance between added object coordinates

def get_contour_bounding_box(contour):
    # returns a bounding box for the contour
    x, y, w, h = cv2.boundingRect(contour)
    return [x, y, x + w, y + h]

def find_median_RGB(img):
    # scans the image and returns the median value of each channel
    median_b = int(np.median(img[:, :, 0]))
    median_g = int(np.median(img[:, :, 1]))
    median_r = int(np.median(img[:, :, 2]))
    median_bgr = (median_b, median_g, median_r)  # openCV uses BGR order

    # print("Median RGB:", median_bgr)
    return median_bgr

def find_suitable_contours(good_contours, contours_picked_data):

    attempts = 0
    max_attempts = 20 # prevent infinite loops if all contours are too close
    
    while attempts < max_attempts:
        if not good_contours: # no good contours available
            return None

        contour_chosen = rd.choice(good_contours)
        
        # calculate centroid of chosen contour
        M_chosen = cv2.moments(contour_chosen)
        if M_chosen['m00'] == 0: # avoid division by zero if contour area is 0
            attempts += 1
            continue
        centroid_chosen = [M_chosen['m10'] // M_chosen['m00'], M_chosen['m01'] // M_chosen['m00']]
        
        too_close = False
        for picked_contour in contours_picked_data:
            M_picked = cv2.moments(picked_contour)
            if M_picked['m00'] == 0: continue # skip if picked contour has zero area

            picked_centroid = [M_picked['m10'] // M_picked['m00'], M_picked['m01'] // M_picked['m00']]

            x_diff = centroid_chosen[0] - picked_centroid[0]
            y_diff = centroid_chosen[1] - picked_centroid[1]
            distance = (x_diff**2 + y_diff**2) ** 0.5
            
            if distance < THRESHOLD_CONTOUR_DISTANCE:
                too_close = True
                break
        
        if not too_close:
            return contour_chosen
        
        attempts += 1
        
    return None # return None if no suitable contour found after max_attempts

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
def coordinate_to_add_local_tracking(object_path, base_img, picked_coordinates):

    median_BGR = find_median_RGB(cv2.imread(object_path, cv2.IMREAD_COLOR)) # Read object for its median color
    list_of_coordinates = find_closest_pixels(base_img, median_BGR)
    
    rd.shuffle(list_of_coordinates)
    
    for potential_y, potential_x in list_of_coordinates:
        too_close = False
        for picked_y, picked_x in picked_coordinates:
            # check distance for current object (object_img) from previously placed ones
            x_diff = abs(potential_x - picked_x)
            y_diff = abs(potential_y - picked_y)
            if x_diff < THRESHOLD_OBJECT_COORDINATE_DISTANCE and y_diff < THRESHOLD_OBJECT_COORDINATE_DISTANCE:
                too_close = True
                break
        
        if not too_close:
            return [potential_y, potential_x]
        
    # if all the coordinates are too close, simple return a random coordinate from the list then
    return rd.choice(list_of_coordinates)

def color_adjust_object(object_img, base_img, target_coordinate, alpha): # alpha determines the degree to color blend
    """
    Adjusts the colors of object_img to blend with the base_img region.
    Returns a new object_img with adjusted colors.
    """
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


# --- main API Functions for Flask integration ---

def apply_contour_manipulation(original_img_array, num_of_changes=1):

    img_modified = original_img_array.copy()

    img_modified = cv2.resize(img_modified, (640, 640))

    gray_img = cv2.cvtColor(img_modified, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.bilateralFilter(gray_img, 3, 50, 50)
    thresh = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((5, 5), np.uint8)
    final_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(final_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 

    good_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if MIN_AREA_FOR_CONTOURS <= area <= MAX_AREA_FOR_CONTOURS:
            good_contours.append(contour)
    
    differences_coords = []
    contours_indices_picked_for_spacing = [] 

    for i in range(num_of_changes):
        contour_chosen = find_suitable_contours(good_contours, contours_indices_picked_for_spacing)
        if contour_chosen is None:
            break 

        contours_indices_picked_for_spacing.append(contour_chosen)

        differences_coords.append(get_contour_bounding_box(contour_chosen))

        transform_type = rd.choice(list(DICT_TRANSFORMS.keys()))
        
        if transform_type == 1:
            print(f"Applying color change to contour {i+1}...")
            img_modified = change_color(img_modified, contour_chosen)
        elif transform_type == 2:
            expansion_factor = rd.uniform(1.4, 1.5)
            print(f"Applying expansion (factor {expansion_factor}) to contour {i+1}...")
            img_modified = expand_contour(img_modified, contour_chosen, expansion_factor)
    
    return img_modified, differences_coords

def apply_object_addition(original_img_array, num_objects=1, alpha=0.5, intended_width=30):

    img_modified = original_img_array.copy()

    img_modified = cv2.resize(img_modified, (640, 640))

    items_list = [f for f in os.listdir(OBJECTS_DIR) if f.endswith(('.png', '.PNG'))]
    if not items_list:
        print(f"Error: No PNG objects found in '{OBJECTS_DIR}'. Please ensure the 'objects' folder exists and contains PNGs.")
        return original_img_array, []

    num_objects_to_add = min(num_objects, len(items_list), 3) # cap at 3 or less

    selected_files = rd.sample(items_list, num_objects_to_add)
    
    added_object_differences = []
    picked_coordinates_for_spacing = [] # store [y, x] for spacing checks

    for file_name in selected_files:
        object_path = os.path.join(OBJECTS_DIR, file_name)

        # get smart coordinate ensuring spacing
        smart_coordinate_yx = coordinate_to_add_local_tracking(object_path, img_modified, picked_coordinates_for_spacing)
        picked_coordinates_for_spacing.append(smart_coordinate_yx) # Add to list for next check


        obj_img_raw = cv2.imread(object_path, cv2.IMREAD_UNCHANGED)
        if obj_img_raw is None:
            print(f"Skipping {file_name}: Failed to read object image.")
            continue
        
        h_orig, w_orig, _ = obj_img_raw.shape
        if w_orig == 0: continue 
        
        h_resized = int(h_orig * intended_width / w_orig)
        w_resized = intended_width
        

        img_modified = paste_object(img_modified, object_path, smart_coordinate_yx, alpha, intended_width)
        
        x1 = smart_coordinate_yx[1]
        y1 = smart_coordinate_yx[0]
        x2 = x1 + w_resized
        y2 = y1 + h_resized

        added_object_differences.append([x1, y1, x2, y2])
        print(f"Added object '{file_name}' at: {[x1, y1, x2, y2]}")

    return img_modified, added_object_differences