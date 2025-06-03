import cv2
import numpy as np
import random as rd
import os
from sklearn.neighbors import NearestNeighbors

# --- Configuration ---
# Assuming 'objects' directory is a direct subfolder of 'backend'
# Adjust this path if your 'objects' folder is located elsewhere
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

# --- Helper Functions (from original files, slightly modified for flexibility) ---

def get_contour_bounding_box(contour):
    """
    Calculates the bounding box [x1, y1, x2, y2] for a given contour.
    """
    x, y, w, h = cv2.boundingRect(contour)
    return [x, y, x + w, y + h]

def find_median_RGB(img):
    # scans the image and returns the median value of each channel
    median_b = int(np.median(img[:, :, 0]))
    median_g = int(np.median(img[:, :, 1]))
    median_r = int(np.median(img[:, :, 2]))
    median_bgr = (median_b, median_g, median_r)  # openCV uses BGR order
    return median_bgr

def find_suitable_contours(good_contours, contours_picked_data):
    """
    Randomly selects a contour from the good_contours that has not been picked before
    or is not too close to those already picked.
    contours_picked_data: list of already picked contour NumPy arrays.
    """
    
    attempts = 0
    max_attempts = 20 # Prevent infinite loops if all contours are too close
    
    while attempts < max_attempts:
        if not good_contours: # No good contours available
            return None

        contour_chosen = rd.choice(good_contours)
        
        # Calculate centroid of chosen contour
        M_chosen = cv2.moments(contour_chosen)
        if M_chosen['m00'] == 0: # Avoid division by zero if contour area is 0
            attempts += 1
            continue
        centroid_chosen = [M_chosen['m10'] // M_chosen['m00'], M_chosen['m01'] // M_chosen['m00']]
        
        too_close = False
        for picked_contour in contours_picked_data:
            M_picked = cv2.moments(picked_contour)
            if M_picked['m00'] == 0: continue # Skip if picked contour has zero area

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
        
    return None # Return None if no suitable contour found after max_attempts

def change_color(img_array, contour):
    """
    Changes the color of a specific contour to blend with its surroundings.
    Modifies img_array in place.
    """
    mask_img = np.zeros(img_array.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask_img, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    kernel = np.ones((25, 25), np.uint8)
    dilated_mask = cv2.dilate(mask_img, kernel, iterations=1)
    ROI_mask = cv2.bitwise_xor(dilated_mask, mask_img)

    ROI_pixels = cv2.bitwise_and(img_array, img_array, mask=ROI_mask)
    
    # Ensure there are pixels in ROI_mask to calculate median from
    if np.any(ROI_mask == 255):
        median_surrounding_color = np.median(ROI_pixels[ROI_mask == 255], axis=0).astype(np.uint8)
        img_array[mask_img == 255] = median_surrounding_color
    
    return img_array

def expand_contour(img_array, contour, expansion_factor):
    """
    Expands a given contour by a factor and pastes it back into the image.
    Modifies img_array in place.
    """
    mask_img = np.zeros(img_array.shape[:2], dtype=np.uint8) 
    mask_img = cv2.drawContours(mask_img, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)  
    
    ROI_masked = cv2.bitwise_and(img_array, img_array, mask=mask_img)

    x, y, width, height = cv2.boundingRect(contour)

    # Ensure crop dimensions are positive
    if width <= 0 or height <= 0:
        return img_array

    cropped = ROI_masked[y:y+height, x:x+width]
    
    new_width = int(width * expansion_factor)
    new_height = int(height * expansion_factor)

    # Ensure new dimensions are positive before resizing
    if new_width <=0 or new_height <=0:
        return img_array

    expanded_crop = cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Calculate paste coordinates, adjusting for boundaries
    paste_y = y
    paste_x = x
    
    img_h, img_w, _ = img_array.shape

    # Adjust paste coordinates if expanded crop goes out of bounds
    if paste_y + new_height > img_h:
        paste_y = img_h - new_height
    if paste_x + new_width > img_w:
        paste_x = img_w - new_width
    
    paste_y = max(0, paste_y)
    paste_x = max(0, paste_x)

    # Get the region from the original image that the expanded crop will cover
    target_region = img_array[paste_y:paste_y + expanded_crop.shape[0], paste_x:paste_x + expanded_crop.shape[1]]

    # Create a mask for the black pixels in the expanded crop (where original contour was transparent)
    black_mask_expanded = np.all(expanded_crop == 0, axis=-1)

    # Replace black pixels in expanded_crop with corresponding pixels from the target region
    # Ensure dimensions match before assignment
    if expanded_crop.shape == target_region.shape:
        expanded_crop[black_mask_expanded] = target_region[black_mask_expanded]
    else: # If dimensions don't match (due to boundary adjustments), slice accordingly
        min_h = min(expanded_crop.shape[0], target_region.shape[0])
        min_w = min(expanded_crop.shape[1], target_region.shape[1])
        
        expanded_crop_sliced = expanded_crop[:min_h, :min_w]
        target_region_sliced = target_region[:min_h, :min_w]
        black_mask_expanded_sliced = black_mask_expanded[:min_h, :min_w]

        expanded_crop_sliced[black_mask_expanded_sliced] = target_region_sliced[black_mask_expanded_sliced]
        expanded_crop = expanded_crop_sliced # Update expanded_crop to the sliced version


    # Paste the (potentially sliced) expanded crop back into the original image
    img_array[paste_y:paste_y + expanded_crop.shape[0], paste_x:paste_x + expanded_crop.shape[1]] = expanded_crop

    return img_array

def find_closest_pixels(img, target_bgr, k=25):
    height, width, _ = img.shape
    pixels = img.reshape(-1, 3)
    coordinates = np.array([(y, x) for y in range(height) for x in range(width)])

    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(pixels)
    distances, indices = knn.kneighbors([target_bgr])
    matched_coordinates = coordinates[indices[0]]
    return matched_coordinates.tolist()

def coordinate_to_add_local_tracking(object_path, base_img, picked_coordinates):
    """
    Randomly picks a coordinate from the list of closest pixels, ensuring it's not too close
    to previously picked coordinates.
    picked_coordinates: list of [y, x] tuples of already chosen object insertion points.
    """
    median_BGR = find_median_RGB(cv2.imread(object_path, cv2.IMREAD_COLOR)) # Read object for its median color
    list_of_coordinates = find_closest_pixels(base_img, median_BGR)
    
    rd.shuffle(list_of_coordinates)
    
    for potential_y, potential_x in list_of_coordinates:
        too_close = False
        for picked_y, picked_x in picked_coordinates:
            # Check distance for current object (object_img) from previously placed ones
            x_diff = abs(potential_x - picked_x)
            y_diff = abs(potential_y - picked_y)
            if x_diff < THRESHOLD_OBJECT_COORDINATE_DISTANCE and y_diff < THRESHOLD_OBJECT_COORDINATE_DISTANCE:
                too_close = True
                break
        
        if not too_close:
            return [potential_y, potential_x] # Return [y, x]

    # If all found coordinates are too close, simply return a random coordinate
    # from the initially generated list (this might overlap)
    if list_of_coordinates: # Ensure list is not empty
        return rd.choice(list_of_coordinates)
    return [0,0] # Fallback if no coordinates found (shouldn't happen with k=25 unless image is tiny)

def color_adjust_object(object_img, base_img, target_coordinate, alpha):
    """
    Adjusts the colors of object_img to blend with the base_img region.
    Returns a new object_img with adjusted colors.
    """
    y, x = target_coordinate
    h, w, _ = object_img.shape

    # Ensure patch dimensions don't go out of bounds
    base_patch_h = min(h, base_img.shape[0] - y)
    base_patch_w = min(w, base_img.shape[1] - x)
    
    if base_patch_h <= 0 or base_patch_w <= 0:
        return object_img # Return original object image if target patch is invalid

    base_patch = base_img[y:y+base_patch_h, x:x+base_patch_w]
    object_patch_to_blend = object_img[:base_patch_h, :base_patch_w] # Slice object_img to match patch size

    # Vectorized blending
    base_pixel_float = base_patch.astype(np.float32)
    object_pixel_float = object_patch_to_blend.astype(np.float32)
    adjusted_pixel_float = object_pixel_float + (base_pixel_float - object_pixel_float) * alpha
    adjusted_img_patch = np.clip(adjusted_pixel_float, 0, 255).astype(np.uint8)

    # Create a full-sized adjusted_img with the blended patch
    full_adjusted_img = object_img.copy()
    full_adjusted_img[:base_patch_h, :base_patch_w] = adjusted_img_patch

    return full_adjusted_img

def paste_object(base_img_array, object_path, target_coordinate, alpha=0.5, intended_width=30):
    """
    Pastes a PNG object onto the base_img_array with blending and transparency.
    Modifies base_img_array in place.
    """
    object_img_raw = cv2.imread(object_path, cv2.IMREAD_UNCHANGED)
    if object_img_raw is None:
        print(f"Warning: Could not read object image at {object_path}. Skipping paste.")
        return base_img_array # Return original if object can't be read

    h_orig, w_orig, c_orig = object_img_raw.shape

    if w_orig == 0: # Avoid division by zero
        print(f"Warning: Object {object_path} has zero width. Skipping paste.")
        return base_img_array

    intended_height = int(h_orig * intended_width / w_orig)
    
    # Ensure valid dimensions for resizing
    if intended_width <= 0 or intended_height <= 0:
        print(f"Warning: Calculated object dimensions for {object_path} are invalid ({intended_width}x{intended_height}). Skipping paste.")
        return base_img_array

    object_img_resized = cv2.resize(object_img_raw, (intended_width, intended_height), interpolation=cv2.INTER_AREA)

    if object_img_resized.shape[2] < 4: # No alpha channel, assume fully opaque
        b,g,r = cv2.split(object_img_resized)
        mask = np.full((intended_height, intended_width), 255, dtype=np.uint8) # Full white mask
    else:
        b,g,r,a = cv2.split(object_img_resized) 
        mask = a # Alpha channel is the mask

    object_img_rgb = cv2.merge((b, g, r))

    y, x = target_coordinate
    height, width, _ = object_img_rgb.shape # Get dimensions after resize

    base_h, base_w, _ = base_img_array.shape

    # Define the region of interest (ROI) in the base image where the object will be pasted
    # These coordinates are absolute on the base_img_array
    paste_y1 = max(0, y)
    paste_x1 = max(0, x)
    
    # Calculate the end coordinates of the paste region, clamping to image boundaries
    paste_y2 = min(base_h, y + height)
    paste_x2 = min(base_w, x + width)

    # Calculate the actual dimensions of the region to paste onto
    actual_paste_h = paste_y2 - paste_y1
    actual_paste_w = paste_x2 - paste_x1

    if actual_paste_h <= 0 or actual_paste_w <= 0: # Object entirely outside or zero-sized paste area
        return base_img_array

    # Slice object, mask, and base ROI to match the actual paste area
    # This ensures that if the object goes off screen, we only use the portion that fits
    object_img_slice = object_img_rgb[0:actual_paste_h, 0:actual_paste_w]
    mask_slice = mask[0:actual_paste_h, 0:actual_paste_w]
    
    roi = base_img_array[paste_y1:paste_y2, paste_x1:paste_x2]

    # Adjust object color for blending ONLY the portion that will be pasted
    # We pass (paste_y1, paste_x1) as target_coordinate for blending, as this is the top-left of the actual paste area
    object_img_slice_blended = color_adjust_object(object_img_slice, base_img_array, (paste_y1, paste_x1), alpha)

    # Perform blending using the mask
    mask_inverse = cv2.bitwise_not(mask_slice)
    background = cv2.bitwise_and(roi, roi, mask=mask_inverse)
    foreground = cv2.bitwise_and(object_img_slice_blended, object_img_slice_blended, mask=mask_slice)
    blended_img = cv2.add(background, foreground)

    base_img_array[paste_y1:paste_y2, paste_x1:paste_x2] = blended_img
    return base_img_array

# --- Main API Functions for Flask Integration ---

def apply_contour_manipulation(original_img_array, num_of_changes=1):
    """
    Applies contour-based color change or expansion to the image.
    Returns the modified image array and a list of bounding boxes for differences.
    """
    img_modified = original_img_array.copy()

    # Resize image for consistency, assuming the game's working resolution is 640x640
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
    contours_indices_picked_for_spacing = [] # Store actual contour objects for spacing checks

    for i in range(num_of_changes):
        contour_chosen = find_suitable_contours(good_contours, contours_indices_picked_for_spacing)
        if contour_chosen is None:
            break # No suitable contour found after attempts

        # Add to picked list to avoid future overlap
        contours_indices_picked_for_spacing.append(contour_chosen)

        # Get bounding box of the chosen contour *before* modification
        # This will be the difference area.
        differences_coords.append(get_contour_bounding_box(contour_chosen))

        transform_type = rd.choice(list(DICT_TRANSFORMS.keys()))
        
        if transform_type == 1: # Change color
            print(f"Applying color change to contour {i+1}...")
            img_modified = change_color(img_modified, contour_chosen)
        elif transform_type == 2: # Expand contour
            expansion_factor = rd.uniform(1.4, 1.5)
            print(f"Applying expansion (factor {expansion_factor}) to contour {i+1}...")
            img_modified = expand_contour(img_modified, contour_chosen, expansion_factor)
    
    return img_modified, differences_coords

def apply_object_addition(original_img_array, num_objects=1, alpha=0.5, intended_width=30):
    """
    Adds external objects to the image.
    Returns the modified image array and a list of bounding boxes for added objects.
    """
    img_modified = original_img_array.copy()

    # Ensure base image is resized to a standard size (e.g., 640x640) for consistency
    img_modified = cv2.resize(img_modified, (640, 640))

    items_list = [f for f in os.listdir(OBJECTS_DIR) if f.endswith(('.png', '.PNG'))]
    if not items_list:
        print(f"Error: No PNG objects found in '{OBJECTS_DIR}'. Please ensure the 'objects' folder exists and contains PNGs.")
        return original_img_array, []

    # Limit the number of objects to add for game playability
    num_objects_to_add = min(num_objects, len(items_list), 3) # Cap at 3 or less

    selected_files = rd.sample(items_list, num_objects_to_add)
    
    added_object_differences = []
    picked_coordinates_for_spacing = [] # Store [y, x] for spacing checks

    for file_name in selected_files:
        object_path = os.path.join(OBJECTS_DIR, file_name)

        # Get smart coordinate ensuring spacing
        smart_coordinate_yx = coordinate_to_add_local_tracking(object_path, img_modified, picked_coordinates_for_spacing)
        picked_coordinates_for_spacing.append(smart_coordinate_yx) # Add to list for next check

        # Store the object's dimensions *after* it's resized by paste_object
        # We need to load it here to get its dimensions after the resize that paste_object does
        obj_img_raw = cv2.imread(object_path, cv2.IMREAD_UNCHANGED)
        if obj_img_raw is None:
            print(f"Skipping {file_name}: Failed to read object image.")
            continue
        
        h_orig, w_orig, _ = obj_img_raw.shape
        if w_orig == 0: continue # Avoid division by zero for aspect ratio calc
        
        # Calculate the size the object will be *after* resizing
        h_resized = int(h_orig * intended_width / w_orig)
        w_resized = intended_width
        
        # Apply paste operation
        # This function modifies img_modified in place
        img_modified = paste_object(img_modified, object_path, smart_coordinate_yx, alpha, intended_width)
        
        # Calculate the bounding box based on the paste location and *resized* dimensions
        x1 = smart_coordinate_yx[1]
        y1 = smart_coordinate_yx[0]
        x2 = x1 + w_resized
        y2 = y1 + h_resized

        added_object_differences.append([x1, y1, x2, y2])
        print(f"Added object '{file_name}' at: {[x1, y1, x2, y2]}")

    return img_modified, added_object_differences