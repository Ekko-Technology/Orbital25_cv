import cv2
import numpy as np
import random as rd
import datetime
import os
import sys
# import add_objects


# in built with setMOuseCallback from opencv detecting mouse events, where the 5 arguments are to be given as below
def click_events(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Left click at ({x}, {y})")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"({x}, {y})", (x+12, y+2), font, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
        # Draw a circle at the clicked position
        cv2.circle(img, (x, y), 10, (0, 255, 0), 3)  
        cv2.imshow('image', img)
    elif event == cv2.EVENT_RBUTTONDOWN:
        print(f"Right click at ({x}, {y})")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"({x}, {y})", (x+10, y+2), font, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
        # Draw a red cross on the position
        cross_size = 8
        cv2.line(img, (x-cross_size, y-cross_size), (x+cross_size, y+cross_size), (0, 0, 255), 2)
        cv2.line(img, (x+cross_size, y-cross_size), (x-cross_size, y+cross_size), (0, 0, 255), 2)
        cv2.imshow('image', img)


def process_image():
    # set read image as a global variable for compatibility with other functions
    global img

    valid_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    jpg_files = [files for files in os.listdir("sample_imgs") if files.endswith(valid_formats)]
    print("files loaded:", jpg_files[:5], "...")

    if jpg_files:
        # randomly select a filename from the list of files
        selected_file = rd.choice(jpg_files)
        # use opencv to return the matrix of the image
        img = cv2.imread(os.path.join('sample_imgs', selected_file))

        h, w, c = img.shape 

        # set height and width of image if approriate
        if w > 960 or h > 720 or w < 600 or h < 400:
            img = cv2.resize(img, (640, 480)) # resize image if need be

        print("height:", h)
        print("width:", w)
        print("channels:", c)
        # print(cv2.split(img)) # split the image into its channels

        # put text on image
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_1 = f"Width: {str(w)}"
        text_2 = f"Height: {str(h)}"
        dateTime = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        cv2.putText(img, text_1, (10, 20), font, 0.5, (255, 100, 100), 1, cv2.LINE_AA) # image, text, position, font, font_scale, color, thickness
        cv2.putText(img, text_2, (10, 40), font, 0.5, (255, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(img, dateTime, (10, 60), font, 0.5, (255, 100, 100), 1, cv2.LINE_AA)

        print("Press 'esc' to exit and 's' to save the image")
        cv2.imshow('image', img)

        # call the function click_events when the mouse is clicked
        cv2.setMouseCallback('image', click_events)

        # wait for unlimited seconds
        k = cv2.waitKey(0)
        # if 'esc' key is pressed, exit the program. if 's' key is pressed, save the image
        if k == 27:
            cv2.destroyAllWindows()
            print("Exiting without saving.")
        elif k == ord('s'):
            os.makedirs("saved_img", exist_ok=True)  # create directory if it doesn't exist
            cv2.imwrite(f"saved_img/{selected_file}_copy.png", img)
            print("Image saved under saved_img directory")



def test_events_on_blackboard():
    global img
    for event in dir(cv2):
        if event.startswith('EVENT'):
            print(event)
    img = np.zeros((480, 640, 3), np.uint8)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_events)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    
    process_image()
    # test_events_on_blackboard()
    # process_video()
    
    











def check_camera_avail():
    print("Checking available camera indices...")
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Camera found at index {i}")
            cap.release()
        else:
            print(f"❌ No camera at index {i}")

def process_video():
    cap = cv2.VideoCapture(0)
    # comment out to save recorded video in mp4
    # output = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640, 480)) # takes in video file name to save, codec to compress video to correct format, frames per second, and dimensions of the video
    print("Press esc to exit the video stream")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video")
            break

        # comment out to save recorded video
        # output.write(frame)

        # image filtering
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display video frame
        cv2.imshow('Video', gray)

        # waits for an ascii code keyboard key to be pressed
        key = cv2.waitKey(1)
        # Wait till 'esc' key is pressed to exit
        if key == 27:  # ESC key
            break
    
    cap.release()
    cv2.destroyAllWindows()