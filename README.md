# Orbital25_cv
Orbital Project aiming to use computer vision to automate filters and other edits in the game of spot the difference


Backend:

OpenCV is an open sourced computer vision tool used for image processing and computer vision tasks. In this project, the tool is use extensively on making subtle image modifications for the spot the difference game. Detecting objects is done using openCV's contour detection which is done after several image pre-processing steps to reduce image noise for optimal edge/contour detections. The modified contour coordinates will then be returned to the flask backend for front end integration.


Acknowledgements
https://medium.com/swlh/contours-in-images-a58b4c12c0ff
