
Edge detection finds the main lines and shapes in an image. 
It works by finding places where the brightness of the image changes quickly. 
In this assignment, each image is first changed to grayscale and blurred to reduce noise. 
Then Canny edge detection is used to show only the most important edges.

Image blending combines the edge image with the original color image. 
OpenCV’s `addWeighted()` function mixes the two images together. 
This makes the edges easy to see while still keeping the original colors of the image.




