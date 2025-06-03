# Skypulse Interview
This repository contains a coding exercise intended to assess the problem-solving and coding skills of potential team members.

## The Task:
Given a shakey video, your task is to add a feature to stabilize the video, as every two consecutive frames have been taken from slightly different camera positions.

### Specifications: 
1) Real-time Processing: Your code should run online, analyzing each frame and displaying it rather than processing the entire video at once.
2) Alignment: Align all frames relative to the first frame to correct the shakiness.

### Evaluation Criteria:

Your work will be evaluated based on the following:
1) Quality of your solution.
2) Ability to improve algorithm weakness
3) Code quality - How clean, well-structured, and readable is the code.


Please create your own branch to implement your solution and submit it via pull request

Good luck!




# How To run:

    poetry install  

    poetry run python main.py

    This will run the stabilizer on your configured input video and save the stabilized output alongside the original for comparison

# General thougths and approach: 
## Why I Went with This Method
I decided to go with a classic feature-tracking method (using good features to track + optical flow + affine transformation) because it’s fast and works in real-time on low-power hardware like the Raspberry Pi. I also liked that it’s easy to understand and debug — I can see exactly what’s happening and adjust each part.
The goal was to keep the visual flow steady throughout the video, starting from the original frame. By making small adjustments from one frame to the next, the motion stays smooth and consistent over time, while keeping things efficient enough for online use.

### Other algorithms:

Full trajectory smoothing (e.g., with Kalman filters or L1 optimization): These can give more stable results but require buffering a large number of frames — which could slow down the real-time requirement.

Deep learning-based stabilizers: Great for complex scenes or moving objects, but too heavy for Raspberry Pi and harder to interpret/debug in current production time.

3D or mesh-based methods: Can handle parallax but are much more complex, computationally expensive, and overkill for many use cases.


## Making the Code Modular
One of the things I paid attention to was writing the code in a way that makes it easy to tweak. For example, it’s pretty straightforward to swap the feature detector (e.g., ORB instead of Shi-Tomasi), try a different transform model (like homography or even full trajectory smoothing), or change how the motion is estimated. This helps adapt the algorithm to different inputs — especially if the camera or scene changes.

## Limitations
This method has some downsides:

-Some drift over time – Since the algorithm keeps track of motion from frame to frame, little tracking mistakes can build up over time, especially in longer videos.

-If there’s a lot of movement in the scene , the tracking can get confused and the stabilization might struggle.

- There is some smoothing — a simple moving average over recent frames — which helps reduce jitter, though it’s not perfect for really shaky footage

- If there’s a lot of depth variation (parallax), it can warp parts of the image.

- Some deep learning methods could potentially do better in very tricky videos, but I went with this approach for speed, clarity, and platform compatibility.

## How I measured Success:
To see how well it worked, I:

- Compared my stabilized video side-by-side with the “ground truth” (intended output) video

- compared transformation graphs (dx, dy, angle, scale) between the original, stabilized, and intended videos to measure the effectiveness of motion correction.

- Plotted and saved the "good features" (cv2.goodFeaturesToTrack) used for alignment to make sure they were stable.

- Looked at whether objects stayed centered and whether the camera jitter was reduced.

*In a larger version of this project, I would use example benchmark videos and basic metrics like frame difference or motion smoothness to measure how well the stabilization works





