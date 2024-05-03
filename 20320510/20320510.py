import cv2
import numpy as np
import os


# Extract keyframes dynamically based on the size of the video file
def extract_key_frames(video_path):
    # Get the size of the input video in MB
    file_size = os.path.getsize(video_path) / (1024 * 1024)
    # Initialise the VideoCapture object
    video_capture = cv2.VideoCapture(video_path)
    # Calculate the total number of frames in the video
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use different keyframe extraction strategies based on the video size
    # 1.For videos under 10MB, extract 20% of the total number of frames as keyframes
    if file_size <= 10:
        key_frames_count = int(total_frames * 0.20)
    # 2.For videos between 10MB and 20MB, extract 10% of the total number of frames as keyframes
    elif file_size <= 20:
        key_frames_count = int(total_frames * 0.10)
    # 3.For videos over 20MB, extract a fixed number of 40 frames
    else:
        key_frames_count = 40

    # Calculate the frame interval
    frame_interval = total_frames // key_frames_count
    key_frames = []

    #  Extract keyframes by the frame index
    for i in range(key_frames_count + 1):  # Add the last frame
        frame_id = i * frame_interval if i < key_frames_count else total_frames - 1
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = video_capture.read()
        if ret:
            key_frames.append(frame)
    # Release the VideoCapture object
    video_capture.release()
    return key_frames


# Filter the frames based on the number of feature points per frame and
# apply Gaussian blur to the selected frames.
def filter_frames(frames):
    # Create SIFT feature detector
    sift = cv2.SIFT_create()
    filtered_frames = []
    min_threshold = 200
    max_threshold = 17000
    for frame in frames:
        # Compute SIFT feature points on each frame
        keypoints, descriptors = sift.detectAndCompute(frame, None)
        num_keypoints = len(keypoints)
        # Filter out keyframes based on number of feature points
        if min_threshold <= num_keypoints <= max_threshold:
            # Adjust the kernel size of the Gaussian blur dynamically
            if num_keypoints < 2000:
                ksize = (3, 3)  # Fewer feature points, use smaller kernel
            else:
                ksize = (5, 5)  # More feature points, use larger kernel
            blurred_frame = cv2.GaussianBlur(frame, ksize, 0)
            filtered_frames.append(blurred_frame)
    return filtered_frames


# Stitch keyframes into a panorama by using the OpenCV Stitcher class
def stitch_frames(key_frames):
    # Create a Stitcher instance
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    # Perform stitching operation
    status, pano = stitcher.stitch(key_frames)
    # Check the stitching status
    if status == cv2.Stitcher_OK:
        return pano
    else:
        print("Stitching failed: ", status)
        return None


# Resize the panorama
def resize_panorama(pano, height=1500):
    # Calculate the new size ratio
    height_ratio = height / pano.shape[0]
    new_width = int(pano.shape[1] * height_ratio)
    # Resize the panorama
    pano_resized = cv2.resize(pano, (new_width, height))
    return pano_resized


# Perform a series of image enhancement processes to cope with
# the generation of panoramas under conditions of blurred video
def enhance_image(image):
    # Apply Gaussian blur to suppress noise
    def suppress_noise(img):
        return cv2.GaussianBlur(img, (3, 3), 0)

    # Apply dynamic range adjustment to improve the brightness
    # and contrast of panorama
    def adjust_dynamic_range(img, gamma=1.1):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)

    # Sharpen the image
    def sharpen_image(img):
        blurred = cv2.GaussianBlur(img, (0, 0), 3)
        return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

    # Apply the above enhancements
    noise_suppressed = suppress_noise(image)
    range_adjusted = adjust_dynamic_range(noise_suppressed)
    sharpened = sharpen_image(range_adjusted)

    return sharpened

# 主程序
if __name__ == "__main__":
    # Disable OpenCL acceleration to ensure compatibility with OpenCV features
    cv2.ocl.setUseOpenCL(False)
    videos = ["video_1.mp4", "video_2.mp4", "video_3.mp4"]

    for video in videos:
        print(f"Processing {video}")
        # Extract keyframes
        key_frames = extract_key_frames(video)
        # Filter keyframes
        filtered_frames = filter_frames(key_frames)
        # Stitch the filtered frames into a panorama
        panorama = stitch_frames(filtered_frames)
        if panorama is not None:
            # Enhance the stitched panorama
            enhanced_panorama = enhance_image(panorama)
            # Resize the enhanced panorama
            panorama_resized = resize_panorama(enhanced_panorama)
            # Define the output file path
            output_video = os.path.splitext(video)[0] + '_panorama.jpg'
            # Save the final panorama image
            cv2.imwrite(output_video, panorama_resized)
            print(f"Panorama saved successfully as {output_video}.")
        else:
            print("Error during the panorama creation process.")
