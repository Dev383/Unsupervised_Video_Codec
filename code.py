import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# Parameters
video_path = "/content/drive/MyDrive/video_compression/VideoCompressionDataset/AlitaBattleAngel.mkv"  # Replace with your video path
block_size = (16, 16)
num_clusters = 15
seconds_to_process = 1
frame_rate = 24  # Change according to your video's frame rate

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Cannot open video")

# Define the helper functions (image_to_blocks, calculate_average_color, etc.)

# Helper functions
def image_to_blocks(img, block_size):
    img_height, img_width, _ = img.shape
    blocks = []
    for y in range(0, img_height, block_size[1]):
        for x in range(0, img_width, block_size[0]):
            block = img[y:y+block_size[1], x:x+block_size[0]]
            blocks.append(block)
    return blocks

def calculate_average_color(blocks):
    average_colors = []
    for block in blocks:
        average_color = np.mean(block, axis=(0, 1))
        average_colors.append(average_color)
    return np.array(average_colors)

def reconstruct_image_from_clusters(cluster_assignments, cluster_centers, block_size, frame_shape):
    reconstructed_img = np.zeros(frame_shape, dtype=np.uint8)
    block_index = 0
    for y in range(0, frame_shape[0], block_size[1]):
        for x in range(0, frame_shape[1], block_size[0]):
            if y + block_size[1] <= frame_shape[0] and x + block_size[0] <= frame_shape[1]:
                reconstructed_img[y:y + block_size[1], x:x + block_size[0]] = cluster_centers[cluster_assignments[block_index]]
            block_index += 1
    return reconstructed_img

def get_motion_vector(block, ref_frame, block_pos_x, block_pos_y, search_window):
    block_height, block_width, _ = block.shape
    ref_height, ref_width, _ = ref_frame.shape
    min_sad = float('inf')
    motion_vector = (0, 0)
    for y in range(max(0, block_pos_y - search_window), min(block_pos_y + search_window, ref_height - block_height)):
        for x in range(max(0, block_pos_x - search_window), min(block_pos_x + search_window, ref_width - block_width)):
            candidate_block = ref_frame[y:y + block_height, x:x + block_width]
            sad = np.sum(np.abs(block.astype(int) - candidate_block.astype(int)))
            if sad < min_sad:
                min_sad = sad
                motion_vector = (x - block_pos_x, y - block_pos_y)
    return motion_vector

def encode_video_data(cluster_assignments, motion_vectors, block_size, frame_shape):
    motion_model = np.array(motion_vectors).flatten().tolist()
    return {
        'clusters': cluster_assignments.tolist(),
        'motion_model': motion_model,
        'block_size': block_size,
        'frame_shape': frame_shape
    }

def calculate_motion_vectors(first_frame, next_frame, block_size):
    blocks_first = image_to_blocks(first_frame, block_size)
    motion_vectors = []
    for idx, block in enumerate(blocks_first):
        block_y = (idx // (first_frame.shape[1] // block_size[0])) * block_size[1]
        block_x = (idx % (first_frame.shape[1] // block_size[0])) * block_size[0]
        mv = get_motion_vector(block, next_frame, block_x, block_y, 5)
        motion_vectors.append(mv)
    return motion_vectors
def apply_motion_vectors_to_frame(frame, motion_vectors, block_size):
    # Create a new frame which is a copy of the original
    new_frame = np.zeros_like(frame)
    num_blocks_y, num_blocks_x = frame.shape[0] // block_size[1], frame.shape[1] // block_size[0]

    for idx, motion_vector in enumerate(motion_vectors):
        block_y = (idx // num_blocks_x) * block_size[1]
        block_x = (idx % num_blocks_x) * block_size[0]
        new_block_y, new_block_x = block_y + motion_vector[1], block_x + motion_vector[0]

        # Check if the new block position is within the frame boundaries
        if 0 <= new_block_x < frame.shape[1] - block_size[0] and 0 <= new_block_y < frame.shape[0] - block_size[1]:
            new_frame[new_block_y:new_block_y + block_size[1], new_block_x:new_block_x + block_size[0]] = frame[block_y:block_y + block_size[1], block_x:block_x + block_size[0]]

    return new_frame

def calculate_residual_frame(actual_frame, predicted_frame):
    """Calculate the residual (difference) frame."""
    residual = cv2.subtract(actual_frame, predicted_frame)
    return residual

def apply_residual_frame(predicted_frame, residual):
    """Apply the residual frame to the predicted frame."""
    reconstructed_frame = cv2.add(predicted_frame, residual)
    return reconstructed_frame

# Read and process the first frame
ret, first_frame = cap.read()
if not ret:
    raise IOError("Cannot read the first frame")

blocks_first = image_to_blocks(first_frame, block_size)
average_colors_first = calculate_average_color(blocks_first)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(average_colors_first)
cluster_assignments_first = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# Initialize storage for original and decompressed frames
original_frames = [first_frame]
compressed_data = []
count = 0

# Process video for the first ten seconds
for _ in range(seconds_to_process * frame_rate - 1):
    ret, next_frame = cap.read()
    if not ret:
        break

    count+=1
    print(f"Processing frame {count}")
    original_frames.append(next_frame)
    motion_vectors = calculate_motion_vectors(first_frame, next_frame, block_size)
    encoded_data = encode_video_data(cluster_assignments_first, motion_vectors, block_size, first_frame.shape)
    compressed_data.append(encoded_data)

    first_frame = next_frame


# Compression loop with progress updates
total_frames = len(original_frames) - 1  # Total number of frames to process
compressed_data = []

for i in range(1, len(original_frames)):
    current_frame = original_frames[i - 1]
    next_frame = original_frames[i]

    # Calculate motion vectors
    motion_vectors = calculate_motion_vectors(current_frame, next_frame, block_size)

    # Apply motion vectors to get the predicted frame
    predicted_frame = apply_motion_vectors_to_frame(current_frame, motion_vectors, block_size)

    # Calculate the residual frame
    residual_frame = calculate_residual_frame(next_frame, predicted_frame)

    # Encode and store motion vectors and residual frame
    encoded_data = encode_video_data(cluster_assignments_first, motion_vectors, block_size, current_frame.shape)
    encoded_data['residual'] = residual_frame.tolist()  # Store residual frame
    compressed_data.append(encoded_data)

    # Print progress
    progress_percentage = (i / total_frames) * 100
    print(f"Compressing frame {i}/{total_frames} ({progress_percentage:.2f}%)")
# Decompression loop with residuals
decompressed_frames = []
reference_frame = original_frames[0]

for encoded_data in compressed_data:
    motion_vectors = np.array(encoded_data['motion_model']).reshape(-1, 2)

    # Ensure that residual data is converted back to the appropriate format
    residual_frame = np.array(encoded_data['residual'], dtype=np.uint8)

    # Apply motion vectors and residual frame
    predicted_frame = apply_motion_vectors_to_frame(reference_frame, motion_vectors, block_size)
    reconstructed_frame = apply_residual_frame(predicted_frame, residual_frame)

    decompressed_frames.append(reconstructed_frame)
    reference_frame = reconstructed_frame
import cv2

output_video_path = 'reconstructed_video.avi'
codec = cv2.VideoWriter_fourcc(*'XVID')
output_frame_rate = frame_rate

# Check the number of frames and their size
print(f"Total frames to write: {len(decompressed_frames)}")
if decompressed_frames:
    print(f"Frame size: {decompressed_frames[0].shape}")

output_size = (decompressed_frames[0].shape[1], decompressed_frames[0].shape[0])

# Create a VideoWriter object
out = cv2.VideoWriter(output_video_path, codec, output_frame_rate, output_size)

# Write each frame to the video file
for idx, frame in enumerate(decompressed_frames):
    print(f"Writing frame {idx + 1}/{len(decompressed_frames)}")
    out.write(frame)

# Release the VideoWriter
out.release()

print(f"Video reconstructed and saved to {output_video_path}")