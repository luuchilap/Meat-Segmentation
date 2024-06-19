import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Define paths
valid_images_path = "C:\\Users\\ADMIN\\Segmentation1\\split\\val\\images"
output_dir = "C:\\Users\\ADMIN\\Segmentation1\\masking"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Initialize YOLOv8 model
best_model = YOLO("C:\\Users\\ADMIN\\Segmentation1\\best.pt")

# List all images in the validation folder
image_files = [file for file in os.listdir(valid_images_path) if file.endswith('.JPG')]

# Select 9 images at equal intervals
num_images = len(image_files)
selected_images = [image_files[i] for i in range(0, num_images, max(num_images // 9, 1))]

# Initialize the subplot for combined image and masks
fig, axes = plt.subplots(3, 3, figsize=(10, 7))
fig.suptitle('Validation Set Inferences with Segmentation Masks on Black Background', fontsize=24)

# Define colors for each label (meat and meatless)
color_map = {
    0: (255, 0, 0),
    1: (0, 255, 0)
}

# Perform inference on each selected image and display it
for i, ax in enumerate(axes.flatten()):
    if i < len(selected_images):
        image_path = os.path.join(valid_images_path, selected_images[i])
        results = best_model.predict(source=image_path, imgsz=640)

        # Load the original image
        original_image = cv2.imread(image_path)
        height, width, _ = original_image.shape

        # Create a blank (black) image
        black_background = np.zeros((height, width, 3), dtype=np.uint8)

        # Combine all masks into one image
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes  # Extract boxes containing class labels
            classes = boxes.cls.cpu().numpy().astype(int)
            combined_mask = np.zeros_like(black_background)

            for j, mask in enumerate(masks):
                binary_mask = (mask > 0).astype(np.uint8) * 255
                class_id = classes[j]
                color = color_map.get(class_id, (255, 255, 255))  # Default white for unknown classes
                colored_mask = np.zeros_like(black_background)
                for k in range(3):  # Apply the color to the mask
                    colored_mask[:, :, k] = binary_mask / 255 * color[k]
                combined_mask = cv2.addWeighted(combined_mask, 1, colored_mask, 1, 0)

            # Overlay the combined mask on the black background
            combined_image = cv2.addWeighted(black_background, 1, combined_mask, 1, 0)

            # Save the combined image with masks
            output_image_path = os.path.join(output_dir, f'combined_masked_{os.path.basename(image_path)}')
            cv2.imwrite(output_image_path, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

            # Display the combined image with masks
            ax.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
        else:
            # If no masks, just display the black background
            ax.imshow(black_background)

        ax.axis('off')
        ax.set_title(f'Image {i+1}')

    else:
        # If fewer than 9 images are found
        ax.axis('off')

plt.tight_layout()
plt.show()
