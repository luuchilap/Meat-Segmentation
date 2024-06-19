from ultralytics import YOLO
import cv2
import os
from matplotlib import pyplot as plt

# Load the trained model
model = YOLO("C:\\Users\\ADMIN\\Segmentation1\\best.pt")

# Path to the validation images
val_images_path = "C:\\Users\\ADMIN\\Segmentation1\\split\\val\\images"

# Get the list of validation images
val_images = [os.path.join(val_images_path, img) for img in os.listdir(val_images_path) if img.endswith(('.png', '.JPG', '.jpeg'))]

# Function to plot images
def plot_image_with_prediction(img_path, result):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title('Original Image')

    # Plot the prediction
    plt.figure(figsize=(10, 10))
    plt.imshow(result.masks.masked_img(img_rgb))
    plt.axis('off')
    plt.title('Segmented Image')

# Run inference on each validation image and display results
# for img_path in val_images:
#     result = model(img_path)
#     plot_image_with_prediction(img_path, result[0])
# result = model("C:\\Users\\ADMIN\\Segmentation1\\split\\val\\images\\13.JPG")

#Show image
import cv2

# # Load the image
# image_path = 'C:\\Users\\ADMIN\\Segmentation1\\split\\val\\images\\13.JPG'
# image = cv2.imread(image_path)

# # Display the image
# cv2.imshow('Image', image)

# # Wait for a key press and close the image window
# cv2.waitKey(0)
# cv2.destroyAllWindows()
def show_image(image_path):
    image = cv2.imread(image_path)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# image_path = os.path.join(val_images_path, "13.JPG")
# # print(os.path.join(val_images_path, "13.JPG"))
# model.predict(image_path, save=True, imgsz=320, conf=0.5)
for image in os.listdir(val_images_path):
    model.predict(os.path.join(val_images_path, image), save=True)
