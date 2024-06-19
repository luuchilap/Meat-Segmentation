import os
import cv2
import numpy as np

input_dir = './tmp/masks'
output_dir = './tmp/labels'

for j in os.listdir(input_dir):
    image_path = os.path.join(input_dir, j)
    # Load the mask
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("title", mask)
    # cv2.waitKey()
    # break
    H, W = mask.shape
    unique_classes = np.unique(mask)
    polygons_dict = {}

    for cls in unique_classes:
        if cls == 0:  # Skip background class
            continue
        # print(cls)
        # Create a binary mask for the current class
        class_mask = (mask == cls).astype(np.uint8) * 255
        _, binary_mask = cv2.threshold(class_mask, 1, 255, cv2.THRESH_BINARY)
        
        # Find contours for the current class
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert the contours to polygons
        polygons = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                polygon = []
                for point in cnt:
                    x, y = point[0]
                    polygon.append(x / W)
                    polygon.append(y / H)
                # print(polygon)
                polygons.append(polygon)
                
        
        # Store polygons with the class label
        polygons_dict[cls] = polygons

    # Save the polygons to a file
    output_path = os.path.join(output_dir, '{}.txt'.format(j[:-4]))
    with open(output_path, 'w') as f:
        for cls, polygons in polygons_dict.items():
            for polygon in polygons:
                for p_, p in enumerate(polygon):
                    # print(f"p_, p: {p_}, {p}")
                    
                    if p_ == 0:
                        if (cls == 80): cls = 0
                        if (cls == 97): cls = 1
                        f.write('{} {} '.format(cls, p))
                        # print('{}'.format(cls))
                        # break
                    elif p_ == len(polygon) - 1:
                        f.write('{}\n'.format(p))
                    else:
                        # print('{} '.format(p))
                        f.write('{} '.format(p))

        f.close()
