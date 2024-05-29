import os
from PIL import Image

# Function to resize images
def resize_image(image_path, output_path, size=(64, 64)):
    with Image.open(image_path) as img:
        img = img.resize(size, Image.LANCZOS)
        img.save(output_path)

# Directory containing the original images
input_dir = 'circle_images'  # Update this path with your input image directory
output_dir = 'circle_images_64'  # Update this path with your output image directory

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through all images in the directory
for image_name in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_name)
    if image_path.endswith('.png') or image_path.endswith('.jpg') or image_path.endswith('.jpeg'):
        output_path = os.path.join(output_dir, image_name)
        resize_image(image_path, output_path)

print("Image resizing complete. Resized images saved in:", output_dir)
