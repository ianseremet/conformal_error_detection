from pillow_heif import open_heif
from PIL import Image
import cv2
import numpy as np
import os

#declaration of directories 
input_dir = "C:\\Users\\ianse\\Projects\\sample_image_input"
output_dir_png = "C:\\Users\\ianse\\Projects\\sample_image_output"
output_dir_crop = "C:\\Users\\ianse\\Projects\\cropped_image_output"

#image conversion to png (accepted type by CVAT)
def convert_heic_to_png(heic_folder, png_output_folder):
    if not os.path.exists(png_output_folder):
        os.makedirs(png_output_folder)

    for filename in os.listdir(heic_folder):
        if filename.lower().endswith('.heic') or filename.lower().endswith('.jpg'):
            heic_file_path = os.path.join(heic_folder, filename)
            output_file_path = os.path.join(png_output_folder, f"{os.path.splitext(filename)[0]}.png")

            try:
                heif_file = open_heif(heic_file_path)
                image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data)
                image.save(output_file_path, "PNG")
                print(f"Converted {filename} to {output_file_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

#png cropping based on red channel 
def crop_png_images(png_folder, cropped_output_folder):
    r_thresh = 120 #edit this value to 
    window_width = 800
    window_height = 800
    push_y = 800
    push_x = 800

    def get_average_red_value(window):
        return np.mean(window[:, :, 2])

    if not os.path.exists(cropped_output_folder):
        os.makedirs(cropped_output_folder)

    for filename in os.listdir(png_folder):
        if filename.lower().endswith('.png') or filename.lower().endswith('.jpg'):
            img_path = os.path.join(png_folder, filename)
            filename_rev = os.path.splitext(filename)[0]

            img = cv2.imread(img_path)
            if img is not None:
                print(f"Loaded image: {filename} with dimensions: {img.shape}")
                height, width, _ = img.shape

                for y in range(0, height - window_height + 1, push_y):
                    for x in range(0, width - window_width + 1, push_x):
                        window = img[y:y+window_height, x:x+window_width]
                        avg_r = get_average_red_value(window)
                        print(f"Window at ({x}, {y}) has average red value: {avg_r}")

                        if avg_r > r_thresh:
                            cropped_img_name = f"{filename_rev}_crop_{x}_{y}_r{int(avg_r)}.png"
                            output_path = os.path.join(cropped_output_folder, cropped_img_name)
                            success = cv2.imwrite(output_path, window)
                            if success:
                                print(f"Saved cropped image: {cropped_img_name}")
                            else:
                                print(f"Failed to save cropped image: {cropped_img_name}")
            else:
                print(f"Failed to load image: {filename}")

#call functions
convert_heic_to_png(input_dir, output_dir_png)
crop_png_images(output_dir_png, output_dir_crop)
