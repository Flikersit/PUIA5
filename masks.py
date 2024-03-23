import json
import numpy as np
from PIL import Image
from pathlib import Path
import skimage
import os
import shutil
from skimage.io import imsave
import imageio

def create_mask(image_info, annotations, output_folder):
    # Create an empty mask as a numpy array
    mask_np = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)

    # Counter for the object number
    object_number = 1

    for ann in annotations:
        if ann['image_id'] == image_info['id']:
            # Extract segmentation polygon
            for seg in ann['segmentation']:
                if ann["category_id"] == 5:
                    # Convert polygons to a binary mask and add it to the main mask
                    rr, cc = skimage.draw.polygon(seg[1::2], seg[0::2], mask_np.shape)
                    mask_np[rr, cc] = ann["category_id"]
                    object_number += 1 #We are assigning each object a unique integer value (labeled mask)
    
    name, extension = os.path.splitext(image_info['file_name'])  # Разделяем имя файла и его расширение
    # Save the numpy array as a JPEG using imsave function
    print(name)
    result = name.split("/")
    print(result)
    if len(result) == 2:
        res = result[1]
    else:
        res = result[0]
    mask_path = os.path.join(output_folder, res + "_mask.png")  # Добавляем "_mask" к имени файла
    print(mask_path)
    max_value = 10-1
    normalized_mask = (mask_np / max_value) * 255
    integer_mask = normalized_mask.astype(np.uint8)
    imsave(mask_path, integer_mask)
    imageio.imwrite(mask_path, integer_mask, format='PNG')




def main():
    directory_path = Path(r'C:\Users\User\Desktop\pilsen_pigs_2023_cvat_backup\workspase')
    count = 1
    for project_dir in directory_path.iterdir():
        if project_dir.is_dir():
            for item in project_dir.glob('instances_default.json'):
                mask_output_folder = Path(rf'C:\Users\User\Desktop\pilsen_pigs_2023_cvat_backup\masks\{count}')
                print(item.name)
                with item.open() as f:
                    data = json.load(f)
                    images = data['images']
                    annotations = data['annotations']
                    count += 1
                for img in images:
                    # Create the masks
                    create_mask(img, annotations, mask_output_folder)

main()
