import cv2
import os

def crop_images(paths):
  for path in paths:
    result_folder = f'{path}/../{path.split("/")[-1]}_cropped'
    if not os.path.exists(result_folder):
      os.makedirs(result_folder)
    image_paths = list(filter(lambda x: x.endswith('.png'), os.listdir(path)))

    for img_path in image_paths:
      print(f'Processing image "{img_path}"...')
      img = cv2.imread(f'{path}/{img_path}')
      cropped_img = img[100:984, 940:1824] # Should be square
      cv2.imwrite(f'{result_folder}/{img_path}', cropped_img)
      print(f'Image stored at {result_folder}/{img_path}"\n')

def crop_single_image(img_path):
  print(f'Processing image "{img_path}"...')
  img = cv2.imread(img_path)
  cropped_img = img[100:984, 940:1824] # Should be square
  cv2.imwrite(f'{img_path.split(".")[0]}_cropped.png', cropped_img)
  print(f'Image stored at {img_path.split(".")[0]}_cropped.png')
