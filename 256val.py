import cv2
import os
import shutil

input_dir = '/Users/johnfernandez/Documents/Spring 23/CS 4701/Guess-a-Sketch/FinalVal'

with open('folders.txt', 'r') as f:
    output_dirs = f.read().splitlines()

batch_size = 8
dir_counter = 0
file_counter = 0

for root, dirs, files in os.walk(input_dir):
    for filename in files:
        if filename.endswith('.jpg') or filename.endswith('.png'):

            img_path = os.path.join(root, filename)
            img = cv2.imread(img_path)

            # blur1 = cv2.GaussianBlur(img, (5, 5), 0)

            scaled1 = cv2.pyrDown(img)

            resized1 = cv2.resize(scaled1, (512, 512))

            # blur2 = cv2.GaussianBlur(resized1, (5, 5), 0)

            scaled2 = cv2.pyrDown(resized1)

            # Constructing output path
            output_dir = os.path.join(os.path.dirname(
                root), '256Val', os.path.basename(root))

            # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Calculate the output file path
            output_path = os.path.join(
                output_dir, f'{filename[:-4]}.png')

            # Save the processed image to the output file
            cv2.imwrite(output_path, scaled2)

            file_counter += 1

            if file_counter >= len(output_dirs) * batch_size:
                file_counter = 0
                dir_counter += 1

                if dir_counter >= len(output_dirs):
                    break

    if dir_counter >= len(output_dirs):
        break
