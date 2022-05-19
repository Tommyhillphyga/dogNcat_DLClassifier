
import torch
import cv2 
import numpy as np
import os
import glob
from PIL import Image

#load files from the folder

path = 'petImages'
main_folder = os.listdir(path)
cats = []
dogs= []
for i in range(len(main_folder)):
    sub_folder = os.listdir(os.path.join(path, main_folder[i]))
    #print(sub_folder)
    for file in sub_folder:
        full_path = os.path.join(path, main_folder[i], file)
        if i == 0:
            cats.append(full_path)
        elif i == 1:
            dogs.append(full_path)
print(len(dogs))
# for file in dogs:
#     name = file.split('\\')[-1]
#     print
#     try:
#         from PIL import ImageFile
#         ImageFile.LOAD_TRUNCATED_IMAGES=True
#         Image.open(file)
#     except Exception as e:
#         print("An exception is raised:", e)
#         print(name, ': is corrupted image')



# no_of_images = len(files)

# #create a shuffling index
# shuffle = np.random.permutation(no_of_images)

# #create a validation, train and test directories
# _dir = ['valid','train']


# for dir in _dir:
#     os.mkdir(os.path.join(path, dir))
#     for folder in ['dogs','cats']:
#         sub_path = os.path.join(path, dir, folder)
#         print(sub_path)
#         os.mkdir(sub_path)
            
# except:
#     pass
#     folders = []
#     # copy small datset into the train folder
#     for i in shuffle[:2000]:
#         folder = files[i].split("\\")[-1].split(".")[0]
        
#         image = files[i].split("\\")[-1]
#         print({folder:image})
#         os.rename(files[i], os.path.join(path, 'valid', folder, image))

#     for i in shuffle[2000:]:
#         folder = files[i].split("/")[-1].split(".")[0]
#         image = files[i].split("/")[-1]
#         os.rename(files[i], os.path.join(path, 'train', folder, image))
# print(shuffle)


