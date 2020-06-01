# For yolov3 annotations
import pandas as pd
import os
import numpy
import glob
df = pd.read_csv("data.csv")
def convert(df):
    yolo_box = []
    for i in range(0, len(df)):
        dw = 1. / df['width'][i]
        dh = 1. / df['height'][i]
        center_x = ((df['xmin'][i] + df['xmax'][i]) / 2.0)*dw
        center_y = ((df['ymin'][i] + df['ymax'][i]) / 2.0)*dh
        w = (df['xmax'][i] - df['xmin'][i])*dw
        h = (df['ymax'][i] - df['ymin'][i])*dh
        yolo_box.append([center_x, center_y, w, h])
    return yolo_box

df['yolo_box'] = convert(df)
#print(df.head())

unique_img_ids = df.image_id.unique()
#print(len(unique_img_ids))
if not os.path.exists("yolo_train_annotations"):
    os.makedirs("yolo_train_annotations")

folder_location = "yolo_train_annotations"
#change  unique_img_ids[:2] to unique_img_ids to iterate through all images
for img_id in unique_img_ids: # loop through all unique image ids. Remove the slice to do all images
    #print(img_id)
    filt_df = df.query("image_id == @img_id") # filter the df to a specific id
    #print(filt_df.shape[0])
    all_boxes = filt_df.yolo_box.values
    file_name = "{}/{}.txt".format(folder_location,img_id) # specify the name of the folder and get a file name

    s = "0 %s %s %s %s \n" # the first number is the identifier of the class. If you are doing multi-class, make sure to change that
    with open(file_name, 'a') as file: # append lines to file
        for i in all_boxes:
            new_line = (s % tuple(i))
            file.write(new_line)

all_imgs = glob.glob("images/*.jpg")
all_imgs = [i.split("/")[-1].replace(".jpg", "") for i in all_imgs]
print(len(unique_img_ids))
print(len(all_imgs))
positive_imgs = df.image_id.unique().astype(str)
print(len(positive_imgs))
if len(positive_imgs) != len(all_imgs):
	negative_images = set(all_imgs) - set(positive_imgs)
	print("All images:, positive images:, Negative images:",len(all_imgs), len(positive_imgs), len(negative_images))
	for i in list(negative_images):
		file_name = "yolo_train_annotations/{}.txt".format(i)
		#print(file_name)
		with open(file_name, 'w') as fp:
			pass

