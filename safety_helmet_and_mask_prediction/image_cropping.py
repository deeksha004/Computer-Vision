import pandas as pd
import os
import numpy
import glob
import cv2
import matplotlib.pyplot as plt

classes = ["Head"]
annotation_location = "yolo_train_annotations"
image_location = "images"
crop_out_location = "train_heads"
with_head_loc = '{}/with_heads'.format(crop_out_location)
without_head_loc = '{}/without_heads'.format(crop_out_location)
images = glob.glob('images/*.jpg')
data = pd.read_csv("data.csv")
emply_file_ids =[]
colors = (0,255,0)
if not os.path.exists(crop_out_location):
    os.makedirs(crop_out_location)
    os.makedirs(with_head_loc)
    os.makedirs(without_head_loc)
    
def draw_bounding_box(img, class_id, xmin, ymin, x_plus_w, y_plus_h):
    label = class_id
    color = colors#[class_id]
    cv2.rectangle(img, (xmin, ymin), (x_plus_w,y_plus_h), color, 3)
    print(xmin, ymin, x_plus_w, y_plus_h)
    cv2.putText(img, label, (xmin+20,ymin+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    plt.imshow(img)
    plt.show()
#print(data.iloc[155:160,:])
#print(data.loc[data["image_id"]==48])  
print(data.iloc[0:50,:])
#print(data["has_safety_helmet"][0])  
#def crop_head(image)
def main():
	annotation_files = glob.glob('{}/*.txt'.format(annotation_location))
	annotation_files.sort()
	index=0
	for files in annotation_files:
		image_id = files.split("/")[-1].replace(".txt", "")
		image = cv2.imread('{}/{}.jpg'.format(image_location, image_id))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		height, width, channels = image.shape
		f = open(files, "r")
		count=0
		if os.stat(files).st_size != 0:
			for line in f:
				detection = [word for word in line.split()]
				center_x = float(detection[1]) * width
				center_y = float(detection[2]) * height
				w = float(detection[3]) * width
				h = float(detection[4]) * height
				x = center_x - w / 2
				y = center_y - h / 2
				print(round(x), round(y), round(x+w), round(y+h))
				cropped = image[round(y):round(y+h), round(x):round(x+w)]
				cropped=cv2.resize(cropped,(28,28))
				if data["has_safety_helmet"][index]=='yes':
					label= with_head_loc
					plt.imsave('{}/{}_{}.png'.format(label,image_id,count), cropped)
				elif data["has_safety_helmet"][index]=='no':
					label=without_head_loc
					plt.imsave('{}/{}_{}.png'.format(label,image_id,count), cropped)
				index=index+1
				count=count+1
				#plt.imshow(cropped)
				#plt.show()
				#draw_bounding_box(image, "Head", round(x), round(y), round(x+w), round(y+h))
		else:
			emply_file_ids.append(image_id)
		f.close()
main()
