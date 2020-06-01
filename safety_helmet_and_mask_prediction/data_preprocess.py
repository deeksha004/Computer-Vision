import pandas as pd
import numpy as np

import xml.etree.ElementTree as et 

xtree = et.parse("annotations.xml")
xroot = xtree.getroot()
#data = pd.DataFrame(columns=['image_id', 'label', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax', 'has_mask', 'has_safety_helmet'])
rows = []
count =0
for image in xroot:
    id_ = image.attrib.get("id")
    image_id = id_
    width = image.attrib.get("width")
    height= image.attrib.get("height")
    #print(id_)
    for bb in image:
        label = bb.attrib.get("label")
        if label == "head":
            xmin= bb.attrib.get("xtl")
            ymin= bb.attrib.get("ytl")
            xmax= bb.attrib.get("xbr")
            ymax= bb.attrib.get("ybr")
            for at in bb:
                if at.attrib['name']=='mask':
                    has_mask = at.text if at is not None else None
                elif at.attrib['name']=='has_safety_helmet':
                    has_safety_helmet= at.text if at is not None else None
                elif at.attrib['name']!='mask':
                    has_mask = 'None'
                elif at.attrib['name']!='has_safety_helmet':
                    has_safety_helmet = 'None'
            #print([image_id, label, width, height])
            rows.append([image_id, label, width, height, xmin, ymin, xmax, ymax, has_mask, has_safety_helmet])
columns=['image_id', 'label', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax', 'has_mask', 'has_safety_helmet']
df = pd.DataFrame(rows, columns = columns)
df[['width', 'height', 'xmin', 'ymin', 'xmax', 'ymax']]=df[['width', 'height', 'xmin', 'ymin', 'xmax', 'ymax']].astype(float)
df.to_csv("data.csv", index=False)
