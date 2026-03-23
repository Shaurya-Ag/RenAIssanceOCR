import cv2
import json
import csv
from functools import cmp_to_key
import math
import os
import unicodedata

def compare_rectangles(a,b):
    y1 = (a[1] + a[3] + a[5] + a[7]) / 4
    y2 = (b[1] + b[3] + b[5] + b[7]) / 4

    x1 = (a[0] + a[2] + a[4] + a[6]) / 4
    x2 = (b[0] + b[2] + b[4] + b[6]) / 4

    h1 = (
        math.hypot(a[0] - a[6], a[1] - a[7]) +
        math.hypot(a[2] - a[4], a[3] - a[5])
    ) / 2

    h2 = (
        math.hypot(b[0] - b[6], b[1] - b[7]) +
        math.hypot(b[2] - b[4], b[3] - b[5])
    ) / 2

    threshold = min(h1, h2) * 0.5

    if abs(y1 - y2) <= threshold:
        return x1 - x2

    return y1 - y2

def sort_boxes(rects):
    rects.sort(key=cmp_to_key(compare_rectangles))  
                            # y  x
def crop_image(img, rects):
    if img is None:
        print("Error: Image not found.")
        return None
    crops=[]
    if not rects:
        return rects
    for rect in rects:
        x1, y1, x2, y2 , x3, y3, x4, y4  = rect # Clockwise from top-left
        x_r=max(x2,x3)
        x_l=min(x1,x4)
        y_u=min(y1,y2)
        y_d=max(y3,y4)
        crop=img[y_u:y_d, x_l:x_r,:]
        crops.append(crop)
    return crops


prev_crops=[]
image_number=0
file_image_number={}

csv_file=open(r'final_dataset\labels.csv','w',newline='')
writer=csv.writer(csv_file)
all_chars=set()

for filename in os.listdir(r'.\data\boxes'):
    if filename.endswith('.txt'):
        
        print("Processing",filename)
        
        with open(rf'.\data\boxes\{filename}','r') as f:
            lines=f.readlines()
            rects=[]
            for line in lines:
                if not line.strip():
                    continue
                rect=list(map(int, line.strip().split(','))) # It is really a csv under a .txt name
                rects.append(rect)
            if not rects:
                print("No boxes for file:",filename)
                continue
            
            sort_boxes(rects)
            
            img=cv2.imread(os.path.join(r".\data\images", filename.replace('.txt','.jpeg').replace('res_','')))
            crops=crop_image(img, rects)
                        
            if not crops:
                raise ValueError("Image loading error:",filename)            
            
            
            transcript_file=open(os.path.join(r".\data\transcripts", filename),'r')
            transcript=transcript_file.read().strip().split()
            transcript_file.close()
            i=0
            crop=0
            while crop<len(crops) and i<len(transcript):
                label=transcript[i]
                cv2.imwrite(os.path.join(r".\final_dataset\images", f"{image_number}_{i}.jpeg"), crops[crop])
                writer.writerow([f"{image_number}_{i}", label])
                i += 1
                crop+=1
                all_chars.update(unicodedata.normalize("NFC", label))

            file_image_number[filename] = image_number
            image_number += 1
all_chars = sorted(all_chars)
char_to_index = {label: idx for idx, label in enumerate(all_chars)}
with open('new_tokeniser.json', 'w', encoding='utf-8') as f:
    json.dump(char_to_index, f)
with open('file_to_image_number.json', 'w') as f:
    json.dump(file_image_number, f)