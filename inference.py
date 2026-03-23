import warnings
warnings.filterwarnings("ignore")
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from PIL import Image
from torchvision.transforms import v2
from Model import CRNNModel, ResizeWithPadding
from Trainer import CRNNTrainer
import json
import subprocess
from functools import cmp_to_key
import math
import os

IMG_MEAN = 0.8477857112884521
IMG_STD = 0.24946823716163635
BLANK_IDX = 78
tmp_dir = "tmp_craft_results"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

argparser = argparse.ArgumentParser(description="Inference script for Renaissance OCR")
argparser.add_argument("--model_path", type=str, default=r"CRNNmodel.pth", help="Path to the trained CRNN model")
argparser.add_argument("--craft_model_path", type=str, default = r'CRAFT-pytorch\weights\craft_mlt_25k.pth', help="Path to the trained model")
argparser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing input images")
argparser.add_argument("--tokeniser", type=str, default=r'new_tokeniser.json', help="Path to the tokeniser file")
argparser.add_argument("--result_folder", type=str, default='inference_results', help="Folder to store transcription results")
args = argparser.parse_args()

if not os.path.exists(args.image_folder):
    print(f"Error: Image folder '{args.image_folder}' does not exist.")
    exit(1)
if not os.path.exists(args.craft_model_path):
    print(f"Error: Model file '{args.craft_model_path}' does not exist.")
    exit(1)
if not os.path.exists(args.tokeniser):
    print(f"Error: Tokeniser file '{args.tokeniser}' does not exist.")
    exit(1)
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
if not os.path.exists(args.result_folder):
    os.makedirs(args.result_folder)

def load_model(model_path, device):
    model = CRNNModel(num_classes=79)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_tokeniser(tokeniser_path):
    with open(tokeniser_path, "r", encoding="utf-8") as f:
        idx_to_label = json.load(f)
    return idx_to_label

def apply_CRAFT(image_folder, result_folder):
    subprocess.run(["python", "CRAFT-pytorch\\test.py", "--trained_model", args.craft_model_path, "--test_folder", image_folder, "--result_folder", tmp_dir], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def preprocess_image(image):
    transform = v2.Compose([
        ResizeWithPadding(32, 128),
        v2.ToTensor(),
        v2.Normalize(mean=[IMG_MEAN], std=[IMG_STD])
    ])
    image = transform(image)
    return image

def decode_predictions(preds, idx_to_label):
    pred_labels = []
    for pred in preds:
        pred_text = ""
        prev_idx = None
        for idx in pred:
            if idx != prev_idx and idx != BLANK_IDX:
                pred_text += idx_to_label[idx.item()]
            prev_idx = idx
        pred_labels.append(pred_text)
    return pred_labels


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
        print("Error: Rectangles not found.")
        return rects
    for rect in rects:
        x1, y1, x2, y2 , x3, y3, x4, y4  = rect # Clockwise from top-left
        x_r=max(x2,x3)
        x_l=min(x1,x4)
        y_u=min(y1,y2)
        y_d=max(y3,y4)
        crop=img[y_u:y_d, x_l:x_r]
        crops.append(crop)
    return crops



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device)
    char_to_idx = load_tokeniser(args.tokeniser)
    idx_to_label = {idx: char for char, idx in char_to_idx.items()}
    apply_CRAFT(args.image_folder, args.result_folder)
    
    for image_name in os.listdir(args.image_folder):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue  # Skip non-image files
        
        image_path = os.path.join(args.image_folder, image_name)
        result_path = os.path.join(tmp_dir, os.path.splitext(image_name)[0] + ".txt")
        
        if not os.path.exists(result_path):
            print(f"Warning: CRAFT result for '{image_name}' not found. Skipping.")
            continue
        lines = None
        with open(result_path, "r") as f:
            lines = f.readlines()
        
        rects = []
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) == 8:
                rects.append(list(map(int, parts[:8])))
        
        sort_boxes(rects)
        img = cv2.imread(image_path.replace('.txt','.jpeg'), cv2.IMREAD_GRAYSCALE)
        crops = crop_image(img, rects)
        print("Processing image:", image_name)
        words = []
        for crop in crops:
            if crop is None:
                continue
            pil_crop = Image.fromarray(crop)
            input_tensor = preprocess_image(pil_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                preds = model(input_tensor)
            pred_labels = decode_predictions(preds.argmax(dim=-1), idx_to_label)
            words.append(''.join(pred_labels))
        final_text = ' '.join(words)
        with open(os.path.join(args.result_folder, os.path.splitext(image_name)[0] + ".txt"), "w", encoding="utf-8") as f:
            f.write(final_text)
if __name__ == "__main__":
    main()