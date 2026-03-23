import torch.nn as nn
import torch.nn.functional as F_nn
import torchvision.transforms.functional as F


class ResizeWithPadding:
    def __init__(self, height=32, max_width=128):
        self.height = height
        self.max_width = max_width

    def __call__(self, img):
        w, h = img.size

        # scale width according to new height
        new_w = int(w * (self.height / h))
        img = F.resize(img, (self.height, new_w))

        # pad or clamp
        if new_w < self.max_width:
            pad_width = self.max_width - new_w
            img = F.pad(img, (0, 0, pad_width, 0), fill=255)
        else:
            img = F.resize(img, (self.height, self.max_width))

        return img
class CRNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding='same', bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d((2, 1), (2, 1))
                
        self.embedding = nn.Linear(128 * 8, 256)
        self.ln = nn.LayerNorm(256)
        
        
        self.lstm = nn.LSTM(256, 128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = F_nn.leaky_relu(self.bn1(self.conv1(x)))
        x = F_nn.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        x = F_nn.leaky_relu(self.bn3(self.conv3(x)))
        x = F_nn.leaky_relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        B, C, H, W = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(B, W, C * H)
        x = self.ln(self.embedding(x))
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.permute(1, 0, 2)
        x = F_nn.log_softmax(x, dim=2)
        return x