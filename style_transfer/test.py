from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torchsummary
import numpy as np
from collections import namedtuple

import copy
import time
import glob
import re
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# from google.colab import drive
# drive.mount("/content/gdrive")

# style_image_location = "/content/gdrive/My Drive/Colab Notebooks/sai/data/download.jpg" #FIXME
#
# style_image_sample = Image.open(style_image_location, 'r')
# display(style_image_sample)

batch_size = 8
random_seed = 10
num_epochs = 64
initial_lr = 1e-3
checkpoint_dir = "./ckp/" #FIXME

content_weight = 1e5
style_weight = 1e10
log_interval = 50
checkpoint_interval = 500


class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out

class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.encoder = nn.Sequential()

        self.encoder.add_module('conv1', ConvLayer(3, 32, kernel_size=9, stride=1))
        self.encoder.add_module('in1', nn.InstanceNorm2d(32, affine=True))
        self.encoder.add_module('relu1', nn.ReLU())

        self.encoder.add_module('conv2', ConvLayer(32, 64, kernel_size=3, stride=2))
        self.encoder.add_module('in2', nn.InstanceNorm2d(64, affine=True))
        self.encoder.add_module('relu2', nn.ReLU())

        self.encoder.add_module('conv3', ConvLayer(64, 128, kernel_size=3, stride=2))
        self.encoder.add_module('in3', nn.InstanceNorm2d(128, affine=True))
        self.encoder.add_module('relu3', nn.ReLU())

        # Residual layers
        self.residual = nn.Sequential()

        for i in range(5):
            self.residual.add_module('resblock_%d' % (i + 1), ResidualBlock(128))

        # Upsampling Layers
        self.decoder = nn.Sequential()
        self.decoder.add_module('deconv1', UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2))
        self.decoder.add_module('in4', nn.InstanceNorm2d(64, affine=True))
        self.encoder.add_module('relu4', nn.ReLU())

        self.decoder.add_module('deconv2', UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2))
        self.decoder.add_module('in5', nn.InstanceNorm2d(32, affine=True))
        self.encoder.add_module('relu5', nn.ReLU())

        self.decoder.add_module('deconv3', ConvLayer(32, 3, kernel_size=9, stride=1))

    def forward(self, x):
        encoder_output = self.encoder(x)
        residual_output = self.residual(encoder_output)
        decoder_output = self.decoder(residual_output)

        return decoder_output


""" Util Functions """


def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    display(img)
    img.save(filename)


def post_process_image(data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    # img = Image.fromarray(img)

    return img


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

np.random.seed(random_seed)
torch.manual_seed(random_seed)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
])
#
# print(glob.glob("/content/gdrive/My Drive/Colab Notebooks/sai/train/val2017/*"))
#
# train_dataset = datasets.ImageFolder("/content/gdrive/My Drive/Colab Notebooks/sai/train", transform) #FIXME
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

transformer = TransformerNet()
vgg = VGG16(requires_grad=False).to(device)

optimizer = torch.optim.Adam(transformer.parameters(), initial_lr)
mse_loss = nn.MSELoss()

style_image_location = "./img/fire.jpg"

style = load_image(filename=style_image_location, size=None, scale=None)
style = style_transform(style)
style = style.repeat(batch_size, 1, 1, 1).to(device)

features_style = vgg(normalize_batch(style))
gram_style = [gram_matrix(y) for y in features_style]


# capture = cv2.VideoCapture(0)
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# while True:
#     ret, frame = capture.read()
#     cv2.imshow("VideoFrame", frame)
#     if cv2.waitKey(1) > 0:
#         break
#
# capture.release()
# cv2.destroyAllWindows()



with torch.no_grad():
    style_model = TransformerNet()

    ckpt_model_path = os.path.join(checkpoint_dir, "ckpt_epoch_15_batch_id_500.pth")  # FIXME
    checkpoint = torch.load(ckpt_model_path, map_location=device)

    # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
    for k in list(checkpoint.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del checkpoint[k]

    style_model.load_state_dict(checkpoint['model_state_dict'])
    style_model.to(device)

    cap = cv2.VideoCapture(0)  # FIXME
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    # cap.set(cv2.CAP_PROP_FPS, 10)

    frame_cnt = 0

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('/content/gdrive/My Drive/Colab Notebooks/sai/data/test_result.mp4', fourcc, 60.0, (1920,1080)) #FIXME
    # out = cv2.VideoWriter('/content/gdrive/My Drive/Colab Notebooks/sai/data/test_result2.mp4', fourcc, 30.0,
    #                       (1080, 720))  # FIXME

    # cv2.waitKey(1000)
    print('start')
    while (cap.isOpened()):

        ret, frame = cap.read()

        try:
            frame = frame[:, :, ::-1] - np.zeros_like(frame)
        except:
            cv2.waitKey(1)
            cv2.imshow("frame",np.zeros((640,480,3)))
            continue
            # break

        # if cv2.waitKey(1) > 0:
        #     break

        # print(frame_cnt, "th frame is loaded!")

        # content_image = frame
        # content_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Lambda(lambda x: x.mul(255))
        # ])
        # content_image = content_transform(content_image)
        # content_image = content_image.unsqueeze(0).to(device)
        #
        # output = style_model(content_image)
        # # save_image("/content/gdrive/My Drive/Colab Notebooks/data/vikendi_video_result/" + str(frame_cnt) +".png", output[0]) #FIXME
        # # out.write(post_process_image(output[0]))
        cv2.imshow("frame", frame)

        frame_cnt += 1

    cap.release()
    cv2.destroyAllWindows()

