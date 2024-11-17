import numpy as np
import cv2
import os
import pickle
import sys
import math

from PIL import Image
import matplotlib.pyplot as plt
from torchvision.io import read_image

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms

#from tensorboardX import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

torch.set_default_dtype(torch.float32)


class SkeToImageTransform:
    def __init__(self, image_size):
        self.imsize = image_size

    def __call__(self, ske):
        #image = Image.new('RGB', (self.imsize, self.imsize), (255, 255, 255))
        image = white_image = np.ones((self.imsize, self.imsize, 3), dtype=np.uint8) * 255
        ske.draw(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('Image', image)
        # key = cv2.waitKey(-1)
        return image



class VideoSkeletonDataset(Dataset):
    def __init__(self, videoSke, ske_reduced, source_transform=None, target_transform=None):
        self.videoSke = videoSke
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.ske_reduced = ske_reduced

    def __len__(self):
        return self.videoSke.skeCount()

    def __getitem__(self, idx):
        # Create skeleton image
        ske_image = np.ones((64, 64, 3), dtype=np.uint8) * 255
        Skeleton.draw_reduced(self.videoSke.ske[idx].reduce(), ske_image)
        
        # Transform skeleton image
        if self.source_transform:
            ske_image = self.source_transform(ske_image)
        
        # Get and transform target image
        target_image = Image.open(self.videoSke.imagePath(idx))
        if self.target_transform:
            target_image = self.target_transform(target_image)
        
        return ske_image, target_image



def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)




class GenNNSkeToImage(nn.Module):
    def __init__(self):
        super(GenNNSkeToImage, self).__init__()
        self.model = nn.Sequential(
            # Input: (3, 64, 64)
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # (64, 32, 32)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # (512, 4, 4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # (3, 64, 64)
            nn.Tanh()
        )
        print(self.model)

    def forward(self, x):
        return self.model(x)









class GenVanillaNNfromImage():
    def __init__(self, videoSke, loadFromFile=False):
        image_size = 64
        self.netG = GenNNSkeToImage()
        self.filename = 'best_modelFromImage.pth'

        # Transform for the skeleton image
        self.ske_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # Transform for the target image
        self.tgt_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, 
                                            source_transform=self.ske_transform, 
                                            target_transform=self.tgt_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=16, shuffle=True)

        if loadFromFile and os.path.isfile(self.filename):
            print("GenVanillaNN: Load=", self.filename)
            print("GenVanillaNN: Current Working Directory: ", os.getcwd())
            # Load the state dict instead of the full model
            state_dict = torch.load(self.filename)
            self.netG.load_state_dict(state_dict)

    def train(self, n_epochs=20, patience=10):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        best_loss = float('inf')
        best_model = None
        epochs_no_improve = 0
        
        for epoch in range(n_epochs):
            epoch_loss = 0
            for i, (ske_images, target_images) in enumerate(self.dataloader):
                # Forward pass
                outputs = self.netG(ske_images)
                loss = criterion(outputs, target_images)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(self.dataloader)
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}')
            
            # Check if this is the best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model = self.netG.state_dict()
                epochs_no_improve = 0
                torch.save(best_model, 'best_model.pth')
                print(f"New best model saved with loss: {best_loss:.4f}")
            else:
                epochs_no_improve += 1
            
            # Early stopping
            if epochs_no_improve == patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Save a sample image every 5 epochs
            if (epoch + 1) % 5 == 0:
                with torch.no_grad():
                    sample_ske_image = next(iter(self.dataloader))[0][:1]
                    sample_output = self.netG(sample_ske_image)
                    save_image(sample_output, f'sample_epoch_{epoch+1}.png', normalize=True)
        
        # Load the best model
        self.netG.load_state_dict(torch.load('best_model.pth'))
        print(f"Training completed. Best model loaded with loss: {best_loss:.4f}")

    def generate(self, ske):
        # Create an image with the skeleton drawn
        image = np.ones((64, 64, 3), dtype=np.uint8) * 255
        Skeleton.draw_reduced(ske.reduce(), image)
        
        # Transform the skeleton image
        ske_image = self.ske_transform(image).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            generated_image = self.netG(ske_image)
        
        return self.tensor2image(generated_image[0])

    @staticmethod
    def tensor2image(tensor):
        numpy_image = tensor.detach().numpy()
        # Réorganiser les dimensions (C, H, W) en (H, W, C)
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        # passage a des images cv2 pour affichage
        numpy_image = cv2.cvtColor(np.array(numpy_image), cv2.COLOR_RGB2BGR)
        denormalized_image = numpy_image * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
        denormalized_output = denormalized_image * 1
        return denormalized_output






if __name__ == '__main__':
    force = False
    n_epoch = 2000  # 200
    train = 1 #False
    #train = True

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"
    print("GenVanillaNN: Current Working Directory=", os.getcwd())
    print("GenVanillaNN: Filename=", filename)
    print("GenVanillaNN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    if train:
        # Train
        gen = GenVanillaNNfromImage(targetVideoSke, loadFromFile=False)
        gen.train(n_epoch)
    else:
        gen = GenVanillaNNfromImage(targetVideoSke, loadFromFile=True)    # load from file        
