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







class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
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
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),  # (1, 1, 1)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)


class GenGAN():
    def __init__(self, videoSke, loadFromFile=False):
        image_size = 64
        self.netG = GenNNSkeToImage()
        self.netD = Discriminator()
        self.filename_G = 'best_model_G.pth'
        self.filename_D = 'best_model_D.pth'

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

        if loadFromFile and os.path.isfile(self.filename_G) and os.path.isfile(self.filename_D):
            print(f"GenVanillaNN: Load G={self.filename_G}, D={self.filename_D}")
            
            # Load and set state dict for Generator
            state_dict_G = torch.load(self.filename_G)
            self.netG.load_state_dict(state_dict_G)
            
            # Load and set state dict for Discriminator
            state_dict_D = torch.load(self.filename_D)
            self.netD.load_state_dict(state_dict_D)
        else:
            self.netG.apply(init_weights)
            self.netD.apply(init_weights) 
    def train(self, n_epochs=100, patience=20):
        criterion = nn.BCELoss()
        optimizer_G = optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
        # Scheduler de Taux d'Apprentissage 
        scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.1) 
        scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=10, gamma=0.1)
        best_loss_G = float('inf')
        best_model_G = None
        epochs_no_improve = 0
        
        for epoch in range(n_epochs):
            for i, (ske_images, real_images) in enumerate(self.dataloader):
                batch_size = real_images.size(0)
                real_label = torch.full((batch_size, 1), 1, dtype=torch.float32)
                fake_label = torch.full((batch_size, 1), 0, dtype=torch.float32)

                # Train Discriminator
                optimizer_D.zero_grad()
                
                output = self.netD(real_images)
                loss_D_real = criterion(output, real_label)
                loss_D_real.backward()

                fake_images = self.netG(ske_images)
                output = self.netD(fake_images.detach())
                loss_D_fake = criterion(output, fake_label)
                loss_D_fake.backward()

                loss_D = loss_D_real + loss_D_fake
                optimizer_D.step()

                # Train Generator
                optimizer_G.zero_grad()
                output = self.netD(fake_images)
                loss_G = criterion(output, real_label)
                loss_G.backward()
                optimizer_G.step()
            optimizer_G.step() 
            scheduler_D.step() 
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}')
            
            # Early stopping for Generator
            if loss_G.item() < best_loss_G:
                best_loss_G = loss_G.item()
                best_model_G = self.netG.state_dict()
                epochs_no_improve = 0
                torch.save(best_model_G, 'best_model_G.pth')
                torch.save(self.netD.state_dict(), 'best_model_D.pth')
                print(f"New best model saved with Generator loss: {best_loss_G:.4f}")
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve == patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Save sample images
            if (epoch + 1) % 5 == 0:
                with torch.no_grad():
                    sample_ske_image = next(iter(self.dataloader))[0][:1]
                    sample_output = self.netG(sample_ske_image)
                    save_image(sample_output, f'sample_epoch_{epoch+1}.png', normalize=True)
        
        # Load the best model
        self.netG.load_state_dict(torch.load('best_model_G.pth'))
        self.netD.load_state_dict(torch.load('best_model_D.pth'))
        print(f"Training completed. Best Generator model loaded with loss: {best_loss_G:.4f}")

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
    @staticmethod
    def tensor2image(tensor):
        # Convertir le tenseur en numpy array et réorganiser les dimensions de CHW à HWC
        numpy_image = tensor.cpu().detach().numpy()
        numpy_image = np.transpose(numpy_image, (1, 2, 0))  # CHW -> HWC

        # Conversion de [-1, 1] à [0, 255] sans normalisation
        #numpy_image = (numpy_image * 127.5 + 127.5).astype(np.uint8)

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
        gen = GenGAN(targetVideoSke, loadFromFile=False)
        gen.train(n_epochs=1000, patience=20)
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)

    # Test with a second video
    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)