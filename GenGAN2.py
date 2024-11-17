import numpy as np
import cv2
import os
import sys
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from VideoSkeleton import VideoSkeleton
from Skeleton import Skeleton

torch.set_default_dtype(torch.float32)

class SkeToImageTransform:
    def __init__(self, image_size):
        self.imsize = image_size

    def __call__(self, ske):
        image = np.ones((self.imsize, self.imsize, 3), dtype=np.uint8) * 255
        ske.draw(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
        ske_image = np.ones((64, 64, 3), dtype=np.uint8) * 255
        Skeleton.draw_reduced(self.videoSke.ske[idx].reduce(), ske_image)
        if self.source_transform:
            ske_image = self.source_transform(ske_image)
        target_image = Image.open(self.videoSke.imagePath(idx))
        if self.target_transform:
            target_image = self.target_transform(target_image)
        return ske_image, target_image

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.conv_block(x)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, num_residual_blocks=9):
        super(Generator, self).__init__()
        model = [
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]

        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        model += [nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(3, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, stride=1, padding=1)  # Changed stride to 1 and added padding
        )

    def forward(self, img):
        return self.model(img)



class GenGAN:
    def __init__(self, videoSke, loadFromFile=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.netG = Generator().to(self.device)
        self.netD = Discriminator().to(self.device)

        self.filename_G = 'DanceGenGAN2_G64.pth'
        self.filename_D = 'DanceGenGAN2_D64.pth'

        self.ske_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.tgt_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, 
                                            source_transform=self.ske_transform, 
                                            target_transform=self.tgt_transform)
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=16, shuffle=True)

        if loadFromFile and os.path.isfile(self.filename_G) and os.path.isfile(self.filename_D):
            self.netG.load_state_dict(torch.load(self.filename_G, map_location=self.device))
            self.netD.load_state_dict(torch.load(self.filename_D, map_location=self.device))
        else:
            self.netG.apply(init_weights)
            self.netD.apply(init_weights)

    def train(self, n_epochs=100, patience=20):
        criterion_GAN = nn.MSELoss()
        criterion_pixel = nn.L1Loss()

        optimizer_G = optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

        scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.1)
        scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.1)

        best_loss_G = float('inf')
        epochs_no_improve = 0

        for epoch in range(n_epochs):
            for i, (ske_images, real_images) in enumerate(self.dataloader):
                ske_images = ske_images.to(self.device)
                real_images = real_images.to(self.device)

                valid = torch.ones((ske_images.size(0), 1, 3, 3), requires_grad=False).to(self.device)
                fake = torch.zeros((ske_images.size(0), 1, 3, 3), requires_grad=False).to(self.device)


             

                # Train Generator
                optimizer_G.zero_grad()
                gen_imgs = self.netG(ske_images)
                # print("Output of netD(gen_imgs):", self.netD(gen_imgs).shape)
                # print("Size of 'valid':", valid.shape)
                # print("Size of 'fake':", fake.shape)

                loss_GAN = criterion_GAN(self.netD(gen_imgs), valid)
                loss_pixel = criterion_pixel(gen_imgs, real_images)
                loss_G = loss_GAN + 100 * loss_pixel
                loss_G.backward()
                optimizer_G.step()

                # Train Discriminator
                optimizer_D.zero_grad()
                loss_real = criterion_GAN(self.netD(real_images), valid)
                loss_fake = criterion_GAN(self.netD(gen_imgs.detach()), fake)
                loss_D = (loss_real + loss_fake) / 2
                loss_D.backward()
                optimizer_D.step()

            scheduler_G.step()
            scheduler_D.step()

            print(f'Epoch [{epoch+1}/{n_epochs}], Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}')

            # Early stopping
            if loss_G.item() < best_loss_G:
                best_loss_G = loss_G.item()
                torch.save(self.netG.state_dict(), self.filename_G)
                torch.save(self.netD.state_dict(), self.filename_D)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

            # Save sample images
            if (epoch + 1) % 5 == 0:
                with torch.no_grad():
                    sample_ske_image = next(iter(self.dataloader))[0][:1].to(self.device)
                    sample_output = self.netG(sample_ske_image)
                    save_image(sample_output, f'sample_epoch_{epoch+1}.png', normalize=True)

        print(f"Training completed. Best models saved as {self.filename_G} and {self.filename_D}")

    def generate(self, ske):
        image = np.ones((64, 64, 3), dtype=np.uint8) * 255
        Skeleton.draw_reduced(ske.reduce(), image)
        ske_image = self.ske_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            generated_image = self.netG(ske_image)
            save_image(generated_image, f'generated.png', normalize=True)
        return self.tensor2image(generated_image[0])

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
    n_epoch = 2000
    train = True

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"

    print("GenGAN: Current Working Directory=", os.getcwd())
    print("GenGAN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    if train:
        gen = GenGAN(targetVideoSke, loadFromFile=False)
        gen.train(n_epochs=n_epoch, patience=20)
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)

    # Test with a second video
    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        nouvelle_taille = (256, 256)
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)
