import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from wikiart_dataset_class import wikiart_dataset
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from wikiart.Discriminator_Generator import Discriminator, Generator

# hyperparameters
learning_rate = 0.0005
mini_batch_size = 64
image_size = 64
channels_img = 3
channels_noise = 256
number_of_epochs = 1000

#number of channels of the generator and discriminator
features_d = 16
features_g = 16

# This will transform data to tensor format which is pytorch's expexted format


my_transforms = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
# Here we download the dataset and transfrom it, train=True will only download traning dataset

#data set
dataset = wikiart_dataset(csv_file='Impressionism_csv.csv', root_dir= './', transform = my_transforms)
train_set, test_set = torch.utils.data.random_split(dataset, [6325, 2000])

# Loading the training data
train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = mini_batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size = mini_batch_size, shuffle = False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#instantiate D and G networks
netD = Discriminator(channels_img, features_d).to(device)
netG = Generator(channels_noise, channels_img, features_g).to(device)


#optimizer
optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))

netD.train()
netG.train()

criterion = nn.BCELoss()

real_label = 1
fake_label = 0

fixed_noise = torch.randn(64, channels_noise, 1, 1).to(device)

step = 0
print("starting to train...")

for epoch in range(number_of_epochs):
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        batch_size = data.shape[0]

    #Training Discriminator on max log(D(X)) + log(1 - D(G(z)))
        netD.zero_grad()
        label = (torch.ones(batch_size)*0.9).to(device)
        output = netD(data).reshape(-1)
        lossD_real = criterion(output, label)
        D_x = output.mean().item()

        noise = torch.randn(batch_size, channels_noise, 1, 1).to(device)
        fake = netG(noise)
        label = (torch.ones(batch_size) * 0.1).to(device)

        output = netD(fake.detach()).reshape(-1)
        lossD_fake = criterion(output, label)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

    #Training generator: maximize log(D(G(z)))
        netG.zero_grad()
        label = torch.ones(batch_size).to(device)
        output = netD(fake).reshape(-1)
        lossG = criterion(output, label)
        lossG.backward()
        optimizerG.step()

        #Print losses  and print results
        if batch_idx % 100 == 0:
            step += 1
            print(f"Epoch [{epoch}/{number_of_epochs}] Batch {batch_idx}/{len(train_loader)} \
                  Loss D: {lossD:.4f}, loss G: {lossG:.4f} D(x): {D_x:.4f}")

            with torch.no_grad():
                idx = 0
                fake = netG(fixed_noise)
                img_grid_real = data[idx].cpu().numpy()
                img_grid_fake = fake.detach()[idx].cpu().numpy()
                plt.imshow(img_grid_real.transpose(1, 2, 0))
                #plt.savefig(f'gan_real{epoch}_{batch_idx}.png')
                plt.imshow(img_grid_fake.transpose(1, 2, 0))
                plt.savefig(f'gan_fake_Impressionism{epoch}_{batch_idx}.png')
