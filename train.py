import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision import transforms

import AdaIN_net as net

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


def normalize_losses(loss_list):
    mean = np.mean(loss_list)
    std = np.std(loss_list)
    normalized_losses = [(x - mean) / std for x in loss_list]
    return normalized_losses


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.transform = transform
        if 'wikiart' in root:
            self.paths = list(Path(self.root).glob('*'))
        else:
            self.paths = list(Path(self.root).glob('*'))

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = 1e-5 / (1.0 + 5e-5 * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def loss_plot(c_losses, s_losses, t_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(t_losses, label='Content + Style', color='blue')
    plt.plot(c_losses, label='Content', color='orange')
    plt.plot(s_losses, label='Style', color='green')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AdaIN Style + Content Transfer")

    parser.add_argument('-content_dir', type=str, required=True, help='Directory path to a batch of content images')
    parser.add_argument('-style_dir', type=str, required=True, help='Directory path to a batch of content images')

    parser.add_argument('-gamma', type=float, default=1.0,
                        help='Gamma parameter')
    parser.add_argument('-e', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('-b', type=int, default=20,
                        help='batch size')
    parser.add_argument('-l', type=str, default='encoder.pth',
                        help='load encoder')
    parser.add_argument('-s', type=str, default='decoder.pth',
                        help='save decoder')
    parser.add_argument('-p', type=str, default='decoder.png',
                        help='Value for p')
    parser.add_argument('-cuda', type=str, help='cuda', default='Y')
    args = parser.parse_args()

    print("Args:", args)

    print("Cuda avaliable: ", torch.cuda.is_available())

    if torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    encoder = net.encoder_decoder.encoder
    encoder.load_state_dict(torch.load(args.l, map_location=device))
    network = net.AdaIN_net(encoder)
    network.to(device)
    network.train()

    content_tf = train_transform()
    style_tf = train_transform()

    COCO_data = FlatFolderDataset(args.content_dir, content_tf)
    WIKI_data = FlatFolderDataset(args.style_dir, style_tf)

    num_images = 10000

    COCO_data = torch.utils.data.Subset(COCO_data, list(range(num_images)))
    WIKI_data = torch.utils.data.Subset(WIKI_data, list(range(num_images)))

    content_loader = data.DataLoader(COCO_data, batch_size=args.b, shuffle=True, num_workers=2)
    style_loader = data.DataLoader(WIKI_data, batch_size=args.b, shuffle=True, num_workers=2)

    content_losses = []
    style_losses = []
    total_losses = []

    avg_content_losses = []
    avg_style_losses = []
    avg_total_losses = []

    optimizer = torch.optim.Adam(network.decoder.parameters(), lr=1e-5)

    print("Device: ", device)

    for epoch in range(args.e):
        adjust_learning_rate(optimizer, iteration_count=epoch)
        print(f"Epoch {epoch + 1}/{args.e}")
        epoch_content_loss, epoch_total_loss, epoch_style_loss = 0, 0, 0
        epoch_start = time.time()

        for content_images, style_images in zip(content_loader, style_loader):
            content_images = content_images.to(device)
            style_images = style_images.to(device)

            optimizer.zero_grad()

            loss_c, loss_s = network(content_images, style_images)
            loss = loss_c + args.gamma * loss_s

            loss.backward()
            optimizer.step()

            content_losses.append(loss_c.item())
            style_losses.append(loss_s.item())
            total_losses.append(loss.item())

            epoch_content_loss += loss_c.item()
            epoch_style_loss += loss_s.item()
            epoch_total_loss += loss.item()

        epoch_end = time.time()

        avg_content_loss = sum(content_losses[-len(content_loader):]) / len(content_loader)
        avg_style_loss = sum(style_losses[-len(style_loader):]) / len(style_loader)
        avg_total_loss = sum(total_losses[-len(content_loader):]) / len(content_loader)

        avg_content_losses.append(avg_content_loss)
        avg_style_losses.append(avg_style_loss)
        avg_total_losses.append(avg_total_loss)

        print(f"Content Loss: {avg_content_loss:.2f}, Style Loss: {avg_style_loss:.2f}, "
              f"Total Loss: {avg_total_loss:.2f}")
        print(f"Time Taken for epoch: {epoch_end - epoch_start:.2f} seconds\n")

        torch.save(net.encoder_decoder.decoder.state_dict(), args.s)

    normalized_content_losses = normalize_losses(avg_content_losses)
    normalized_style_losses = normalize_losses(avg_style_losses)
    normalized_total_losses = normalize_losses(avg_total_losses)

    loss_plot(normalized_content_losses, normalized_style_losses, normalized_total_losses, args.p)
