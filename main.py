import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import loadmat
from scipy.io import savemat
from torch import optim
from torch.autograd import Variable
from vit_pytorch_indian_Houston import ViT
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import time
import os

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Houston'], default='Indian', help='dataset to use')
parser.add_argument('--flag_test', choices=['test', 'train'], default='train', help='testing mark')
parser.add_argument('--mode', choices=['ViT', 'CAF'], default='ViT', help='mode choice')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=5, help='number of evaluation')
parser.add_argument('--patches', type=int, default=1, help='number of patches')
parser.add_argument('--band_patches', type=int, default=1, help='number of related band')
parser.add_argument('--epoches', type=int, default=300, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--channels_band', type=int, default=0, help='channels_band')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_train_test(label_data, train_ratio=0.1):
    num_classes = np.max(label_data)
    train_mask = np.zeros_like(label_data)
    test_mask = np.zeros_like(label_data)
    
    for i in range(1, num_classes + 1):
        idx = np.where(label_data == i)[0]
        np.random.shuffle(idx)
        split = int(len(idx) * train_ratio)
        train_idx = idx[:split]
        test_idx = idx[split:]
        train_mask[train_idx] = i
        test_mask[test_idx] = i
    
    return train_mask, test_mask

def prepare_data(input_data, train_mask, test_mask):
    height, width, bands = input_data.shape
    
    train_idx = np.where(train_mask > 0)
    test_idx = np.where(test_mask > 0)
    
    x_train = input_data[train_idx[0], train_idx[1], :]
    x_test = input_data[test_idx[0], test_idx[1], :]
    
    y_train = train_mask[train_idx] - 1
    y_test = test_mask[test_idx] - 1
    
    return x_train, x_test, y_train, y_test

def normalize_data(input_data):
    normalized = np.zeros_like(input_data, dtype=np.float32)
    for i in range(input_data.shape[2]):
        band = input_data[:, :, i]
        normalized[:, :, i] = (band - np.min(band)) / (np.max(band) - np.min(band))
    return normalized

def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    
    if args.dataset == 'Indian':
        data = loadmat('./data/IndianPine.mat')
    elif args.dataset == 'Pavia':
        data = loadmat('./data/Pavia.mat')
    elif args.dataset == 'Houston':
        data = loadmat('./data/Houston.mat')
    
    input_data = data['input']
    label_data = data['TR'] + data['TE']
    num_classes = np.max(label_data)
    
    input_normalized = normalize_data(input_data)
    train_mask, test_mask = split_train_test(label_data)
    x_train, x_test, y_train, y_test = prepare_data(input_normalized, train_mask, test_mask)
    
    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()
    
    train_dataset = Data.TensorDataset(x_train, y_train)
    test_dataset = Data.TensorDataset(x_test, y_test)
    
    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = ViT(
        image_size=args.patches,
        near_band=args.band_patches,
        num_patches=input_data.shape[2],
        num_classes=num_classes,
        channels_band=args.channels_band,
        dim=64,
        depth=5,
        heads=4,
        mlp_dim=8,
        dropout=0.1,
        emb_dropout=0.1,
        mode=args.mode
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches//10, gamma=args.gamma)
    
    best_acc = 0.0
    print(f"Starting training... Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    for epoch in range(args.epoches):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        if epoch % args.test_freq == 0:
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            acc = 100. * correct / total
            print(f'Epoch: {epoch}, Test Acc: {acc:.2f}%')
            
            if acc > best_acc:
                print(f'Saving best model... Accuracy: {acc:.2f}%')
                best_acc = acc
                torch.save(model.state_dict(), f'best_model_{args.dataset}.pt')
        
        scheduler.step()

if __name__ == '__main__':
    main()
