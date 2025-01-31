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
parser.add_argument('--gpu_id', default='3', help='gpu id')
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

# Set GPU device first
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

# Initialize CUDA
torch.cuda.init()

# Simple GPU check
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
else:
    raise RuntimeError("No CUDA device available. Please check your GPU setup")

# Set device
device = torch.device("cuda")

def set_device(gpu_id):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    return device

def chooose_train_and_test_point(train_data, test_data, true_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data==(i+1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]]
    total_pos_train = total_pos_train.astype(int)

    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]]
    total_pos_test = total_pos_test.astype(int)

    for i in range(num_classes+1):
        each_class = []
        each_class = np.argwhere(true_data==i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes+1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)

    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true

def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    print(f"[INFO] Patch size: {patch}")
    print(f"[INFO] Mirror image shape: [{mirror_hsi.shape[0]}, {mirror_hsi.shape[1]}, {mirror_hsi.shape[2]}]")
    return mirror_hsi

def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image

def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=float)
    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape
    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]
    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band

def train_and_test_data(mirror_image, band, train_point, test_point, true_point, patch=5, band_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    for k in range(true_point.shape[0]):
        x_true[k,:,:,:] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)
    print(f"[INFO] x_train shape = {x_train.shape}, type = {x_train.dtype}")
    print(f"[INFO] x_test shape = {x_test.shape}, type = {x_test.dtype}")
    print(f"[INFO] x_true shape = {x_true.shape}, type = {x_test.dtype}")
    
    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    x_true_band = gain_neighborhood_band(x_true, band, band_patch, patch)
    print(f"[INFO] x_train_band shape = {x_train_band.shape}, type = {x_train_band.dtype}")
    print(f"[INFO] x_test_band shape = {x_test_band.shape}, type = {x_test_band.dtype}")
    print(f"[INFO] x_true_band shape = {x_true_band.shape}, type = {x_true_band.dtype}")
    
    x_train_band = torch.from_numpy(x_train_band.transpose(0,2,1)).type(torch.FloatTensor)
    x_test_band = torch.from_numpy(x_test_band.transpose(0,2,1)).type(torch.FloatTensor) 
    x_true_band = torch.from_numpy(x_true_band.transpose(0,2,1)).type(torch.FloatTensor)
    
    return x_train_band, x_test_band, x_true_band

def train_and_test_label(number_train, number_test, number_true, num_classes):
    y_train = []
    y_test = []
    y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    for i in range(num_classes+1):
        for j in range(number_true[i]):
            y_true.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_true = np.array(y_true)
    print(f"[INFO] y_train: shape = {y_train.shape}, type = {y_train.dtype}")
    print(f"[INFO] y_test: shape = {y_test.shape}, type = {y_test.dtype}")
    print(f"[INFO] y_true: shape = {y_true.shape}, type = {y_true.dtype}")
    return y_train, y_test, y_true

class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res, target, pred.squeeze()

def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.cuda(non_blocking=True)
        batch_target = batch_target.cuda(non_blocking=True)
        
        optimizer.zero_grad()
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.item(), n)
        top1.update(prec1[0].item(), n)
        tar = np.append(tar, t.cpu().numpy())
        pre = np.append(pre, p.cpu().numpy())
        
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
            
    return top1.avg, objs.avg, tar, pre

def valid_epoch(model, valid_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
            batch_data = batch_data.cuda(non_blocking=True)
            batch_target = batch_target.cuda(non_blocking=True)
            
            batch_pred = model(batch_data)
            loss = criterion(batch_pred, batch_target)

            prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
            n = batch_data.shape[0]
            objs.update(loss.item(), n)
            top1.update(prec1[0].item(), n)
            tar = np.append(tar, t.cpu().numpy())
            pre = np.append(pre, p.cpu().numpy())
            
    return tar, pre

def test_epoch(model, test_loader, criterion):
    pre = np.array([])
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
            batch_data = batch_data.cuda(non_blocking=True)
            batch_pred = model(batch_data)
            
            _, pred = batch_pred.topk(1, 1, True, True)
            pp = pred.squeeze()
            pre = np.append(pre, pp.cpu().numpy())
            
    return pre

def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA

def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA

if __name__ == "__main__":
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    if args.dataset == 'Indian':
        data = loadmat('./data/IndianPine.mat')
    elif args.dataset == 'Pavia':
        data = loadmat('./data/Pavia.mat')
    elif args.dataset == 'Houston':
        data = loadmat('./data/Houston.mat')
    else:
        raise ValueError("Unknown dataset")

    color_mat = loadmat('./data/AVIRIS_colormap.mat')
    TR = data['TR']
    TE = data['TE']
    input = data['input']
    label = TR + TE
    num_classes = np.max(TR)

    color_mat_list = list(color_mat)
    color_matrix = color_mat[color_mat_list[3]]

    input_normalize = np.zeros(input.shape)
    for i in range(input.shape[2]):
        input_max = np.max(input[:,:,i])
        input_min = np.min(input[:,:,i])
        input_normalize[:,:,i] = (input[:,:,i]-input_min)/(input_max-input_min)

    height, width, band = input.shape
    print(f"[INFO] Data dimensions - Height: {height}, Width: {width}, Bands: {band}")

    total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = chooose_train_and_test_point(TR, TE, label, num_classes)
    mirror_image = mirror_hsi(height, width, band, input_normalize, patch=args.patches)
    x_train_band, x_test_band, x_true_band = train_and_test_data(mirror_image, band, total_pos_train, total_pos_test, total_pos_true, patch=args.patches, band_patch=args.band_patches)
    y_train, y_test, y_true = train_and_test_label(number_train, number_test, number_true, num_classes)

    # Move model and criterion to device
    model = ViT(
        image_size = args.patches,
        near_band = args.band_patches,
        num_patches = band,
        num_classes = num_classes,
        channels_band = args.channels_band,
        dim = 64,
        depth = 5,
        heads = 4,
        mlp_dim = 8,
        dropout = 0.1,
        emb_dropout = 0.1,
        mode = args.mode
    ).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches//10, gamma=args.gamma)

    # Move data to GPU with error checking
    try:
        x_train = x_train_band.cuda()
        y_train = torch.from_numpy(y_train).long().cuda()
        x_test = x_test_band.cuda()
        y_test = torch.from_numpy(y_test).long().cuda()
        x_true = x_true_band.cuda()
        y_true = torch.from_numpy(y_true).long().cuda()
    except RuntimeError as e:
        print(f"Error moving data to GPU: {e}")
        raise

    # Create data loaders
    Label_train = Data.TensorDataset(x_train, y_train)
    Label_test = Data.TensorDataset(x_test, y_test)
    Label_true = Data.TensorDataset(x_true, y_true)

    label_train_loader = Data.DataLoader(Label_train, batch_size=args.batch_size, shuffle=True)
    label_test_loader = Data.DataLoader(Label_test, batch_size=args.batch_size, shuffle=True)
    label_true_loader = Data.DataLoader(Label_true, batch_size=100, shuffle=False)

    if args.flag_test == 'test':
        if args.mode == 'ViT':
            model.load_state_dict(torch.load('./log_locality/ViT.pt'))
        elif (args.mode == 'CAF') and (args.patches == 7):
            model.load_state_dict(torch.load('./log/IP.pt'))
        else:
            raise ValueError("Wrong Parameters")
            
        model.eval()
        tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
        OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)

        pre_u = test_epoch(model, label_true_loader, criterion)
        prediction_matrix = np.zeros((height, width), dtype=float)
        for i in range(total_pos_true.shape[0]):
            prediction_matrix[total_pos_true[i,0], total_pos_true[i,1]] = pre_u[i] + 1
        
        plt.subplot(1,1,1)
        plt.imshow(prediction_matrix, colors.ListedColormap(color_matrix))
        plt.xticks([])
        plt.yticks([])
        plt.show()
        
        savemat('matrix.mat', {'P':prediction_matrix, 'label':label})

    elif args.flag_test == 'train':
        print(f"Starting training on {device}")
        best_acc = 0.0
        for epoch in range(args.epoches):
            model.train()
            train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer)
            
            if (epoch % args.test_freq == 0) or (epoch == args.epoches - 1):
                model.eval()
                tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
                OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
                
                if OA2 > best_acc:
                    best_acc = OA2
                    torch.save(model.state_dict(), "./log/IP.pt")
                print(f"Epoch: {epoch + 1:03d} | Test Acc: {OA2:.4f}")
                
            scheduler.step()
                                                                

    print("\n[RESULTS] Final Metrics:")
    print(f"[RESULTS] OA: {OA2:.4f} | AA: {AA_mean2:.4f} | Kappa: {Kappa2:.4f}")
    print(f"[RESULTS] Class-wise AA: {AA2}")

    print("\n[CONFIG] Training Parameters:")
    for k, v in vars(args).items():
        print(f"[CONFIG] {k}: {v}")

    # At the start of training
    torch.cuda.empty_cache()
    print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")

    # Inside training loop
    if epoch % 5 == 0:
        torch.cuda.empty_cache()
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
