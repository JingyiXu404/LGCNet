import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import platform
from time import *
from tqdm import tqdm, trange
import torch.cuda
from dataset import FNFDataset
from torchvision.transforms import transforms
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib
import argparse
import warnings
import psnr
import math
import cv2
import freeze
warnings.filterwarnings(action='ignore')
matplotlib.use('agg')
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from MDN import MDN as Net
def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False
def padding_size(x, d):
    x = x + 2
    return math.ceil(x / d) * d - x
def pad(img):
    h, w = img.shape[2], img.shape[3]
    h_psz = padding_size(h, 4)
    w_psz = padding_size(w, 4)
    padding = torch.nn.ReflectionPad2d((0, w_psz, 0, h_psz))
    img = padding(img)
    return img
def data_process_npy(data_in):
    data_out=data_in.detach().float().cpu().numpy()
    data_out=np.transpose(data_out,(0,2,3,1))
    data_out = data_out.squeeze()
    return data_out*255

def get_overall_run_params():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size-train", dest="batch_size_train", default=32, type=int)
    parser.add_argument("--batch-size-test", dest="batch_size_test", default=1, type=int)
    parser.add_argument("--sigma", dest="sigma", default=25, type=int)
    parser.add_argument("--lr", dest="lr", default=1 * 1e-4, type=float, help="learning rate")
    parser.add_argument("--epoch", dest="EPOCH", default=10000, type=int, help="epochs")
    parser.add_argument("--phase", dest="phase", default='test', type=str, help="train or test")
    parser.add_argument("--dataset", dest="dataset", default='FAID', type=str, help="FAID, MID, DPD")
    parser.add_argument("--patch-size", dest="patch_size", default=128, type=int, help="size of train and test dataset")
    parser.add_argument("--gpu", dest="gpu", default=False, type=str2bool)
    parser.add_argument("--iter", dest="iter", default=3, type=int)
    params = parser.parse_args()
    return params
def cal_multi_loss(lossfn, preds,gt):
        losses = None
        for i, pred in enumerate(preds):
            loss = lossfn(pred, gt)
            if i != len(preds) - 1:
                loss *= (1 / (len(preds) - 1))
            if i == 0:
                losses = loss
            else:
                losses += loss
        return losses
def load_network(load_path,network,iter):
    if isinstance(network, nn.DataParallel):
        network = network.module
    network.head.load_state_dict(torch.load(load_path + 'head.pth'), strict=True)

    for i in range(iter):
        state_dict_x = torch.load(load_path + 'up_m_' + str(i) + '.pth')
        network.update[i].up_m.load_state_dict(state_dict_x, strict=True)
        state_dict_hypa = torch.load(load_path + 'hypa_' +str(i)+ '.pth')
        network.hypa_list[i].load_state_dict(state_dict_hypa, strict=True)

def save(save_dir, net,iter):
    if isinstance(net, nn.DataParallel):
        net = net.module
    save_network(save_dir, net.head, 'head')
    for i in range(iter):
        save_network(save_dir,net.update[i].up_m,'up_m_' + str(i))
        save_network(save_dir, net.hypa_list[i], 'hypa_' +str(i))
def save_best(save_dir, net,iter):
    if isinstance(net, nn.DataParallel):
        net = net.module
    save_network(save_dir, net.head, 'head',True)
    for i in range(iter):
        save_network(save_dir,net.update[i].up_m,'up_m_' + str(i),True)
        save_network(save_dir, net.hypa_list[i], 'hypa_' +str(i),True)
def save_network(save_dir,network, network_label,best=False):
        filename = '{}.pth'.format(network_label)
        if best:
            os.makedirs(save_dir+'/best/', exist_ok=True)
            save_path = os.path.join(save_dir+'/best/', filename)
        else:
            save_path = os.path.join(save_dir, filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path, _use_new_zipfile_serialization=False)

def get_freeze_iter1(net):
    if isinstance(net, nn.DataParallel):
        net = net.module
    freeze.freeze_by_names(net, ('head'))
    freeze.freeze_by_names(net.hypa_list, ('0'))
def get_freeze_iter2(net):
    if isinstance(net, nn.DataParallel):
        net = net.module
    freeze.freeze_by_names(net.update, ('0'))
    freeze.freeze_by_names(net.hypa_list, ('1'))
def get_freeze_iter3(net):
    if isinstance(net, nn.DataParallel):
        net = net.module
    freeze.freeze_by_names(net.update, ('1'))
    freeze.freeze_by_names(net.hypa_list, ('2'))
def process_networks(params):
    # "cuda" if torch.cuda.is_available() else "cpu" instead of that I force to use cuda here
    device = torch.device("cuda")
    logging.info('Using device "{}"'.format(device))
    n_layer, n_feat = 7, 64
    EPOCH = params.EPOCH
    phase = params.phase
    sigma = params.sigma
    ITER = params.iter
    model_ft = Net(num_of_layers=ITER)

    if params.gpu:
        model_ft = nn.DataParallel(model_ft).to(device)
    model_ft = model_ft.to(device)

    if not os.path.exists('./Results_Images/'+params.dataset+'_N'+str(sigma)+'/'):
        os.makedirs('./Results_Images/'+params.dataset+'_N'+str(sigma)+'/')

    print("loading dataset-------------------------------------")
    root_path_test_1 = './dataset/'+params.dataset+'/test_nonflash/'
    root_path_test_2 = './dataset/'+params.dataset+'/test_flash/'

    dr_dataset_test = FNFDataset(root1=root_path_test_1, root2=root_path_test_2, sigma=sigma,patch_size=params.patch_size, normalization=False, transform=False, type='test')
    test_loader = DataLoader(dr_dataset_test, batch_size=params.batch_size_test, num_workers=0, shuffle=False)
    data_loaders = {'test': test_loader}

    if phase == 'test':
        if os.path.exists('./Results_models/'+params.dataset+'_N' + str(sigma) + '/head.pth'):
            print("loading model-------------------------------------")
            load_network('./Results_models/'+params.dataset+'_N' + str(sigma) + '/', model_ft, ITER)
        with torch.no_grad():
            val_bar = tqdm(data_loaders['test'])
            y = []
            pred = []
            output = []
            filename = []
            label = []
            batch_ind = 0
            for target, guide, gt in val_bar:
                batch_ind += 1
                model_ft = model_ft.eval()
                noise = data_process_npy(target)
                target = target.to(device)
                guide = guide.to(device)
                gt = gt.to(device)
                h, w = gt.size()[-2:]
                output,outelse = model_ft(torch.cat([target, guide], dim=1))

                gt = data_process_npy(gt)
                y = np.append(y, gt)

                for i in range(len(output)):
                    if i < len(output)-1:
                        continue
                    zi = data_process_npy(output[i][...,:h,:w])
                    path_output = './Results_Images/'+params.dataset+'_N' + str(sigma) + '/' + str(batch_ind) + '_z' + str(i + 1) + '.png'
                    cv2.imwrite(path_output, zi)
                pred = np.append(pred, zi)
            PSNR = psnr.psnr(y, pred)

            print("Test: PSNR=%.2fdB" % PSNR)


if __name__ == '__main__':
    params = get_overall_run_params()
    process_networks(params)