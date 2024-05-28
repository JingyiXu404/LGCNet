import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info
import basicblock as B
from utils_image import *
from utils import *
import time
from math import ceil
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).expand(3,1,3,3)
        kernely = torch.FloatTensor(kernely).expand(3,1,3,3)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)
def laplace_weight(N,C_in,C_out):
    kernelx = [[-1, -1, -1],
                [-1, 8 , -1],
                [-1, -1, -1]]
    kernelx = torch.FloatTensor(kernelx).expand(N,C_out,C_in,3,3)
    weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
    return weightx

class HeadNet(nn.Module):
    def __init__(self, in_nc, nc_x, out_nc, d_size):
        super(HeadNet, self).__init__()
        self.head_z = nn.Sequential(
            nn.Conv2d(in_nc * 2,nc_x[0],d_size,padding=(d_size - 1) // 2,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nc_x[0], out_nc, d_size, padding=1, bias=False))
        self.head_m = nn.Sequential(
            nn.Conv2d(in_nc * 2, nc_x[0], d_size, padding=(d_size - 1) // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nc_x[0], out_nc, d_size, padding=1, bias=False))
        self.head_u = nn.Sequential(
            nn.Conv2d(in_nc * 2, nc_x[0], d_size, padding=(d_size - 1) // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nc_x[0], out_nc, d_size, padding=1, bias=False))
        self.head_a = nn.Sequential(
            nn.Conv2d(in_nc * 2, nc_x[0], d_size, padding=(d_size - 1) // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nc_x[0], out_nc, d_size, padding=1, bias=False))
        self.head_w = nn.Sequential(
            nn.Conv2d(in_nc * 2, nc_x[0], d_size, padding=(d_size - 1) // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nc_x[0], out_nc, d_size, padding=1, bias=False))
    def forward(self, x,y):
        z = self.head_z(torch.cat([x,y], dim=1))
        m = self.head_m(torch.cat([x,y], dim=1))
        u = self.head_u(torch.cat([x, y], dim=1))
        a = self.head_a(torch.cat([x, y], dim=1))
        w = self.head_w(torch.cat([x, y], dim=1))
        return z,a,w,m,u
class HyPaNet(nn.Module):
    def __init__(self,in_nc: int = 1,nc: int = 256,out_nc: int = 3,):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_nc*2, nc, 1, padding=0, bias=True), nn.Sigmoid(),
            nn.Conv2d(nc, out_nc, 1, padding=1, bias=True), nn.Softplus())
    def forward(self, x, y):
        x = torch.cat([x,y], dim=1)
        x = (x - 0.098) / 0.0566
        x = self.mlp(x) + 1e-6
        return x
class Update_z(nn.Module):
    def __init__(self):
        super(Update_z, self).__init__()

    def forward(self, X,Y,Z,A,W,M,U,S, lambda1=0, eta=0, beta=0, size_x=0):
        """
            XYZAWMU: N, 1, C_in, H, W, 2  torch.Size([10, 1, 3, 128, 65, 2])
            S: N, C_out, C_in, H, W, 2 torch.Size([10, 3, 3, 128, 65, 2])
            lambda1: N, H, W, 1
            eta: N, H, W, 1
            rho: N, H, W, 1
        """

        lambda1 = reshape_params3(lambda1,Z)
        eta = reshape_params3(eta,Z)
        beta = reshape_params3(beta,Z)
        lr = lambda1/beta
        er = eta/beta
        ler = lr + er
        _S = cconj(S)
        _Z = cmul(S,(cmul(_S,Y)+A-W)) + lr * X + er * (M-U)
        factor1 = _Z / ler
        numerator = cmul(_S, _Z)
        denominator = csum(ler * cmul(_S, S), ler.squeeze(-1)**2)
        factor2 = cmul(S, cdiv(numerator, denominator))
        Z = (factor1 - factor2).mean(1)
        return torch.irfft(Z, 2, signal_sizes=list(size_x))
class Update_aw(nn.Module):
    def __init__(self,C_in):
        super(Update_aw, self).__init__()
        kernel = [[-1, -1, -1],
                  [-1, 8, -1],
                  [-1, -1, -1]]
        kernel = torch.FloatTensor(kernel).expand(1, C_in, 3, 3)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
    def forward(self,z,y,w,lam):
        lam = reshape_params4(lam, z)
        out_z = F.conv2d(z, self.weight, padding=1)
        out_y = F.conv2d(y, self.weight, padding=1)
        out_a = out_z - out_y + w
        out_a = torch.mul(torch.sign(out_a), F.relu(torch.abs(out_a) - lam))
        out_w = w + out_z - out_y - out_a
        return out_a,out_w
class Update_m(nn.Module):
    def __init__(self,
                 in_nc = 3,
                 nc_x= [64, 128, 256, 512],
                 nb= 4):
        super(Update_m, self).__init__()
        self.encode = nn.Sequential(
            B.conv(in_nc+3, nc_x[0], bias=False, mode='C'),
            B.conv(nc_x[0], nc_x[0], bias=False, mode='R'),
            B.conv(nc_x[0], nc_x[0], bias=False, mode='C'),)
        self.m_down1 = B.sequential(
            *[
                B.ResBlock(nc_x[0], nc_x[0], bias=False, mode='CRC')
                for _ in range(nb)
            ], B.downsample_strideconv(nc_x[0], nc_x[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(
            *[
                B.ResBlock(nc_x[1], nc_x[1], bias=False, mode='CRC')
                for _ in range(nb)
            ], B.downsample_strideconv(nc_x[1], nc_x[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(
            *[
                B.ResBlock(nc_x[2], nc_x[2], bias=False, mode='CRC')
                for _ in range(nb)
            ], B.downsample_strideconv(nc_x[2], nc_x[3], bias=False, mode='2'))

        self.m_body = B.sequential(*[
            B.ResBlock(nc_x[-1], nc_x[-1], bias=False, mode='CRC')
            for _ in range(nb)
        ])

        self.m_up3 = B.sequential(
            B.upsample_convtranspose(nc_x[3], nc_x[2], bias=False, mode='2'),
            *[
                B.ResBlock(nc_x[2], nc_x[2], bias=False, mode='CRC')
                for _ in range(nb)
            ])
        self.m_up2 = B.sequential(
            B.upsample_convtranspose(nc_x[2], nc_x[1], bias=False, mode='2'),
            *[
                B.ResBlock(nc_x[1], nc_x[1], bias=False, mode='CRC')
                for _ in range(nb)
            ])
        self.m_up1 = B.sequential(
            B.upsample_convtranspose(nc_x[1], nc_x[0], bias=False, mode='2'),
            *[
                B.ResBlock(nc_x[0], nc_x[0], bias=False, mode='CRC')
                for _ in range(nb)
            ])

        self.m_tail = B.conv(nc_x[0], in_nc, bias=False, mode='C')
    def forward(self, x,gamma):
        gamma = reshape_params4(gamma, x)
        x0 = x
        x1 = self.encode(torch.cat([x, gamma], dim=1))
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1) + x0
        return x
class Stage(nn.Module):
    # for single input, one stage
    def __init__(self, in_nc=3, nc_x=[64, 128, 256, 512],nb=4):
        super(Stage, self).__init__()
        self.up_m = Update_m(in_nc=in_nc, nc_x=nc_x, nb=nb)
        self.up_z = Update_z()
        self.up_aw = Update_aw(C_in = in_nc)

    def forward(self, z, y, x, a, w, m, u, lambda1=0, eta=0, rho=0, beta=0, gamma=0):
        """
                    z, y, x, a, w, m, u: N, C_in, H, W torch.Size([10, 3, 128, 128])
                    s: N, C_out, C_in, d_size, d_size
                    lambda1/eta: 1, 1, 1, 1
                    reg: float
                """
        # get laplace operator
        size_x = np.array(list(x.shape[-2:]))
        N = x.shape[0]
        C_in = x.shape[1]
        s = laplace_weight(N,C_in,C_in)
        # FFT
        X,Y,Z,A,W,M,U,S = self.rfft_xd(z, y, x, a, w, m, u, s)

        # update z
        z = self.up_z(X,Y,Z,A,W,M,U,S, lambda1=lambda1, eta=eta, beta=beta,size_x=size_x)
        # update a and w
        a,w = self.up_aw(z,y,w,lam=rho/beta)
        # update m
        m = self.up_m(z+u, eta/gamma)
        # update u
        u = u + z -m
        return z, a, w, m, u

    def rfft_xd(self, z, y, x, a, w, m, u, s):
        X = torch.rfft(x, 2).unsqueeze(1)
        Y = torch.rfft(y, 2).unsqueeze(1)
        Z = torch.rfft(z, 2).unsqueeze(1)
        A = torch.rfft(a, 2).unsqueeze(1)
        W = torch.rfft(w, 2).unsqueeze(1)
        M = torch.rfft(m, 2).unsqueeze(1)
        U = torch.rfft(u, 2).unsqueeze(1)
        S = p2o(s, z.shape[-2:])
        return X,Y,Z,A,W,M,U,S
class MDN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_of_layers=3,nc_x=[64, 128, 256, 512],nb=4,d_size=3):
        super(MDN, self).__init__()
        self.num_of_layers = num_of_layers
        self.head = HeadNet(in_nc=in_channels, nc_x=nc_x, out_nc=out_channels, d_size=d_size)
        self.hypa_list: nn.ModuleList = nn.ModuleList()
        self.update: nn.ModuleList = nn.ModuleList()
        for _ in range(num_of_layers):
            # self.hypa_list.append(HyPaNet(in_nc=in_channels, out_nc=3))
            self.hypa_list.append(HyPaNet(in_nc=in_channels, out_nc=15))
            self.update.append(Stage(in_nc=in_channels, nc_x=nc_x,nb=nb))
    def forward(self, xy):
        c = xy.shape[1]
        x = xy[:, :3, :, :]
        y = xy[:, 3:c, :, :]
        h, w = y.size()[-2:]
        paddingBottom = int(ceil(h / 8) * 8 - h)
        paddingRight = int(ceil(w / 8) * 8 - w)
        y = F.pad(y, [0, paddingRight, 0, paddingBottom], mode='circular')
        x = F.pad(x, [0, paddingRight, 0, paddingBottom], mode='circular')

        z,a,w,m,u = self.head(x,y) # initialize z0,a0,w0,m0,u0
        preds = []
        preds_else = [[a,w,m,u]]
        for i in range(self.num_of_layers):
            hypas = self.hypa_list[i](z,y)
            lambda1 = hypas[:, 0:3].unsqueeze(-1)
            eta = hypas[:, 3:6].unsqueeze(-1)
            rho = hypas[:, 6:9].unsqueeze(-1)
            beta = hypas[:, 9:12].unsqueeze(-1)
            gamma = hypas[:, 12:15].unsqueeze(-1)
            z, a, w, m, u = self.update[i](z,y,x,a,w,m,u,lambda1,eta,rho,beta,gamma)#z, y, x, a, w, m, u,lambda1, eta, rho
            preds.append(z)
            preds_else.append([a,w,m,u])
        return preds,preds_else

if __name__ == '__main__':
    net = MDN()
    # print(net)
    x = torch.rand([10, 3, 128, 128])
    y = torch.rand([10, 3, 128, 128])
    xy = torch.rand([10, 6, 128, 128])
    # z = torch.rand([10, 3, 128, 128])
    # a = torch.rand([10, 3, 128, 128])
    # w = torch.rand([10, 3, 128, 128])
    # m = torch.rand([10, 3, 128, 128])
    # u = torch.rand([10, 3, 128, 128])
    # a_out = net(xy)
    # print(a_out.shape)
    macs, params = get_model_complexity_info(net, (6, 128,128), as_strings=True,print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
