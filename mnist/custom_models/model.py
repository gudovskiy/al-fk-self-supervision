import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# if you want to use kornia, uncomment these lines:
# sys.path.append("../")
# from kornia.geometry.transform.imgwarp import get_rotation_matrix2d, warp_affine

TH = 28
TW = TH
TC = 1
CL = 10

class NetMC(nn.Module):
    def __init__(self, l, c):
        super(NetMC, self).__init__()
        self.l = l
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, c)
        self.mpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = F.dropout(self.relu(self.mpool(self.conv1(x ))), p=0.25, training=True, inplace=True)
        x2 = F.dropout(self.relu(self.mpool(self.conv2(x1))), p=0.25, training=True, inplace=True)
        x3 = x2.view(-1, 320)
        x4 = F.dropout(self.relu(self.fc1(x3)), p=0.5, training=True, inplace=True)
        x = self.fc2(x4)
        # make a list for multiscale aggregation
        if   self.l==90:
            f = [x1, x2, x4, x]
        elif self.l==80:
            f = [x1, x2, x4]
        elif self.l==30:
            f = [x1, x2]
        elif self.l==20:
            f = [x2]
        elif self.l==10:
            f = [x]
        elif self.l==0:
            f = []
        else:
            print('Wrong descriptor length in model.py')
            sys.exit(0)
        #
        return x, f

class NetMSA(nn.Module):
    def __init__(self, l, c):
        super(NetMSA, self).__init__()
        self.l = l
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, c)
        self.mpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.mpool(self.conv1(x)))
        x2 = self.relu(self.mpool(self.conv2(x1)))
        x3 = x2.view(-1, 320)
        x4 = self.relu(self.fc1(x3))
        x = self.fc2(x4)
        # make a list for multiscale aggregation
        if   self.l==90:
            f = [x1, x2, x4, x]
        elif self.l==80:
            f = [x1, x2, x4]
        elif self.l==30:
            f = [x1, x2]
        elif self.l==20:
            f = [x2]
        elif self.l==10:
            f = [x]
        elif self.l==0:
            f = []
        else:
            print('Wrong descriptor length in model.py')
            sys.exit(0)
        #
        return x, f


def lenet(L=20, UNSUP=False, MC=False):
    #
    if UNSUP:
        model = NetMSA(l=L, c=4)
    elif MC:
        model = NetMC( l=L, c=CL)
    else:
        model = NetMSA(l=L, c=CL)
    #
    return model


def test(args, model, device, loader, epoch, unsup=False):
    model.eval()
    D = len(loader.dataset)
    if unsup:
        R  = 4 # 4 angles as in Gidaris18
        D *= R
        unsup_target = torch.tensor([0, 1, 2, 3], dtype=torch.long).to(device).unsqueeze(1) # 4x1
    #
    test_loss = 0.
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target, index) in enumerate(loader):
            data = data.to(device)
            if unsup: # unsupervised pretraining
                B = data.size(0)
                target = unsup_target.repeat(1, B).view(-1) # R*Bx1
                data = data.repeat(R, 1, 1, 1)
                data[1*B:2*B, ...] = torch.rot90(data[1*B:2*B, ...], 1, [3, 2]) # 90  deg
                data[2*B:3*B, ...] = torch.flip( data[2*B:3*B, ...],    [2, 3]) # 180 deg
                data[3*B:4*B, ...] = torch.rot90(data[3*B:4*B, ...], 1, [2, 3]) # 270 deg
            else: # supervised training
                target = target.to(device)
            #
            output, _ = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= D
    accuracy = 100. * correct / D
    print('\nTest Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        epoch, test_loss, correct, D, accuracy))

    return accuracy


def get_miss(args, model, device, loader):
    model.eval()
    D = len(loader.dataset)
    test_loss = 0
    correct = 0
    index_miss = torch.tensor([], dtype=torch.long).to(device)
    with torch.no_grad():
        for data, target, index in loader:
            indexT = index.long()
            data, target, indexT = data.to(device), target.to(device), indexT.to(device)
            output, _ = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max probability
            match = pred.eq(target.view_as(pred))
            mask = match.le(0)
            index_miss = torch.cat([index_miss, torch.masked_select(indexT.view(-1, 1), mask)])
            correct += match.sum().item()
    
    test_loss /= D
    accuracy = 100. * correct / D
    print('\nGet descr: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, D, accuracy))

    return index_miss


def get_miss_and_cm(args, model, device, loader):
    model.eval()
    D = len(loader.dataset)
    test_loss = 0
    correct = 0
    index_miss = torch.tensor([], dtype=torch.long).to(device)
    gt = torch.zeros(D, dtype=torch.long).to(device)
    pr = torch.zeros(D, dtype=torch.long).to(device)
    with torch.no_grad():
        for data, target, index in loader:
            indexT = index.long()
            data, target, indexT = data.to(device), target.to(device), indexT.to(device)
            output, _ = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max probability
            match = pred.eq(target.view_as(pred))
            mask = match.le(0)
            index_miss = torch.cat([index_miss, torch.masked_select(indexT.view(-1, 1), mask)])
            gt[index] = target.view(-1)
            pr[index] = pred.view(-1)
            correct += match.sum().item()
    
    test_loss /= D
    accuracy = 100. * correct / D
    print('\nGet descr: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, D, accuracy))

    return index_miss, gt, pr


def train(args, model, device, loader, optimizer, epoch, unsup=False):
    model.train()
    D = len(loader.dataset)
    exp_lr_scheduler(optimizer, epoch, lr_decay=args.lr_decay, lr_decay_epoch=args.lr_decay_epoch)
    if unsup:
        R  = 4 # 4 angles as in Gidaris18
        D *= R
        unsup_target = torch.tensor([0, 1, 2, 3], dtype=torch.long).to(device).unsqueeze(1) # 4x1
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    for batch_idx, (data, target, index) in enumerate(loader):
        data = data.to(device)
        if unsup: # unsupervised pretraining
            B = data.size(0)
            target = unsup_target.repeat(1, B).view(-1) # R*Bx1
            data = data.repeat(R, 1, 1, 1)
            data[1*B:2*B, ...] = torch.rot90(data[1*B:2*B, ...], 1, [3, 2]) # 90  deg
            data[2*B:3*B, ...] = torch.flip( data[2*B:3*B, ...],    [2, 3]) # 180 deg
            data[3*B:4*B, ...] = torch.rot90(data[3*B:4*B, ...], 1, [2, 3]) # 270 deg
        # supervised training
        else:
            target = target.to(device)
        #
        output, _ = model(data)
        loss = F.cross_entropy(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, lr = {:.6f}'.format(
                epoch, batch_idx * len(data), D,
                100. * batch_idx / D, loss.item(), lr))


def gen_mc(args, model, optimizer, device, loader, prefix, save_file):
    D = len(loader.dataset)
    K = args.sample_steps # or "T" in code
    if args.ensemble_size > 1:
        descr = torch.zeros(D, CL, 2)
    else:
        descr = torch.zeros(D)
    #
    model.eval()
    test_loss = 0
    correct = 0
    r = 1e-8 # regularization for log() to avoid nan
    #
    for batch_idx, (data, target, index) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        B = data.size(0) # batch size
        outputT = torch.zeros(B, CL, K).to(device)
        #
        for k in range(K):
            output, _ = model(data)
            outputT[:,:,k] = output.detach()
        # calc MC accuracy
        outputAve = torch.mean(outputT, 2)
        test_loss += F.cross_entropy(outputAve, target, reduction='sum').item() # sum up batch loss
        pred = outputAve.max(1, keepdim=True)[1] # get the index of the max probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        # MC
        p = F.softmax(outputT, dim=1) # BxCxT
        if args.ensemble_size > 1:
            if 'ent' in args.subtype_method:
                pT = torch.sum(p, 2)/K # BxC
                descr[index,:,0] = pT.detach().cpu()
            elif 'bald' in args.subtype_method:
                pT = torch.sum(p,                             2)/K # BxC
                pL = torch.sum(torch.mul(p, torch.log2(p+r)), 2)/K # BxC
                descr[index,:,0] = pT.detach().cpu()
                descr[index,:,1] = pL.detach().cpu()
            elif 'var' in args.subtype_method:
                pMax, _ = torch.max(p, 1) # BxT
                pMax = torch.unsqueeze(pMax, 1).repeat(1, p.size(1), 1) # BxCxT
                pEq = torch.eq(p, pMax).float() # BxCxT
                pSum = torch.sum(pEq, 2)/K # BxC
                descr[index,:,0] = pSum.detach().cpu()
            else:
                sys.exit(0)
                print('Wrong uncert with ensembles method!')
        else:
            if 'ent' in args.subtype_method:
                pT = torch.sum(p, 2)/K # BxC
                max_entropy = -torch.sum(torch.mul(pT, torch.log2(pT+r)), 1) # B
                descr[index] = max_entropy.detach().cpu()
            elif 'bald' in args.subtype_method:
                pT = torch.sum(p, 2)/K # BxC
                max_entropy = -torch.sum(torch.mul(pT, torch.log2(pT+r)), 1) # B
                bald = max_entropy + torch.sum(torch.sum(torch.mul(p, torch.log2(p+r)), 1), 1)/K # B
                descr[index] = bald.detach().cpu()
            elif 'std' in args.subtype_method:
                pS = torch.std(p, 2) # BxC
                mean_std = torch.mean(pS, 1) # B
                descr[index] = mean_std.detach().cpu()
            elif 'var' in args.subtype_method:
                pMax, _ = torch.max(p, 1)
                pMax = torch.unsqueeze(pMax, 1).repeat(1, p.size(1), 1)
                pEq = torch.eq(p, pMax).float()
                pSum = torch.sum(pEq, 2)/K
                fMax, _ = torch.max(pSum, 1)
                var_ratio = 1 - fMax
                descr[index] = var_ratio.detach().cpu()
            else:
                sys.exit(0)
                print('Wrong uncert without ensembles method!')
        #
        if index[0] % args.log_interval == 0:
            print('Generating MC for sample #', index[0])

    test_loss /= D
    accuracy = 100. * correct / D
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        prefix, test_loss, correct, D, accuracy))
    #
    torch.save(descr, save_file)


def gen_descr(args, model, optimizer, device, loader, prefix, save_file, val_file):
    D = len(loader.dataset)
    L = args.descriptor_length
    if (prefix == 'val') and ('map' in args.subtype_method):
        descr = torch.zeros(D, L, 4)
    elif (prefix == 'train') and ('map' in args.subtype_method):
        descr = torch.zeros(D, L, 2)
        val_descr = torch.load(val_file)
        fvec_val = val_descr[...,0].to(device) # MxL
        grad_val = val_descr[...,2].to(device) # MxL
        M = fvec_val.size(0)
    else:
        descr = torch.zeros(D, L, 2)
    #
    model.eval()
    gen_loss = 0.
    correct = 0
    pseudo_correct = 0
    eps = 1e-10 # small regularization constant
    r = 1e-6 # regularization for to avoid zero-vector noise
    #
    print('PREFIX/BATCH =', prefix, args.batch)
    if (args.subtype_method == 'none') or ('grad' in args.subtype_method):
        K = 1
    elif ('ent' in args.subtype_method) or ('ami' in args.subtype_method) or ('cor' in args.subtype_method):
        K = args.sample_steps
        #gamPath = torch.linspace(1.0/K, 1.0, steps=K) # gamma
        gamPath = torch.ones(K)
        #gamPath = torch.randn(K) / 5.0 + 1.0 # gamma [-?:1:?]
        #briPath = torch.randn(K) / 100.0 # brightness
        briPath = torch.randn(K, TC, TH, TW) / 100.0 # brightness
        angPath = torch.randn(K) * 5 # rotation angle in rads [+/-5 deg]
        #angPath = torch.linspace(-5, 5, steps=K) # rotation angle in rads [+/-5 deg]

        # define the rotation center
        center = torch.ones(1, 2)
        center[..., 0] = TW / 2  # x
        center[..., 1] = TH / 2  # y
        center = center.to(device)
        # define the scale factor
        scale = torch.ones(1).to(device)
        #
        gamPath = gamPath.to(device)
        briPath = briPath.to(device)
        angPath = angPath.to(device)
    else:
        print('Wrong Pseudo-Label Estimation Method!', args.subtype_method)
        sys.exit(0)

    # make a list for multiscale aggregation
    if   L == 90:
        I = [0, 10, 30, 80, 90]
    elif L == 80:
        I = [0, 10, 30, 80]
    elif L == 30:
        I = [0, 10, 30]
    elif L == 20:
        I = [0, 20]
    elif L == 10:
        I = [0, 10]
    elif L == 0:
        I = []
    else:
        print('Wrong descriptor length in model.py')
        sys.exit(0)

    if ('ent' in args.subtype_method) or ('ami' in args.subtype_method) or ('cor' in args.subtype_method):
        targetEst = torch.zeros(D, CL).to(device)
        for batch_idx, (data, target, index) in enumerate(loader):
            data = data.to(device)
            
            B = data.size(0)
            fM = torch.zeros((K,     B,  L)).to(device)
            gM = torch.zeros((K, CL, B,  L)).to(device)

            # generate samples
            for pi, pv in enumerate(gamPath):
                
                # set sample strategies
                gamma = gamPath[pi] # (1.0/gamPath[pi]) * torch.ones_like(data)
                const = briPath[pi]
                angle = angPath[pi]

                if 0: # Kornia
                    # compute the transformation matrix
                    M = get_rotation_matrix2d(center.repeat(B,1), angle.repeat(B), scale.repeat(B))
                    # apply the transformation to original image
                    data_rot = warp_affine(data, M, dsize=(TH, TW))
                else: # naive rotations
                    # Calculate rotation for each target pixel
                    y_mid = np.floor((data.size(2)) / 2.)
                    x_mid = np.floor((data.size(3)) / 2.)

                    # Use meshgrid for pixel coords
                    xv, yv = torch.meshgrid(torch.arange(data.size(2)), torch.arange(data.size(3)))
                    xv = xv.contiguous().to(device)
                    yv = yv.contiguous().to(device)
                    src_ind = torch.cat((
                        (xv.float() - x_mid).view(-1, 1),
                        (yv.float() - y_mid).view(-1, 1)),
                        dim=1
                    ).to(device)
                    rot = torch.zeros(2, 2).to(device)
                    rot[0,0] = torch.cos(angle)
                    rot[0,1] = torch.sin(angle)
                    rot[1,0] =-torch.sin(angle)
                    rot[1,1] = torch.cos(angle)                    
                    # Calculate indices using rotation matrix
                    src_ind = torch.matmul(src_ind, rot.t())
                    src_ind = torch.round(src_ind)
                    src_ind += torch.tensor([[x_mid, y_mid]]).to(device)
                    # Set out of bounds indices to limits
                    src_ind[src_ind < 0] = 0.
                    src_ind[:, 0][src_ind[:, 0] >= data.size(2)] = float(data.size(2)) - 1
                    src_ind[:, 1][src_ind[:, 1] >= data.size(3)] = float(data.size(3)) - 1
                    #
                    data_rot = torch.zeros_like(data)
                    src_index = src_ind.long()
                    data_rot[:, :, xv.view(-1), yv.view(-1)] = data[:, :, src_index[:, 0], src_index[:, 1]]

                # forward
                if K == 1:
                    yV, fL = model(data)
                else:
                    yV, fL = model(data_rot + const)
                    #yV, fL = model(torch.pow(data_rot, gamma) + const)
                
                # generate feature vector
                fV = torch.tensor([]).to(device)
                for j in range(len(fL)):
                    if fL[j].dim() > 2:
                        f = F.avg_pool2d(fL[j].detach(), fL[j].size(-1)).view(fL[j].size(0), -1)
                    else:
                        f = fL[j].detach()
                    #
                    fV = torch.cat([fV, f], dim=1) # BxL
                #
                fM[pi] = fV
                #
                for ci in range(CL): # sweep thru every class
                    cV = ci * torch.ones(B, dtype=torch.long).to(device)
                    loss = F.cross_entropy(yV, cV, reduction='sum')
                    optimizer.zero_grad()
                    gL = torch.autograd.grad(loss, fL, retain_graph=True)
                    gV = torch.tensor([]).to(device)
                    for j in range(len(gL)):
                        if gL[j].dim() > 2:
                            g = F.avg_pool2d(gL[j].detach(), gL[j].size(-1)).view(gL[j].size(0), -1)
                        else:
                            g = gL[j].detach()
                        #
                        gV = torch.cat([gV, g], dim=1) # 1xL
                    #
                    gM[pi, ci] = gV
                #
            if args.subtype_method == 'cor':
                # trace of cross-covariance
                fT = fM.permute(1,2,0)
                gT = gM.permute(2,1,3,0)

                fT = fT.reshape(B, -1,  1) # BxL*Tx1
                gT = gT.reshape(B, CL, -1) # BxCLxL*T

                cov = torch.bmm( gT, fT ).squeeze() # BxCLxL*T * BxL*Tx1 = BxCL
                targetEst[index] = cov
            #
            elif ('Diag' in args.subtype_method):
                # cross-covariances
                fT = fM.permute(1,2,0) # BxLxT
                gT = gM.permute(1,2,3,0)#.repeat(1,1,1,T) # CLxBxLxT
                ami = torch.zeros(B, CL).to(device)
                #
                if 'ami' in args.subtype_method:
                    covZZ = torch.bmm( fT, torch.transpose(fT, 1,2) ) # BxLxL
                    dZZ = torch.diagonal(covZZ, offset=0, dim1=1, dim2=2) # BxL
                    logZZ = torch.log(dZZ) # BxL
                    ami += torch.sum(logZZ, 1).unsqueeze(-1).repeat(1,CL) # BxCL
                for cl in range(CL):
                    covGG = torch.bmm( gT[cl], torch.transpose(gT[cl], 1,2) ) # BxLxL
                    dGG = torch.diagonal(covGG, offset=0, dim1=1, dim2=2) # BxL
                    logGG = torch.log(dGG) # BxL
                    ami[:,cl] += torch.sum(logGG, 1)
                    if 'ami' in args.subtype_method:
                        cT = torch.cat([fT, gT[cl]], dim=1)
                        covCC = torch.bmm( cT, torch.transpose(cT, 1,2) ) # Bx2Lx2L
                        dCC = torch.diagonal(covCC, offset=0, dim1=1, dim2=2) # Bx2L
                        logCC = torch.log(dCC) # Bx2L
                        ami[:,cl] -= torch.sum(logCC, 1)
                #
                targetEst[index] = ami
            elif ('Full' in args.subtype_method):
                # cross-covariances
                fT = fM.permute(1,2,0).double() # BxLxT
                gT = gM.permute(1,2,3,0).double()#.repeat(1,1,1,T) # CLxBxLxT
                
                eps = 1e-8 # small regularization constant
                regD = (1e-44 * torch.ones(1)).double().to(device) # very small constant
                ami = torch.zeros(B, CL).double().to(device)
                #
                if 'ami' in args.subtype_method:
                    covZZ = torch.bmm( fT, torch.transpose(fT, 1,2) ) + eps * torch.eye(1*L).double().to(device)
                    lZZ = torch.cholesky(covZZ) # BxLxL
                    dZZ = torch.diagonal(lZZ, offset=0, dim1=1, dim2=2)
                    dFlatZZ = dZZ.flatten()                    
                    pdZZ = regD * torch.ones_like(dFlatZZ)
                    isfinZZ = torch.isfinite(dFlatZZ)
                    idxfinZZ = isfinZZ.nonzero()
                    pdZZ[idxfinZZ] = dFlatZZ[idxfinZZ]
                    dSqrZZ = pdZZ.reshape(-1,1*L)**2
                    logSqrZZ = torch.log(dSqrZZ)
                    ami += torch.sum(logSqrZZ, 1).unsqueeze(-1).repeat(1,CL) # BxCL
                for cl in range(CL):
                    covGG = torch.bmm( gT[cl], torch.transpose(gT[cl], 1,2) ) + eps * torch.eye(1*L).double().to(device)
                    lGG = torch.cholesky(covGG) # BxLxL
                    dGG = torch.diagonal(lGG, offset=0, dim1=1, dim2=2)
                    dFlatGG = dGG.flatten()                    
                    pdGG = regD * torch.ones_like(dFlatGG)
                    isfinGG = torch.isfinite(dFlatGG)
                    idxfinGG = isfinGG.nonzero()
                    pdGG[idxfinGG] = dFlatGG[idxfinGG]
                    dSqrGG = pdGG.reshape(-1,1*L)**2
                    logSqrGG = torch.log(dSqrGG)
                    ami[:,cl] += torch.sum(logSqrGG, 1)
                    if 'ami' in args.subtype_method:
                        cT = torch.cat([fT, gT[cl]], dim=1)
                        covCC = torch.bmm( cT, torch.transpose(cT, 1,2) ) + eps * torch.eye(2*L).double().to(device)
                        lCC = torch.cholesky(covCC) # Bx2Lx2L
                        dCC = torch.diagonal(lCC, offset=0, dim1=1, dim2=2)
                        dFlatCC = dCC.flatten()
                        pdCC = regD * torch.ones_like(dFlatCC)
                        isfinCC = torch.isfinite(dFlatCC)
                        idxfinCC = isfinCC.nonzero()
                        pdCC[idxfinCC] = dFlatCC[idxfinCC]
                        dSqrCC = pdCC.reshape(-1,2*L)**2
                        logSqrCC = torch.log(dSqrCC)
                        ami[:,cl] -= torch.sum(logSqrCC, 1)
                        #
                targetEst[index] = ami.float()
    #
    for batch_idx, (data, target, index) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        B = data.size(0)
        # forward
        output, feat_vec = model(data)
        # accuracy
        pred = (F.softmax(output, dim=1)).max(1, keepdim=True)[1] # get the index of the max probability
        eq = pred.eq(target.view_as(pred))
        correct += eq.sum().item()
        # generate feature vector
        fV = torch.tensor([]).to(device)
        for j in range(len(feat_vec)):
            if feat_vec[j].dim() > 2:
                f = F.avg_pool2d(feat_vec[j].detach(), feat_vec[j].size(-1)).view(feat_vec[j].size(0), -1)
            else:
                f = feat_vec[j].detach()
            #
            fV = torch.cat([fV, f], dim=1) # BxL
        # generate label
        if ('map' in args.subtype_method):
            q = 0
            if prefix == 'train': # train
                # estimate label first by finding similar example in val
                label = torch.argmax(output, dim=1) # pseudo-label generated by naive prob maximization
                loss = F.cross_entropy(output, label, reduction='sum')
                optimizer.zero_grad()
                feat_grad = torch.autograd.grad(loss, feat_vec, retain_graph=True)
                fG = torch.tensor([]).to(device)
                for j in range(len(feat_vec)):
                    if feat_vec[j].dim() > 2:
                        g = F.avg_pool2d(feat_grad[j].detach(), feat_grad[j].size(-1)).view(feat_grad[j].size(0), -1)
                    else:
                        g = feat_grad[j].detach()
                    #
                    fG = torch.cat([fG, g], dim=1) # 1xL
                #
                fA, gA = fV, fG
                fB, gB = fvec_val, grad_val

                for i in range(1, len(I)): # sum of distance metrics for multi-scale case
                    fA[:, I[i-1]:I[i]] -= torch.mean(fA[:, I[i-1]:I[i]], 1, keepdim=True)
                    fB[:, I[i-1]:I[i]] -= torch.mean(fB[:, I[i-1]:I[i]], 1, keepdim=True)
                    gA[:, I[i-1]:I[i]] -= torch.mean(gA[:, I[i-1]:I[i]], 1, keepdim=True)
                    gB[:, I[i-1]:I[i]] -= torch.mean(gB[:, I[i-1]:I[i]], 1, keepdim=True)
                    fA[:, I[i-1]:I[i]] /= (torch.std(fA[:, I[i-1]:I[i]], 1, keepdim=True) + eps)
                    fB[:, I[i-1]:I[i]] /= (torch.std(fB[:, I[i-1]:I[i]], 1, keepdim=True) + eps)
                    gA[:, I[i-1]:I[i]] /= (torch.std(gA[:, I[i-1]:I[i]], 1, keepdim=True) + eps)
                    gB[:, I[i-1]:I[i]] /= (torch.std(gB[:, I[i-1]:I[i]], 1, keepdim=True) + eps)
                #
                fSim = torch.mm(fA, fB.t()) # BxL * LxM = BxM
                gSim = torch.mm(gA, gB.t()) # BxL * LxM = BxM
                #
                if   args.subtype_method == 'gradFmap':
                    sim = fSim
                elif args.subtype_method == 'gradGmap':
                    sim = fSim + gSim
                else:
                    print('Wrong map gradXmap pseudo labels!')
                    sys.exit(0)
                #
                sim_idx = torch.argmax(sim, dim=1) # Bx1
                labelEst = val_descr[sim_idx,0,3].long().to(device)
                '''K = 3 # experimental: search in local neighborhood
                sim_val, sim_idx = torch.topk(sim, K, dim=1, largest=True) # BxK
                sim_idx_flat = sim_idx.reshape(-1) # BKx1
                labelSel = val_descr[sim_idx_flat,0,3].reshape(B,K).long().to(device) # BxK
                labelHist = torch.zeros((B,CL)).long().to(device)
                for cl in range(CL):
                    labelHist[:, cl] = torch.sum(torch.eq(labelSel, cl), dim=1).long()
                #
                hist_idx = torch.argmax(labelHist, dim=1) # Bx1
                labelEst = hist_idx # Bx1'''
                loss = F.cross_entropy(output, labelEst, reduction='sum')
                gen_loss += loss.item() # sum up batch loss
                pseudo_correct += labelEst.eq(target).sum().item()
                optimizer.zero_grad()
                feat_grad = torch.autograd.grad(loss, feat_vec, retain_graph=True)
                fG = torch.tensor([]).to(device)
                for j in range(len(feat_vec)):
                    if feat_vec[j].dim() > 2:
                        g = F.avg_pool2d(feat_grad[j].detach(), feat_grad[j].size(-1)).view(feat_grad[j].size(0), -1)
                    else:
                        g = feat_grad[j].detach()
                    #
                    fG = torch.cat([fG, g], dim=1) # 1xL
                #
                descr[index, :, 1] = fG.cpu()
            elif prefix == 'val':
                label = torch.stack([target, torch.argmax(output, dim=1)])
                for l in range(0,2):
                    loss = F.cross_entropy(output, label[l], reduction='sum')
                    if l==0:
                        gen_loss += loss.item() # sum up batch loss
                        pseudo_correct += label[l].eq(target).sum().item()
                    optimizer.zero_grad()
                    feat_grad = torch.autograd.grad(loss, feat_vec, retain_graph=True)
                    fG = torch.tensor([]).to(device)
                    for j in range(len(feat_vec)):
                        if feat_vec[j].dim() > 2:
                            g = F.avg_pool2d(feat_grad[j].detach(), feat_grad[j].size(-1)).view(feat_grad[j].size(0), -1)
                        else:
                            g = feat_grad[j].detach()
                        #
                        fG = torch.cat([fG, g], dim=1) # 1xL
                    descr[index, :, l+1] = fG.cpu()
                descr[index, 0, 3] = target.float().cpu()
                #
            if index[0] % args.log_interval == 0:
                print('Generating CL descr for sample #', index[0], q)
        else:
            if ('Abl' in args.subtype_method):
                label = target # true label
                q = 1
            elif (args.subtype_method == 'none') or (args.subtype_method == 'grad'):
                label = torch.argmax(output, dim=1) # pseudo-label generated by naive prob maximization
                q = 0
            elif ('ent' in args.subtype_method) or ('ami' in args.subtype_method) or ('cor' in args.subtype_method):
                label = torch.argmax(targetEst[index], dim=1) # pseudo-label generated by more advanced methods
                q = 0
            else:
                print('Wrong pseudo label estimation method!')
                sys.exit(0)
            #
            loss = F.cross_entropy(output, label, reduction='sum')
            gen_loss += loss.item() # sum up batch loss
            pseudo_correct += label.eq(target).sum().item()
            # Fisher score
            if 'none' not in args.subtype_method:
                optimizer.zero_grad()
                feat_grad = torch.autograd.grad(loss, feat_vec)
                fG = torch.tensor([]).to(device)
                for j in range(len(feat_vec)):
                    if feat_vec[j].dim() > 2:
                        g = F.avg_pool2d(feat_grad[j].detach(), feat_grad[j].size(-1)).view(feat_grad[j].size(0), -1)
                    else:
                        g = feat_grad[j].detach()
                    #
                    fG = torch.cat([fG, g], dim=1) # 1xL
                #
                descr[index, :, 1] = fG.cpu()
                #
            if index[0] % args.log_interval == 0:
                print('Generating descr for sample #', index[0], q)
        #
        descr[index, :, 0] = fV.cpu()
    gen_loss /= D
    accuracy = 100. * correct / D
    pseudo_accuracy = 100. * pseudo_correct / D
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Pseudo Acc.: {:.2f}%\n'.format(
        prefix, gen_loss, correct, D, accuracy, pseudo_accuracy))
    #
    torch.save(descr, save_file)


def save(model, acc, epoch, checkpoint_file):
    state = {   'state_dict': model.state_dict(),
                'acc': acc,
                'epoch': epoch}

    torch.save(state, checkpoint_file)


def exp_lr_scheduler(optimizer, epoch, lr_decay, lr_decay_epoch):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch:
        return optimizer
    
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer
