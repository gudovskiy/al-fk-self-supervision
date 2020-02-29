import os, time
import torch
import torch.nn.functional as F

TH = 224
TW = TH
TC = 3
CL = 1000
V = 50000

def validate(args, loader, model, criterion, unsup=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    loader_len = int(loader._size / args.batch_size)
    B = args.batch_size
    topK = (1, 5)
    if unsup:
        topK = (1, 1) # only 4 classes
        R  = 4 # 4 angles as in Gidaris18
        unsup_target = torch.tensor([0, 1, 2, 3], dtype=torch.long).cuda().unsqueeze(1) # 4x1
    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(loader):
            input = data[0]["data"]
            input = input.cuda()
            # unsupervised pretraining
            if unsup:
                target = unsup_target.repeat(1, B).view(-1) # R*Bx1
                input = input.repeat(R, 1, 1, 1)
                input[1*B:2*B, ...] = torch.rot90(input[1*B:2*B, ...], 1, [3, 2]) # 90  deg
                input[2*B:3*B, ...] = torch.flip( input[2*B:3*B, ...],    [2, 3]) # 180 deg
                input[3*B:4*B, ...] = torch.rot90(input[3*B:4*B, ...], 1, [2, 3]) # 270 deg
            # supervised training
            else:
                target = data[0]["label"].squeeze().cuda().long()            
            #
            target = target.cuda(async=True)
            # compute output
            output, _ = model(input)
            loss = criterion(output, target)
            
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=topK)
            reduced_loss = loss.data
            losses.update(to_python_float(reduced_loss), B)
            top1.update(to_python_float(prec1), B)
            top5.update(to_python_float(prec5), B)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.local_rank == 0 and i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      #'Speed {2:.3f} ({3:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, loader_len,
                       args.world_size * args.batch_size / batch_time.val,
                       args.world_size * args.batch_size / batch_time.avg,
                       batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return [top1.avg, top5.avg]


def get_miss(args, loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    loader_len = int(loader._size / args.batch_size)
    B = args.batch_size
    topK = (1, 5)
    index_miss = torch.tensor([], dtype=torch.long).cuda()
    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(loader):
            input = data[0]["data"]
            input = input.cuda()
            B = input.size(0)
            target = data[0]["label"].squeeze().cuda().long()
            target = target.cuda(async=True)

            # compute output
            output, _ = model(input)
            loss = criterion(output, target)

            # compute miss indices
            pred = (F.softmax(output, dim=1)).max(1, keepdim=True)[1] # get the index of the max probability
            match = pred.eq(target.view_as(pred))
            indexT = (i*B + (match == 0).nonzero()[:,0]).long()
            index_miss = torch.cat([index_miss, indexT.view(-1)])
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=topK)
            reduced_loss = loss.data
            losses.update(to_python_float(reduced_loss), B)
            top1.update(to_python_float(prec1), B)
            top5.update(to_python_float(prec5), B)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.local_rank == 0 and i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      #'Speed {2:.3f} ({3:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, loader_len,
                       args.world_size * args.batch_size / batch_time.val,
                       args.world_size * args.batch_size / batch_time.avg,
                       batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return index_miss


def train(args, loader, model, criterion, optimizer, epoch, unsup=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    loader_len = int(loader._size / args.batch_size)
    B = args.batch_size
    topK = (1, 5)
    if unsup:
        topK = (1, 1) # only 4 classes
        R  = 4 # 4 angles as in Gidaris18
        unsup_target = torch.tensor([0, 1, 2, 3], dtype=torch.long).cuda().unsqueeze(1) # 4x1
    # update learning rate
    end = time.time()
    for i, data in enumerate(loader):
        input = data[0]["data"]
        input = input.cuda()
        # unsupervised pretraining
        if unsup:
            lr = adjust_unsup_learning_rate(args, optimizer, epoch, i, loader_len)
            target = unsup_target.repeat(1, B).view(-1) # R*Bx1
            input = input.repeat(R, 1, 1, 1)
            input[1*B:2*B, ...] = torch.rot90(input[1*B:2*B, ...], 1, [3, 2]) # 90  deg
            input[2*B:3*B, ...] = torch.flip( input[2*B:3*B, ...],    [2, 3]) # 180 deg
            input[3*B:4*B, ...] = torch.rot90(input[3*B:4*B, ...], 1, [2, 3]) # 270 deg
        # supervised training
        else:
            lr = adjust_learning_rate(args, optimizer, epoch, i, loader_len)
            target = data[0]["label"].squeeze().cuda().long()            

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output, _ = model(input)
        loss = criterion(output, target)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=topK)
        reduced_loss = loss.data
        losses.update(to_python_float(reduced_loss), B)
        top1.update(to_python_float(prec1), B)
        top5.update(to_python_float(prec5), B)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0 and i > 1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  #'Speed {:.2f} ({:.2f})\t'
                  'LR {3:.6f}\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.2f} ({top1.avg:.2f})\t'
                  'Prec@5 {top5.val:.2f} ({top5.avg:.2f})'.format(
                   epoch, i, loader_len, lr,
                   args.world_size * args.batch_size / batch_time.val,
                   args.world_size * args.batch_size / batch_time.avg,
                   batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def gen_mc(args, loader, model, criterion, optimizer, prefix, save_file):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    #
    loader_len = int(loader._size / args.batch_size)
    D = int(loader._size)
    K = args.sample_steps # or "T" in code
    B = args.batch_size
    #
    if loader._size % args.batch_size == 0:
        DD = D
    else:
        DD = (loader_len + 1)*B
    #
    if args.ensemble_size > 1:
        descr = torch.zeros(DD, CL, 2)
    else:
        descr = torch.zeros(DD)
    #
    topK=(1, 5)
    model.eval()
    r = 1e-8 # regularization for log() to avoid nan
    #
    end = time.time()
    for i, data in enumerate(loader):
        input = data[0]["data"]
        input = input.cuda()
        target = data[0]["label"].squeeze().cuda().long()
        # measure data loading time
        data_time.update(time.time() - end)
        outputT = torch.zeros(B, CL, K).cuda()
        #
        for k in range(K):
            output, _ = model(input)
            outputT[:,:,k] = output.detach()
        # calc MC accuracy
        outputAve = torch.mean(outputT, 2)
        loss = criterion(outputAve, target)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputAve.data, target, topk=topK)
        reduced_loss = loss.data
        losses.update(to_python_float(reduced_loss), B)
        top1.update(to_python_float(prec1), B)
        top5.update(to_python_float(prec5), B)
        # MC
        index = range(i*B, (i+1)*B)
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
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        #
        if args.local_rank == 0 and i % args.print_freq == 0:
            print('Gen MC Descr: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    #'Speed {2:.3f} ({3:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, loader_len,
                    args.world_size * args.batch_size / batch_time.val,
                    args.world_size * args.batch_size / batch_time.avg,
                    batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))
    #
    torch.save(descr[:D], save_file)


def gen_descr(args, loader, model, criterion, optimizer, prefix, save_file, val_file):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    pseudo_top1 = AverageMeter()
    pseudo_top5 = AverageMeter()
    #
    loader_len = int(loader._size / args.batch_size)
    D = int(loader._size)
    L = args.descriptor_length
    B = args.batch_size
    #
    if loader._size % args.batch_size == 0:
        DD = D
    else:
        DD = (loader_len + 1)*B
    #
    if (prefix == 'val') and ('map' in args.subtype_method):
        descr = torch.zeros(DD, L, 4)
    elif (prefix == 'train') and ('map' in args.subtype_method):
        descr = torch.zeros(DD, L, 2)
        val_descr = torch.load(val_file)
        fvec_val = val_descr[...,0].cuda() # MxL
        grad_val = val_descr[...,2].cuda() # MxL
        M = fvec_val.size(0)
    else:
        descr = torch.zeros(DD, L, 2)
    #
    topK=(1, 5)
    model.eval()
    eps = 1e-10 # small regularization constant
    # make a list for multiscale aggregation
    if   L == 448:
        I = [0, 64, 192, 448]
    elif L == 512:
        I = [0, 512]
    elif L == 768:
        I = [0, 256, 768]
    elif L == 0:
        I = []
    else:
        print('Wrong descriptor length in model.py')
        sys.exit(0)
    #
    end = time.time()
    for i, data in enumerate(loader):
        input = data[0]["data"]
        input = input.cuda()
        target = data[0]["label"].view(-1).cuda().long() # real label
        index = range(i*B, (i+1)*B)
        # measure data loading time
        data_time.update(time.time() - end)
        # forward
        output, feat_vec = model(input)
        # accuracy
        prec1, prec5 = accuracy(output.data, target, topk=topK)
        top1.update(to_python_float(prec1), B)
        top5.update(to_python_float(prec5), B)
        # generate feature vector
        fV = torch.tensor([]).cuda()
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
                label = torch.argmax(output, dim=1)
                loss = criterion(output, label)
                optimizer.zero_grad()
                feat_grad = torch.autograd.grad(loss, feat_vec, retain_graph=True)
                fG = torch.tensor([]).cuda()
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
                for j in range(1, len(I)): # sum of distance metrics for multi-scale case
                    fA[:, I[j-1]:I[j]] -= torch.mean(fA[:, I[j-1]:I[j]], 1, keepdim=True)
                    fB[:, I[j-1]:I[j]] -= torch.mean(fB[:, I[j-1]:I[j]], 1, keepdim=True)
                    gA[:, I[j-1]:I[j]] -= torch.mean(gA[:, I[j-1]:I[j]], 1, keepdim=True)
                    gB[:, I[j-1]:I[j]] -= torch.mean(gB[:, I[j-1]:I[j]], 1, keepdim=True)
                    fA[:, I[j-1]:I[j]] /= (torch.std(fA[:, I[j-1]:I[j]], 1, keepdim=True) + eps)
                    fB[:, I[j-1]:I[j]] /= (torch.std(fB[:, I[j-1]:I[j]], 1, keepdim=True) + eps)
                    gA[:, I[j-1]:I[j]] /= (torch.std(gA[:, I[j-1]:I[j]], 1, keepdim=True) + eps)
                    gB[:, I[j-1]:I[j]] /= (torch.std(gB[:, I[j-1]:I[j]], 1, keepdim=True) + eps)
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
                labelEst = val_descr[sim_idx,0,3].long().cuda()
                loss = criterion(output, labelEst)
                #
                labelAcc = (torch.arange(output.size(1)).long().unsqueeze(0).repeat(output.size(0),1).cuda() == labelEst.unsqueeze(1)).float()
                pseudo_prec1, pseudo_prec5 = accuracy(labelAcc, target, topk=topK)
                pseudo_top1.update(to_python_float(pseudo_prec1), B)
                pseudo_top5.update(to_python_float(pseudo_prec5), B)
                reduced_loss = loss.data
                losses.update(to_python_float(reduced_loss), B)
                optimizer.zero_grad()
                feat_grad = torch.autograd.grad(loss, feat_vec, retain_graph=True)
                fG = torch.tensor([]).cuda()
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
                    loss = criterion(output, label[l])
                    if l==0:
                        pseudo_prec1, pseudo_prec5 = accuracy(output.data, label[l], topk=topK)
                        pseudo_top1.update(to_python_float(pseudo_prec1), B)
                        pseudo_top5.update(to_python_float(pseudo_prec5), B)
                        reduced_loss = loss.data
                        losses.update(to_python_float(reduced_loss), B)
                    #
                    optimizer.zero_grad()
                    feat_grad = torch.autograd.grad(loss, feat_vec, retain_graph=True)
                    fG = torch.tensor([]).cuda()
                    for j in range(len(feat_vec)):
                        if feat_vec[j].dim() > 2:
                            g = F.avg_pool2d(feat_grad[j].detach(), feat_grad[j].size(-1)).view(feat_grad[j].size(0), -1)
                        else:
                            g = feat_grad[j].detach()
                        #
                        fG = torch.cat([fG, g], dim=1) # 1xL
                    descr[index, :, l+1] = fG.cpu()
                descr[index, 0, 3] = target.float().cpu()
        else:
            if ('Abl' in args.subtype_method):
                label = target # true label
                q = 1
            else:
                label = torch.argmax(output, dim=1) # pseudo-label
                q = 0
            #
            pseudo_prec1, pseudo_prec5 = accuracy(output.data, label, topk=topK)
            loss = criterion(output, label)
            reduced_loss = loss.data
            losses.update(to_python_float(reduced_loss), B)

            pseudo_top1.update(to_python_float(pseudo_prec1), B)
            pseudo_top5.update(to_python_float(pseudo_prec5), B)
            # Fisher score
            if 'none' not in args.subtype_method:
                optimizer.zero_grad()
                feat_grad = torch.autograd.grad(loss, feat_vec)
                fG = torch.tensor([]).cuda()
                for j in range(len(feat_vec)):
                    if feat_vec[j].dim() > 2:
                        g = F.avg_pool2d(feat_grad[j].detach(), feat_grad[j].size(-1)).view(feat_grad[j].size(0), -1)
                    else:
                        g = feat_grad[j].detach()
                    #
                    fG = torch.cat([fG, g], dim=1) # 1xL
                #
                descr[index, :, 1] = fG.cpu()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        #
        if i % args.print_freq == 0:
            print('\nGen Descr: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    #'Speed {2:.3f} ({3:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                    'Pseudo prec@1 {pseudo_top1.val:.3f} ({pseudo_top1.avg:.3f})\t'
                    'Pseudo prec@5 {pseudo_top5.val:.3f} ({pseudo_top5.avg:.3f})'.format(
                    i, loader_len,
                    args.world_size * args.batch_size / batch_time.val,
                    args.world_size * args.batch_size / batch_time.avg,
                    batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5, pseudo_top1=pseudo_top1, pseudo_top5=pseudo_top5))
        #
        descr[index, :, 0] = fV.cpu()
    #
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Pseudo prec@1 {pseudo_top1.avg:.3f} Pseudo prec@5 {pseudo_top5.avg:.3f}'
            .format(top1=top1, top5=top5, pseudo_top1=pseudo_top1, pseudo_top5=pseudo_top5))
    #
    torch.save(descr[:D], save_file)


# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


def save_checkpoint(state, filename):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(args, optimizer, epoch, step, len_epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 5: # warm up
        lr = args.lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)
    elif epoch < 30:
        lr = args.lr
    elif epoch < 50:
        lr = args.lr * 1e-1
    elif epoch < 57:
        lr = args.lr * 1e-2
    else:
        lr = args.lr * 1e-3
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


def adjust_unsup_learning_rate(args, optimizer, epoch, step, len_epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 2: # warm up
        lr = args.lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)
    elif epoch < 6:
        lr = args.lr
    elif epoch < 10:
        lr = args.lr * 1e-1
    elif epoch < 14:
        lr = args.lr * 1e-2
    else:
        lr = args.lr * 1e-3
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def reduce_tensor(world_size, tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt
