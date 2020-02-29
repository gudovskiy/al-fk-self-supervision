import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
from sklearn.metrics import accuracy_score

import sampler





class Solver:
    def __init__(self, args, test_dataloader):
        self.args = args
        self.test_dataloader = test_dataloader

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        self.sampler = sampler.AdversarySampler(self.args.budget)


    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label, _ in dataloader:
                    #print(img.size())
                    yield img, label
        else:
            while True:
                for img, _, _ in dataloader:
                    #print(img.size())
                    yield img


    def train(self, querry_dataloader, task_model, vae, discriminator, unlabeled_dataloader):
        #print(len(querry_dataloader), len(unlabeled_dataloader))
        labeled_data = self.read_data(querry_dataloader)
        unlabeled_data = self.read_data(unlabeled_dataloader, labels=False)

        if self.args.dataset == 'ImageNet':
            optim_vae = optim.Adam(vae.parameters(), lr=1e-3)
            optim_task_model = optim.Adam(task_model.parameters(), lr=1e-3)
            optim_discriminator = optim.Adam(discriminator.parameters(), lr=1e-3)
        else:
            optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
            optim_task_model = optim.Adam(task_model.parameters(), lr=5e-4)
            optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)

        vae.train()
        discriminator.train()
        task_model.train()

        if self.args.cuda:
            vae = vae.cuda()
            discriminator = discriminator.cuda()
            task_model = task_model.cuda()
        
        change_lr_iter = self.args.train_iterations // 25
        best_acc = 0.0
        for param in optim_vae.param_groups:
            lr = param['lr']
        print('Iter = 0 of', self.args.train_iterations, change_lr_iter)
        for iter_count in range(self.args.train_iterations):
            if iter_count is not 0 and iter_count % change_lr_iter == 0:
                for param in optim_vae.param_groups:
                    param['lr'] = param['lr'] * 0.9
                    lr = param['lr']
                for param in optim_task_model.param_groups:
                    param['lr'] = param['lr'] * 0.9

                for param in optim_discriminator.param_groups:
                    param['lr'] = param['lr'] * 0.9
            #print('Reading data')
            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)
            #print(self.args.cuda)
            if self.args.cuda:
                labeled_imgs = labeled_imgs.cuda()
                unlabeled_imgs = unlabeled_imgs.cuda()
                labels = labels.cuda()

            # task_model step
            preds = task_model(labeled_imgs)
            task_loss = self.ce_loss(preds, labels)
            optim_task_model.zero_grad()
            task_loss.backward()
            optim_task_model.step()

            #print(labeled_imgs.size())

            # VAE step
            #print('VAE step')
            for count in range(self.args.num_vae_steps):
                origin, recon, z, mu, logvar = vae(labeled_imgs)
                unsup_loss = self.vae_loss(origin, recon, mu, logvar, self.args.beta)
                unlab_origin, unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
                transductive_loss = self.vae_loss(unlab_origin, unlab_recon, unlab_mu, unlab_logvar, self.args.beta)
            
                labeled_preds = discriminator(mu)
                unlabeled_preds = discriminator(unlab_mu)
                
                #lab_real_preds = torch.ones(labeled_imgs.size(0))
                #unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
                
                lab_real_preds = torch.ones(labeled_imgs.size(0), 1)
                unlab_real_preds = torch.ones(unlabeled_imgs.size(0), 1)
                #print(labeled_preds.size(), unlabeled_preds.size(), lab_real_preds.size(), unlab_real_preds.size())

                if self.args.cuda:
                    lab_real_preds = lab_real_preds.cuda()
                    unlab_real_preds = unlab_real_preds.cuda()
                
                #print(labeled_preds, lab_real_preds)
                #print(unlabeled_preds, unlab_real_preds)

                dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + \
                        self.bce_loss(unlabeled_preds, unlab_real_preds)
                total_vae_loss = unsup_loss + transductive_loss + self.args.adversary_param * dsc_loss
                optim_vae.zero_grad()
                total_vae_loss.backward()
                optim_vae.step()

                # sample new batch if needed to train the adversarial network
                if count < (self.args.num_vae_steps - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)

                    if self.args.cuda:
                        labeled_imgs = labeled_imgs.cuda()
                        unlabeled_imgs = unlabeled_imgs.cuda()
                        labels = labels.cuda()
            #print('Discriminator step')
            # Discriminator step
            for count in range(self.args.num_adv_steps):
                with torch.no_grad():
                    _, _, _, mu, _ = vae(labeled_imgs)
                    _, _, _, unlab_mu, _ = vae(unlabeled_imgs)
                
                labeled_preds = discriminator(mu)
                unlabeled_preds = discriminator(unlab_mu)
                
                #lab_real_preds = torch.ones(labeled_imgs.size(0))
                #unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))
                
                lab_real_preds = torch.ones(labeled_imgs.size(0), 1)
                unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0), 1)
                #print(labeled_preds.size(), unlabeled_preds.size(), lab_real_preds.size(), unlab_real_preds.size())

                if self.args.cuda:
                    lab_real_preds = lab_real_preds.cuda()
                    unlab_fake_preds = unlab_fake_preds.cuda()
                
                #print(labeled_preds, lab_real_preds)
                #print(unlabeled_preds, unlab_real_preds)
                dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + \
                        self.bce_loss(unlabeled_preds, unlab_fake_preds)

                optim_discriminator.zero_grad()
                dsc_loss.backward()
                optim_discriminator.step()

                # sample new batch if needed to train the adversarial network
                if count < (self.args.num_adv_steps - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)

                    if self.args.cuda:
                        labeled_imgs = labeled_imgs.cuda()
                        unlabeled_imgs = unlabeled_imgs.cuda()
                        labels = labels.cuda()
            
            #print('Iter =', iter_count)
            #if (iter_count % 2000 == 0 and self.args.dataset != 'ImageNet') or (iter_count % 10000 == 0 and self.args.dataset == 'ImageNet'):
            if iter_count % 2000 == 0:
                acc = self.test(task_model)
                if acc > best_acc:
                    best_acc = acc
                #
                print('Training iteration: {} with lr = {:.8f}'.format(iter_count, lr))
                print('Curr/Best accuracy: {:.2f}/{:.2f}'.format(acc, best_acc))
                print('Curr loss: task = {:.6f}, vae = {:.6f}, discr = {:.6f}'.format(task_loss.item(), total_vae_loss.item(), dsc_loss.item()))

        final_accuracy = best_acc
        return final_accuracy, vae, discriminator


    def sample_for_labeling(self, vae, discriminator, unlabeled_dataloader):
        querry_indices = self.sampler.sample(vae, 
                                             discriminator, 
                                             unlabeled_dataloader, 
                                             self.args.cuda)

        return querry_indices
                

    def test(self, task_model):
        task_model.eval()
        total, correct = 0, 0
        for imgs, labels, _ in self.test_dataloader:
            if self.args.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                preds = task_model(imgs)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        return correct / total * 100


    def vae_loss(self, x, recon, mu, logvar, beta):
        #print(recon.size(), x.size())
        MSE = self.mse_loss(recon, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD
