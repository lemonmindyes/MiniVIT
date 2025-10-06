import math
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from dataloader import ImageNetDataset
from vit import VIT


if __name__ == '__main__':
    # param
    path = '/root/autodl-tmp/imagenet'
    batch_size = 512
    base_lr = 1e-3
    weight_decay = 0.05
    epoch = 300
    warmup_epoch = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = torch.amp.autocast() if device == 'cuda' else nullcontext()

    train_dataset = ImageNetDataset(path = path, split = 'train')
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 8,
                              prefetch_factor = 2, pin_memory = False, persistent_workers = True)
    val_dataset = ImageNetDataset(path = path, split = 'val')
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = 8,
                            prefetch_factor = 2, pin_memory = False, persistent_workers = True)

    config = Config()
    model = VIT(config = config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total params:{total_params}')

    loss_func = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled = device == 'cuda')
    opt = torch.optim.AdamW(model.parameters(), lr = base_lr, weight_decay = weight_decay)

    total_step = epoch * len(train_loader)
    warmup_step = warmup_epoch * len(train_loader)
    if not os.path.exists('train_log.txt'):
        with open('train_log.txt', 'a', encoding='utf-8') as f:
            f.write(f'Epoch, Step, Loss, Lr, Time\n')
    if not os.path.exists('val_log.txt'):
        with open('val_log.txt', 'a', encoding='utf-8') as f:
            f.write(f'Epoch, Acc\n')
    try:
        checkpoint = torch.load('checkpoint.bin')
        model.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['opt'])
        scaler.load_state_dict(checkpoint['scaler'])
    except:
        checkpoint = {'epoch': 0, 'global_step': 0}
    global_step = checkpoint['global_step']

    for e in range(checkpoint['epoch'], epoch):
        model.train()
        start_time = time.time()
        for step, (img, label) in enumerate(train_loader):
            if e < warmup_epoch:
                lr = base_lr * global_step / warmup_step
            else:
                progress = (global_step - warmup_step) / (total_step - warmup_step)
                lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

            # update lr
            for param_group in opt.param_groups:
                param_group['lr'] = lr

            img, label = img.to(device), label.to(device)
            with ctx:
                out = model(img)
                loss = loss_func(out, label)

            opt.zero_grad(set_to_none = True) # zero gradient
            scaler.scale(loss).backward() # scale loss to fit opt's expected gradient scale
            scaler.unscale_(opt) # unscale gradient
            scaler.step(opt) # step opt
            scaler.update() # update scaler

            global_step += 1

            if step % 100 == 0:
                print(f'Epoch:{e + 1}/{epoch}, Step:{step + 1}/{len(train_loader)}, Loss:{loss.item():.4f}, '
                      f'Lr:{opt.param_groups[0]["lr"]:.6f}, Time:{time.time() - start_time:.2f}')
                with open('train_log.txt', 'a', encoding = 'utf-8') as f:
                    f.write(f'{e + 1}, {step + 1}, {loss.item():.4f}, {opt.param_groups[0]["lr"]:.6f}, '
                            f'{time.time() - start_time:.2f}\n')
        model.eval()
        total_acc = 0
        total_num = 0
        for step, (img, label) in enumerate(val_loader):
            img, label = img.to(device), label.to(device)
            with torch.no_grad(), ctx:
                out = model(img)
            pred = out.argmax(dim = 1)
            total_acc += (pred == label).sum().item()
            total_num += img.shape[0]
        print(f'Epoch:{e + 1}/{epoch}, Acc:{total_acc / total_num:.4f}, Time:{time.time() - start_time:.2f}')
        with open('val_log.txt', 'a', encoding = 'utf-8') as f:
            f.write(f'{e + 1}, {total_acc / total_num:.4f}\n')

        checkpoint = {
            'epoch': e,
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'scaler': scaler.state_dict(),
            'global_step': global_step
        }
        torch.save(checkpoint, 'checkpoint.bin.pt')
        os.replace('checkpoint.bin.pt', 'checkpoint.bin')





