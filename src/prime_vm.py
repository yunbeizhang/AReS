from functools import partial
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
import argparse
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import prepare_padding_data, prepare_watermarking_data, IMAGENETNORMALIZE, prepare_plain_data
from model.prepare_model import prepare_pretrained_model, prepare_student_model
from reprogramming import *
from mapping import *
from cfg import *

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dataset', choices=["cifar10", "cifar100", "dtd", "flowers102", "ucf101", "food101", "gtsrb", "svhn", "eurosat", "oxfordpets", "stanfordcars", "sun397"], default="sun397")
    p.add_argument('--teacher', type=str, default="resnet101")
    p.add_argument('--student', type=str, default="resnet18")
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--mode', choices=['linear', 'full'], default="linear")
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--criterion', choices=['kl', 'ce', 'l2_prob', 'l2_logit'], default='kl')
    args = p.parse_args()
    print(args)
    start_time = time.time()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    save_path = os.path.join(f'{results_path}', 'distilled', args.mode, f'vm_{args.teacher}_{args.dataset}', f'{args.student}_{args.criterion}_{args.seed}')
    
    imgsize = 224
    padding_size = imgsize / 2

    # Data
    train_preprocess = transforms.Compose([
        transforms.Resize((imgsize + 4, imgsize + 4)),
        transforms.RandomCrop(imgsize),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std']),
    ])
    test_preprocess = transforms.Compose([
        transforms.Resize((imgsize, imgsize)),
        transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std']),
    ])
    loaders, class_names = prepare_plain_data(args.dataset, data_path=data_path, preprocess=train_preprocess,
                                                    test_process=test_preprocess)

    # SRC Network
    teacher_net = prepare_pretrained_model(args.teacher)
    teacher_net = teacher_net.to(device)
    teacher_net.requires_grad_(False)
    teacher_net.eval()

    # TRG Network
    student_net, optimizer = prepare_student_model(args.student, args.mode, args.lr)
    student_net = student_net.to(device)
        
    # Optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    os.makedirs(save_path, exist_ok=True)

    # Train
    best_loss = 1e10
    scaler = GradScaler()
    train_loss_list = []
    test_loss_list = []
    best_epoch = 0

    for epoch in range(args.epochs):
        teacher_net.eval()
        student_net.train()
        total_num = 0
        true_num = 0
        loss_sum = 0
        pbar = tqdm(loaders['train'], total=len(loaders['train']), desc=f"Training Epo {epoch}", ncols=100)
        for x, y in pbar:
            pbar.set_description_str(f"Training Epo {epoch}", refresh=True)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_output = teacher_net(x)
                teacher_prob = F.softmax(teacher_output, dim=1)
                
            student_output = student_net(x)
            student_prob = F.softmax(student_output, dim=1)
            
            if args.criterion == 'kl':
                loss = F.kl_div(student_prob.log(), teacher_prob, reduction="batchmean")
            elif args.criterion == 'ce':
                loss = F.cross_entropy(student_output, teacher_prob, reduction='mean')
            elif args.criterion == 'l2_prob':
                loss = F.mse_loss(student_prob, teacher_prob, reduction='mean')
            elif args.criterion == 'l2_logit':
                loss = F.mse_loss(student_output, teacher_output, reduction='mean')
            else:
                raise NotImplementedError(f'{args.criterion} is not supported')


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix_str(f"Training Loss {loss.item():.4f}")
            train_loss_list.append(loss.item())
        scheduler.step()

        # Test
        teacher_net.eval()
        student_net.eval()
        total_num = 0
        true_num = 0
        pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Testing Epo {epoch}", ncols=100)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                student_output = student_net(x)
                teacher_output = teacher_net(x)
            teacher_prob = F.softmax(teacher_output, dim=1)
            student_prob = F.softmax(student_output, dim=1)
            
            if args.criterion == 'kl':
                loss = F.kl_div(student_prob.log(), teacher_prob, reduction="batchmean")
            elif args.criterion == 'ce':
                loss = F.cross_entropy(student_output, teacher_prob, reduction='mean')
            elif args.criterion == 'l2_prob':
                loss = F.mse_loss(student_prob, teacher_prob, reduction='mean')
            elif args.criterion == 'l2_logit':
                loss = F.mse_loss(student_output, teacher_output, reduction='mean')
            else:
                raise NotImplementedError(f'{args.criterion} is not supported')

            pbar.set_postfix_str(f"Testing Loss {loss.item():.4f}")
            test_loss_list.append(loss.item())
        eval_loss = loss.item()
        # Save CKPT
        state_dict = {
            "model": student_net.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
        }
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_epoch = epoch
            state_dict['best_loss'] = best_loss
            print(f"Best Model Found, Save to {os.path.join(save_path, 'best.pth')}")
            torch.save(state_dict, os.path.join(save_path, 'best.pth'))
            
    final_log = {
        'args': args,
        'train_loss': train_loss_list,
        'test_loss': test_loss_list,
    }
    end_time = time.time()
    total_time = end_time - start_time
    torch.save(final_log, os.path.join(save_path, f'complete_{best_epoch}_{total_time:.0f}.pth'))
