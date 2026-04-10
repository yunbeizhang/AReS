import argparse
from functools import partial
import os
from torch.cuda.amp import autocast, GradScaler
import clip

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import prepare_watermarking_data, prepare_watermarking_data_few_shot, prepare_plain_data_few_shot, single_round_API_inference
from reprogramming import *
from mapping import *
from cfg import *
from data import DEFAULT_TEMPLATE, ENSEMBLE_TEMPLATES, get_separate_text_embedding, get_text_ensemble_embedding, CUSTOM_TEMPLATES
from model.prepare_model import prepare_student_model, prepare_vlm_student_model

import time

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dataset', choices=["cifar10", "cifar100", "dtd", "flowers102", "ucf101", "food101", "gtsrb", "svhn", "eurosat", "oxfordpets", "stanfordcars", "sun397"], default="sun397")
    p.add_argument('--num_samples_per_class', type=int, default=16, help="Number of samples per class for few-shot learning. Set to -1 for full dataset.")
    p.add_argument('--mode', choices=['linear', 'full'], default="linear")
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--criterion', choices=['kl', 'ce', 'l2_prob', 'l2_logit', 'sce'], default='kl')
    p.add_argument('--student', type=str, default="resnet18")
    p.add_argument('--epochs', type=int, default=100)
    args = p.parse_args()
    
    start_time = time.time()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(args.seed)
    save_path = os.path.join(f'{results_path}_vlm', f'distilled_{args.num_samples_per_class}_shot', f'vlm_{args.mode}', args.dataset, f'{args.student}_{args.criterion}_{args.seed}')

    model, preprocess = clip.load("ViT-B/16")
    convert_models_to_fp32(model)
    model.eval()
    model.requires_grad_(False)

    # loaders, class_names = prepare_watermarking_data(dataset=args.dataset, data_path=data_path, preprocess=preprocess, test_process=preprocess)
    # loaders, class_names = prepare_watermarking_data_few_shot(dataset=args.dataset, data_path=data_path, preprocess=preprocess, test_process=preprocess, num_samples_per_class=args.num_samples_per_class, seed=args.seed)
    # templates = [DEFAULT_TEMPLATE] + ENSEMBLE_TEMPLATES
    # txt_emb = torch.cat(get_saparate_text_embedding(class_names, templates, model))
    
    loaders, class_names = prepare_plain_data_few_shot(dataset=args.dataset, data_path=data_path, preprocess=preprocess, test_process=preprocess, shuffle=True, num_samples_per_class=args.num_samples_per_class, seed=args.seed)

    templates = [CUSTOM_TEMPLATES[args.dataset]]
    txt_emb = get_text_ensemble_embedding(class_names, templates, model)
    emb_names = np.array([f"T{i // len(class_names)} {class_names[i % len(class_names)]}" for i in range(txt_emb.size(0))])

    def network(x):
        x_emb = model.encode_image(x)
        x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
        logits = model.logit_scale.exp() * x_emb @ txt_emb
        return logits

    distillation_loaders = single_round_API_inference(loaders, network, device=device, batch_size=64, num_workers=4)

    student_net, optimizer = prepare_vlm_student_model(args.student, args.mode, args.lr, num_classes=len(class_names))
    student_net = student_net.to(device)

    # Optimizer
    t_max = args.epochs * len(distillation_loaders['train'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    os.makedirs(save_path, exist_ok=True)

    best_loss = 1e10
    best_epoch = 0
    scaler = GradScaler()
    for epoch in range(args.epochs):
        student_net.train()
        total_num = 0
        true_num = 0
        loss_sum = 0
        pbar = tqdm(distillation_loaders['train'], total=len(distillation_loaders['train']), desc=f"{args.dataset} Train Epo {epoch}", ncols=100)

        for i, (x, y, logit) in enumerate(pbar):
            pbar.set_description_str(f"Training Epo {epoch}", refresh=True)
            x, y, logit = x.to(device), y.to(device), logit.to(device)
            optimizer.zero_grad()
            teacher_prob = F.softmax(logit, dim=1)
            
            student_output = student_net(x)
            student_prob = F.softmax(student_output, dim=1)
            
            if args.criterion == 'kl':
                loss = F.kl_div(student_prob.log(), teacher_prob, reduction="batchmean")
            elif args.criterion == 'ce':
                loss = F.cross_entropy(student_output, teacher_prob, reduction='mean')
            elif args.criterion == 'l2_prob':
                loss = F.mse_loss(student_prob, teacher_prob, reduction='mean')
            elif args.criterion == 'l2_logit':
                loss = F.mse_loss(student_output, logit, reduction='mean')
            else:
                raise NotImplementedError(f'{args.criterion} is not supported')

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
            pbar.set_postfix_str(f"Training Loss {loss.item():.4f}")
            scheduler.step()

        # Test
        student_net.eval()
        total_num = 0
        true_num = 0
        pbar = tqdm(distillation_loaders['test'], total=len(distillation_loaders['test']), desc=f"{args.dataset} Test Epo {epoch}", ncols=100)
        fx0s = []
        ys = []
        for x, y, logit in pbar:
            x, y, logit = x.to(device), y.to(device), logit.to(device)
            ys.append(y)
            with torch.no_grad():
                student_output = student_net(x)
                
            teacher_prob = F.softmax(logit, dim=1)
            student_prob = F.softmax(student_output, dim=1)
            
            if args.criterion == 'kl':
                loss = F.kl_div(student_prob.log(), teacher_prob, reduction="batchmean")
            elif args.criterion == 'ce':
                loss = F.cross_entropy(student_output, teacher_prob, reduction='mean')
            elif args.criterion == 'l2_prob':
                loss = F.mse_loss(student_prob, teacher_prob, reduction='mean')
            elif args.criterion == 'l2_logit':
                loss = F.mse_loss(student_output, logit, reduction='mean')
            else:
                raise NotImplementedError(f'{args.criterion} is not supported')

            pbar.set_postfix_str(f"Testing Loss {loss.item():.4f}")
            
        eval_loss = loss.item()
        running_time = time.time() - start_time
        # Save CKPT
        state_dict = {
            "model": student_net.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            'args': args,
            "running_time": running_time,
        }
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_epoch = epoch
            state_dict['best_loss'] = best_loss
            print(f"Best Model Found, Save to {os.path.join(save_path, 'best.pth')}")
            torch.save(state_dict, os.path.join(save_path, 'best.pth'))

    end_time = time.time()
    total_running_time = end_time - start_time
    torch.save(state_dict, os.path.join(save_path, f'complete_{total_running_time:.0f}_{best_epoch}.pth'))
