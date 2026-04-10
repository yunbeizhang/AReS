from functools import partial
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
import argparse
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import prepare_padding_data, prepare_watermarking_data, IMAGENETNORMALIZE
from reprogramming import *
from mapping import *
from cfg import *
from model.prepare_model import prepare_pretrained_model, prepare_vlm_distilled_model

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--reprogramming', choices=["padding", "watermarking"], default="padding")
    p.add_argument('--mapping', choices=["rlm", "flm", "ilm", "blm", "blmp"], default="blmp")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dataset', choices=["cifar10", "cifar100", "dtd", "flowers102", "ucf101", "food101", "gtsrb", "svhn", "eurosat", "oxfordpets", "stanfordcars", "sun397"], default="sun397")
    p.add_argument('--model', default="resnet18")
    
    p.add_argument('--distilled', action='store_true')
    p.add_argument('--vlm_distilled', action='store_true')
    p.add_argument('--num_samples_per_class', type=int, default=16, help="Number of samples per class for few-shot learning. Set to -1 for full dataset.")
    
    p.add_argument('--teacher', type=str, default="resnet101")
    p.add_argument('--student', type=str, default="resnet18")
    p.add_argument('--mode', choices=['linear', 'full'], default="linear")
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--criterion', choices=['kl', 'ce', 'l2_prob','l2_logit'], default='kl')
    args = p.parse_args()
    start_time = time.time()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    
    args.restore_weight = None
    if args.distilled:
        args.restore_weight = os.path.join(f'{results_path}', 'distilled', args.mode, f'vm_{args.teacher}_{args.dataset}', f'{args.student}_{args.criterion}_{args.seed}', 'best.pth')
    elif args.vlm_distilled:
        args.restore_weight = os.path.join(f'{results_path}_vlm', 'distilled_16_shot', f'vlm_{args.mode}', args.dataset, f'{args.student}_{args.criterion}_{args.seed}', 'best.pth')
    
    if args.restore_weight is not None:
        if args.distilled:
            if args.reprogramming == "padding":
                save_path = os.path.join(results_path, 'oracle', 'distilled', args.mode, f'vm_{args.teacher}_{args.dataset}', f'{args.student}_{args.criterion}_{args.seed}')
            elif args.reprogramming == "watermarking":
                save_path = os.path.join(results_path, 'oracle', 'distilled', args.mode, f'vm_{args.teacher}_{args.dataset}', f'{args.student}_{args.criterion}_{args.seed}_{args.reprogramming}')
                
        elif args.vlm_distilled:
            save_path = os.path.join(f'{results_path}_vlm', 'oracle', 'distilled_16_shot', f'vlm_{args.mode}', args.dataset, f'{args.student}_{args.criterion}_{args.seed}')
    else:
        save_path = os.path.join(results_path, 'oracle', 'pretrained', f'vm_{args.model}_{args.dataset}',   f'{args.mapping}_{args.reprogramming}_{args.seed}')

    imgsize = 224
    padding_size = imgsize / 2

    # Data
    if args.reprogramming == "padding":
        loaders, configs = prepare_padding_data(args.dataset, data_path=data_path)
        class_names = configs['class_names']
        normalize = transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std'])
    elif args.reprogramming == "watermarking":
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
        loaders, class_names = prepare_watermarking_data(args.dataset, data_path=data_path, preprocess=train_preprocess,
                                                     test_process=test_preprocess)

    # Network
    if args.vlm_distilled:
        network = prepare_vlm_distilled_model(args.model, num_classes=len(class_names), restore_weight=args.restore_weight)
    else:
        network = prepare_pretrained_model(args.model, restore_weight=args.restore_weight)
    network = network.to(device)
    network.requires_grad_(False)
    network.eval()

    # Visual Prompt
    if args.reprogramming == "padding":
        visual_prompt = PaddingVR(imgsize, mask=configs['mask'], normalize=normalize).to(device)
    elif args.reprogramming == "watermarking":
        visual_prompt = WatermarkingVR(imgsize, padding_size).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(visual_prompt.parameters(), lr=config_vm['lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5 * config_vm['epoch']), int(0.72 * config_vm['epoch'])], gamma=0.1)

    os.makedirs(save_path, exist_ok=True)

    # Train
    best_acc = 0.
    scaler = GradScaler()

    # Label Mapping for RLM, fLM
    if args.mapping == "rlm":
        mapping_matrix = torch.randperm(1000)[:len(class_names)]
        label_mapping = partial(label_mapping_base, mapping_sequence=mapping_matrix)
    elif args.mapping == 'flm':
        mapping_matrix = one2one_mappnig_matrix(visual_prompt, network, loaders['train'])
        label_mapping = partial(label_mapping_base, mapping_sequence=mapping_matrix)


    for epoch in range(config_vm['epoch']):
        # Label Mapping for ILM, BLM, BLM++
        if args.mapping == 'ilm':
            mapping_matrix = one2one_mappnig_matrix(visual_prompt, network, loaders['train'])
            label_mapping = partial(label_mapping_base, mapping_sequence=mapping_matrix)
        elif args.mapping == 'blm':
            mapping_matrix = blm_reweight_matrix(visual_prompt, network, loaders['train'], lap=config_vm['blm']['lap'])
            label_mapping = partial(label_mapping_calculation, mapping_matrix=mapping_matrix)
        elif args.mapping == 'blmp':
            mapping_matrix = blmp_reweight_matrix(visual_prompt, network, loaders['train'], lap=config_vm['blmp']['lap'], k=int(len(class_names) * config_vm['blmp']['topk_ratio']))
            label_mapping = partial(label_mapping_calculation, mapping_matrix=mapping_matrix)

        visual_prompt.train()
        total_num = 0
        true_num = 0
        loss_sum = 0
        train_acc_list, test_acc_list = [], []
        pbar = tqdm(loaders['train'], total=len(loaders['train']), desc=f"Training Epo {epoch}", ncols=100)
        for x, y in pbar:
            pbar.set_description_str(f"Training Epo {epoch}", refresh=True)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                fx = label_mapping(network(visual_prompt(x)))
                loss = F.cross_entropy(fx, y, reduction='mean')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            loss_sum += loss.item() * fx.size(0)
            pbar.set_postfix_str(f"Training Acc {100 * true_num / total_num:.2f}%")
            train_acc_list.append(true_num / total_num)
        scheduler.step()

        # Test
        visual_prompt.eval()
        total_num = 0
        true_num = 0
        pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Testing Epo {epoch}", ncols=100)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                fx0 = network(visual_prompt(x))
                fx = label_mapping(fx0)
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            acc = true_num / total_num
            pbar.set_postfix_str(f"Testing Acc {100 * acc:.2f}%, Best Acc {100 * best_acc:.2f}%")
            test_acc_list.append(acc)

        running_time = time.time() - start_time
        # Save CKPT
        state_dict = {
            'args': args,
            "visual_prompt_dict": visual_prompt.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "mapping_matrix": mapping_matrix,
            "running_time": running_time,
        }
        if acc > best_acc:
            best_acc = acc
            state_dict['best_acc'] = best_acc
            torch.save(state_dict, os.path.join(save_path, 'best.pth'))
            
    end_time = time.time()
    total_time = end_time - start_time
    final_log = {
        'args': args,
        "visual_prompt_dict": visual_prompt.state_dict(),
        'train_acc_list': train_acc_list,
        'test_acc_list': test_acc_list,
        'best_acc': best_acc,
    }
    torch.save(final_log, os.path.join(save_path, f'complete_{best_acc*100:.1f}_{total_time:.0f}.pth'))