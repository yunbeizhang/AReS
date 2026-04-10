from functools import partial
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import prepare_padding_data, prepare_watermarking_data, IMAGENETNORMALIZE
from model.prepare_model import prepare_pretrained_model
from reprogramming import *
from mapping import *
from cfg import *

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--reprogramming', choices=["padding", "watermarking"], default="padding")
    p.add_argument('--mapping', choices=["rlm", "flm", "ilm", "blm", "blmp"], default="blmp")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dataset', choices=["cifar10", "cifar100", "dtd", "flowers102", "ucf101", "food101", "gtsrb", "svhn", "eurosat", "oxfordpets", "stanfordcars", "sun397"], default="sun397")
    p.add_argument('--model', default="resnet18")
    p.add_argument('--vp_src', type=str, default="resnet18")
    p.add_argument('--fixed_lm', action='store_true')
    
    p.add_argument('--distilled_model', action='store_true')
    p.add_argument('--distilled_vp', action='store_true')
    p.add_argument('--teacher', type=str, default="resnet101")
    p.add_argument('--student', type=str, default="resnet18")
    p.add_argument('--mode', choices=['linear', 'full'], default="linear")
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--criterion', choices=['kl', 'ce', 'l2_prob', 'l2_logit'], default='kl')
    args = p.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    
    args.restore_model = None
    args.restore_vp = None
    if args.distilled_model:
        args.restore_model = os.path.join(f'{results_path}', 'distilled', args.mode, f'vm_{args.teacher}_{args.dataset}', f'{args.student}_{args.criterion}_{args.seed}', 'best.pth')
    
    if args.distilled_vp:
        if args.reprogramming == "padding":
            args.restore_vp = os.path.join(results_path, 'oracle', 'distilled', args.mode, f'vm_{args.teacher}_{args.dataset}', f'{args.student}_{args.criterion}_{args.seed}', 'best.pth')
        elif args.reprogramming == "watermarking":
            args.restore_vp = os.path.join(results_path, 'oracle', 'distilled', args.mode, f'vm_{args.teacher}_{args.dataset}', f'{args.student}_{args.criterion}_{args.seed}_{args.reprogramming}', 'best.pth')
    else:
        args.restore_vp = os.path.join(results_path, 'oracle', 'pretrained', f'vm_{args.vp_src}_{args.dataset}', f'{args.mapping}_{args.reprogramming}_{args.seed}', 'best.pth')

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
    
    network = prepare_pretrained_model(args.model, args.restore_model)
    network = network.to(device)
    network.requires_grad_(False)
    network.eval()

    # Visual Prompt
    if args.reprogramming == "padding":
        visual_prompt = PaddingVR(imgsize, mask=configs['mask'], normalize=normalize).to(device)
    elif args.reprogramming == "watermarking":
        visual_prompt = WatermarkingVR(imgsize, padding_size).to(device)


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
        
    print(f'Restoring VP from {args.restore_vp}')
    state_dict = torch.load(args.restore_vp)
    visual_prompt.load_state_dict(state_dict['visual_prompt_dict'])
    mapping_matrix = state_dict['mapping_matrix']


    for epoch in range(1):
        # Label Mapping for ILM, BLM, BLM++
        if args.mapping == 'ilm':
            if not args.fixed_lm:
                mapping_matrix = one2one_mappnig_matrix(visual_prompt, network, loaders['train'])
            label_mapping = partial(label_mapping_base, mapping_sequence=mapping_matrix)
        elif args.mapping == 'blm':
            if not args.fixed_lm:
                mapping_matrix = blm_reweight_matrix(visual_prompt, network, loaders['train'], lap=config_vm['blm']['lap'])
            label_mapping = partial(label_mapping_calculation, mapping_matrix=mapping_matrix)
        elif args.mapping == 'blmp':
            if not args.fixed_lm:
                mapping_matrix = blmp_reweight_matrix(visual_prompt, network, loaders['train'], lap=config_vm['blmp']['lap'], k=int(len(class_names) * config_vm['blmp']['topk_ratio']))
            label_mapping = partial(label_mapping_calculation, mapping_matrix=mapping_matrix)
        
        visual_prompt.eval()
        total_num = 0
        true_num = 0
        correct = 0
        pbar = tqdm(loaders['train'], total=len(loaders['train']), desc=f"Training Epo {epoch}", ncols=100)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                fx0 = network(visual_prompt(x))
                fx = label_mapping(fx0)
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            correct += true_num
            acc = true_num / total_num
            pbar.set_postfix_str(f"Training Acc {100 * acc:.2f}% on {args.model}")
        print(f'RESULTS\tTrain\t{args.vp_src}\t{args.model}\t{acc*100:.2f}')
        
        # Test
        visual_prompt.eval()
        total_num = 0
        true_num = 0
        correct = 0
        pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Testing Epo {epoch}", ncols=100)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                fx0 = network(visual_prompt(x))
                fx = label_mapping(fx0)
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            correct += true_num
            acc = true_num / total_num
            pbar.set_postfix_str(f"Testing Acc {100 * acc:.2f}% on {args.model}")
            

        print(f'RESULTS\tTest\t{args.vp_src}\t{args.model}\t{acc*100:.2f}')
        