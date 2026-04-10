import argparse
from functools import partial
import os
from torch.cuda.amp import autocast, GradScaler
import clip

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import prepare_watermarking_data, prepare_plain_data
from reprogramming import *
from mapping import *
from cfg import *
from data import DEFAULT_TEMPLATE, ENSEMBLE_TEMPLATES, get_separate_text_embedding, get_text_ensemble_embedding, CUSTOM_TEMPLATES

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dataset', choices=["cifar10", "cifar100", "dtd", "flowers102", "ucf101", "food101", "gtsrb", "svhn", "eurosat", "oxfordpets", "stanfordcars", "sun397"], default="sun397")
    args = p.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(args.seed)

    model, preprocess = clip.load("ViT-B/16")
    convert_models_to_fp32(model)
    model.eval()
    model.requires_grad_(False)

    loaders, class_names = prepare_plain_data(dataset=args.dataset, data_path=data_path, preprocess=preprocess, test_process=preprocess)
    templates = [CUSTOM_TEMPLATES[args.dataset]]
    txt_emb = get_text_ensemble_embedding(class_names, templates, model)
    print(len(templates), "templates")
    print(f"{len(class_names)} Class Names: {class_names}")
    print(f"Text Embedding Shape: {txt_emb.shape}")
    emb_names = np.array([f"T{i // len(class_names)} {class_names[i % len(class_names)]}" for i in range(txt_emb.size(0))])

    def network(x):
        x_emb = model.encode_image(x)
        x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
        logits = model.logit_scale.exp() * x_emb @ txt_emb
        return logits

    total_num = 0
    true_num = 0
    pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Testing Epo 1", ncols=100)
    fx0s = []
    ys = []
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        ys.append(y)
        with torch.no_grad():
            fx0 = network(x)
            fx = fx0
        total_num += y.size(0)
        true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
        acc = true_num / total_num
        pbar.set_postfix_str(f"Testing Acc {100 * acc:.2f}%")

    print("*" * 50)
    print(f"{args.dataset}\t{100 * acc:.2f}")
