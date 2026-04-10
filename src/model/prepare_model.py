import torchvision.models as models
import torch
import timm
import re

vit_model_dict = {
    'vitb32': 'vit_base_patch32_224',
    'vitb16': 'vit_base_patch16_224',
    'vitl32': 'vit_large_patch32_224',
    'vitl16': 'vit_large_patch16_224',
    'vith14': 'vit_huge_patch14_224',
    'vitb32_clip': 'vit_base_patch32_clip_224',
    'vitb16_clip': 'vit_base_patch16_clip_224',
    'vitl14_clip': 'vit_large_patch14_clip_224',
    'vith14_clip': 'vit_huge_patch14_clip_224',
}

def prepare_pretrained_model(model_name, restore_weight=None):
    if 'vit' in model_name:
        # model_dict  = {'b': 'base', 'l': 'large', 'h': 'huge'}
        # patch_size = re.findall(r'(\d+)', model_name)[0]
        # model_size = model_name[3]
        # model = timm.create_model(f'vit_{model_dict[model_size]}_patch{patch_size}_224', pretrained=True)
        model = timm.create_model(vit_model_dict[model_name], pretrained=True)
        
    elif 'resnet' in model_name:
        num_layers = re.findall(r'(\d+)', model_name)[0]
        pretrain_weights = models.__dict__[f'ResNet{num_layers}_Weights'].IMAGENET1K_V1
        model = models.__dict__[model_name](weights=pretrain_weights)
        
    else:
        raise NotImplementedError(f'{model_name} is not supported')
    
    if restore_weight is not None:
        state_dict = torch.load(restore_weight)['model']
        model.load_state_dict(state_dict)
        assert model_name in restore_weight, f'{model_name} is not in {restore_weight}'
        print(f'Restored model weight from {restore_weight}')
        
    return model

def prepare_vlm_distilled_model(model_name, num_classes, restore_weight=None):
    if 'vit' in model_name:
        # model_dict  = {'b': 'base', 'l': 'large', 'h': 'huge'}
        # patch_size = re.findall(r'(\d+)', model_name)[0]
        # model_size = model_name[3]
        # model = timm.create_model(f'vit_{model_dict[model_size]}_patch{patch_size}_224', pretrained=True)
        model = timm.create_model(vit_model_dict[model_name], pretrained=True)
        model.head = torch.nn.Linear(model.head.in_features, num_classes)
        
    elif 'resnet' in model_name:
        num_layers = re.findall(r'(\d+)', model_name)[0]
        pretrain_weights = models.__dict__[f'ResNet{num_layers}_Weights'].IMAGENET1K_V1
        model = models.__dict__[model_name](weights=pretrain_weights)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        
    else:
        raise NotImplementedError(f'{model_name} is not supported')
    
    if restore_weight is not None:
        state_dict = torch.load(restore_weight)['model']
        model.load_state_dict(state_dict)
        assert model_name in restore_weight, f'{model_name} is not in {restore_weight}'
        print(f'Restored model weight from {restore_weight}')
        
    return model

def prepare_student_model(model_name, mode='linear', lr=1e-3):
    encoder_params, clf_params = [], []
    if 'vit' in model_name:
        # patch_size = re.findall(r'(\d+)', model_name)[0]
        # model = timm.create_model(f'vit_base_patch{patch_size}_224', pretrained=True)
        model = timm.create_model(vit_model_dict[model_name], pretrained=True)
        model.head = torch.nn.Linear(model.head.in_features, model.head.out_features)
        
        for name, param in model.named_parameters():
            if 'head' not in name:
                encoder_params.append(param)
            else:
                clf_params.append(param)
    elif 'resnet' in model_name:
        num_layers = re.findall(r'(\d+)', model_name)[0]
        pretrain_weights = models.__dict__[f'ResNet{num_layers}_Weights'].IMAGENET1K_V1
        model = models.__dict__[model_name](weights=pretrain_weights)
        model.fc = torch.nn.Linear(model.fc.in_features, model.fc.out_features)
        
        for name, param in model.named_parameters():
            if 'fc' not in name:
                encoder_params.append(param)
            else:
                clf_params.append(param)
        
    else:
        raise NotImplementedError(f'{model_name} is not supported')
    
    if mode == 'linear':
        for param in encoder_params:
            param.requires_grad = False
        for param in clf_params:
            param.requires_grad = True    
        optimizer = torch.optim.AdamW(clf_params, lr=lr)
        
    elif mode == 'full':
        for param in encoder_params:
            param.requires_grad = True
        for param in clf_params:
            param.requires_grad = True
            
        optimizer = torch.optim.AdamW(
            [
                {'params': encoder_params, 'lr': lr/10},
                {'params': clf_params, 'lr': lr}
            ]
        )
    else:
        raise NotImplementedError(f'{mode} is not supported')
    
    return model, optimizer

def prepare_vlm_student_model(model_name, mode='linear', lr=1e-3, num_classes=1000):
    encoder_params, clf_params = [], []
    if 'vit' in model_name:
        # patch_size = re.findall(r'(\d+)', model_name)[0]
        # model = timm.create_model(f'vit_base_patch{patch_size}_224', pretrained=True)
        model = timm.create_model(vit_model_dict[model_name], pretrained=True)
        model.head = torch.nn.Linear(model.head.in_features, num_classes)
        
        for name, param in model.named_parameters():
            if 'head' not in name:
                encoder_params.append(param)
            else:
                clf_params.append(param)
    elif 'resnet' in model_name:
        num_layers = re.findall(r'(\d+)', model_name)[0]
        pretrain_weights = models.__dict__[f'ResNet{num_layers}_Weights'].IMAGENET1K_V1
        model = models.__dict__[model_name](weights=pretrain_weights)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        
        for name, param in model.named_parameters():
            if 'fc' not in name:
                encoder_params.append(param)
            else:
                clf_params.append(param)
        
    else:
        raise NotImplementedError(f'{model_name} is not supported')
    
    if mode == 'linear':
        for param in encoder_params:
            param.requires_grad = False
        for param in clf_params:
            param.requires_grad = True    
        optimizer = torch.optim.AdamW(clf_params, lr=lr)
        
    elif mode == 'full':
        for param in encoder_params:
            param.requires_grad = True
        for param in clf_params:
            param.requires_grad = True
            
        optimizer = torch.optim.AdamW(
            [
                {'params': encoder_params, 'lr': lr/10},
                {'params': clf_params, 'lr': lr}
            ]
        )
    else:
        raise NotImplementedError(f'{mode} is not supported')
    
    return model, optimizer