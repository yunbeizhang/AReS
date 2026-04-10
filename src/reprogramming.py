import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
from torch.nn import Parameter
from torchvision import transforms
import torchvision.models as models

class PaddingVR(nn.Module):
    def __init__(self, out_size, mask, init='zero', normalize=None):

        super(PaddingVR, self).__init__()
        assert mask.shape[0] == mask.shape[1]
        in_size = mask.shape[0]
        self.out_size = out_size
        if init == "zero":
            self.program = torch.nn.Parameter(data=torch.zeros(3, out_size, out_size))
        elif init == "randn":
            self.program = torch.nn.Parameter(data=torch.randn(3, out_size, out_size))
        self.normalize = normalize

        self.l_pad = int((out_size-in_size+1)/2)
        self.r_pad = int((out_size-in_size)/2)

        mask = np.repeat(np.expand_dims(mask, 0), repeats=3, axis=0)
        mask = torch.Tensor(mask)
        self.register_buffer("mask", F.pad(mask, (self.l_pad, self.r_pad, self.l_pad, self.r_pad), value=1)) # register a buffer that should not to be considered a model parameter

    def forward(self, x):
        x = F.pad(x, (self.l_pad, self.r_pad, self.l_pad, self.r_pad), value=0) + torch.sigmoid(self.program) * self.mask
        if self.normalize is not None:
            x = self.normalize(x)
        return x


class WatermarkingVR(nn.Module):
    def __init__(self, size, pad):
        super(WatermarkingVR, self).__init__()

        self.size = size
        self.program = torch.nn.Parameter(data=torch.zeros(3, size, size))

        if size > 2*pad:
            mask = torch.zeros(3, size-2*pad, size-2*pad)
            self.register_buffer("mask", F.pad(mask, [pad for _ in range(4)], value=1))
        elif size == 2*pad:
            mask = torch.ones(3, size, size)
            self.register_buffer("mask", mask)

    def forward(self, x):
        x = x + self.program * self.mask
        return x

from transformers import ViTFeatureExtractor, ResNetModel, ViTModel, ViTMAEModel, ViTForImageClassification

class CoordinatorINIT(nn.Module):
    def __init__(self, args=None):
        super(CoordinatorINIT, self).__init__()
        # self.args = args
        
        act = nn.GELU #if args.TRAINER.BLACKVIP.ACT == 'gelu' else nn.ReLU
        e_out_dim = 0 #args.TRAINER.BLACKVIP.E_OUT_DIM
        src_dim = 1568 #args.TRAINER.BLACKVIP.SRC_DIM

        self.enc = EncoderManual(e_out_dim, act=act, gap=False)
        self.dec = DecoderManual(0, src_dim=e_out_dim, act=act, arch='vit-base')
    
    def forward(self, x):
        z = self.enc(x)
        wrap = self.dec(z)
        return wrap, z


class Coordinator(nn.Module):
    def __init__(self, args=None):
        super(Coordinator, self).__init__()
        # self.args = args        
        self.backbone = args.backbone #'vit-mae-base' #args.TRAINER.BLACKVIP.PT_BACKBONE
        print(f'backbone: {self.backbone}')
        act = nn.GELU #if args.TRAINER.BLACKVIP.ACT == 'gelu' else nn.ReLU
        src_dim = 1568 #args.TRAINER.BLACKVIP.SRC_DIM

        z_dim = 768
        if self.backbone == 'vit-mae-base':   #! SSL-MAE VIT-B (n param: 86M)
            self.enc_pt = ViTForImageClassification.from_pretrained("facebook/vit-mae-base")
        elif self.backbone == 'vit-base' or self.backbone == 'vit-base16':       #! SUP VIT-B
            # self.enc_pt = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            self.enc_pt = timm.create_model('vit_base_patch16_224', pretrained=True)
        elif self.backbone == 'vit-base32':       #! SUP VIT-B
            # self.enc_pt = ViTModel.from_pretrained("google/vit-base-patch32-224-in21k")
            self.enc_pt = timm.create_model('vit_base_patch32_224', pretrained=True)
        elif self.backbone == 'dino-resnet-50': #! SSL-DINO RN50 (n param: 23M)
            self.enc_pt = ResNetModel.from_pretrained("Ramos-Ramos/dino-resnet-50")
            z_dim = 2048
        elif self.backbone == 'resnet18': #! SUP RN18 (n param: 11M)
            self.enc_pt = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.enc_pt.fc = nn.Identity()
            z_dim = 512
        elif self.backbone == 'resnet50':
            self.enc_pt = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.enc_pt.fc = nn.Identity()
            z_dim = 2048
            src_dim = z_dim + 1088 # 49 * 32 * 2 - z_dim
        else: raise ValueError('not implemented')

        self.dec = DecoderManual(z_dim, src_dim, act=act, arch=self.backbone)

    def forward(self, x):
        with torch.no_grad():
            if self.backbone == 'vit-mae-base':
                #! (N, 197, 768) => pick [CLS] => (N, 768)
                out = self.enc_pt(x, output_hidden_states=True)
                z = out.hidden_states[-1][:,0,:]
            elif self.backbone == 'vit-base' or self.backbone == 'vit-base16' or self.backbone == 'vit-base32':
                #! (N, 197, 768) => pick [CLS] => (N, 768)
                # out = self.enc_pt(x)
                # z = out.last_hidden_state[:,0,:] # for ViTModel
                
                out = self.enc_pt.forward_features(x)  # for timm vit models
                z = out[:,0,:]  # for timm vit models, directly get the first token (CLS)
                # print(out.shape, z.shape)
                # Note: timm vit models return (N, 768) directly, no need to access last_hidden_state
            elif self.backbone == 'dino-resnet-50':
                #! (N, 2048, 7, 7) => pool => (N, 2048)
                out_temp = self.enc_pt(x)
                zdim_ = out_temp.last_hidden_state.shape[1]
                out = out_temp.pooler_output.reshape(-1, zdim_)
                z = out
            elif self.backbone == 'resnet18' or self.backbone == 'resnet50':
                #! (N, 2048, 7, 7) => pool => (N, 2048)
                #! (N, 512, 7, 7) => pool => (N, 512)
                out_temp = self.enc_pt(x)
                zdim_ = out_temp.shape[1]
                out = out_temp.reshape(-1, zdim_)
                z = out
            else: raise ValueError
        
        wrap = self.dec(z)
        return wrap, z


class DecoderManual(nn.Module):
    def __init__(self, i_dim, src_dim, act=nn.GELU, arch='vit-base'):
        super(DecoderManual, self).__init__()
        if i_dim: self.shared_feature = 1
        else:     self.shared_feature = 0
        if self.shared_feature:
            #! start from 7*7*16(784:16) or 7*7*32(1568:800) or 7*7*64(3,136:2368)
            if (src_dim % 49) != 0: raise ValueError('map dim must be devided with 7*7')
            self.p_trigger = torch.nn.Parameter(torch.Tensor(1, src_dim - i_dim))
            torch.nn.init.uniform_(self.p_trigger, a=0.0, b=0.1) # can be tuned
            src_c = src_dim // 49
        else:
            src_c = src_dim
        
        bias_flag = False
        body_seq = []
        
        if arch in ['vit-mae-base', 'vit-base', 'vit-base16', 'vit-base32']:
            if src_c >= 64:    g_c = 64
            else:              g_c = src_c
            body_seq              +=  [nn.ConvTranspose2d(src_c, 64, 2, 2, 0, groups=g_c),
                                       nn.ConvTranspose2d(64, 64, kernel_size=1, bias=bias_flag)]
            body_seq              +=  [nn.BatchNorm2d(64), act()]
            body_seq              +=  [nn.ConvTranspose2d(64, 64, 2, 2, 0, groups=64),
                                       nn.ConvTranspose2d(64, 32, kernel_size=1, bias=bias_flag)]
            body_seq              +=  [nn.BatchNorm2d(32), act()]
            body_seq              +=  [nn.ConvTranspose2d(32, 32, 2, 2, 0, groups=32),
                                       nn.ConvTranspose2d(32, 32, kernel_size=1, bias=bias_flag)]
            body_seq              +=  [nn.BatchNorm2d(32), act()]
            body_seq              +=  [nn.ConvTranspose2d(32, 32, 2, 2, 0, groups=32),
                                       nn.ConvTranspose2d(32, 16, kernel_size=1, bias=bias_flag)]
            body_seq              +=  [nn.BatchNorm2d(16), act()]
            body_seq              +=  [nn.ConvTranspose2d(16, 3, 2, 2, 0, bias=bias_flag)]  
        elif arch in ['dino-resnet-50', 'resnet18', 'resnet50']:
            body_seq              +=  [nn.ConvTranspose2d(src_c, 64, 2, 2, 0, groups=32),
                                       nn.ConvTranspose2d(64, 64, kernel_size=1, bias=bias_flag)]
            body_seq              +=  [nn.BatchNorm2d(64), act()]
            body_seq              +=  [nn.ConvTranspose2d(64, 64, 2, 2, 0, groups=64),
                                       nn.ConvTranspose2d(64, 32, kernel_size=1, bias=bias_flag)]
            body_seq              +=  [nn.BatchNorm2d(32), act()]            
            body_seq              +=  [nn.ConvTranspose2d(32, 32, 2, 2, 0, groups=32),
                                       nn.ConvTranspose2d(32, 32, kernel_size=1, bias=bias_flag)]
            body_seq              +=  [nn.BatchNorm2d(32), act()]
            body_seq              +=  [nn.ConvTranspose2d(32, 32, 2, 2, 0, groups=32),
                                       nn.ConvTranspose2d(32, 16, kernel_size=1, bias=bias_flag)]
            body_seq              +=  [nn.BatchNorm2d(16), act()]
            body_seq              +=  [nn.ConvTranspose2d(16, 3, 2, 2, 0, bias=bias_flag)]
        else: raise ValueError('not implemented')
        self.body   = nn.Sequential(*body_seq)

    def forward(self, z):
        if self.shared_feature:
            N = z.shape[0]
            D = self.p_trigger.shape[1]
            p_trigger = self.p_trigger.repeat(N, 1)
            z_cube = torch.cat((z, p_trigger), dim=1)
            z_cube = z_cube.reshape(N, -1, 7, 7)
        else:
            return self.body(z)
        return self.body(z_cube)


class EncoderManual(nn.Module):
    def __init__(self, out_dim, act=nn.GELU, gap=False):
        super(EncoderManual, self).__init__()        
        bias_flag = False
        body_seq = []
        body_seq              +=  [nn.Conv2d(3, 32, 3, 1, 1),
                                    nn.Conv2d(32, 32, 2, 2, 0, bias=bias_flag)]
        body_seq              +=  [nn.BatchNorm2d(32), act()]
        body_seq              +=  [nn.Conv2d(32, 32, 3, 1, 1),
                                    nn.Conv2d(32, 64, 2, 2, 0, bias=bias_flag)]
        body_seq              +=  [nn.BatchNorm2d(64), act()]
        body_seq              +=  [nn.Conv2d(64, 64, 3, 1, 1),
                                    nn.Conv2d(64, 64, 2, 2, 0, bias=bias_flag)]
        body_seq              +=  [nn.BatchNorm2d(64), act()]
        body_seq              +=  [nn.Conv2d(64, 64, 3, 1, 1),
                                    nn.Conv2d(64, 128, 2, 2, 0, bias=bias_flag)]
        body_seq              +=  [nn.BatchNorm2d(128), act()]
        body_seq              +=  [nn.Conv2d(128, 128, 3, 1, 1),
                                   nn.Conv2d(128, out_dim, 2, 2, 0, bias=bias_flag)]
        body_seq              +=  [nn.BatchNorm2d(out_dim), act()]
        if gap:     body_seq  +=  [nn.AdaptiveAvgPool2d((1, 1))]
        self.body   = nn.Sequential(*body_seq)

    def forward(self, x):
        return self.body(x)

class BlackVIP(nn.Module):
    def __init__(self, args=None):
        super(BlackVIP, self).__init__()
        # self.args = args
        self.p_eps = args.p_eps
        self.coordinator = Coordinator(args=args)
    
    def forward(self, x):
        prompt, _  = self.coordinator(x)
        prompted_image = prompt * self.p_eps + x
        return prompted_image
    
inv_normalize = transforms.Normalize(
                                    mean=[-0.48145466/0.26862954, 
                                          -0.4578275/0.26130258, 
                                          -0.40821073/0.27577711],
                                    std=[1/0.26862954, 
                                         1/0.26130258, 
                                         1/0.27577711]
                                    )
    
   
from torch.autograd import Variable
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        #if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum() 
