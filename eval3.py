import argparse
import random

import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import image_dataset
from LRP_util import *
from modules.resnet import resnet50
from modules.vgg import vgg16_bn, vgg19_bn
from torchvision.datasets import ImageNet
from models.jbuparam_model import PlainGuidedUpsampler, JBUParamModel

# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('--models', type=str, default='resnet50',
                    help='resnet50 or vgg16 or vgg19')
parser.add_argument('--cam', type=str, default='relevancecam',
                    help='cam method')
parser.add_argument('--upsample', type=str, default=None,
                    help='upsample method')
parser.add_argument('--target_layer', type=str, default='layer2',
                    help='target_layer')
parser.add_argument('--target_class', type=int, default=None,
                    help='target_class')
parser.add_argument('--seed', type=int, default=0,
                    help='seed')
parser.add_argument('--inp_size', type=int, default=224,
                    help='image side length')
args = parser.parse_args()

###########################################################################################################################
model_arch = args.models

if model_arch == 'vgg16':
    model = vgg16_bn(pretrained=True).cuda().eval()  #####
    target_layer = model.features[int(args.target_layer)]
elif model_arch == 'vgg19':
    model = vgg19_bn(pretrained=True).cuda().eval()  #####
    target_layer = model.features[int(args.target_layer)]
elif model_arch == 'resnet50':
    model = resnet50(pretrained=True).cuda().eval() #####
    if args.target_layer == 'layer1':
        target_layer = model.layer1
    elif args.target_layer == 'layer2':
        target_layer = model.layer2
        c = 512
    elif args.target_layer == 'layer3':
        target_layer = model.layer3
    elif args.target_layer == 'layer4':
        target_layer = model.layer4
        c = 2048
#######################################################################################################################
scale = 2

if args.upsample == 'jbu-plain':
    upsampler = PlainGuidedUpsampler(radius=2, sigma_spatial=0.6, sigma_range_factor=1.4)
elif args.upsample == 'jbu-sigma':
    shell = JBUParamModel(transform='NONE', error_fn='mean_cosine_distance', use_fixed_sigmas=False, transform_layers=2, featurizer='resnet50-relevancecam', target_layer='layer4', learn_guidance=False, upsample_factor=scale)
    upsampler = shell.load_from_checkpoint(
        "/data/scratch-oc40/fus/relevance_cam/lightning_logs/version_2/checkpoints/epoch=4-step=1601459.ckpt")
    upsampler = upsampler.cuda()
elif args.upsample == 'jbu-transformed':
    shell = JBUParamModel(transform='POST', error_fn='mean_cosine_distance', use_fixed_sigmas=True, transform_layers=2, featurizer='resnet50-relevancecam', target_layer='layer4', learn_guidance=False, upsample_factor=scale)
    upsampler = shell.load_from_checkpoint(
        "/data/scratch/fus/trainval/lightning_logs/version_68/checkpoints/epoch=0-step=320291.ckpt")
    upsampler = upsampler.cuda()
elif args.upsample == 'jbu-guidance':
    shell = JBUParamModel(transform='NONE', error_fn='mean_cosine_distance', use_fixed_sigmas=True, transform_layers=0, featurizer='resnet50-relevancecam', target_layer='layer4', learn_guidance=True, upsample_factor=scale)
    upsampler = shell.load_from_checkpoint(
        # "/data/scratch/fus/trainval/lightning_logs/version_62/checkpoints/epoch=4-step=400364.ckpt")
        "/data/scratch/fus/trainval/lightning_logs/version_64/checkpoints/epoch=0-step=160145.ckpt")
elif args.upsample == 'jbu-sigma-transformed-guidance':
    shell = JBUParamModel(transform='POST', error_fn='mean_cosine_distance', use_fixed_sigmas=False, transform_layers=2, featurizer='resnet50-relevancecam', target_layer='layer4', learn_guidance=True, upsample_factor=scale)
    upsampler = shell.load_from_checkpoint(
        "/data/scratch-oc40/fus/relevance_cam/lightning_logs/version_6/checkpoints/epoch=1-step=160145.ckpt")
    upsampler = upsampler.cuda()
else:
    upsampler = None

seed = args.seed
print(f'Seed: {seed}')
torch.manual_seed(seed)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

value = dict()
def forward_hook(module, input, output):
    value['activations'] = output
def backward_hook(module, input, output):
    value['gradients'] = output[0]

target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

normalize_tensor = transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])

imagenet_train_224 = transforms.Compose([
    transforms.Resize((args.inp_size, args.inp_size)),
    transforms.ToTensor(),
    normalize_tensor
])
n = 1
cache = f'./val_paths_{model_arch}.txt'
# cache = './dataset/imagenet/val.txt'
dataset = image_dataset.ImageDataset('./dataset/imagenet', cache, transform=imagenet_train_224)
loader = DataLoader(dataset, batch_size=n, shuffle=False, worker_init_fn=seed_worker, generator=g)

weight_dict = dict()
weight_dict['y'] = list()
weight_dict['relevance'] = list()

# data_idx = []

i = 0
k = 0
with tqdm(total=2000) as pbar:
    for j, (idx, img, label) in enumerate(loader):
        if i == 2000:
            break
        img = img.cuda()
        label = label.cuda()

        # img_path_long = './picture/{}'.format('snake2.JPG')
        # img = cv2.imread(img_path_long,1)
        # img = np.float32(cv2.resize(img, (224, 224))) / 255
        # in_tensor = preprocess_image(img).cuda()

        relevance_cam, output = model(img, args.target_layer, [args.target_class], upsampler=upsampler, scale=scale)
        maxindex = torch.argmax(output, dim=-1)
        if args.cam == 'relevance_cam':
            relevance_cam = normalize(relevance_cam)
            cam = torch.nn.Upsample(size=(224, 224), mode='bilinear')(relevance_cam)

            # cam = tensor2image(relevance_cam)
        elif args.cam == 'score_cam':
            Score_CAM_class = ScoreCAM(model, target_layer)
            score_map, _ = Score_CAM_class(img, class_idx=maxindex)
            cam = score_map
            # cam = score_map.detach().cpu().numpy()

        elif args.cam == 'grad_cam':
            output[:, maxindex].sum().backward(retain_graph=True)
            activation = value['activations']  # [1, 2048, 7, 7]
            gradient = value['gradients']  # [1, 2048, 7, 7]

            gradient_ = torch.mean(gradient, dim=(2, 3), keepdim=True)
            grad_cam = activation * gradient_
            grad_cam = torch.sum(grad_cam, dim=(0, 1))
            grad_cam = torch.clamp(grad_cam, min=0)
            cam = torch.nn.Upsample(size=(224, 224), mode='bilinear')(grad_cam.unsqueeze(0).unsqueeze(0))

        elif args.cam == 'grad_cam_pp':
            output[:, maxindex].sum().backward(retain_graph=True)
            activation = value['activations']  # [1, 2048, 7, 7]
            gradient = value['gradients']  # [1, 2048, 7, 7]
            gradient_2 = gradient ** 2
            gradient_3 = gradient ** 3

            alpha_numer = gradient_2
            alpha_denom = 2 * gradient_2 + torch.sum(activation * gradient_3, axis=(2, 3), keepdims=True)  # + 1e-2
            alpha = alpha_numer / alpha_denom
            w = torch.sum(alpha * torch.clamp(gradient, 0), axis=(2, 3), keepdims=True)
            grad_campp = activation * w
            grad_campp = torch.sum(grad_campp, dim=(0, 1))
            grad_campp = torch.clamp(grad_campp, min=0)
            cam = torch.nn.Upsample(size=(224, 224), mode='bilinear')(grad_campp.unsqueeze(0).unsqueeze(0))

        cam_sort = torch.sort(cam.reshape(n, -1), dim=-1)[0]
        # cam = cam * (cam > cam_sort[:, -int(224*224*0.5)].unsqueeze(1).unsqueeze(1).unsqueeze(1)) ## top 50% pixels
        cam = (cam > cam_sort[:, -int(224*224*0.5)].unsqueeze(1).unsqueeze(1).unsqueeze(1)) ## top 50% pixels

        # cam_in = Variable(preprocess_image(img * (cam[...,np.newaxis]))).cuda()
        img = torch.nn.Upsample(size=(224, 224), mode='bilinear')(img)
        masked_img = img * cam

        masked_out = F.softmax(model(masked_img), dim=1)
        original_out = F.softmax(output, dim=1)

        original_pred = torch.argmax(original_out, dim=-1)

        # if y_maxindex == label[0]:
        weight_dict['y'].extend(original_out[torch.arange(original_out.size(0)), original_pred].detach().cpu().tolist())
        weight_dict['relevance'].extend(masked_out[torch.arange(masked_out.size(0)), original_pred].detach().cpu().tolist())
        # data_idx.append(idx)
        i += n
        pbar.update(n)

original_out = np.array(weight_dict['y'])
relevance_dict = np.array(weight_dict['relevance'])
posit_r = (original_out - relevance_dict) > 0
relevance_aver = (original_out - relevance_dict) * posit_r / original_out
# r_denom = relevance_aver > 0

print("Average Drop:", np.sum(relevance_aver) * 100 / 2000)

print("Average Increase:", (1-posit_r.mean()) * 100)
