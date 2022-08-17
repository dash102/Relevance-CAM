import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch import nn

from PIL import Image
from LRP_util import *
from demo_jbuparam_model import JBUParamModel, PlainGuidedUpsampler

from demo_layers import Conv2d, BatchNorm2d, ReLU, MaxPool2d, Sequential, Bottleneck, conv1x1, Linear
from demo_utils import feature_pca, mean_cosine_distance
import torch.utils.model_zoo as model_zoo


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc = Linear(512 * block.expansion, 1000)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(*layers)

    def upsample(self, source, guidance_unscaled, upsampler, scale):
        _, _, H, W = source.shape
        guidance = F.interpolate(guidance_unscaled, size=(H * scale, W * scale), mode='bilinear', antialias=True)
        return upsampler(source, guidance)

    def forward(self, inp, upsampler=None):
        x = self.conv1(inp)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        f, ax = plt.subplots(1, 6, figsize=(12, 4))
        H = 7
        W = 7
        scale = 2

        ax[0].set_title('Input RGB\n(3, 224, 224)')
        ax[0].imshow(inp[0].permute(1, 2, 0).detach().cpu())

        ax[1].set_title('Guidance\n(3, 14, 14)')
        guidance = F.interpolate(inp, size=(H * scale, W * scale), mode='bilinear', antialias=True)
        ax[1].imshow(guidance[0].permute(1, 2, 0).detach().cpu())

        ax[2].set_title('Low-res features\n(2048, 7, 7)')
        source_pca, pcas = feature_pca(layer4, return_pca=True)
        ax[2].imshow(source_pca[0])

        comp = nn.Upsample(scale_factor=2)(inp)
        x_comp = self.conv1(comp)
        x_comp = self.bn1(x_comp)
        x_comp = self.relu(x_comp)
        x_comp = self.maxpool(x_comp)

        layer1_comp = self.layer1(x_comp)
        layer2_comp = self.layer2(layer1_comp)
        layer3_comp = self.layer3(layer2_comp)
        layer4_comp = self.layer4(layer3_comp)

        ax[3].set_title('Features from 2x RGB\n(2048, 14, 14)')
        ax[3].imshow(feature_pca(layer4_comp, pcas=pcas)[0])

        upsampler_plain = PlainGuidedUpsampler(radius=2, sigma_spatial=0.6, sigma_range_factor=1.4)

        ax[4].set_title('Plain JBU\n(2048, 14, 14)')
        upsampled_plain = self.upsample(layer4, inp, upsampler_plain, scale)
        ax[4].imshow(feature_pca(upsampled_plain, pcas=pcas)[0])

        ax[5].set_title('JBU+MLP\n(2048, 14, 14)')
        upsampled_transformed = self.upsample(layer4, inp, upsampler, scale)
        ax[5].imshow(feature_pca(upsampled_transformed, pcas=pcas)[0])

        for i in range(6):
            ax[i].set_axis_off()

        plt.show()

        print()
        print('mean cosine distance: JBU + learned transform')
        print(mean_cosine_distance(layer4_comp, self.upsample(layer4, inp, upsampler, scale)))
        print('mean cosine distance: plain JBU')
        print(mean_cosine_distance(layer4_comp, self.upsample(layer4, inp, upsampler_plain, scale)))


def resnet50(pretrained=False):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    return model

model = resnet50(pretrained=True).cuda().eval()
target_layer = model.layer4
c = 2048

shell = JBUParamModel(transform='POST', error_fn='mean_cosine_distance', use_fixed_sigmas=True, transform_layers=2, featurizer='resnet50-relevancecam', target_layer='layer4')
upsampler_transformed = shell.load_from_checkpoint("/data/scratch/fus/trainval/lightning_logs/version_36/checkpoints/epoch=2-step=240218.ckpt")
upsampler_transformed = upsampler_transformed.cuda()


normalize_tensor = transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])

imagenet_train_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize_tensor
])

path = 'ILSVRC2012_val_00032952.JPEG'
img = Image.open(path)
inp = imagenet_train_224(img).cuda().unsqueeze(0)

model(inp, upsampler=upsampler_transformed)
