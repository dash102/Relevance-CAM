import pytorch_lightning as pl
import torch.nn
import torch.nn as nn
import torchvision

from jbu import jbu_mark, jbu_original, jbu_crf
from models import resnet_infocam, resnet_relevancecam
from utils.util import *
import utils.metrics
from utils.display import build_grid

class MLPTransformModel(nn.Module):
    def __init__(self, layers, channels=2048):
        super().__init__()
        all_layers = [nn.Conv2d(channels, channels, kernel_size=1)]
        for i in range(1, layers):
            all_layers.extend([nn.ReLU(),
                               nn.Conv2d(channels, channels, kernel_size=1)])
        self.conv = nn.Sequential(*all_layers)

        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=1e-8)
                nn.init.normal_(m.bias, mean=0.0, std=1e-8)

    def forward(self, x):
        return self.conv(x)


class LinearTransformModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2048, 2048, kernel_size=1)
        nn.init.normal_(self.conv.weight, mean=0.0, std=1e-8)
        nn.init.normal_(self.conv.bias, mean=0.0, std=1e-8)

    def forward(self, x):
        return self.conv(x)


class CRFFeaturizer(nn.Module):
    def __init__(self, model, num_splits, size=None):
        super().__init__()
        layers = list(model.children())
        self.size = size
        self.featurizers_for_guidance = []
        first_chunk = 7 - num_splits
        self.featurizers_for_guidance.append(nn.Sequential(*layers[:first_chunk]))
        for remaining in range(first_chunk, 8):
            self.featurizers_for_guidance.append(layers[remaining])
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        guidance_features = []
        for i in range(len(self.featurizers_for_guidance) - 1):
            x = self.featurizers_for_guidance[i](x)
            guidance_features.append(torch.nn.Upsample(size=self.size, mode='bilinear')(x))
        source = self.featurizers_for_guidance[-1](x)
        return guidance_features, source


class Featurizer(nn.Module):
    def __init__(self, model, guidance_layer, mode=None):
        super().__init__()
        self.mode = mode
        if self.mode is None:
            layers = list(model.children())
            self.featurizer_for_guidance = nn.Sequential(*layers[:guidance_layer])
            self.featurizer_remaining = nn.Sequential(*layers[guidance_layer:-2])
        else:
            self.model = model
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):  # mode for relevanceCAM
        if self.mode is None:
            guidance_features = self.featurizer_for_guidance(x)
            features = self.featurizer_remaining(guidance_features)
            return guidance_features, features
        else:
            return None, self.model(x, self.mode)


class GuidanceModel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class JBUParamModel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("JBUParamModel")

        parser.add_argument('--upsample_factor', type=int, default=2)
        parser.add_argument('--lr', type=float, default=3e-4)

        parser.add_argument('--use_target', action='store_true')
        parser.add_argument('--no-use_target', action='store_false')
        parser.set_defaults(use_target=False)

        parser.add_argument('--use_fixed_sigmas', action='store_true')
        parser.add_argument('--no-use_fixed_sigmas', action='store_false')
        parser.set_defaults(use_target=False)

        parser.add_argument('--learn_guidance', action='store_true')
        parser.add_argument('--no-learn_guidance', action='store_false')
        parser.set_defaults(learn_guidance=False)

        parser.add_argument('--transform', type=lambda loc: TransformLocation[loc], choices=list(TransformLocation),
                            required=True, help='type of feature transform: NONE, PRE, POST')
        parser.add_argument('--radius', default=2, type=int,
                            help='number of PCA components to upsample. If == source channels, no PCA is performed')

        parser.add_argument('--error_fn', type=str, required=True, help='metric to evaluate JBU accuracy')
        parser.add_argument('--featurizer', type=str, help='resnet50, resnet50-relevancecam, or resnet50-infocam')
        parser.add_argument('--target_layer', type=str, help='layer2, layer4 for resnet50. 23, 43 for vgg16.')
        parser.add_argument('--batch_size', default=8, type=int)

        parser.add_argument('--transform_layers', type=int, default=2)

        return parent_parser

    def __init__(self, **args):
        super().__init__()
        self.save_hyperparameters()
        if not self.hparams.use_fixed_sigmas:
            weights = torch.empty(2, 1)
            torch.nn.init.uniform_(weights[0], a=0.5, b=3)
            torch.nn.init.uniform_(weights[1], a=0.5, b=1.5)
            self.weights = nn.Parameter(weights)
            print('Learning weights:', self.weights)
        self.constraint_fn = torch.abs
        if self.hparams.transform != TransformLocation.NONE:
            self.transform_model = MLPTransformModel(self.hparams.transform_layers, channels=2048)
        self.metric = getattr(utils.metrics, self.hparams.error_fn)

        # mapping scale to (layer of featurizer that will produce features for guidance, # channels)
        # scale_mapping = {16: (4, 112), 8: (5, 256), 4: (6, 512), 2: (7, 1024)}
        # layer_i, c = scale_mapping[self.hparams.upsample_factor]
        if self.hparams.featurizer == 'resnet50':
            self.featurizer = Featurizer(torchvision.models.resnet50(pretrained=True).to(device), None)
        elif self.hparams.featurizer == 'resnet50-infocam':
            self.featurizer = Featurizer(resnet_infocam.resnet50(pretrained=True,
                                num_classes=200).to(device), None)
        elif self.hparams.featurizer == 'resnet50-relevancecam':
            self.featurizer = Featurizer(resnet_relevancecam.resnet50(pretrained=True).to(device), None, mode=self.hparams.target_layer)
        else:
            raise ValueError(f'{self.hparams.featurizer} is not a valid featurizer arg')
        # if self.hparams.learn_guidance:
        #     self.guidance_model = GuidanceModel(c + 3)

    def forward(self, source, guidance):
        if self.hparams.use_fixed_sigmas:
            sigma_spatial = torch.tensor([0.6]).to(source.device)
            sigma_range_factor = torch.tensor([1.4]).to(source.device)
        else:
            sigma_spatial, sigma_range_factor = self.weights
        sigma_range = sigma_range_factor * torch.std(guidance, dim=(1, 2, 3))
        if self.hparams.transform == TransformLocation.PRE:
            source = self.transform_model(source) + source
        sigma_spatial = self.constraint_fn(sigma_spatial)
        sigma_range = self.constraint_fn(sigma_range)
        out = jbu_mark(source, guidance, radius=self.hparams.radius, sigma_spatial=sigma_spatial, sigma_range=sigma_range).squeeze()
        if len(out.shape) < 4:
            out = out.unsqueeze(0)
        if self.hparams.transform == TransformLocation.POST:
            out = self.transform_model(out) + out
        return out

    def training_step(self, batch, batch_idx):
        source_input, guidance_img, comparison_input = batch
        _, comparison = self.featurizer(comparison_input)
        guidance_features, source = self.featurizer(source_input)
        if self.hparams.learn_guidance:
            guidance_features = torch.nn.Upsample(size=guidance_img.shape[-2:], mode='bilinear')(guidance_features)
            guidance = self.guidance_model(torch.cat([guidance_img, guidance_features], dim=1))
        else:
            guidance = guidance_img
        jbu_source = self(source, guidance)
        loss_batch = self.metric(jbu_source, comparison)
        loss = loss_batch.mean()
        self.log('train_loss', loss)
        if self.global_step % self.hparams.log_every_n_steps == 0:
            self.logger.experiment.add_scalar('loss/train', loss, self.global_step)
            self.logger.experiment.add_image('train_samples', build_grid(source, comparison, jbu_source, guidance, wh=14),
                                             self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        source_input, guidance_img, comparison_input = batch
        _, comparison = self.featurizer(comparison_input)
        guidance_features, source = self.featurizer(source_input)
        if self.hparams.learn_guidance:
            guidance_features = torch.nn.Upsample(size=guidance_img.shape[-2:], mode='bilinear')(guidance_features)
            guidance = self.guidance_model(torch.cat([guidance_img, guidance_features], dim=1))
        else:
            guidance = guidance_img
        jbu_source = self(source, guidance)
        loss_batch = self.metric(jbu_source, comparison)
        loss = loss_batch.mean()
        self.log('val_loss', loss)
        self.logger.experiment.add_scalar('loss/val', loss, self.global_step)
        self.logger.experiment.add_image('val_samples', build_grid(source, comparison, jbu_source, guidance, wh=14),
                                         self.global_step)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class JBUCRFParamModel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("JBUParamModel")

        parser.add_argument('--upsample_factor', type=int, default=2)
        parser.add_argument('--lr', type=float, default=3e-4)

        parser.add_argument('--radius', default=2, type=int,
                            help='number of PCA components to upsample. If == source channels, no PCA is performed')

        parser.add_argument('--error_fn', type=str, required=True, help='metric to evaluate JBU accuracy')
        parser.add_argument('--batch_size', default=8, type=int)

        parser.add_argument('--num_guidances', type=int, required=True, help='number of guidance components')
        parser.add_argument('--transform', type=lambda loc: TransformLocation[loc], choices=list(TransformLocation),
                            required=True, help='type of feature transform: NONE, PRE, POST')
        parser.add_argument('--transform_layers', type=int, default=2)

        return parent_parser

    def __init__(self, **args):
        super().__init__()
        self.save_hyperparameters()
        weights = torch.empty(self.hparams.num_guidances + 2, 1)
        self.constraint_fn = torch.abs
        torch.nn.init.uniform_(weights[0], a=0.5, b=3)
        for i in range(1, len(weights)):
            torch.nn.init.uniform_(weights[i], a=0.5, b=1.5)
        self.weights = nn.Parameter(weights)

        self.metric = getattr(utils.metrics, self.hparams.error_fn)

        resnet = torchvision.models.resnet50(pretrained=True).to(device)
        resnet = resnet.eval()
        self.featurizer = Featurizer(resnet, self.hparams.num_guidances - 1, size=(224, 224))
        self.sigma_spatial_path = []
        self.sigma_range_path = []

        self.splits = [3, 1024, 512, 256, 112][:self.hparams.num_guidances + 1]
        if self.hparams.transform != TransformLocation.NONE:
            self.transform_model = MLPTransformModel(self.hparams.transform_layers)

    def forward(self, source, guidance):
        sigma_spatial = self.weights[:1]
        sigma_range_factors = self.weights[1:]
        sigma_spatial = self.constraint_fn(sigma_spatial)
        sigma_range_factors = self.constraint_fn(sigma_range_factors)
        self.sigma_spatial_path.append(self.weights[0].item())
        self.sigma_range_path.append(self.weights[1].item())
        out = jbu_crf(source, guidance, radius=self.hparams.radius,
                      sigma_spatial=sigma_spatial, sigma_range_factors=sigma_range_factors, splits=self.splits)
        if len(out.shape) < 4:
            out = out.unsqueeze(0)
        if self.hparams.transform == TransformLocation.POST:
            out = self.transform_model(out) + out
        return out

    def training_step(self, batch, batch_idx):
        source_input, guidance_img, comparison_input = batch
        _, comparison = self.featurizer(comparison_input)
        guidance_features, source = self.featurizer(source_input)
        guidance = guidance_img
        if self.hparams.num_guidances > 0:
            guidance = torch.cat([guidance_img, *guidance_features], dim=1)
        jbu_source = self(source, guidance)
        loss_batch = self.metric(jbu_source, comparison)
        loss = loss_batch.mean()
        self.log('train_loss', loss)
        if self.global_step % self.hparams.log_every_n_steps == 0:
            self.logger.experiment.add_scalar('loss/train', loss, self.global_step)
            self.logger.experiment.add_image('train_samples', build_grid(source, comparison, jbu_source, guidance, wh=224),
                                             self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        source_input, guidance_img, comparison_input = batch
        _, comparison = self.featurizer(comparison_input)
        guidance_features, source = self.featurizer(source_input)
        guidance = guidance_img
        if self.hparams.num_guidances > 0:
            guidance = torch.cat([guidance_img, *guidance_features], dim=1)
        jbu_source = self(source, guidance)
        loss_batch = self.metric(jbu_source, comparison)
        loss = loss_batch.mean()
        self.log('val_loss', loss)
        if self.global_step % self.hparams.log_every_n_steps == 0:
            self.logger.experiment.add_scalar('loss/val', loss, self.global_step)
            self.logger.experiment.add_image('val_samples', build_grid(source, comparison, jbu_source, guidance, wh=224),
                                             self.global_step)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class PlainGuidedUpsampler(nn.Module):
    def __init__(self, radius=2, sigma_spatial=2.5, sigma_range=None, sigma_range_factor=None):
        super().__init__()
        self.radius = radius
        self.sigma_spatial = sigma_spatial
        self.sigma_range = sigma_range
        self.sigma_range_factor = sigma_range_factor

    def forward(self, source, guidance):
        if self.sigma_range_factor is not None:
            sigma_range = self.sigma_range_factor * torch.std(guidance, dim=(1, 2, 3))
        else:
            sigma_range = self.sigma_range
        out = jbu_mark(source, guidance, radius=self.radius,
                       sigma_spatial=self.sigma_spatial, sigma_range=sigma_range)
        return out


class GuidedUpsampler(nn.Module):
    def __init__(self, transform_layers, channels=2048, guidance_channels=2048, radius=2):
        super().__init__()
        # weights = torch.empty(2, 1)
        self.radius = radius
        self.constraint_fn = torch.abs
        # torch.nn.init.uniform_(weights[0], a=0.5, b=3)
        # torch.nn.init.uniform_(weights[1], a=0.5, b=1.5)
        # self.weights = nn.Parameter(weights)
        self.transform_model = MLPTransformModel(transform_layers, channels=channels)  # transform_layers
        self.metric = getattr(utils.metrics, 'mean_cosine_distance')  # error_fn
        # self.guidance_model = GuidanceModel(guidance_channels + 3)

    def forward(self, source, guidance_input):
        # guidance = self.guidance_model(guidance_input)
        guidance = guidance_input
        sigma_spatial = 0.6
        sigma_range_factor = 1.4
        sigma_range = sigma_range_factor * torch.std(guidance, dim=(1, 2, 3))
        # sigma_spatial = self.constraint_fn(sigma_spatial)
        # sigma_range = self.constraint_fn(sigma_range)
        out = jbu_mark(source, guidance, radius=self.radius, sigma_spatial=sigma_spatial, sigma_range=sigma_range).squeeze()
        if len(out.shape) < 4:
            out = out.unsqueeze(0)
        out = self.transform_model(out) + out
        return out
