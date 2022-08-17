import numbers
from sklearn.decomposition import PCA
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from demo_layers import Conv2d
import math


def feature_pca(features, components=3, pcas=None, return_pca=False):
    b, c, h, w = features.shape
    features_permuted = features.detach().cpu().permute(0, 2, 3, 1)
    features_flat_batched = features_permuted.reshape(b, h * w, c)
    if pcas is None:
        # Fit and transform separate PCA per sample
        pcas = []
        for i in range(b):
            pca = PCA(n_components=components)
            pca.fit(features_flat_batched[i])
            pcas.append(pca)
    features_pca = np.array([pcas[min(i, len(pcas) - 1)].transform(features_flat_batched[i]) for i in range(b)])
    features_pca = features_pca - np.min(features_pca, axis=(1, 2), keepdims=True)
    features_pca /= np.max(features_pca, axis=(1, 2), keepdims=True)
    features_pca = features_pca.reshape(b, h, w, components)
    return (features_pca, pcas) if return_pca else features_pca


def cosine_distance(x1, x2, dim=1):
    cos_sim = F.cosine_similarity(x1, x2, dim=dim).unsqueeze(dim)
    return 1 - cos_sim


def mean_cosine_distance(x1, x2, dim=1):
    return torch.mean(cosine_distance(x1, x2, dim=dim), dim=(-1, -2, -3))


def build_spatial_kernel(sigma_spatials, radius):
    dist_range = torch.arange(radius * 2 + 1) - radius
    x, y = torch.meshgrid(dist_range, dist_range)
    x = x.cuda()
    y = y.cuda()
    num = (x ** 2 + y ** 2)
    denom = 2 * sigma_spatials ** 2
    if len(sigma_spatials) > 1:
        num = torch.repeat_interleave(num.unsqueeze(0), repeats=len(sigma_spatials), dim=0)
        denom = denom.unsqueeze(1).unsqueeze(1)
    else:
        denom = denom.squeeze()
    return torch.exp(-num / denom)

def apply_range_kernel(x, sigma_ranges, parameter_search:bool = False):
    num = x ** 2
    denom = 2 * sigma_ranges ** 2
    if len(sigma_ranges) > 1:
        if parameter_search:
            num = torch.repeat_interleave(num, repeats=len(sigma_ranges), dim=1)
            denom = denom.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(0)
        else:
            for i in range(len(num.shape) - 1):
                denom = denom.unsqueeze(1)
    else:
        denom = denom.squeeze()
    return torch.exp(-num / denom)


def setup_parameter_search(sigma_spatial, sigma_range):
    return torch.repeat_interleave(sigma_spatial, len(sigma_range)), sigma_range.repeat(len(sigma_spatial))


def setup_sigma(sigma):
    if isinstance(sigma, numbers.Number):
        return torch.tensor([sigma])
    elif isinstance(sigma, torch.Tensor):
        return sigma
    return sigma


def jbu(source, guidance, radius=2, sigma_spatial=2.5, sigma_range=None, parameter_search=False, epsilon=1e-8):
    GB, GC, GH, GW = guidance.shape
    SB, SC, SH, SQ = source.shape

    assert (SB == GB)

    scale = SH / GH
    diameter = radius * 2 + 1

    sigma_spatial = setup_sigma(sigma_spatial).to(guidance.device)
    sigma_range = setup_sigma(sigma_range if sigma_range is not None else torch.std(guidance, dim=(1, 2, 3))).cuda()
    if parameter_search:
        sigma_spatial, sigma_range = setup_parameter_search(sigma_spatial, sigma_range)

    dilation = int(math.ceil(1 / scale))
    padding = radius * dilation

    # create high-res copy of low-res source to access floating-point coordinates
    hr_source = torch.nn.Upsample((GH, GW), mode='bilinear', align_corners=False)(source)

    guidance = F.normalize(guidance, dim=1)
    guidance_padded = F.pad(guidance, pad=[padding] * 4, mode='reflect')
    hr_source_padded = F.pad(hr_source, pad=[padding] * 4, mode='reflect')

    kernel_spatial = build_spatial_kernel(sigma_spatial, radius).cuda() \
        .reshape(-1, diameter * diameter, 1, 1)

    range_queries = torch.nn.Unfold(diameter, dilation=dilation)(guidance_padded) \
        .reshape((GB, GC, diameter * diameter, GH, GW)) \
        .permute(0, 1, 3, 4, 2)

    if GC == 1:
        range_kernel = 1 - torch.einsum("bchwp,bchw->bphw", range_queries, guidance)
        range_kernel -= torch.amin(range_kernel, dim=(1, 2, 3), keepdim=True)
        range_kernel /= torch.amax(range_kernel, dim=(1, 2, 3), keepdim=True)
    else:
        range_kernel = (2 - torch.einsum("bchwp,bchw->bphw", range_queries, guidance))

    # range_kernel = torch.exp(-range_kernel**2 / (2 * sigma_range ** 2).squeeze())
    range_kernel = apply_range_kernel(range_kernel, sigma_range, parameter_search).squeeze()

    combined_kernel = range_kernel * kernel_spatial
    combined_kernel /= combined_kernel.sum(1, keepdim=True).clamp(epsilon)

    source_patches = torch.nn.Unfold(diameter, dilation=dilation)(hr_source_padded) \
        .view((SB, SC, diameter * diameter, GH, GW))

    upsampled = torch.einsum('bphw,bcphw->bchw', combined_kernel, source_patches)
    return upsampled


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
        out = jbu(source, guidance, radius=self.radius,
                       sigma_spatial=self.sigma_spatial, sigma_range=sigma_range)
        return out