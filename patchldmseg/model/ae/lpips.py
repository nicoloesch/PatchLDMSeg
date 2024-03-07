"""Adapted from https://github.com/SongweiGe/TATS"""

"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""


from collections import namedtuple
from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
import requests
import os
import hashlib

from typing import Literal, Optional, Tuple, List

from patchldmseg.model.ae.vqgan_torch import NLayerDiscriminator
from patchldmseg.utils.misc import Stage

URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

def l1_pixel_loss(x, y):
    return torch.abs(x-y)

def l2_pixel_loss(x, y):
    return torch.pow(x-y, 2)

class VQLPIPSWithDiscriminator(nn.Module):
    r"""Vector-quantized LPIPS with Discriminator following
    Rombach et al. and adapted to 3D with Khader et al

    Parameters
    ----------
    dimensions : int
        Number of dimensions of the input data. Either 2 or 3.
    disc_conditional : bool
        Whether the discriminator is conditional or not. Make sure
        to adapt disc_in_channels accordingly.
    disc_factor : float
        Factor by which the discriminator loss is multiplied.
    disc_layers : int
        Number of layers of the discriminator.
    disc_loss : str
        Discriminator loss. Either 'hinge' or 'vanilla'.
    disc_num_df : int
        Number of hidden channels of the first layer of the discriminator.
    disc_in_channels : int
        Number of input channels of the discriminator.
    disc_start_step : int
        Step at which the discriminator is activated.
    disc_weight : float
        Weight of the discriminator loss.
    gan_image_weight : float
        Weight of the image GAN loss.
    gan_volume_weight : float, optional
        Weight of the volume GAN loss. Only being used if dimensions == 3.
    perceptual_loss : str
        Perceptual loss. Either 'lpips', 'clips' or 'dists'.
    perceptual_loss_weight : float
        Weight of the perceptual loss.
    pixel_loss : str
        Pixel loss. Either 'l1' or 'l2'. This is the reconstruction loss
    pixel_loss_weight : float
        Weight of the pixel loss.
    use_fake_3d: bool
        Whether fake 3D for the perceptual loss is used. If True, the perceptual loss
        takes a slice of each axis and averages the perceptual loss across the slices.
        If false, only axial slices are utilised.
    """

    def __init__(
            self,
            dimensions: int,
            disc_layers: int,
            disc_loss: Literal['hinge', 'vanilla'],
            disc_num_df: int,
            disc_in_channels: int,
            disc_start_epoch: int,
            gan_image_weight: float,
            gan_volume_weight: float,
            gan_feature_weight: float,
            perceptual_loss: Literal['lpips', 'clips', 'dists'],
            perceptual_loss_weight: float,
            pixel_loss: Literal['l1', 'l2'],
            pixel_loss_weight: float,
            disc_conditional: bool = False,
            use_fake_3d: bool = False,
            ) -> None:
        super().__init__()

        assert disc_loss in ['hinge', 'vanilla']
        assert pixel_loss in ['l1', 'l2']
        assert perceptual_loss in ['lpips',] # Implement the others

        # Params
        self.dimensions = dimensions

        # Perceptual Loss
        if perceptual_loss == 'lpips':
            self.perceptual_model = LPIPS().eval()
        self.perceptual_loss_weight = perceptual_loss_weight
        self.use_fake_3d = use_fake_3d

        # Pixel Loss
        if pixel_loss == 'l1':
            self.pixel_loss = l1_pixel_loss
        elif pixel_loss == 'l2':
            self.pixel_loss = l2_pixel_loss
        else:
            raise NotImplementedError(f"Pixel loss {pixel_loss} not implemented")
        self.pixel_loss_weight = pixel_loss_weight

        # Discriminator
        self.image_discriminator = NLayerDiscriminator(
            input_channels=disc_in_channels,
            hidden_channels=disc_num_df,
            dimensions=2,
            num_layers=disc_layers,
            norm_layer=nn.BatchNorm2d
        )

        if dimensions == 3:
            self.volume_discriminator = NLayerDiscriminator(
                input_channels=disc_in_channels,
                hidden_channels=disc_num_df,
                dimensions=3,
                num_layers=disc_layers,
                norm_layer=nn.BatchNorm3d
            )
        else:
            self.volume_discriminator = nn.Identity()

        self.gan_image_weight = gan_image_weight
        self.gan_volume_weight = gan_volume_weight
        self.gan_feature_weight = gan_feature_weight

        self.disc_start_epoch = disc_start_epoch
        self.disc_conditional = disc_conditional

        if disc_loss == 'hinge':
            self.disc_loss = hinge_d_loss
        elif disc_loss == 'vanilla':
            self.disc_loss = vanilla_d_loss
        else:
            raise NotImplementedError(f"Discriminator loss {disc_loss} not implemented")
    
    @staticmethod
    def _extract_2d(
        dimensions: int,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        cond_tensor: Optional[torch.Tensor] = None,
        use_fake_3d: bool = False
        ) -> Tuple[
            List[torch.Tensor], List[torch.Tensor], 
            Optional[torch.Tensor], Optional[torch.Tensor],
            List[Optional[torch.Tensor]], Optional[torch.Tensor]]:
        r"""This function extracts 2D and 3D tensors from the input
        required for the discriminator and autoencoder loss"""

        assert dimensions in [2, 3]
        if use_fake_3d:
            assert dimensions == 3, "Fake 3D only works for 3D models"

        if dimensions == 2:
            assert x.dim() == 4, "Input must be 4D tensor (B, C, H, W)"
            assert x_recon.dim() == 4, "Reconstruction must be 4D tensor (B, C, H, W)"

            x_2d = x
            x_2d_recon = x_recon
            x_3d = None
            x_3d_recon = None
            x_2d_cond_tensor = cond_tensor
            x_3d_cond_tensor = None

        elif dimensions == 3:
            B, C, H, W, D = x.shape
            x_2d_cond_tensor = [None] * dimensions
            x_3d_cond_tensor = cond_tensor

            if use_fake_3d:
                indices_2d = [
                    torch.randint(0, H, (B,), device=x.device).reshape(-1, 1, 1, 1, 1).repeat(1,C,1,W,D),
                    torch.randint(0, W, (B,), device=x.device).reshape(-1, 1, 1, 1, 1).repeat(1,C,H,1,D),
                    torch.randint(0, D, (B,), device=x.device).reshape(-1, 1, 1, 1, 1).repeat(1,C,H,W,1)]
                
                x_2d = [torch.gather(x, dim_iter, indices).squeeze(dim_iter) for dim_iter, indices in zip([2,3,4], indices_2d)]
                x_2d_recon = [torch.gather(x_recon, dim_iter, indices).squeeze(dim_iter) for dim_iter, indices in zip([2,3,4], indices_2d)]
                
                if cond_tensor is not None:
                    x_2d_cond_tensor = [torch.gather(cond_tensor, dim_iter, indices).squeeze(dim_iter) for dim_iter, indices in zip([2,3,4], indices_2d)]

            else:
                indices_2d = torch.randint(
                    0, D, (B, ), device=x.device
                    ).reshape(-1, 1, 1, 1, 1).repeat(1,C,H,W, 1)
                x_2d = torch.gather(x, -1, indices_2d).squeeze(-1)
                x_2d_recon = torch.gather(x_recon, -1, indices_2d).squeeze(-1)

                if cond_tensor is not None:
                    x_2d_cond_tensor = torch.gather(cond_tensor, -1, indices_2d).squeeze(-1)

            x_3d = x
            x_3d_recon = x_recon
        else:
            raise RuntimeError(f"Dimensions must be 2 or 3 not {dimensions}")

        if not isinstance(x_2d, list):
            x_2d = [x_2d]
        if not isinstance(x_2d_recon, list):
            x_2d_recon = [x_2d_recon]
        if not isinstance(x_2d_cond_tensor, list):
            x_2d_cond_tensor = [x_2d_cond_tensor]

        return (
            x_2d, x_2d_recon, 
            x_3d, x_3d_recon,
            x_2d_cond_tensor, x_3d_cond_tensor
        )


    def forward(self,
                x: torch.Tensor,
                stage: Stage,
                x_recon: torch.Tensor,
                mode: Literal['ae', 'disc'],
                commitment_loss: torch.Tensor,
                epoch: int,
                condition_tensor: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, dict]:

        discriminator_factor = 1.0 if epoch >= self.disc_start_epoch else 0.0
        
        # Reconstruction error
        rec_loss = self.pixel_loss(x, x_recon).mean() * self.pixel_loss_weight

        # Unpacking the potential 3D Tensor
        x_2d, x_2d_recon, x_3d, x_3d_recon, x_2d_cond, x_3d_cond = (
            self._extract_2d(
                dimensions=self.dimensions, 
                x=x, 
                x_recon=x_recon,
                cond_tensor=condition_tensor,
                use_fake_3d=self.use_fake_3d))

        if mode == 'ae':
            ae_loss, dict_to_log = self.autoencoder_loss(
                x_2d=x_2d, 
                x_2d_recon=x_2d_recon, 
                stage=stage,
                discriminator_factor=discriminator_factor,
                x_3d=x_3d, 
                x_3d_recon=x_3d_recon,
                condition_tensor_2d=x_2d_cond,
                condition_tensor_3d=x_3d_cond,
                rec_loss=rec_loss)
            
            dict_to_log.update({
                f'loss/commitment_{stage.value}': commitment_loss,
            })
            
            loss = rec_loss + ae_loss + commitment_loss

        elif mode == 'disc':
            loss, dict_to_log = self.discriminator_loss(
                x_2d=x_2d, 
                x_2d_recon=x_2d_recon, 
                stage=stage,
                discriminator_factor=discriminator_factor,
                x_3d=x_3d, 
                x_3d_recon=x_3d_recon,
                condition_tensor_2d=x_2d_cond,
                condition_tensor_3d=x_3d_cond,
            )

        else:
            raise RuntimeError(f"Mode must be 'ae' or 'disc' not {mode}")

        dict_to_log.update({
            f'loss/{stage.value}_{mode}': loss,
        })

        return loss, dict_to_log
    
    def _ae_process(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        discriminator: nn.Module,
        gan_weight: float,
        cond_tensor: Optional[torch.Tensor] = None,
        feat_weights: float = 4.0 / (3 + 1)
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Run single processing of the autoencder either for image
        or volume as both steps are exactly the same and 
        only depnd on the x, x_recon and the discriminator"""

        if cond_tensor is not None:
            assert self.disc_conditional, "Discriminator must be conditional if condition tensor is given"
            x_recon = torch.cat((x_recon, cond_tensor), dim=1)
        else:
            assert not self.disc_conditional, "Discriminator must be unconditional if no condition tensor is given"

        # Discriminator Loss
        logits_fake, pred_fake = discriminator(x_recon)
        g_loss_weighted = -torch.mean(logits_fake) * gan_weight

        # GAN feature matching loss
        # Tune features such that we get the same prediction results on the discriminator
        gan_feat_loss = torch.tensor([0.0]).to(x.device)
        if gan_weight > 0:
            _, pred_real = discriminator(x)
            # NOTE: Could be replaced by L1 loss with reduction = sum
            for i in range(len(pred_fake)-1):
                gan_feat_loss += (
                    feat_weights * 
                    F.l1_loss(pred_fake[i], pred_real[i].detach()) * 
                    (gan_weight > 0))

        # NOTE: The gan_feat_loss here is either video_gan_feat_loss or image_gan_feat_loss
        # depending on dimension - will be combined later

        return g_loss_weighted, gan_feat_loss
        
    def autoencoder_loss(
            self, 
            x_2d: List[torch.Tensor], 
            x_2d_recon: List[torch.Tensor], 
            stage: Stage,
            discriminator_factor: float,
            rec_loss: torch.Tensor,
            x_3d: Optional[torch.Tensor] = None,
            x_3d_recon: Optional[torch.Tensor] = None,
            condition_tensor_2d: List[Optional[torch.Tensor]] = [None],
            condition_tensor_3d: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        r"""Calculates the autoencoder loss based on the perceptual loss, the gan feature loss and the g_loss"""

        assert isinstance(x_2d, list) and isinstance(x_2d_recon, list), "Input and reconstruction must be a list of tensors"
        assert all([t_2d.dim() == 4 for t_2d in x_2d]), "Input must be a list of 4D tensors (B, C, H, W)"
        assert all([t_2d_recon.dim() == 4 for t_2d_recon in x_2d_recon]), "Reconstruction must be a list of 4D tensors (B, C, H, W)"
        device = x_2d[0].device

        if self.use_fake_3d:
            assert self.dimensions == 3, "Fake 3D only works for 3D models"

        # Perceptual Loss
        perceptual_loss = torch.tensor([0.0]).to(device)
        if self.perceptual_loss_weight > 0.0:
            perceptual_loss = (
                torch.mean(
                    torch.stack([self.perceptual_model(t_2d, t_2d_recon).mean() for t_2d, t_2d_recon in zip(x_2d, x_2d_recon)]))
                * self.perceptual_loss_weight)
        
        # rec and perceptual are iid. -> mean(A) + mean(B) = mean(A+B)
        nll_loss = rec_loss + perceptual_loss

        g_image_loss_weighted = []
        image_gan_feat_loss = []

        for t_2d, t_2d_recon, t_2d_cond in zip(x_2d, x_2d_recon, condition_tensor_2d):
            # Discriminator Loss already weighted
            g_image_loss_weighted_it, image_gan_feat_loss_it = self._ae_process(
                x=t_2d, x_recon=t_2d_recon, 
                discriminator=self.image_discriminator, 
                gan_weight=self.gan_image_weight,
                cond_tensor=t_2d_cond)
            g_image_loss_weighted.append(g_image_loss_weighted_it)
            image_gan_feat_loss.append(image_gan_feat_loss_it)

        g_image_loss_weighted = torch.mean(torch.stack(g_image_loss_weighted))
        image_gan_feat_loss = torch.mean(torch.stack(image_gan_feat_loss))
        
        # Autoencoder loss
        ae_loss = discriminator_factor * g_image_loss_weighted

        # GAN feature matching loss - tune features such that we get the same prediction result on the discirminator
        gan_feat_loss = (
            discriminator_factor * self.gan_feature_weight *  image_gan_feat_loss)

        dict_to_log = {}

        if self.dimensions == 3:
            assert x_3d is not None and x_3d.dim() == 5, "Input must be 5D tensor (B, C, H, W, D)"
            assert x_3d_recon is not None and x_3d_recon.dim() == 5, "Input must be 5D tensor (B, C, H, W, D)"
            assert self.dimensions == 3, "Model must be 3D"

            g_volume_loss_weighted, volume_gan_feat_loss = self._ae_process(
                x=x_3d, x_recon=x_3d_recon, 
                discriminator=self.volume_discriminator, 
                gan_weight=self.gan_volume_weight,
                cond_tensor=condition_tensor_3d)
            
            ae_loss += discriminator_factor * g_volume_loss_weighted
            gan_feat_loss += (
                discriminator_factor * self.gan_feature_weight * volume_gan_feat_loss.squeeze())

            dict_to_log.update({            
                f'loss/{stage.value}_g_volume': g_volume_loss_weighted,
                f'loss/{stage.value}_volume_gan_feat': volume_gan_feat_loss,
            })

        # Log afterwards as I am overwriting some variables in the 3D processing
        dict_to_log.update({
            f'loss/{stage.value}_g_image': g_image_loss_weighted,
            f'loss/{stage.value}_image_gan_feat': image_gan_feat_loss,
            f'loss/{stage.value}_ae': ae_loss,
            f'loss/{stage.value}_gan_feat': gan_feat_loss,
            f'loss/{stage.value}_perceptual': perceptual_loss,
            f'loss/{stage.value}_nll': nll_loss,
            f'loss/{stage.value}_rec': rec_loss
        })
        return ae_loss + perceptual_loss + gan_feat_loss, dict_to_log
    
    def _dl_process(
        self,
        x: torch.Tensor, 
        x_recon: torch.Tensor, 
        discriminator: nn.Module,
        cond_tensor: Optional[torch.Tensor] = None):
        r"""Run single processing of the discriminator either for image
        or volume as both steps are exactly the same and only
        depend on the x, x_recon and the discriminator"""

        if cond_tensor is not None:
            assert self.disc_conditional, "Discriminator must be conditional if condition tensor is given"
            x_recon = torch.cat((x_recon.detach(), cond_tensor), dim=1)
            x = torch.cat((x.detach(), cond_tensor), dim=1)
        else:
            assert not self.disc_conditional, "Discriminator must be unconditional if no condition tensor is given"

        # Discriminator Loss
        logits_real, _ = discriminator(x.detach())
        logits_fake, _ = discriminator(x_recon.detach())
        disc_loss = self.disc_loss(logits_real, logits_fake)
        return disc_loss

    def discriminator_loss(
            self, 
            x_2d: List[torch.Tensor],
            x_2d_recon: List[torch.Tensor],
            stage: Stage,
            discriminator_factor: float,
            x_3d: Optional[torch.Tensor] = None,
            x_3d_recon: Optional[torch.Tensor] = None,
            condition_tensor_2d: List[Optional[torch.Tensor]] = [None],
            condition_tensor_3d: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, dict]:

        assert isinstance(x_2d, list) and isinstance(x_2d_recon, list), "Input and reconstruction must be a list of tensors"
        assert all([t_2d.dim() == 4 for t_2d in x_2d]), "Input must be a list of 4D tensors (B, C, H, W)"
        assert all([t_2d_recon.dim() == 4 for t_2d_recon in x_2d_recon]), "Reconstruction must be a list of 4D tensors (B, C, H, W)"
        
        discriminator_image_loss = torch.mean(
            torch.stack([self._dl_process(
                x=t_2d, 
                x_recon=t_2d_recon, 
                discriminator=self.image_discriminator, 
                cond_tensor=t_2d_cond) for t_2d, t_2d_recon, t_2d_cond in zip(x_2d, x_2d_recon, condition_tensor_2d)]
            )
        )
        
        discriminator_loss = (
            discriminator_factor * 
            discriminator_image_loss * self.gan_image_weight)

        dict_to_log = {}

        if self.dimensions == 3:
            assert x_3d is not None and x_3d.dim() == 5, "Input must be 5D tensor (B, C, H, W, D)"
            assert x_3d_recon is not None and x_3d_recon.dim() == 5, "Input must be 5D tensor (B, C, H, W, D)"
            assert self.dimensions == 3, "Model must be 3D"

            discriminator_volume_loss = self._dl_process(
                x=x_3d, x_recon=x_3d_recon, 
                discriminator=self.volume_discriminator,
                cond_tensor=condition_tensor_3d)
            discriminator_loss += (
                discriminator_factor * 
                discriminator_volume_loss * self.gan_volume_weight)

            dict_to_log.update({
                f'loss/{stage.value}_disc_volume': discriminator_volume_loss,
            }) 

        dict_to_log.update({
            f'loss/{stage.value}_disc_image': discriminator_image_loss,
            f'loss/{stage.value}_disc': discriminator_loss,
        })

        return discriminator_loss, dict_to_log 
       

def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(
            name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


class LPIPS(nn.Module):
    r"""Taken from 
    https://github.com/CompVis/taming-transformers/taming/modules/losses/lpips.py"""
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "cache"))
        self.load_state_dict(torch.load(
            ckpt, map_location=torch.device("cpu")), strict=False)
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name, os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "cache"))
        model.load_state_dict(torch.load(
            ckpt, map_location=torch.device("cpu")), strict=False)
        return model

    def forward(self, input, target):
        assert input.shape == target.shape
        B, C, *spat = input.shape
        # Create RGB Tensor
        input = input.reshape(B*C, 1, *spat).expand(B*C, 3, *spat)
        target = target.reshape(B*C, 1, *spat).expand(B*C, 3, *spat)

        in0_input, in1_input = (self.scaling_layer(
            input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(
                outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True)
               for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor(
            [-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor(
            [.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1,
                             padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, weights=models.vgg.VGG16_Weights.IMAGENET1K_V1):
        super(vgg16, self).__init__()
        vgg_pretrained_features: nn.Sequential = models.vgg16(weights=weights).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2,
                          h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)
