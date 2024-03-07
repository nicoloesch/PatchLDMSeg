#VQGAN Implementation from Khader et. al

from pytorch_lightning.utilities.types import STEP_OUTPUT
from .base_model import BaseModel
from patchldmseg.model.ae import EMAVectorQuantizer, VQGANDecoder, VQGANEncoder, VQLPIPSWithDiscriminator, VQGANEncoderKhader, VQGANDecoderKhader
from patchldmseg.input.datasets import Dataset, BraTS2023
from patchldmseg.model.basic.conv import ConvolutionalBlock
from patchldmseg.utils.misc import Stage, TASK, SoE
from patchldmseg.utils.visualization import rescale
from patchldmseg.model.diffusion.embedding import PosEmbedding, CoordinateEmbedding
from patchldmseg.utils import metrics

import torch
import torch.nn.functional as F

from typing import Any, Literal, Tuple, Optional, Union



class VQGAN(BaseModel):
    r"""VQGAN model based on a autoencoder architecture. Follows the implementation 
    of Khader et. al. and Rombach et al.

    Parameters
    ----------
    activation : str, optional
        The name of the activation function to be used. MUST match the exact spelling of torch
    act_args : tuple
        Additional arguments of the activation function
    batch_size: int
        The batch size of each step
    channel_factor : tuple
        The multiplicative factor of the channel dimension. It is a multiplicative factor for each
        depth layer of the autoencoder.
    dataset: Dataset
        A dataset object created in the input module/data loader.
    dimensions : int
        Dimensionality of the input. Supported: 2,3
    dropout : int
        Percentage of dropout to be used [0, 100]
    embedding_dim : int
        The dimensionality of the latent embeddings codebook
    hidden_channels : int
        Number of channels the first convolution creates as output channels. The intention behind this is to
        increase the features.
    in_channels : int
        Number of input channels to the first convolution of the entire block
        i.e. the number of channels of the input data
    kernel_size : int
        The size of the kernel for convolution
    norm : str, optional
        The string of the normalization method to be used. Supported are [batch, instance, layer, group, ada_gn]
    num_codes : int
        The number of codes in the codebook
    out_channels : int
        Number of output channels of the network.
    padding : bool
        Whether padding should be used or not. The underlying function determines the optimal padding
    padding_mode : str
        The padding mode to be used. Supported are [zeros, reflect, replicate, circular]
    preactivation : bool
        Whether the activation should precede the convolution
    spatial_factor : int
        The reduction of the spatial dimension by the pooling layer.
    ema_decay: float
        Whether Exponential Moving Average should be used. A value of 0.0 will disable EMA.
        As SWA is provided currently through train_utils (the proper way) and EMA is an add-on,
        SWA will be preferred over EMA -> If both are set, EMA will be ignored!
        https://github.com/Lightning-AI/lightning/issues/10914
    weight_decay: float
        Weight decay of AdamW optimiser. Default: 0.0
    commitment_loss_weight: float
        Weight of the commitment loss. Is the same as the codebook_weight of 
        Rombach et al.
    pos_emb : Literal['lin', 'sin'], optional
        Whether positional embedding is utilised or not. 
        If 'lin' is chosen, the embedding from Bieder is utilised.
        Otherwise, the positional embedding of the transformer is used
    use_fake_3d: bool
        Whether fake 3D for the perceptual loss is used. If True, the perceptual loss
        takes a slice of each axis and averages the perceptual loss across the slices.
        If false, only axial slices are utilised.
    """
    def __init__(
            self,
            activation: Optional[Literal['LeakyReLU', 'ReLU', 'SiLU']],
            act_args: tuple,
            batch_size: int,
            channel_factor: Tuple[int, ...],
            dataset: Dataset,
            dimensions: Literal[2, 3],
            discriminator_channels: int,
            discriminator_layers: int,
            dropout: int,
            embedding_dim: int,
            hidden_channels: int,
            in_channels: int,
            kernel_size: int,
            learning_rate: float,
            num_res_blocks: int,
            patch_size: Tuple[int, ...],
            norm: Optional[Literal['batch', 'instance', 'group', 'layer', 'ada_gn']],
            num_codes: int,
            out_channels: int,
            padding: bool,
            padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'],
            preactivation: bool,
            spatial_factor: int,
            z_channels: int,
            task: TASK = TASK.AE,
            attention_heads: int = 1,
            attention_res: Optional[Tuple[int, ...]] = (8, 16, 32),
            ema_decay: float = 0.0,
            weight_decay: float = 0.0,
            commitment_loss_weight: float = 0.25,
            pixel_loss_weight: float = 1.0,
            perceptual_loss_weight: float = 1.0,
            gan_image_weight: float = 1.0,
            gan_volume_weight: float = 1.0,
            gan_feature_weight: float = 1.0,
            discriminator_start_epoch: int = 10,
            discriminator_loss: Literal['vanilla', 'hinge'] = 'vanilla',
            pixel_loss: Literal['l1', 'l2'] = 'l1',
            accumulate_grad_batches: int = 1,
            gradient_clip_val: float = 1.0,
            gradient_clip_algorithm: Optional[Literal['norm']] = None,
            use_khader: bool = False,
            upd_conv: bool = True,
            sample_every_n_steps: int = 1000,
            pos_emb: Optional[Literal['lin', 'sin']] = None,
            use_fake_3d: bool = False,
            *args,
            **kwargs,
            ):
        super().__init__(batch_size=batch_size,
                         **kwargs)
        
        self._task = task
        self.automatic_optimization = False  # Remove automatic optimisation

        self.parse_input_args(dropout=dropout, in_channels=in_channels, out_channels=out_channels)

        self.save_hyperparameters(ignore=['dataset'])
        self.dataset = dataset

        # Positional Embedding 
        self.pos_emb_mode = pos_emb
        # Required for additional channels with lin emb
        mod_in_channels = in_channels  
        if pos_emb is not None:
            if pos_emb == 'sin':
                assert isinstance(dataset, BraTS2023), "Positional embedding only supported for BraTS"
                # Sinusoidal Positional Embedding of the Attnetion is all you need paper
                pos_emb_ch = 4 * hidden_channels
                self.pos_emb = PosEmbedding(
                    pos_features=hidden_channels,
                    out_features=pos_emb_ch,
                    dimensions=dimensions,
                    activation=activation,
                    act_args=act_args
                )
            elif pos_emb == 'lin':
                self.pos_emb = CoordinateEmbedding(max_spatial_dim=256)  #BraTSTransform resizes to [256,256,128]
                pos_emb_ch = None
                mod_in_channels += 3 # That is what the embedding does
            else:
                raise AttributeError(f"Positional embedding mode {pos_emb} not supported")
        else:
            pos_emb_ch = None
            self.pos_emb = None

        # Step counter
        self._step = 0

        encoder_class, decoder_class = (VQGANEncoderKhader, VQGANDecoderKhader) if use_khader else  (VQGANEncoder, VQGANDecoder)

        self.encoder = encoder_class(
            channel_factor=channel_factor,
            z_channels=z_channels,
            num_res_blocks=num_res_blocks,
            attention_heads=attention_heads,
            attention_res=attention_res,
            patch_size=max(patch_size),
            in_channels=mod_in_channels,
            dimensions=dimensions,
            hidden_channels=hidden_channels,
            activation=activation,
            act_args=act_args,
            dropout=dropout,
            kernel_size=kernel_size,
            norm=norm,
            padding=padding,
            padding_mode=padding_mode,
            preactivation=preactivation,
            spatial_factor=spatial_factor,
            upd_conv=upd_conv,
            emb_ch=pos_emb_ch)

        self.decoder = decoder_class(
            channel_factor=channel_factor,
            z_channels=z_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_heads=attention_heads,
            attention_res=attention_res,
            patch_size=max(patch_size),
            dimensions=dimensions,
            hidden_channels=hidden_channels,
            activation=activation,
            act_args=act_args,
            dropout=dropout,
            kernel_size=kernel_size,
            norm=norm,
            padding=padding,
            padding_mode=padding_mode,
            preactivation=preactivation,
            spatial_factor=spatial_factor,
            upd_conv=upd_conv,
            emb_ch=pos_emb_ch
        )

        #  VQGAN convolutions
        self.pre_vq_conv = ConvolutionalBlock(
            in_channels=self.encoder.out_channels,
            out_channels=embedding_dim,
            dimensions=dimensions,
            kernel_size=1,
            norm=None,
            activation=None,
        )
        self.post_vq_conv = ConvolutionalBlock(
            in_channels=embedding_dim,
            out_channels=self.encoder.out_channels,
            dimensions=dimensions,
            kernel_size=1,
            norm=None,
            activation=None,
        )

        # Codebook for the latent embeddigns
        self.codebook = EMAVectorQuantizer(
            num_codes=num_codes,
            embedding_dim=embedding_dim,
            ema_decay=ema_decay,
            beta=commitment_loss_weight,
        )

        self.loss_criterion = VQLPIPSWithDiscriminator(
            dimensions=dimensions,
            disc_start_epoch=discriminator_start_epoch,
            disc_loss=discriminator_loss,
            pixel_loss=pixel_loss,
            perceptual_loss='lpips',
            disc_in_channels=in_channels,
            disc_layers=discriminator_layers,
            disc_num_df=discriminator_channels,
            perceptual_loss_weight=perceptual_loss_weight,
            pixel_loss_weight=pixel_loss_weight,
            gan_image_weight=gan_image_weight,
            gan_volume_weight=gan_volume_weight,
            gan_feature_weight=gan_feature_weight,
            use_fake_3d=use_fake_3d)
        
        self._is_quantized = False

        # Metrics for testing
        metrics_collection = metrics.get_metrics_collection(
            TASK.SEG,
            class_task='binary',
            threshold=0.5,
            num_classes=2)
        self.test_metrics = metrics_collection.clone(postfix='/test')

        
    def update_step(self, batch_idx):
        r"""This function updates the step counters of the model.
        self.trainer.global_step is incremented for each optimizer step.
        However, we have two optimizers hence why this is inaccurate"""

        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            self._step += 1

    @property
    def z_dim(self):
        r"""The output channel dim of the latent embeddings. This is the same as the embedding dim."""
        return self.pre_vq_conv.out_channels
    
    @property
    def encoder_out_ps(self) -> Tuple[int, ...]:
        r"""Calculates the output spatial dimensions of the encoder"""
        return (self.encoder.out_ps,) * self.get_hparam('dimensions') 

    @property
    def step(self):
        return self._step

    def parse_input_args(
            self, 
            dropout: int,
            in_channels: int,
            out_channels: int, *args, **kwargs) -> None:
        super().parse_input_args(dropout, *args, **kwargs)
        assert out_channels == in_channels, "VQGAN requires the same number of input and output channels"

    def encode(
            self, 
            x: torch.Tensor,
            emb: Optional[torch.Tensor] = None, 
            location: Optional[torch.Tensor] = None,
            quantize: bool = True) -> Union[
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],  # Codebook
                Tuple[torch.Tensor, Optional[torch.Tensor]], # No quantize
                ]:
        
        if emb is None and isinstance(self.pos_emb, PosEmbedding):
            x, emb = self.determine_embedding(x, location)
        
        h = self.encoder(x, emb)
        h = self.pre_vq_conv(h)
        if quantize:
            self._is_quantized = True
            # Returns z_q, loss, perplexity, encodings, encoding_indices
            h = self.codebook(h)

        if isinstance(h, tuple):
            return h + (emb,)  # type: ignore
        return h, emb
    
    def decode(
            self, 
            z_q: torch.Tensor, 
            emb: Optional[torch.Tensor] = None,
            quantize: bool = False):
        if quantize:
            assert not self._is_quantized, "Cannot quantize twice"
            z_q, *_ = self.codebook(z_q)
            self._is_quantized = True

        assert self._is_quantized, "Cannot decode without quantization"

        if emb is None and isinstance(self.pos_emb, PosEmbedding):
            raise RuntimeError("Cannot decode without positional embedding")
        h = self.post_vq_conv(z_q)
        h = self.decoder(h, emb)

        self._is_quantized = False  # Reset for next forward
        return h
    
    def determine_embedding(self, x: torch.Tensor, location: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        emb = None
        if self.pos_emb is not None:
            assert location is not None, "Provide location for positional embedding"
            if self.pos_emb_mode == 'sin':
                assert isinstance(self.pos_emb, PosEmbedding), "Positional embedding must be of type PosEmbedding"
                emb = self.pos_emb(location=location)
            elif self.pos_emb_mode == 'lin':
                assert isinstance(self.pos_emb, CoordinateEmbedding), "Positional embedding must be of type CoordinateEmbedding"
                x = self.pos_emb(x, location)
            else:
                raise AttributeError(f"Positional embedding mode {self.pos_emb_mode} not supported")
            
        return x, emb
    
    def encode_decode(
            self, 
            x: torch.Tensor, 
            location: Optional[torch.Tensor] = None):
        
        x, emb = self.determine_embedding(x, location)
        
        z_q, commitment_loss, perplexity, *_ = self.encode(x=x,  emb=emb, quantize=True) # type: ignore
        x_recon = self.decode(z_q, emb, quantize=False)
        return x_recon, commitment_loss, perplexity

    
    def on_train_start(self) -> None:
        if hasattr(self.loss_criterion, 'perceptual_model'):
            self.loss_criterion.perceptual_model.eval()
        super().on_train_start()
    
    def forward(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Forward pass is not implemented. Use encode and decode instead")
       
    def training_step(self, batch, batch_idx) -> None:

        assert self.trainer.accumulate_grad_batches == 1, "Accumulated gradients are not supported with dual optimizers."

        x = self.get_input_tensor_from_batch(batch, sequences=tuple(self.dataset.img_seq))
        location = self.get_location_tensor_from_batch(batch)

        opt_ae, opt_disc = self.optimizers()

        x_recon, commitment_loss, perplexity = self.encode_decode(
            x=x, location=location)

        assert x.shape == x_recon.shape, f"Input ({x.shape}) and recon ({x_recon.shape}) must have the same shape."


        # Autoencoder Training
        ae_loss, ae_dict_to_log = self.loss_criterion(
            x=x, x_recon=x_recon, stage=Stage.TRAIN,
            mode='ae', epoch=self.trainer.current_epoch,
            commitment_loss=commitment_loss,
        )
        opt_ae.zero_grad()
        self.manual_backward(ae_loss / self.trainer.accumulate_grad_batches)

        # clip gradients
        self.clip_gradients(opt_ae, 
                            gradient_clip_val=self.get_hparam('gradient_clip_val'), 
                            gradient_clip_algorithm=self.get_hparam('gradient_clip_algorithm'))
        opt_ae.step()
        
        # Discriminator Training
        disc_loss, disc_dict_to_log = self.loss_criterion(
            x=x, x_recon=x_recon, stage=Stage.TRAIN,
            mode='disc',
            epoch=self.trainer.current_epoch,
            commitment_loss=commitment_loss,
        )
        
        
        opt_disc.zero_grad()
        self.manual_backward(disc_loss / self.trainer.accumulate_grad_batches)

        # clip gradients
        self.clip_gradients(opt_disc, 
                            gradient_clip_val=self.get_hparam('gradient_clip_val'), 
                            gradient_clip_algorithm=self.get_hparam('gradient_clip_algorithm'))
        opt_disc.step()

        # Log
        self.log_dict({
            **ae_dict_to_log, 
            **disc_dict_to_log, 
            'train/perplexity': perplexity}, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.get_hparam('batch_size'))
        
        self.log_loss(disc_loss + ae_loss, Stage.TRAIN, soe=SoE.STEP)
        
        if (self.step + 1) % self.get_hparam('sample_every_n_steps') == 0:
            self.log_images({"Input": x, 
                             "Recon": rescale(x_recon, -1., 1.), 
                             "Recon_clamp": torch.clamp(x_recon, -1., 1.),
                             "Diff": rescale(torch.abs(x - x_recon), 0, 1)})

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        self.update_step(batch_idx)
        super().on_train_batch_end(outputs, batch, batch_idx)
    
    def validation_step(self, batch, batch_idx) -> None:

        x = self.get_input_tensor_from_batch(batch, sequences=tuple(self.dataset.img_seq))

        location = self.get_location_tensor_from_batch(batch)

        x_recon, commitment_loss, perplexity = self.encode_decode(
            x=x, location=location)

        assert x.shape == x_recon.shape, f"Input ({x.shape}) and recon ({x_recon.shape}) must have the same shape."


        # Autoencoder Output
        ae_loss, ae_dict_to_log = self.loss_criterion(
            x=x, x_recon=x_recon, stage=Stage.VAL,
            mode='ae', epoch=self.trainer.current_epoch,
            commitment_loss=commitment_loss,
        )
        
        # Discriminator Output
        disc_loss, disc_dict_to_log = self.loss_criterion(
            x=x, x_recon=x_recon, stage=Stage.TRAIN,
            mode='disc',
            epoch=self.trainer.current_epoch,
            commitment_loss=commitment_loss,
        )

        # Log
        self.log_dict({
            **ae_dict_to_log, 
            **disc_dict_to_log, 
            'train/perplexity': perplexity}, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.get_hparam('batch_size'))

        self.log_loss(disc_loss + ae_loss, Stage.VAL, soe=SoE.STEP)
        
    
    def configure_optimizers(self) -> Any:
        learning_rate = self.get_hparam('learning_rate')

        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.pre_vq_conv.parameters()) +
            list(self.post_vq_conv.parameters()) +
            list(self.codebook.parameters()),
            lr=learning_rate,
            betas=(0.5, 0.9)
        )

        opt_disc = torch.optim.Adam(
            list(self.loss_criterion.image_discriminator.parameters()) +
            list(self.loss_criterion.volume_discriminator.parameters()),
            lr=learning_rate,
            betas=(0.5, 0.9)
        )
        
        return [opt_ae, opt_disc], []
    
    def test_step(self, batch, batch_idx) -> None:
        x = self.get_input_tensor_from_batch(batch, sequences=tuple(self.dataset.img_seq))
        location = self.get_location_tensor_from_batch(batch)
        x_recon, *_ = self.encode_decode(
            x=x, location=location)
        
        target_seg = self.get_target_seg_tensor_from_batch(batch)

        assert x.shape == x_recon.shape, f"Input ({x.shape}) and recon ({x_recon.shape}) must have the same shape."

        aggr_dict = self._add_batch_to_aggr(
            initial=x,
            healthy=x_recon,
            target=target_seg,
            location=location
        )

        if all([aggr is not None for aggr in aggr_dict.values()]):
            init_tensor = aggr_dict["init"]
            healthy_tensor = aggr_dict["healthy"]
            target_tensor = aggr_dict["target"]

            assert init_tensor is not None and healthy_tensor is not None and target_tensor is not None, "All tensors must be provided"


            anomaly_map = metrics.process_diffusion_tensors(
                input_tensor = init_tensor,
                healthy_tensor = healthy_tensor,
                target_tensor = target_tensor,
                dimensions = self.get_hparam("dimensions")
            )

            self.log_metrics(prediction=anomaly_map,
                             target=target_tensor,
                             stage=Stage.TEST,
                             batch_size=anomaly_map.shape[0])


 