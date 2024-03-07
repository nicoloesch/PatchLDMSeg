import abc
import warnings
import torch
from typing import Any, Tuple, Union, Optional, Literal, Dict, Mapping, Type

from patchldmseg.model.lightning.base_model import BaseModel
from patchldmseg.model.diffusion.backbones import UNetDiffusion
from patchldmseg.model.diffusion.ddpm import DDPMDiffusion
from patchldmseg.model.lightning.vqgan import VQGAN
import patchldmseg.model.diffusion.diffusion_helper as dh
from patchldmseg.model.basic.ema import EMAContextManager
from patchldmseg.utils.misc import Stage, TASK
from patchldmseg.utils import callbacks
from patchldmseg.input.datasets.base_dataset import Dataset



class DummyCM:
    def __init__(self, model: torch.nn.Module):
        self._model = model

    def update(self):
        pass

    def __enter__(self) -> torch.nn.Module:
        return self._model

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Nothing to do here as there is no cleanup necessary atm
        pass


class DiffBase(BaseModel, abc.ABC):
    r"""Basic Model for all Diffusion Models

    Notes
    -----
    F841 noqa is utilised as the init args are saved as hparams with reference
    to the self.hparams attribute. To suppress linter warnings of unused args,
    we specify it here.

    Parameters
    ----------
    activation : str, optional
        The name of the activation function to be used. MUST match the exact spelling of torch
    act_args : tuple
        Additional arguments of the activation function
    attention_ch_per_head : int
        Number of channels per attention head. If this is set to None, attention_heads determines
        the number of attention heads. If it is set to an integer, number of heads
        are determined automatically.
    attention_heads : int
        Number of attention heads used in the AttentionBlock of the U-Net
    attention_res : tuple
        Attention resolutions i.e. where attention should take place based on the patch size.
        I.e. an attention_res of 16 would result to activate attention (if attention is True)
        as soon as the patch size reached 16^dim. If None, no attention will be used.
    batch_size: int
        The batch size of each step
    channel_factor : tuple
        The multiplicative factor of the channel dimension. It is a multiplicative factor for each
        depth layer of the UNet. The input channels are multiplied by this factor.
    concat_sc : bool
        If True, the skip connections of the encoder will be concatenated along the channel dimension
        in the decoder. If False, the skip connections will be summed with the features of the upsampled layer
    conditional_sampling : bool
        If conditional sampling based on the label should be used
    conv_zero_init : bool
        If certain output layers should be initialised with zero weight and bias similar to Dhariwal.
    dataset: Dataset
        A dataset object created in the input module/data loader.
    diffusion_gradient_scale: float
        In the case of guided sampling, how much the gradient of the classifier
        effects the prediction
    diffusion_log_var: bool
        If the calculation of the variance should be carried out in logarithmic units.
        Prevents exploding gradients
    diffusion_loss_type: patchldmseg.model.diffusion.diffusion_helper.LossType
        The type of the loss specified by LossType (SIMPLE, KL or HYBRID)
    diffusion_mean_type: patchldmseg.model.diffusion.diffusion_helper.ModelMeanType
        The mean prediction of the diffusion model (X_PREV, X_0, EPSILON)
    diffusion_noising: str
        The noising schedule utilised for the diffusion process
    diffusion_steps: int
        How many forward diffusion steps should be carried out
    diffusion_var_type: patchldmseg.model.diffusion.diffusion_helper.ModelVarianceType
        The type of the predicted variance (LEARNED, FIXED, LEARNED_RANGE)
    diffusion_verbose: bool
        If the diffusion process should be displayed as a Progressbar.
        Easy for debugging to see the progress of encoding and sampling
    dimensions : int
        Dimensionality of the input. Supported: 2,3
    dropout : int
        Percentage of dropout to be used [0, 100]
    hidden_channels : int
        Number of channels the first convolution creates as output channels. The intention behind this is to
        increase the features before the contraction path of the UNet in a dedicated convolutional layer.
    in_channels : int
        Number of input channels to the first convolution of the entire block
        i.e. the number of channels of the input data
    kernel_size : int
        The size of the kernel for convolution
    norm : str, optional
        The string of the normalization method to be used. Supported are [batch, instance, layer, group, ada_gn]
    num_res_blocks : int
        Number of residual blocks (with optional attention) per layer.
    out_channels : int
        Number of output channels of the network.
    padding : bool
        Whether padding should be used or not. The underlying function determines the optimal padding
    padding_mode : str
        The padding mode, i.e. which values to be used for padding
    patch_size : int
        The patch size of the input patch i.e. the initial patch size prior to the UNet.
        This is required as attention is utilised as attention is performed only on specific
        attention resolutions, specified by `attention_res`
    preactivation : bool
        Whether the activation should precede the convolution
    spatial_factor : int
        The reduction of the spatial dimension by the pooling layer.
    upd_conv : str
        If the model should use convolution for upsampling/downsampling or traditional pooling/interpolation
    ema_decay: float
        Whether Exponential Moving Average should be used. A value of 0.0 will disable EMA.
        As SWA is provided currently through train_utils (the proper way) and EMA is an add-on,
        SWA will be preferred over EMA -> If both are set, EMA will be ignored!
        https://github.com/Lightning-AI/lightning/issues/10914
    weight_decay: float
        Weight decay of AdamW optimiser. Default: 0.0
    """

    def __init__(self,
                 activation: Optional[Literal['LeakyReLU', 'ReLU', 'SiLU']],
                 act_args: tuple,
                 attention_heads: int,
                 attention_ch_per_head: Optional[int],
                 attention_res: Optional[Tuple[int, ...]],
                 backbone_class: Type[UNetDiffusion],
                 batch_size: int,
                 channel_factor: Tuple[int, ...],
                 concat_sc: bool,
                 conditional_sampling: bool,
                 conv_zero_init: bool,
                 dataset: Dataset,
                 diffusion_gradient_scale: float,
                 diffusion_log_var: bool,
                 diffusion_loss_type: dh.LossType,
                 diffusion_mean_type: dh.MeanType,
                 diffusion_noising: str,
                 diffusion_steps: int,
                 diffusion_var_type: dh.VarianceType,
                 diffusion_verbose: bool,
                 dimensions: Literal[2, 3],
                 dropout: int,
                 experiment_name: str,
                 hidden_channels: int,
                 in_channels: int,
                 kernel_size: int,
                 logging_dir: str,
                 norm: Optional[Literal['batch', 'instance', 'group', 'layer', 'ada_gn']],
                 num_res_blocks: int,
                 out_channels: int,
                 padding: bool,
                 padding_mode: str,
                 patch_size: Tuple[int, ...],
                 preactivation: bool,
                 spatial_factor: int,
                 upd_conv: bool,
                 task: TASK,
                 ema_decay: float = 0.9999,
                 learning_rate: float = 1e-4,
                 log_targets: bool = False,
                 weight_decay: float = 0.0,
                 ldm_ckpt: Optional[str] = None,
                 p_unconditional: float = 0.0,
                 **kwargs: Any
                 ):

        super().__init__(
            batch_size=batch_size,
            num_samples_to_log=dataset.num_samples,
        )

        assert 1.0 > p_unconditional >= 0.0, (
            "Classifier free guidance probability must be in [0, 1)"
        )

        # load hparams from checkpoint  for adapted in and output_channels
        # NOTE: Cant do load_from_checkpoint before __init__
        # Overwriting patch_size does not work as it would be a mismatch 
        # between the LightningDataModule and the LightningModule
        self.dm_in_spat_dim = patch_size  # Required for check
        dm_in_ch, dm_out_ch = in_channels, out_channels

        if ldm_ckpt is not None:
            loaded_ldm_ckpt = torch.load(ldm_ckpt, map_location=torch.device('cpu'))  # Load model to CPU as it later gets pushed to the correct GPU
            dm_in_ch = loaded_ldm_ckpt["hyper_parameters"]["embedding_dim"]
            dm_out_ch = dm_in_ch

            # As I have loaded the hparams already I can also check the dataset
            assert (
                loaded_ldm_ckpt["DataModule"]["hparams"]["dataset_str"] ==
                str(dataset)
                ), "Dataset mismatch between LDM and DiffSeg"
            
            ldm_patch_size = loaded_ldm_ckpt["hyper_parameters"]["patch_size"]
            assert (
                ldm_patch_size ==  patch_size
            ), f"Patch Size mismatch. LDM {ldm_patch_size} vs {patch_size}"

            stride = loaded_ldm_ckpt["hyper_parameters"]["spatial_factor"]

            # We need to reduce the patch size for the attention calculations
            for _ in range(len(loaded_ldm_ckpt["hyper_parameters"]["channel_factor"])):
                self.dm_in_spat_dim = tuple(ps // stride for ps in self.dm_in_spat_dim)
        
        self._ldm = None

        # Backbone
        self.backbone = backbone_class(
            dimensions=dimensions,
            hidden_channels=hidden_channels,
            in_channels=dm_in_ch,
            out_channels=dm_out_ch,
            patch_size=self.dm_in_spat_dim,
            activation=activation,
            act_args=act_args,
            attention_heads=attention_heads,
            attention_ch_per_head=attention_ch_per_head,
            attention_res=attention_res,
            channel_factor=channel_factor,
            conditional_sampling=conditional_sampling,
            conv_zero_init=conv_zero_init,
            diffusion_steps=diffusion_steps,
            diffusion_var_type=diffusion_var_type,
            dropout=dropout,
            kernel_size=kernel_size,
            norm=norm,
            num_res_blocks=num_res_blocks,
            padding=padding,
            padding_mode=padding_mode,
            preactivation=preactivation,
            spatial_factor=spatial_factor,
            upd_conv=upd_conv,
            dataset=dataset,
            is_ldm=ldm_ckpt is not None,
            p_unconditional=p_unconditional,
            **kwargs)

        # Diffusion Process
        self.diffusion_process = DDPMDiffusion(
            num_timesteps=diffusion_steps,
            mean_type=diffusion_mean_type,
            var_type=diffusion_var_type,
            loss_type=diffusion_loss_type,
            verbose=diffusion_verbose,
            logging_dir=logging_dir,
            dataset=dataset,
            experiment_name=experiment_name,
            max_in_val=dataset.binary_min_max[-1],
            gradient_scale=diffusion_gradient_scale,
            p_unconditional=p_unconditional
            )
        
        # Save the dataset as it is no longer saved through hparams
        self._dataset = dataset

        # Create the targets lists
        self._log_targets = log_targets
        if log_targets:
            self._train_targets = []
            self._val_targets = []

        # Parameter for sampling
        self._num_samples = dataset.num_samples

        # Task
        self._task = task

        # Moving Average context manager - This will get checked on_train_start for SWA availability
        if ema_decay:
            self._weight_averaged_model_cm = EMAContextManager(self.backbone, beta=ema_decay)
        else:
            self._weight_averaged_model_cm = DummyCM(self.backbone)

    @property
    def task(self) -> TASK:
        return self._task

    @property
    def dataset(self) -> Dataset:
        if not hasattr(self, '_dataset'):
            raise RuntimeError("Make sure to set the dataset in the model __init__ as it is no longer saved to the hparams")
        return self._dataset

    @property
    def weight_averaged_model_cm(self) -> Union[callbacks.SWA, EMAContextManager, DummyCM]:
        r"""Returns the (potentially) weight averaged model"""
        return self._weight_averaged_model_cm

    def _check_for_swa_and_update(self) -> None:
        r"""Updates the context-manager for weight-averaged models. This checks if SWA is in the callbacks
        of the trainer and replaces the context-manager accordingly. This can only be called starting from
        on_train_start as then the trainer is initialised and the callbacks are available

        Notes
        -----
        Can only be called if the training has already started as then the trainer is populated with the callbacks,
        and we can check if SWA is utilised, which then suppresses the pre-initialised EMA.
        """

        uses_swa = any(
            isinstance(cb, callbacks.SWA) for cb in getattr(self.trainer, "callbacks"))

        if uses_swa:
            if self.get_hparam('ema_decay'):
                warnings.warn("SWA and EMA found. Will utilise SWA and omit any EMA-related parameters. "
                              "Make sure to set ema_decay to 0.0 or comment the SWA callback out in train_utils.py.")
            for cb in getattr(self.trainer, "callbacks"):
                if isinstance(cb, callbacks.SWA):
                    # Note: could not overwrite it so had to delete the variable and reassign
                    del self._weight_averaged_model_cm
                    self._weight_averaged_model_cm = cb
                    break

    @property
    def ldm(self) -> Optional[VQGAN]:
        ldm_ckpt = self.get_hparam("ldm_ckpt")
        if ldm_ckpt is not None and self._ldm is None:
            self._ldm = VQGAN.load_from_checkpoint(
                ldm_ckpt, dataset=self._dataset, map_location=self.device)
            
            # Parse Args
            self.parse_ldm_args()

            # Set the Model to evaluation mode as it should not be trainable
            self._ldm.eval()
            for param in self._ldm.parameters():
                param.requires_grad = False
        return self._ldm
    
    def ldm_encode(
            self, 
            tensor: torch.Tensor, 
            location: Optional[torch.Tensor]) -> Tuple[
                torch.Tensor, Optional[torch.Tensor]]:
        r"""Convenience function to encode the tensor with the LDM if present. 
        In addition, the output tensor is scaled to [-1, 1] to be in the correct range
        for the Diffusion model. Needs to be used in conjunction with ldm_decode to 
        get the rescaling in again. Otherwise, the decoder will not be able to decode it properly."""
        emb = None
        if self.ldm is not None:
            with torch.no_grad():
                tensor, *_, emb = self.ldm.encode(
                    x=tensor, location=location, quantize=True)
                
                # Normalise to -1, 1
                tensor = ((tensor - self.ldm.codebook.embedding.weight.min()) / 
                          (self.ldm.codebook.embedding.weight.max() - 
                           self.ldm.codebook.embedding.weight.min())) * 2.0 - 1.0

        return tensor, emb
    
    def ldm_decode(
            self, 
            tensor: torch.Tensor, 
            emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Convenience function to decode the tensor with the LDM if present. The output of the diffusion
        model is rescaled to the appropriate range of the decoder. Needs to be used in conjunction with
        ldm_encode to get the rescaling in again. Otherwise, the decoder will not be able to decode it properly."""
        if self.ldm is not None:
            with torch.no_grad():
                # Denormalise the tensor
                tensor = (((tensor + 1.0) / 2.0) * 
                          (self.ldm.codebook.embedding.weight.max() -
                           self.ldm.codebook.embedding.weight.min())) + self.ldm.codebook.embedding.weight.min()

                tensor = self.ldm.decode(tensor, emb)

        return tensor
    
    def parse_ldm_args(self) -> None:
        r"""Parse the arguments of the latent diffusion model. 
        Important: Do not use the property self.ldm here as it will lead to an infinite loop"""

        if self._ldm is not None:
            assert isinstance(self._ldm, VQGAN), "Unsupported LDM Model"

            # I don't think that makes sense as the dataset is passed into init.
            # It therefore always matches so not a real check
            assert type(self._ldm.dataset) == type(self._dataset), "LDM and DiffSeg must be trained on the same dataset"

            assert (
                self._ldm.get_hparam('dimensions') == 
                self.get_hparam('dimensions')
                ), "LDM and DiffSeg must have the same dimensions"
            
            assert (
                self._ldm.encoder_out_ps == 
                self.dm_in_spat_dim
                ), "Patch size mismatch between LDM and DiffSeg"
            
    def _remove_ldm_from_state_dict(self, ckpt_or_sd: Mapping[str, Any]) -> Dict[str, Any]:
        r"""Removes the LDM from the state_dict of either the state_dict directly or the checkpoint.
        This is required as the LDM cannot be instantiated in the DiffSeg init because of pytorch lightning
        but some checkpoints still contain in as on_save_checkpoint has only been introduced after."""
        from collections import OrderedDict
        state_dict_without_ldm = OrderedDict()
        try:
            # Checkpoint
            for key, value in ckpt_or_sd['state_dict'].items():
                if not 'ldm' in key:
                    state_dict_without_ldm[key] = value

            mod_ckpt_or_sd = {key: val for key, val in ckpt_or_sd.items() if key != 'state_dict'}
            mod_ckpt_or_sd['state_dict'] = state_dict_without_ldm
            return mod_ckpt_or_sd

        except KeyError:
            # State Dict directly
            for key, value in ckpt_or_sd.items():
                if not 'ldm' in key:
                    state_dict_without_ldm[key] = value
            return state_dict_without_ldm
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint = self._remove_ldm_from_state_dict(checkpoint)
        return super().on_save_checkpoint(checkpoint)
    
    def load_state_dict(self, 
                        state_dict: Mapping[str, Any],
                        strict: bool = True):
        
        # Remove LDM from state_dict
        state_dict = self._remove_ldm_from_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            params=self.parameters(), 
            lr=self.get_hparam('learning_rate'),
            weight_decay=self.get_hparam('weight_decay'))
        
        return optimizer
    
        #return {
        #    "optimizer": optimizer,
        #    "lr_scheduler": {
        #        "scheduler": callbacks.WarmUpScheduler(optimizer, warmup=5000),
        #        "frequency": 1,
        #        'interval': 'step',
        #    },
        #}

    def forward(
            self,
            x_t, 
            t, 
            y: Optional[torch.Tensor] = None,
            location: Optional[torch.Tensor] = None) -> Any:
        return self.backbone(x_t=x_t, t=t, y=y, location=location)

    def on_train_start(self) -> None:
        super().on_train_start()
        self._check_for_swa_and_update()

    @abc.abstractmethod
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Perform a single training step"""

    def on_train_batch_end(self, out, batch, batch_idx):
        self._weight_averaged_model_cm.update()

    def on_train_epoch_end(self) -> None:
        if self._log_targets:
            targets = torch.cat(self._train_targets)
            self.log_dict(
                {f"targets/{Stage.TRAIN.value}_{key}": torch.sum(torch.eq(targets, key)).float() for key in [0, 1]},
                sync_dist=True)
            self._train_targets.clear()

    @abc.abstractmethod
    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        r"""Perform a single validation step."""

    def on_validation_epoch_end(self) -> None:
        if self._log_targets:
            targets = torch.cat(self._val_targets)
            self.log_dict(
                {f"targets/{Stage.VAL.value}_{key}": torch.sum(torch.eq(targets, key)).float() for key in [0, 1]},
                sync_dist=True)
            self._val_targets.clear()

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        """Perform a single test step"""
        return super().test_step(batch, batch_idx, dataloader_idx)

    def on_before_optimizer_step(self, optimizer) -> None:
        #from pytorch_lightning.utilities.grads import grad_norm
        #norms = grad_norm(self.backbone, norm_type=2)
        #self.log_dict(norms)
        super().on_before_optimizer_step(optimizer)

    def update_targets(self, val: torch.Tensor, stage: Stage):
        if self._log_targets and hasattr(self, f"_{Stage.TRAIN.value}_targets"):
            getattr(self, f"_{stage.value}_targets").append(val)

