import patchldmseg.utils.metrics
import torch
from typing import (
    Any,
    Mapping, 
    Tuple, 
    Optional, 
    Literal, 
    Dict,
    Union
)
import torch.distributed as dist
import pathlib

from .diffbase import DiffBase
from patchldmseg.model.diffusion.ddpm import DDPMDiffusion
from patchldmseg.model.diffusion.ddim import DDIMDiffusion
from patchldmseg.model.diffusion.edict import EDICTDiffusion
import patchldmseg.model.diffusion.diffusion_helper as dh
from patchldmseg.model.diffusion.backbones import UNetDiffusion
from patchldmseg.utils.misc import Stage, TASK, SoE, create_experiment_name
from patchldmseg.utils import metrics, constants
from patchldmseg.input.datasets import BraTS2023, Dataset

import pathlib

class DiffSeg(DiffBase):
    r"""Diffusion  segmentation with U-Net model as a backbone.

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
        as soon as the patch size reached 16^dim. None results in no attention.
    batch_size: int
        The batch size of each step
    channel_factor : tuple
        The multiplicative factor of the channel dimension. It is a multiplicative factor for each
        depth layer of the UNet. The input channels are multiplied by this factor.
    p_unconditional: float
        Probability of None inputs to the classifier-free model. Typical values are around 0.1 -
        0.2. This is required even in training mode as there are two  models being learned simultaneously 
        and will be checked upon checkpoint loading if it has been set correctly in the checkpoint.
    concat_sc : bool
        If True, the skip connections of the encoder will be concatenated along the channel dimension
        in the decoder. If False, the skip connections will be summed with the features of the upsampled layer
    conditional_sampling : bool
        If conditional sampling based on the label should be used
    conv_zero_init : bool
        If certain output layers should be initialised with zero weight and bias similar to Dhariwal.
    dataset: Dataset
        The utilised dataset. Supported: BraTS2023
    diffusion_gradient_scale: float
        Scale of the gradient of classifier-free guidance
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
    num_val: int
        Number of validation subjects required to determine the optimal sampling frequency
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
    num_encoding_steps: Number of encoding (and decoding) steps to be carried out. If None, the entire
        diffusion process is carried out.
    encoding: Literal['ddim', 'ddpm']
        The encoding mechanism utilised
    decoding: Literal['ddim', 'ddpm']
        The decoding mechanism utilised
    """

    def __init__(self,
                 activation: Optional[Literal['LeakyReLU', 'ReLU', 'SiLU']],
                 act_args: tuple,
                 attention_heads: int,
                 attention_res: Optional[Tuple[int, ...]],
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
                 pid: int,
                 preactivation: bool,
                 spatial_factor: int,
                 task: TASK,
                 upd_conv: bool,
                 learning_rate: float,
                 p_unconditional: float = 0.0,
                 ema_decay: float = 0.0,
                 weight_decay: float = 0.0,
                 eta: float = 0.0,
                 sample_every_n_epoch: int = 1,
                 subsequence_length: Optional[int] = None,
                 num_encoding_steps: int = 500,
                 attention_ch_per_head: Optional[int] = None,
                 ldm_ckpt: Optional[str] = None,
                 pos_emb: Optional[Literal['sin', 'lin']] = None,
                 iteration: int = 0,
                 encoding: Literal['ddim', 'ddpm', 'edict'] = 'ddim',
                 decoding: Literal['ddim', 'ddpm', 'edict'] = 'ddim',
                 edict_weight: float = 0.93,
                 encoding_class: Optional[int] = 0,
                 **kwargs: Any
                 ):

        experiment_name = create_experiment_name(pid, task, diffusion=True)
        backbone_class = UNetDiffusion

        self.parse_input_args(
            dropout=dropout, 
            ema_decay=ema_decay, 
            norm=norm, 
            conditional_sampling=conditional_sampling, 
            dataset=dataset,
            num_encoding_steps=num_encoding_steps,
            diffusion_steps=diffusion_steps,
            subsequence_length=subsequence_length,
            p_unconditional=p_unconditional)       

        super().__init__(activation=activation,
                         act_args=act_args,
                         attention_heads=attention_heads,
                         attention_ch_per_head=attention_ch_per_head,
                         attention_res=attention_res,
                         batch_size=batch_size,
                         backbone_class=backbone_class,
                         channel_factor=channel_factor,
                         concat_sc=concat_sc,
                         conditional_sampling=conditional_sampling,
                         conv_zero_init=conv_zero_init,
                         dataset=dataset,
                         diffusion_gradient_scale=diffusion_gradient_scale,
                         diffusion_log_var=diffusion_log_var,
                         diffusion_loss_type=diffusion_loss_type,
                         diffusion_mean_type=diffusion_mean_type,
                         diffusion_noising=diffusion_noising,
                         diffusion_steps=diffusion_steps,
                         diffusion_var_type=diffusion_var_type,
                         diffusion_verbose=diffusion_verbose,
                         dimensions=dimensions,
                         dropout=dropout,
                         experiment_name=experiment_name,
                         hidden_channels=hidden_channels,
                         in_channels=in_channels,
                         kernel_size=kernel_size,
                         logging_dir=logging_dir,
                         norm=norm,
                         num_res_blocks=num_res_blocks,
                         out_channels=out_channels,
                         padding=padding,
                         padding_mode=padding_mode,
                         patch_size=patch_size,
                         preactivation=preactivation,
                         spatial_factor=spatial_factor,
                         upd_conv=upd_conv,
                         ema_decay=ema_decay,
                         learning_rate=learning_rate,
                         weight_decay=weight_decay,
                         task=task,
                         pos_emb=pos_emb,
                         ldm_ckpt=ldm_ckpt,
                         p_unconditional=p_unconditional)
        
        # Dataset can not be saved as a hparam as it is not picklable. 
        self.save_hyperparameters(ignore=['dataset'])


        # Metrics
        self._num_classes = 2
        metrics_collection = patchldmseg.utils.metrics.get_metrics_collection(
            TASK.SEG,
            class_task='binary',
            threshold=0.5,
            num_classes=self._num_classes)

        self.val_metrics = metrics_collection.clone(postfix='/val')
        self.samples = []
        self.has_logged = False
        self.test_metrics = metrics_collection.clone(postfix='/test')

        # Debug Iterator
        self._iteration = iteration

        # Sample Diffusion Process
        self._sample_diffusion_process = None

    def should_sample(self, batch_idx) -> bool:
        assert self.trainer is not None
        return (self.trainer.current_epoch+1) % self.get_hparam("sample_every_n_epoch") == 0
    
    @property
    def sample_diffusion_process(self) -> Union[DDPMDiffusion, DDIMDiffusion, EDICTDiffusion]:
        r"""Property for the diffusion process utilised for all sampling related
        tasks."""

        if self._sample_diffusion_process is None:
            # NOTE: Currently, we do not support a mixture of encoding and decoding
            assert self.get_hparam('encoding') == self.get_hparam('decoding')

            if self.get_hparam('encoding') == 'ddim':
                self._sample_diffusion_process = DDIMDiffusion.from_ddpm(
                    self.diffusion_process, 
                    eta=self.get_hparam('eta'),
                    subsequence_length=self.get_hparam('subsequence_length')
                    )
            elif self.get_hparam('encoding') == 'edict':
                self._sample_diffusion_process = EDICTDiffusion.from_ddpm(
                    self.diffusion_process, 
                    eta=self.get_hparam('eta'),
                    subsequence_length=self.get_hparam('subsequence_length'),
                    mixing_weight=self.get_hparam('edict_weight')
                )
            elif self.get_hparam('encoding') == 'ddpm':
                self._sample_diffusion_process = self.diffusion_process

            else:
                raise NotImplementedError(f"Encoding {self.get_hparam('encoding')} not supported")

        return self._sample_diffusion_process


    def parse_input_args(
            self, 
            dropout: int, 
            ema_decay: float,
            norm: Optional[Literal['batch', 'instance', 'group', 'layer', 'ada_gn']], 
            conditional_sampling: bool,
            dataset: Dataset,
            diffusion_steps: int,
            num_encoding_steps: Optional[int],
            subsequence_length: Optional[int],
            p_unconditional: float,
            *args, **kwargs) -> None:
        super().parse_input_args(dropout)
        assert 1.0 > ema_decay >= 0.0, 'EMA decay only [0.0, 1.0)'
        assert 1.0 > p_unconditional >= 0.0, 'Classifier-free probability only [0.0, 1.0)'

        assert type(dataset) == BraTS2023, f"Dataset {dataset} not supported"


        # DDIM subsequence
        if subsequence_length is not None:
            assert diffusion_steps >= subsequence_length, "Diffusion steps must be greater than or equal to subsequence length"
            assert diffusion_steps % subsequence_length == 0, "Diffusion steps must be divisible by subsequence_length"
            assert subsequence_length > 0, "Subsequence length must be greater than 0"
            if num_encoding_steps is not None:
                assert num_encoding_steps <= subsequence_length, "Number of encoding steps must be less than or equal to subsequence length"
        else:
            if num_encoding_steps is not None:
                assert num_encoding_steps <= diffusion_steps, "Number of encoding steps must be less than or equal to diffusion steps"


        if conditional_sampling:
            _error_msg = f"conditional_sampling requires `{constants.ADAPTIVE_GROUP_NORM}` norm"
            assert norm == constants.ADAPTIVE_GROUP_NORM, _error_msg

        if p_unconditional > 0.0:
            assert conditional_sampling, "Classifier-free guidance only possible with conditional sampling"

    def _sample_healthy(
            self, 
            batch_dict: Dict[str, torch.Tensor], 
            stage: Stage,  
            **kwargs) -> Dict[str, torch.Tensor]:

        inputs = self.get_input_tensor_from_batch(batch_dict, tuple(self._dataset.img_seq))
        location = self.get_location_tensor_from_batch(batch_dict)

        inputs_ldm, emb = self.ldm_encode(tensor=inputs, location=location)

        target_seg = self.get_target_seg_tensor_from_batch(batch_dict)
        location = self.get_location_tensor_from_batch(batch_dict)

        enc_class = self.get_hparam('encoding_class')
        if enc_class is not None:
            enc_tensor = torch.full((inputs_ldm.shape[0], ), enc_class, device=inputs_ldm.device, dtype=torch.long)
        else:
            enc_tensor = None

        with self.weight_averaged_model_cm as averaged_model:
            averaged_model.eval()
            averaged_model.to(self.device)

            with torch.no_grad():
                healthy = self.sample_diffusion_process.sample_healthy(
                denoise_fn=averaged_model,
                x_0=inputs_ldm,
                first_n_timesteps=self.get_hparam('num_encoding_steps'),
                location=location,
                enc_class=enc_tensor
                )           

        healthy = self.ldm_decode(healthy, emb)

        # Trickery to overcome the mechanism of ldm
        assert self.ldm is not None
        self.ldm._is_quantized = True  # I know that it is quantized
        recon = self.ldm_decode(inputs_ldm, emb)

        if stage == Stage.TEST and self.get_hparam('dimensions') == 3:
            # These are only not None if the entire grid aggregator has been filled
            _btch = self._add_batch_to_aggr(
                initial=inputs,  # Here the original inputs
                healthy=healthy,
                target=target_seg,
                location=location,
                recon=recon,
            )
            healthy, inputs, target_seg, recon = _btch['healthy'], _btch['init'], _btch['target'], _btch['recon']

        if healthy is not None and inputs is not None and target_seg is not None:
            anomaly_map = metrics.process_diffusion_tensors(
                input_tensor=inputs,
                healthy_tensor=healthy,
                target_tensor=target_seg,
                recon_tensor=recon,
                dimensions=self.get_hparam("dimensions"),
            )
            assert anomaly_map.shape == target_seg.shape

            self.log_metrics(prediction=anomaly_map,
                             target=target_seg,
                             stage=stage,
                             batch_size=anomaly_map.shape[0])
            
            samples = {
                'gt': target_seg,
                'healthy': healthy,
                'input': inputs,
                'am': anomaly_map}
            
            if recon is not None:
                samples['recon'] = recon
        else:
            samples = {}

        return samples

    def _step(self, batch, batch_idx: int, stage: Stage) -> torch.Tensor:
        """Process a single train, val or test step"""

        inputs = self.get_input_tensor_from_batch(batch, tuple(self._dataset.img_seq))
        target_class = self.get_target_class_tensor_from_batch(
            batch=batch, dataset=self.dataset, stage=stage)
        location = self.get_location_tensor_from_batch(batch)

        # Generate the timesteps
        timesteps, _ = self.diffusion_process.generate_random_timesteps(
            batch_size=inputs.shape[0],
            device=inputs.device)

        y = target_class if self.get_hparam('conditional_sampling') else None

        # LDM Encoding
        inputs, _ = self.ldm_encode(inputs, location=location)

        loss = self.diffusion_process.calculate_loss(
            x_0=inputs, 
            t=timesteps, 
            denoise_fn=self.backbone, 
            y=y, 
            location=location,
            p_unconditional=self.get_hparam("p_unconditional")
        )

        loss = torch.mean(loss)

        self.log_loss(loss, stage=stage, soe=SoE.STEP)

        # Attend to the respective target containers
        if hasattr(self, f"_{stage.value}_targets"):
            getattr(self, f"_{stage.value}_targets").append(target_class)
        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(
            batch=batch,
            batch_idx=batch_idx,
            stage=Stage.TRAIN)
    

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        validation_loss = self._step(batch, batch_idx, stage=Stage.VAL)

        if self.should_sample(batch_idx):
            self.sample_and_log_images(batch=batch, stage=Stage.VAL)
        return validation_loss
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self._sample_healthy(batch, stage=Stage.TEST)


    def sample_and_log_images(self, batch, stage: Stage) -> None:
        r"""Convenience function for both validation and test steps to sample healthy images and log
        a fraction of them to wandb."""
        samples = self._sample_healthy(batch, stage=stage)
        if not self.has_logged:
            
            if self.is_ddp:
                dist.barrier()

            if samples and len(self.samples) * self._batch_size < self.dataset.num_samples:
                self.samples.append(samples)
            
            if (samples and len(self.samples) * self._batch_size >= self.dataset.num_samples) or self.trainer.is_last_batch:
                merged_samples = self._merge_samples()

                if self.trainer.global_rank == 0:
                    output = {key: tensor[:min(tensor.shape[0], self.dataset.num_samples), ...].cpu() for key, tensor in merged_samples.items()}
                    self.log_images(output)
                
                self.has_logged = True
                self.samples.clear()
    
    def _merge_samples(self) -> dict:
        r"""This function merges samples together by combining the sampled tensors from multiple GPUs
        stored in the self.samples list. The samples are concatenated along the batch dimension and then
        gathered across all processes. The gathered samples are then concatenated along the batch dimension
        and returned for logging"""

        if not self.samples:
            raise RuntimeError("self.samples is empty. Make sure to append the samples to the list.")
    
        aggregate_dict = {key: [] for key in self.samples[0].keys()}

        for sample in self.samples:
            for key, value in sample.items():
                aggregate_dict[key].append(value)

        merged_samples = {key: torch.cat(value, dim=0) for key, value in aggregate_dict.items()}

        if self.is_ddp:
            output = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(output, merged_samples)

            assert all([val is not None for val in output]), "Not all processes have gathered the samples"
            if self.trainer.global_rank == 0:
                output = {
                    key: torch.cat(
                        [val.cpu() for val in values]
                        ) for key, values in zip(output[0].keys(), zip(*[dct.values() for dct in output]))}  # type: ignore
        else:
            output = merged_samples

        return output  # type: ignore

    def on_validation_epoch_end(self) -> None:        
        self.has_logged = False
        self.samples.clear()

    def on_test_end(self) -> None:
        self.has_logged = False
        self.samples.clear()

    def load_state_dict(
            self, 
            state_dict: Mapping[str, Any], 
            strict: bool = True):
        state_dict = self._alter_old_embedding_layer(state_dict)
        return super().load_state_dict(state_dict, strict)
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        ckpt_p_unconditional = checkpoint['hyper_parameters'].get('p_unconditional', 0.0)
        self_unconditional = self.get_hparam('p_unconditional')
        
        assert abs(ckpt_p_unconditional - self_unconditional) < 1e-6, "Checkpoint and model must have the same p_unconditional."
        return super().on_load_checkpoint(checkpoint)

    def _alter_old_embedding_layer(self, state_dict: Mapping[str, Any]):
        r"""With the introduction of the new LabelEmbedding layer of the UNet backbone,
        we need to map the old embedding layer to the new one. As a result, old checkpoints
        are still supported with the LabelEmbedding layer"""

        from collections import OrderedDict
        state_dict_new_label_emb = OrderedDict()

        for key, value in state_dict.items():
            if key.endswith('label_emb.weight'):
                new_key = key.replace('label_emb.weight', 'label_emb.embedding_layer.weight')
                state_dict_new_label_emb[new_key] = value
            else:
                state_dict_new_label_emb[key] = value
        return state_dict_new_label_emb
