# PatchLDMSeg
This is the official PyTorch implementation of the paper "Three-dimensional latent diffusion model for brain tumour segmentation"


## Installation
Installation can be performed after cloning the repository using
```
pip install .
```

All the dependencies are automatically loaded and it should work out of the box.

## Data
This paper utilises the Data from the [BraTS 2023 challenge](https://www.synapse.org/#!Synapse:syn51156910/wiki/621282) and the Glioma sub-challenge. 
After downloading the data, the folder structure has to look like the following:

```
BraTS_2023
├── Glioma
    ├── train
    │   ├── BraTS-GLI-00000-000
    │   │   ├── BraTS-GLI-00000-000-t1c.nii.gz
    │   │   ├── BraTS-GLI-00000-000-t1n.nii.gz
    │   │   ├── BraTS-GLI-00000-000-t2f.nii.gz
    │   │   ├── BraTS-GLI-00000-000-t2w.nii.gz
    │   │   └── BraTS-GLI-00000-000_seg.nii.gz
    │   └── ...
    └── test
        ├── BraTS-GLI-00001-000/
        │   ├── BraTS-GLI-00001-000-t1c.nii.gz
        │   ├── BraTS-GLI-00001-000-t1n.nii.gz
        │   ├── BraTS-GLI-00001-000-t2f.nii.gz
        │   └── BraTS-GLI-00001-000-t2w.nii.gz
        └── ...

```

There is so automatism inbuilt into the entire framework to open and extract certain sequences and extracting them/renaming them into the respective required structure. As a result, you should be able to load the data from Synapse and extract it into the overarching folder structure with the framework modifying each file.

## Usage
### Train First Stage

As the LDM is a two stage model, we first need to train the VQ-GAN. This can be done using the following script:

```bash
python main.py \
fit \
--model='patchldmseg.model.lightning.VQGAN' \
--task=AE \
--diffusion=True \
--datasets_root=<path-to-folder-above-BraTS2023> \
--logging_dir=<directory-to-log> \
--config='config/base_cfg.yaml' \
--config='config/vqgan_brats_3d.yaml' \
--data.batch_size=8 \
--data.patch_size=64 \
--trainer.accumulate_grad_batches=1 \
--trainer.precision='32-true' \
--data.num_train=-1 \
--data.num_val=-1 \
--data.num_test=-1 \
--data.num_pred=-1 \
--data.patches_per_subj=10 \
--model.ema_decay=0.9 \
--model.sample_every_n_steps=2000 \
--model.discriminator_start_epoch=0 \
--model.pos_emb=sin \
--trainer.devices=[0] \
--trainer.logger.project_name=<your-project-name> \
--trainer.logger.experiment_name=<your-experiment-name>
```

The model is connected to wandb and logs the Data there. Please connect to wandb if you require logging (see [Documentation](https://docs.wandb.ai/quickstart))

### Train Second Stage

After training the first stage model, we can now start training the Diffusion model. Similar to the first stage model, training is started using

```bash
python main.py \
fit \
--model='adiff.model.lightning.DiffSeg' \
--task=SEG \
--diffusion=True \
--datasets_root=<path-to-folder-above-BraTS2023> \
--logging_dir=<directory-to-log> \
--config=config/base_cfg.yaml \
--config=config/diffseg_brats_3d.yaml \
--data.batch_size=6 \
--data.conditional_sampling=True \
--data.dataset_str='BraTS_2023' \
--data.brats_2023_subtask=Glioma \
--data.patch_size=64 \
--data.dimensions=3 \
--model.sample_every_n_epoch=250 \
--model.diffusion_var_type=LEARNED_RANGE \
--model.diffusion_loss_type=HYBRID \
--model.num_res_blocks=4 \
--model.channel_factor=[1,1,2,4] \
--data.num_train=-1 \
--data.num_val=-1 \
--data.num_test=-1 \
--data.num_pred=-1 \
--data.use_queue=True \
--data.patches_per_subj=10 \
--trainer.devices=[0] \
--num_workers=8 \
--trainer.precision='32-true' \
--model.pos_emb=sin \
--model.diffusion_verbose=True \
--trainer.min_epochs=250 \
--model.ldm_ckpt=<path-to-ldm-ckpt/epoch.ckpt> \
--trainer.logger.project_name=<your-project-name> \
--trainer.logger.experiment_name=<your-experiment-name>
```

`<path-to-ldm-ckpt/epoch.ckpt>` is stored in `<directory-to-log>/<your-project-name>/<your-experiment-name>` of the VQ-GAN config (see [Definition](#train-first-stage)) or follows the defaults if the attributes are not specified.

### Inference
After training the both models, inference can be performed using

```bash
python main.py \
test \
--config="<path-to-second-stage/config.yaml>" \
--ckpt_path="<path-to-second-stage/epoch.ckpt>" \
--model.ldm_ckpt="<path-to-ldm-ckpt/epoch.ckpt>" \
--datasets_root=<path-to-folder-above-BraTS2023> \
--logging_dir=<directory-to-log> \
--data.batch_size=16 \
--data.num_test=-1 \
--data.patch_overlap=16 \
--model.diffusion_verbose=True \
--model.num_encoding_steps=100 \
--model.encoding=edict \
--model.decoding=edict \
--model.eta=0.0 \
--model.diffusion_gradient_scale=5.0 \
--model.subsequence_length=null \
--trainer.inference_mode=False \
--trainer.devices=[0] \
--trainer.logger.project_name=<your-project-name> \
--trainer.logger.experiment_name=<your-experiment-name>
```