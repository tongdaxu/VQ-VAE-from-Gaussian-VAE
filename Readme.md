# State-of-the-Art VQ-VAE from Gaussian VAE without Training!
* We train a Gaussian VAE, convert it into VQ-VAE with almost 100% codebook usage, and keeps reconstruction performance!
* As flexible to setup as VQ-VAE, supporting: codebook size, codebook dimension, codebook number.
* Pre-trained models can be found in [[Huggingface]](https://huggingface.co/xutongda/GQModel)
* Paper can be found in [[Arxiv]](https://arxiv.org/abs/2512.06609)

# Quick Start 
## Install dependency
* dependency in environment.yaml
    ```bash
    conda env create --file=environment.yaml
    conda activate tokenizer
    ```
## Install this package
* from source
    ```bash
    pip install -e .
    ```
* [optional] CUDA kernel for fast run time
    ```bash
    cd gq_cuda_extension
    pip install --no-build-isolation -e .
    ```
## Download pre-trained model 
* Download model "sd3unet_gq_0.25.ckpt" from [[Huggingface]](https://huggingface.co/xutongda/GQModel):
    ```bash
    mkdir model_256
    mv "sd3unet_gq_0.25.ckpt" ./model_256
    ```
* This is a VQ-VAE with codebook_size=2**16=65536 and codebook_dim=16

## Infer the model as VQ-VAE
* Then use the model as follows
    ```Python
    from PIL import Image
    from torchvision import transforms
    from omegaconf import OmegaConf
    from pit.util import instantiate_from_config
    import torch

    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])
    ])

    img = transform(Image.open("demo.png")).unsqueeze(0).cuda()
    config = OmegaConf.load("./configs/sd3unet_gq_0.25.yaml")
    vae = instantiate_from_config(config.model)
    vae.load_state_dict(
        torch.load("models_256/sd3unet_gq_0.25.ckpt",
            map_location=torch.device('cpu'))["state_dict"],strict=False
        )
    vae = vae.eval().cuda()

    vae.eval()
    z, log = vae.encode(img, return_reg_log=True) 
    img_hat = vae.dequant(log["indices"]) # discrete indices
    img_hat = vae.decode(z) # quantized latent
    ```

## Infer the model as Gaussian VAE
* Alternatively, the model can be used as a Vanilla Gaussian VAE:
    ```Python
    from PIL import Image
    from torchvision import transforms
    from omegaconf import OmegaConf
    from pit.util import instantiate_from_config
    import torch

    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])
    ])

    img = transform(Image.open("demo.png")).unsqueeze(0).cuda()
    config = OmegaConf.load("./configs/sd3unet_gq_0.25.yaml")
    vae = instantiate_from_config(config.model)
    vae.load_state_dict(
        torch.load("models_256/sd3unet_gq_0.25.ckpt",
            map_location=torch.device('cpu'))["state_dict"],strict=False
        )
    vae = vae.eval().cuda()

    vae.eval()

    z = vae.encode(img, return_reg_log=True)[1]["zhat_noquant"] # Gaussian VAE latents
    img_hat = vae.decode(z)
    ```

# Train your own VQ-VAE
* Determine the VQ-VAE parameters:
    * codebook_size: the codebook size, must be 2**N
    * codebook_dimension: the dimension for each codebook
    * codebook_number: number of sub codebook per spatial dimension

* Setup "sd3unet_gq_0.25.yaml" according to VQ-VAE parameters:
    * n_samples: = codebook_size size, must be 2**N
    * group: = codebook_dimension, dim of each codebook
    * z_channels: = codebook_dimension * codebook_number, total dim of codebook

* Setup "sd3unet_gq_0.25.yaml" according to dataset path
    * root: dataset root
    * image_size: target image size
    * batch_size: batch size

* Run the training! The default "sd3unet_gq_0.25.yaml" is setup for codebook_dimension=16, codebook_number=1, codebook_size=2**16=65536
    ```bash
    export WANDB_API_KEY=$YOUR_WANDB_API_KEY
    python main.py --base configs/sd3unet_gq_0.25.yaml --wandb
    ```

* Run the evaluation!
    * After the training, obtain the ckpt in $CKPT_PATH. Then, evaluate the model as 
    ```bash
    python -m torch.distributed.launch --standalone --use-env \
        --nproc-per-node=8 eval.py \
        --bs=16 \
        --img_size 256 \
        --base=/workspace/cogview_dev/xutd/xu/pytorch-image-tokenizer/configs/sd3unet_gq_0.25.yaml \
        --ckpt=$CKPT_PATH \
        --dataset=$IMAGE_FOLDER_PATH
    ```

# Train with VAVAE Like Alignment
* See "configs/sd3unet_gq_0.25_vf.yaml".

# Why it Works?
* The only difference between our Gaussian VAE and vanilla Gaussian VAE is the KL divergence penralization. 
    * The key difference is class "GaussianQuantRegularizer" in "./pit/quantization/gaussian.py".
    * During training, GaussianQuantRegularizer forces each dimension of KL be the same and achieve log(codebook_size).
        ```Python
        kl2 = 1.4426 * 0.5 * (torch.pow(mu, 2) + var - 1.0 - logvar)
        kl2 = kl2.reshape(b,l,self.group,c//self.group)
        kl2 = torch.sum(kl2,dim=2) # sum over group dimension
        kl2_mean, kl2_min, kl2_max = torch.mean(kl2), torch.min(kl2), torch.max(kl2)

        ge = (kl2 > self.log_n_samples + self.tolerance).type(kl2.dtype) * self.lam_max
        eq = (kl2 <= self.log_n_samples + self.tolerance).type(kl2.dtype) * (
            kl2 >= self.log_n_samples - self.tolerance
        ).type(kl2.dtype)
        le = (kl2 < self.log_n_samples - self.tolerance).type(kl2.dtype) * self.lam_min
        kl_loss = torch.sum((ge * kl2 + eq * kl2 + le * kl2), dim=[1,2])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        ```
    * During inference, GaussianQuantRegularizer create a codebook of iid Gaussian, and find the cloest sample to posterior mean.
        ```Python
        q_normal_dist = Normal(mu_q[:, None, :], std_q[:, None, :])
        log_ratios = (
            q_normal_dist.log_prob(self.prior_samples[None])
            - self.normal_log_prob[None] * self.beta
        )
        perturbed = torch.sum(log_ratios, dim=2)
        argmax_indices = torch.argmax(perturbed, dim=1)
        zhat[i : i + bs] = torch.index_select(self.prior_samples, 0, argmax_indices)
        indices[i : i + bs] = argmax_indices
        ```
* Basically we limit the KL divergence of Gaussian VAE close to log2 codebook size. Once this constraint is met, the Gaussian VAE can be converted to VQ-VAE without much loss. 
* For more information, see our paper!


# Contact & Ack
* Largely from https://github.com/Stability-AI/generative-models
* Any questions or comments goes to: x.tongda@nyu.edu
* Or if you have wechat: 18510201763

# Reference 
```
@misc{xu2025vectorquantizationusinggaussian,
      title={Vector Quantization using Gaussian Variational Autoencoder}, 
      author={Tongda Xu and Wendi Zheng and Jiajun He and Jose Miguel Hernandez-Lobato and Yan Wang and Ya-Qin Zhang and Jie Tang},
      year={2025},
      eprint={2512.06609},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.06609}, 
}
```