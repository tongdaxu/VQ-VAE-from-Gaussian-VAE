# State-of-the-Art VQ-VAE from Gaussian VAE without Training!
* We train a Gaussian VAE, convert it into VQ-VAE with almost 100% codebook usage, and keeps reconstruction performance!
* Pre-trained models can be found in [Huggingface](https://huggingface.co/xutongda/GQModel)
* As flexible to setup as VQ-VAE, supporting: codebook size, codebook dimension, codebook number

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
* Download model "sd3unet_gq_0.25.ckpt" from [Huggingface](https://huggingface.co/xutongda/GQModel):
    ```bash
    mkdir model_256
    mv "sd3unet_gq_0.25.ckpt" ./model_256
    ```

## Infer the model as VQ-VAE
* If you put the model other than "./model_256", modify "ckpt" in "./configs/sd3unet_gq_0.25_vq.yaml"
* Then use the model as follows
    ```Python
    from PIL import Image
    from torchvision import transforms
    from omegaconf import OmegaConf
    from pit.util import instantiate_from_config

    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])
    ])

    img = transform(Image.open("demo.png")).unsqueeze(0).cuda()
    config = OmegaConf.load("./configs/sd3unet_gq_0.25_vq.yaml")
    vae = instantiate_from_config(config.model)
    vae = vae.eval().cuda()

    indices = vae.quant(img)[1] # discrete indices
    img_hat = vae.dequant(indices)
    ```

## Infer the model as Gaussian VAE
* If you put the model other than "./model_256", modify "ckpt" in "./configs/sd3unet_gq_0.25_gaussian.yaml"
* Alternatively, the model can be used as a Vanilla Gaussian VAE
* Then use the model as follows
    ```Python
    from PIL import Image
    from torchvision import transforms
    from omegaconf import OmegaConf
    from pit.util import instantiate_from_config

    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])
    ])

    img = transform(Image.open("demo.png")).unsqueeze(0).cuda()
    config = OmegaConf.load("./configs/sd3unet_gq_0.25_gaussian.yaml")
    vae = instantiate_from_config(config.model)
    vae = vae.eval().cuda()

    z = vae.encode(img) # continous representation
    img_hat = vae.decode(z)
    ```

# Train your own VQ-VAE
* Determine the VQ-VAE parameters:
    * codebook_size: the codebook size, must be 2**N
    * codebook_dimension: the dimension for each codebook
    * codebook_number: number of sub codebook per spatial dimension

* Setup "sd3unet_gq_0.25_train.yaml" according to VQ-VAE parameters:
    * n_samples: = codebook_size size, must be 2**N
    * group: = codebook_dimension, dim of each codebook
    * z_channels: = codebook_dimension * codebook_number, total dim of codebook

* Setup "sd3unet_gq_0.25_train.yaml" according to dataset path
    * root: dataset root
    * image_size: target image size
    * batch_size: batch size

* Run the training! The default "sd3unet_gq_0.25_train.yaml" is setup for codebook_dimension=16, codebook_number=1, codebook_size=2**16=65536
    ```bash
    export WANDB_API_KEY=$YOUR_WANDB_API_KEY
    python main.py --base configs/sd3unet_gq_0.25_train.yaml --wandb
    ```

* Run the evaluation!
    * After the training, modfiy the "ckpt" key in "sd3unet_gq_0.25_vq.yaml" and adjust n_samples, group and z_channels accordingly. Then, evaluate the model as 
    ```bash
    python -m torch.distributed.launch --standalone --use-env \
        --nproc-per-node=1 eval.py \
        --bs=16 \
        --img_size 256 \
        --base=/workspace/cogview_dev/xutd/xu/pytorch-image-tokenizer/configs/sd3unet_gq_0.25_vq.yaml \
        --dataset=$IMAGE_FOLDER_PATH
    ```

# Why it Works?
* The only difference between our Gaussian VAE and vanilla Gaussian VAE is the KL divergence penralization. 
    * The key difference in training is class "GaussianQuantTrainRegularizer" in "./pit/quantization/gaussian.py".
    * The key to convert to VQ-VAE is class "GaussianQuantRegularizer" "./pit/quantization/gaussian.py".
* Basically we limit the KL divergence of Gaussian VAE close to log2 codebook size. Once this constraint is met, the Gaussian VAE can be converted to VQ-VAE without much loss. 
* For more information, wait for our paper to come out!


# Contact & Ack
* Largely from https://github.com/Stability-AI/generative-models
* Any questions or comments goes to: x.tongda@nyu.edu
* Or if you have wechat: 18510201763