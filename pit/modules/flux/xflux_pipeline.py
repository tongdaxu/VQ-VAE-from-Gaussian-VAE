from PIL import Image
import torch


from pit.modules.flux.sampling import denoise_controlnet, get_noise, get_schedule, unpack, prepare2, denoise_cat, prepare_control



class XFluxPipelineClean:
    def __init__(self, model_type, device, weight_type):
        self.device = torch.device(device)
        self.model_type = model_type
        self.weight_type=weight_type

        self.image_encoder_path = "openai/clip-vit-large-patch14"
        self.hf_lora_collection = "XLabs-AI/flux-lora-collection"
        self.lora_types_to_names = {
            "realism": "lora.safetensors",
        }
        self.controlnet_loaded = False
        self.ip_loaded = False

    def __call__(self,
                 prompt,
                 neg_prompt,
                 inp_txt, inp_vec,
                 image_prompt: Image = None,
                 controlnet_image: Image = None,
                 width: int = 512,
                 height: int = 512,
                 guidance: float = 4,
                 num_steps: int = 50,
                 seed: int = 123456789,
                 true_gs: float = 3,
                 control_weight: float = 0.9,
                 ip_scale: float = 1.0,
                 neg_ip_scale: float = 1.0,
                 neg_image_prompt: Image = None,
                 timestep_to_start_cfg: int = 0,
                 ):

        return self.forward(
            prompt,
            neg_prompt,
            inp_txt,
            inp_vec,
            width,
            height,
            guidance,
            num_steps,
            seed,
            controlnet_image.to(dtype=self.weight_type),
            timestep_to_start_cfg=timestep_to_start_cfg,
            true_gs=true_gs,
            control_weight=control_weight,
            image_proj=None,
            neg_image_proj=None,
            ip_scale=ip_scale,
            neg_ip_scale=neg_ip_scale,
        )

    def forward(
        self,
        prompt,
        neg_prompt,
        inp_txt,
        inp_vec,
        width,
        height,
        guidance,
        num_steps,
        seed,
        controlnet_image = None,
        timestep_to_start_cfg = 0,
        true_gs = 3.5,
        control_weight = 0.9,
        image_proj=None,
        neg_image_proj=None,
        ip_scale=1.0,
        neg_ip_scale=1.0,
    ):
        x = get_noise(
            inp_txt.shape[0], height, width, device=self.device,
            dtype=self.weight_type, seed=seed
        )
        timesteps = get_schedule(
            num_steps,
            (width // 8) * (height // 8) // (16 * 16),
            shift=True,
        )
        torch.manual_seed(seed)
        with torch.no_grad():
            inp_cond = prepare2(inp_txt, inp_vec, img=x, prompt=prompt)
            neg_inp_cond = prepare2(inp_txt, inp_vec, img=x, prompt=neg_prompt)

            x = denoise_controlnet(
                self.model,
                **inp_cond,
                controlnet=self.controlnet,
                timesteps=timesteps,
                guidance=guidance,
                controlnet_cond=controlnet_image,
                timestep_to_start_cfg=timestep_to_start_cfg,
                neg_txt=neg_inp_cond['txt'],
                neg_txt_ids=neg_inp_cond['txt_ids'],
                neg_vec=neg_inp_cond['vec'],
                true_gs=true_gs,
                controlnet_gs=control_weight,
                image_proj=image_proj,
                neg_image_proj=neg_image_proj,
                ip_scale=ip_scale,
                neg_ip_scale=neg_ip_scale,
                weight_type=self.weight_type,
            )

            x = unpack(x.float(), height, width)
            x = self.ae.decode(x)

        x1 = x.clamp(-1, 1)
        return x1

    def call_plora(
        self,
        prompt,
        neg_prompt,
        inp_txt, inp_vec,
        image_prompt: Image = None,
        controlnet_image: Image = None,
        width: int = 512,
        height: int = 512,
        guidance: float = 4,
        num_steps: int = 50,
        seed: int = 123456789,
        true_gs: float = 3,
        control_weight: float = 0.9,
        ip_scale: float = 1.0,
        neg_ip_scale: float = 1.0,
        neg_image_prompt: Image = None,
        timestep_to_start_cfg: int = 0,
    ):
        x = get_noise(
            inp_txt.shape[0], height, width, device=self.device,
            dtype=self.weight_type, seed=seed
        )
        timesteps = get_schedule(
            num_steps,
            (width // 8) * (height // 8) // (16 * 16),
            shift=True,
        )
        torch.manual_seed(seed)
        with torch.no_grad():

            inp_cond = prepare_control(inp_txt, inp_vec, img=x, control=controlnet_image, prompt=prompt)

            x = denoise_cat(
                self.model,
                **inp_cond,
                timesteps=timesteps,
                guidance=guidance,
                timestep_to_start_cfg=timestep_to_start_cfg,
                image_proj=None,
                neg_image_proj=None,
                ip_scale=ip_scale,
                neg_ip_scale=neg_ip_scale,
                # weight_type=self.weight_type,
            )

            x = unpack(x.float(), height, width)
            x = self.ae.decode(x)

        x1 = x.clamp(-1, 1)
        return x1
