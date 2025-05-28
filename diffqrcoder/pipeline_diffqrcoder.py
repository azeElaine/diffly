from typing import Any, Callable, Dict, List, Optional, Union
import os
import torch
from diffusers import StableDiffusionControlNetPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.models import ControlNetModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.utils import is_torch_xla_available, deprecate
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version
from tqdm import trange

from .image_processor import image_binarize, crop_padding
from .srpg import ScanningRobustPerceptualGuidance


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class DiffQRCoderPipeline(StableDiffusionControlNetPipeline):
    def _generate_logo_mask(self, logo_image: torch.Tensor, padding: int) -> torch.Tensor:  
        batch_size, channels, height, width = logo_image.shape  
  
    # 动态计算Logo尺寸并校验  
        logo_size = max(1, min(height, width) // 4)  
        if padding >= min(height, width) // 2:  
            raise ValueError("Padding exceeds half of image size")  
  
    # 创建布尔类型遮罩（节省内存）  
        mask = torch.zeros((batch_size, height, width),   
                           device=logo_image.device,  
                           dtype=torch.bool)  
  
    # 计算中心区域（适配奇偶尺寸）  
        h_start = (height - logo_size) // 2  
        h_end = h_start + logo_size  
        w_start = (width - logo_size) // 2  
        w_end = w_start + logo_size  
  
    # 批量赋值（无需通道维度）  
        mask[:, h_start:h_end, w_start:w_end] = True  
  
    # Add channel dimension before cropping  
        mask = mask.unsqueeze(1)  # Shape becomes (batch_size, 1, height, width)  
      
    # 裁剪边缘填充  
        return crop_padding(mask, padding)
    def _run_stage1(
        self,
        prompt: Union[str, List[str]] = None,
        qrcode: PipelineImageInput = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        return super().__call__(
            prompt=prompt,
            image=qrcode,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            ip_adapter_image=ip_adapter_image,
            ip_adapter_image_embeds=ip_adapter_image_embeds,
            output_type=output_type,
            return_dict=True,
            cross_attention_kwargs=cross_attention_kwargs,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guess_mode=guess_mode,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            clip_skip=clip_skip,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            **kwargs,
        )

    def _run_stage2(
        self,
        logo_guidance_scale: int = 100,
        logo_image: Optional[PipelineImageInput] = None,  # Add this line 
        prompt: Union[str, List[str]] = None,
        qrcode: PipelineImageInput = None,
        qrcode_module_size: int = 20,
        qrcode_padding: int = 78,
        ref_image: PipelineImageInput = None,
        height: Optional[int] = 512,
        width: Optional[int] =512,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        scanning_robust_guidance_scale: int = 500,
        perceptual_guidance_scale: int = 10,
        srmpgd_num_iteration: Optional[int] = None,
        srmpgd_lr: Optional[float] = 0.1,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        save_intermediate_steps: bool = False,  # 新增参数
        intermediate_dir: str = "output/intermediate",  # 新增参数
        use_custom_postprocess: bool = False,  # 新增参数，控制是否使用自定义后处理
        use_original_qrcode: bool = False,  # 新增参数，控制是否使用原始二维码
        **kwargs,
    ):
        self.srpg = ScanningRobustPerceptualGuidance(
            module_size=qrcode_module_size,
            scanning_robust_guidance_scale=scanning_robust_guidance_scale,
            perceptual_guidance_scale=perceptual_guidance_scale,
            logo_guidance_scale=logo_guidance_scale,  # 新增  
        ).to(self.device).to(self.dtype)
        

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            qrcode,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare image
        if isinstance(controlnet, ControlNetModel):
            qrcode = self.prepare_image(
                image=qrcode,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            height, width = qrcode.shape[-2:]
        elif isinstance(controlnet, MultiControlNetModel):
            qrcodes = []

            # Nested lists as ControlNet condition
            if isinstance(qrcode[0], list):
                # Transpose the nested image list
                qrcode = [list(t) for t in zip(*qrcode)]

            for qrcode_ in qrcode:
                qrcode_ = self.prepare_image(
                    image=qrcode_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                qrcodes.append(qrcode_)

            qrcode = qrcodes
            height, width = qrcode[0].shape[-2:]
        else:
            assert False

        # 5. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        self._num_timesteps = len(timesteps)

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6.5 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None
            else None
        )

        # 7.2 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        is_unet_compiled = is_compiled_module(self.unet)
        is_controlnet_compiled = is_compiled_module(self.controlnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            with torch.enable_grad():
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    # Relevant thread:
                    # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                    if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                        torch._inductor.cudagraph_mark_step_begin()
                    # expand the latents if we are doing classifier free guidance
                    latents = latents.clone().detach().requires_grad_(True)
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # controlnet(s) inference
                    if guess_mode and self.do_classifier_free_guidance:
                        # Infer ControlNet only for the conditional batch.
                        control_model_input = latents
                        control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                        controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                    else:
                        control_model_input = latent_model_input
                        controlnet_prompt_embeds = prompt_embeds

                    if isinstance(controlnet_keep[i], list):
                        cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                    else:
                        controlnet_cond_scale = controlnet_conditioning_scale
                        if isinstance(controlnet_cond_scale, list):
                            controlnet_cond_scale = controlnet_cond_scale[0]
                        cond_scale = controlnet_cond_scale * controlnet_keep[i]

                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=qrcode,
                        conditioning_scale=cond_scale,
                        guess_mode=guess_mode,
                        return_dict=False,
                    )

                    if guess_mode and self.do_classifier_free_guidance:
                        # Inferred ControlNet only for the conditional batch.
                        # To apply the output of ControlNet to both the unconditional and conditional batches,
                        # add 0 to the unconditional batch to keep it unchanged.
                        down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                        mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the original latents x_t -> x_0
                    original_latents = self.scheduler.step(
                        noise_pred,
                        t,
                        latents,
                        **extra_step_kwargs,
                        return_dict=True,
                    ).pred_original_sample

                    original_image = self.vae.decode(
                        original_latents / self.vae.config.scaling_factor,
                        return_dict=True,
                    ).sample
                    

                    # compute the score of Scanning Robust Perceptual Guidance (SRPG)  
                    if logo_image is not None:  
                        # First get the processed and cropped image dimensions  
                        processed_image = crop_padding(self.image_processor.denormalize(original_image), qrcode_padding)  
                        target_height, target_width = processed_image.shape[-2:]  
      
                        target_device = original_image.device  
                        target_dtype = original_image.dtype  
                        # Apply both device and dtype conversion  
                        logo_tensor = self.image_processor.preprocess(logo_image).to(device=target_device, dtype=target_dtype)  
      
                        if logo_tensor.shape[-2:] != (target_height, target_width): 
                            logo_tensor = torch.nn.functional.interpolate(logo_tensor,size=(target_height, target_width),  mode='bilinear', align_corners=False  )  
                        batch_size = logo_tensor.shape[0]  
                        logo_size = max(1, min(target_height, target_width) // 4)  
                        mask = torch.zeros((batch_size, 1, target_height, target_width), device=target_device, dtype=target_dtype)
      
                        h_start = (target_height - logo_size) // 2  
                        h_end = h_start + logo_size  
                        w_start = (target_width - logo_size) // 2  
                        w_end = w_start + logo_size  
      
                        mask[:, :, h_start:h_end, w_start:w_end] = 1.0  
                        logo_mask_tensor = mask  
                    else:  
                        logo_tensor = None  
                        logo_mask_tensor = None
                    score = self.srpg.compute_score(
                        latents=latents,
                        image=crop_padding(self.image_processor.denormalize(original_image), qrcode_padding),
                        qrcode=crop_padding(image_binarize(qrcode[qrcode.size(0) // 2, None]), qrcode_padding),
                        ref_image=crop_padding(ref_image, qrcode_padding),
                        logo_image=logo_tensor,  # 新增  
                        logo_mask=logo_mask_tensor,  # 新增  
                    )

                    timesteps_prev = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    beta_prod_t = 1 - alpha_prod_t
                    alpha_prod_t_prev = self.scheduler.alphas_cumprod[timesteps_prev] if timesteps_prev >= 0 else self.scheduler.final_alpha_cumprod
                    beta_prod_t_prev = 1 - alpha_prod_t_prev

                    noise_pred = noise_pred + (beta_prod_t ** 0.5) * score
                    original_latents = (latents - (beta_prod_t ** 0.5) * noise_pred) / alpha_prod_t ** 0.5
                    latents = (alpha_prod_t_prev ** 0.5) * original_latents + (beta_prod_t_prev ** 0.5) * noise_pred

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)

                    if XLA_AVAILABLE:
                        xm.mark_step()

        # perform Scanning Robust Manifold Projected Gradient Descent (SR-MPGD)
        if srmpgd_num_iteration is not None:
            with torch.enable_grad():
                latents = latents.clone().detach().requires_grad_(True)
                optimizer = torch.optim.SGD([latents], lr=srmpgd_lr)
                
                # 用于保存中间结果的变量
                if save_intermediate_steps:
                    from PIL import Image
                    import torchvision.transforms as T
                    to_pil = T.ToPILImage()
                
                print(f"执行 SR-MPGD 优化，迭代次数: {srmpgd_num_iteration}")
                for i in trange(srmpgd_num_iteration):
                    optimizer.zero_grad()
                    try:
                        # 检查 latents 是否包含 NaN
                        if torch.isnan(latents).any():
                            print(f"迭代 {i}: 检测到 latents 中有 NaN 值，尝试修复...")
                            latents = torch.nan_to_num(latents.detach(), nan=0.0).requires_grad_(True)
                            optimizer = torch.optim.SGD([latents], lr=srmpgd_lr)
                        
                        original_image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
                        
                        # 检查解码后的图像是否包含 NaN
                        if torch.isnan(original_image).any():
                            print(f"迭代 {i}: VAE 解码产生了 NaN 值，尝试修复...")
                            original_image = torch.nan_to_num(original_image, nan=0.0)
                        
                        loss = self.srpg.compute_loss(
                            image=crop_padding(self.image_processor.denormalize(original_image), qrcode_padding),
                            qrcode=crop_padding(image_binarize(qrcode[qrcode.size(0) // 2, None]), qrcode_padding),
                            ref_image=crop_padding(ref_image, qrcode_padding),
                            logo_image=logo_tensor,
                            logo_mask=logo_mask_tensor,
                        )
                        
                        # 检查损失是否为 NaN
                        if torch.isnan(loss):
                            print(f"迭代 {i}: 损失为 NaN，跳过此次迭代...")
                            continue
                        
                        loss.backward()
                        
                        # 检查梯度是否包含 NaN
                        if torch.isnan(latents.grad).any():
                            print(f"迭代 {i}: 梯度包含 NaN 值，使用零梯度...")
                            latents.grad = torch.zeros_like(latents.grad)
                        
                        optimizer.step()
                    except Exception as e:
                        print(f"迭代 {i} 出错: {e}")
                        continue
                    
                    # 保存中间结果（每5次迭代保存一次）
                    if save_intermediate_steps and (i % 5 == 0 or i == srmpgd_num_iteration - 1):
                        with torch.no_grad():
                            try:
                                # 解码当前的潜在表示
                                current_image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
                                
                                # 检查并修复 NaN 值
                                if torch.isnan(current_image).any():
                                    current_image = torch.nan_to_num(current_image, nan=0.0)
                                
                                # 转换为PIL图像
                                current_image_cpu = current_image[0].cpu().detach()
                                
                                # 确保值在合理范围内
                                current_image_cpu = torch.clamp(current_image_cpu, -1.0, 1.0)
                                current_image_cpu = (current_image_cpu + 1.0) / 2.0
                                current_image_cpu = current_image_cpu * 255
                                current_image_cpu = current_image_cpu.to(torch.uint8)
                                
                                pil_image = to_pil(current_image_cpu)
                                
                                # 保存图像
                                save_path = os.path.join(intermediate_dir, f"srmpgd_iter_{i:03d}.png")
                                pil_image.save(save_path)
                                print(f"SR-MPGD 迭代 {i} 的图像已保存到: {save_path}")
                            except Exception as e:
                                print(f"保存迭代 {i} 图像时出错: {e}")
                
                print("SR-MPGD 优化完成")

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if not output_type == "latent":
            try:
                # 如果设置了使用原始二维码，直接基于原始二维码生成图像
                if use_original_qrcode:
                    print("使用原始二维码作为基础图像...")
                    if isinstance(qrcode, torch.Tensor):
                        base_image = qrcode.to(device=latents.device, dtype=latents.dtype)
                        if base_image.shape != (latents.shape[0], 3, self.unet.config.sample_size, self.unet.config.sample_size):
                            # 调整大小和通道
                            base_image = torch.nn.functional.interpolate(
                                base_image, 
                                size=(self.unet.config.sample_size, self.unet.config.sample_size), 
                                mode='bilinear'
                            )
                            if base_image.shape[1] == 1:
                                base_image = base_image.repeat(1, 3, 1, 1)
                        
                        # 添加一些随机变化，但保持二维码的基本结构
                        noise = torch.randn_like(base_image) * 0.1
                        image = base_image + noise
                        image = torch.clamp(image, -1.0, 1.0)
                        has_nsfw_concept = None
                    else:
                        raise ValueError("原始二维码不是张量，无法直接使用")
                else:
                    # 检查 latents 是否包含 NaN 值
                    if torch.isnan(latents).any():
                        print("警告：检测到 latents 中有 NaN 值，尝试修复...")
                        latents = torch.nan_to_num(latents, nan=0.0)
                    
                    # 限制 latents 的范围，防止极端值
                    latents_max = torch.abs(latents).max().item()
                    if latents_max > 10.0:
                        print(f"警告：检测到 latents 中有极端值 ({latents_max})，尝试裁剪...")
                        latents = torch.clamp(latents, -10.0, 10.0)
                    
                    # 解码 latents
                    image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                        0
                    ]
                    
                    # 检查并修复解码后的图像中的 NaN 值
                    if torch.isnan(image).any():
                        print("警告：VAE 解码产生了 NaN 值，尝试使用备用方法...")
                        # 尝试使用不同的缩放因子
                        image = self.vae.decode(latents / 2.0, return_dict=False, generator=generator)[0]
                        
                        # 如果仍然有 NaN，尝试使用更保守的方法
                        if torch.isnan(image).any():
                            print("备用方法仍产生 NaN，尝试使用更保守的方法...")
                            # 使用非常小的缩放因子
                            image = self.vae.decode(latents / 10.0, return_dict=False, generator=generator)[0]
                            
                            # 如果仍然有 NaN，生成基于原始二维码的图像
                            if torch.isnan(image).any():
                                print("所有备用方法都失败，生成基于原始二维码的图像...")
                                # 使用原始二维码作为基础，添加一些随机变化
                                if isinstance(qrcode, torch.Tensor):
                                    base_image = qrcode.to(device=latents.device, dtype=latents.dtype)
                                    if base_image.shape != (latents.shape[0], 3, self.unet.config.sample_size, self.unet.config.sample_size):
                                        # 调整大小和通道
                                        base_image = torch.nn.functional.interpolate(
                                            base_image, 
                                            size=(self.unet.config.sample_size, self.unet.config.sample_size), 
                                            mode='bilinear'
                                        )
                                        if base_image.shape[1] == 1:
                                            base_image = base_image.repeat(1, 3, 1, 1)
                                    
                                    # 添加一些随机变化，但保持二维码的基本结构
                                    noise = torch.randn_like(base_image) * 0.1
                                    image = base_image + noise
                                    image = torch.clamp(image, -1.0, 1.0)
                                else:
                                    # 如果无法使用原始二维码，生成适度的随机图像
                                    image_shape = (latents.shape[0], 3, self.unet.config.sample_size, self.unet.config.sample_size)
                                    image = torch.rand(image_shape, device=latents.device, dtype=latents.dtype) * 0.5 + 0.25
                    
                    image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            except Exception as e:
                print(f"VAE 解码出错: {e}")
                # 尝试使用原始二维码作为基础
                try:
                    print("尝试使用原始二维码作为基础...")
                    if isinstance(qrcode, torch.Tensor):
                        base_image = qrcode.to(device=latents.device, dtype=latents.dtype)
                        if base_image.shape != (latents.shape[0], 3, self.unet.config.sample_size, self.unet.config.sample_size):
                            # 调整大小和通道
                            base_image = torch.nn.functional.interpolate(
                                base_image, 
                                size=(self.unet.config.sample_size, self.unet.config.sample_size), 
                                mode='bilinear'
                            )
                            if base_image.shape[1] == 1:
                                base_image = base_image.repeat(1, 3, 1, 1)
                        
                        # 添加一些随机变化，但保持二维码的基本结构
                        noise = torch.randn_like(base_image) * 0.1
                        image = base_image + noise
                        image = torch.clamp(image, -1.0, 1.0)
                        has_nsfw_concept = None
                    else:
                        raise ValueError("原始二维码不是张量")
                except Exception as e:
                    print(f"使用原始二维码作为基础失败: {e}")
                    # 生成适度的随机图像作为最后的备用
                    print("生成适度的随机图像作为最后的备用...")
                    image_shape = (latents.shape[0], 3, self.unet.config.sample_size, self.unet.config.sample_size)
                    image = torch.rand(image_shape, device=latents.device, dtype=latents.dtype) * 0.5 + 0.25
                    has_nsfw_concept = None
            
            # 添加调试信息，打印像素值范围
            print(f"解码后图像值范围: min={image.min().item() if not torch.isnan(image.min()) else 'nan'}, max={image.max().item() if not torch.isnan(image.max()) else 'nan'}, mean={image.mean().item() if not torch.isnan(image.mean()) else 'nan'}")
            
            # 修复任何剩余的 NaN 值
            if torch.isnan(image).any():
                print("修复解码后图像中的 NaN 值...")
                image = torch.nan_to_num(image, nan=0.0)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        # 添加调试信息，打印是否进行反归一化
        print(f"是否进行反归一化: {do_denormalize}")
        
        # 保存原始图像用于调试
        if save_intermediate_steps:
            try:
                # 手动将图像转换为0-255范围并保存
                debug_image = image.cpu().detach()
                # 确保值在合理范围内
                debug_image = torch.clamp(debug_image, -1.0, 1.0)
                # 转换到0-255范围
                debug_image = ((debug_image + 1.0) * 127.5).round().to(torch.uint8)
                
                import torchvision.utils as vutils
                debug_path = os.path.join(intermediate_dir, "pre_postprocess_image.png")
                vutils.save_image(debug_image.float() / 255.0, debug_path)
                print(f"已保存预处理图像到: {debug_path}")
            except Exception as e:
                print(f"保存调试图像时出错: {e}")
        
        # 使用自定义后处理或默认后处理
        if use_custom_postprocess:
            try:
                from main import custom_postprocess_image
                print("使用自定义后处理函数...")
                if isinstance(image, list):
                    processed_images = [custom_postprocess_image(img) for img in image]
                else:
                    processed_images = [custom_postprocess_image(image[i]) for i in range(image.shape[0])]
                image = processed_images
            except Exception as e:
                print(f"自定义后处理失败，回退到默认处理: {e}")
                image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        else:
            image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        
        # 检查处理后的图像
        if output_type == "pil" and save_intermediate_steps:
            try:
                # 保存处理后的图像的像素值信息
                if isinstance(image, list):
                    first_image = image[0]
                    import numpy as np
                    img_array = np.array(first_image)
                    print(f"处理后图像值范围: min={img_array.min()}, max={img_array.max()}, mean={img_array.mean():.2f}")
                    
                    # 保存一个副本用于调试
                    debug_path = os.path.join(intermediate_dir, "post_process_image.png")
                    first_image.save(debug_path)
                    print(f"已保存后处理图像到: {debug_path}")
            except Exception as e:
                print(f"分析处理后图像时出错: {e}")

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    @torch.no_grad()
    def __call__(  
    self,  
    prompt: Union[str, List[str]] = None,  
    qrcode: PipelineImageInput = None,  
    qrcode_module_size: int = 20,  
    qrcode_padding: int = 78,  
    height: Optional[int] =512,  
    width: Optional[int] =512,  
    num_inference_steps: int = 50,  
    timesteps: List[int] = None,  
    sigmas: List[float] = None,  
    guidance_scale: float = 7.5,  
    negative_prompt: Optional[Union[str, List[str]]] = None,  
    num_images_per_prompt: Optional[int] = 1,  
    eta: float = 0.0,  
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  
    latents: Optional[torch.Tensor] = None,  
    prompt_embeds: Optional[torch.Tensor] = None,  
    negative_prompt_embeds: Optional[torch.Tensor] = None,  
    ip_adapter_image: Optional[PipelineImageInput] = None,  
    ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,  
    logo_image: Optional[PipelineImageInput] = None,  # 新增  
    logo_guidance_scale: int = 100,  # 新增  
    output_type: Optional[str] = "pil",  
    return_dict: bool = True,  
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,  
    controlnet_conditioning_scale: Union[float, List[float]] = 1.0,  
    guess_mode: bool = False,  
    control_guidance_start: Union[float, List[float]] = 0.0,  
    control_guidance_end: Union[float, List[float]] = 1.0,  
    scanning_robust_guidance_scale: int = 500,  
    perceptual_guidance_scale: int = 10,  
    clip_skip: Optional[int] = None,  
    srmpgd_num_iteration: Optional[int] = None,  
    srmpgd_lr: Optional[float] = 0.1,  
    callback_on_step_end: Optional[  
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]  
    ] = None,  
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    save_intermediate_steps: bool = False,  # 新增参数，控制是否保存中间结果
    intermediate_dir: str = "output/intermediate",  # 新增参数，保存中间结果的目录
    use_custom_postprocess: bool = True,  # 新增参数，默认使用自定义后处理
    use_original_qrcode: bool = False,  # 新增参数，控制是否使用原始二维码
    **kwargs,  
):
        print("阶段1: 生成基础二维码图像...")
        stage1_output = self._run_stage1(
            prompt=prompt,
            qrcode=qrcode,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            ip_adapter_image=ip_adapter_image,
            ip_adapter_image_embeds=ip_adapter_image_embeds,
            output_type="pt",
            return_dict=False,
            cross_attention_kwargs=cross_attention_kwargs,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guess_mode=guess_mode,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            clip_skip=clip_skip,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )
        
        # 保存第一阶段的输出图像
        if save_intermediate_steps:
            from PIL import Image
            import torchvision.transforms as T
            
            print("保存阶段1输出图像...")
            # 将Tensor转换为PIL图像
            to_pil = T.ToPILImage()
            stage1_image = to_pil(stage1_output.images[0].cpu().detach())
            
            # 保存图像
            stage1_path = os.path.join(intermediate_dir, "stage1_output.png")
            stage1_image.save(stage1_path)
            print(f"阶段1图像已保存到: {stage1_path}")
            
            # 保存原始二维码图像
            if isinstance(qrcode, Image.Image):
                qrcode_path = os.path.join(intermediate_dir, "original_qrcode.png")
                qrcode.save(qrcode_path)
                print(f"原始二维码已保存到: {qrcode_path}")
        
        print("阶段2: 应用扫描鲁棒性和感知引导...")
        stage2_output = self._run_stage2(
            logo_image=logo_image,  # Add this line  
            logo_guidance_scale=logo_guidance_scale,  # Add this line too 
            prompt=prompt,
            qrcode=qrcode,
            qrcode_module_size=qrcode_module_size,
            qrcode_padding=qrcode_padding,
            ref_image=stage1_output.images,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            ip_adapter_image=ip_adapter_image,
            ip_adapter_image_embeds=ip_adapter_image_embeds,
            output_type=output_type,
            return_dict=return_dict,
            cross_attention_kwargs=cross_attention_kwargs,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guess_mode=guess_mode,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            scanning_robust_guidance_scale=scanning_robust_guidance_scale,
            perceptual_guidance_scale=perceptual_guidance_scale,
            clip_skip=clip_skip,
            srmpgd_num_iteration=srmpgd_num_iteration,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            save_intermediate_steps=save_intermediate_steps,
            intermediate_dir=intermediate_dir,
            use_custom_postprocess=use_custom_postprocess,
            use_original_qrcode=use_original_qrcode,
        )
        return stage2_output
