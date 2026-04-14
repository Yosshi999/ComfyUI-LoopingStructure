from inspect import cleandoc
from comfy_api.latest import io
import torch
from torch import Tensor
from typing import Tuple, List, Literal

def retrieve_linspace_parameters(samples: Tensor) -> Tuple[Tensor, Tensor, int]:
    """Helper function to retrieve linspace parameters from img_ids.
    Args:
        samples (torch.Tensor): The token indices of shape (batch_size, num_tokens).
    Returns:
        start (torch.Tensor): linspace start for each batch (batch_size,).
        end (torch.Tensor): linspace end for each batch (batch_size,).
        steps (int): linspace steps for each batch.
    """
    
    distr = torch.unique(samples, dim=1)  # (batch_size, num_unique == steps)
    start = samples.min(1).values  # (batch_size,)
    end = samples.max(1).values  # (batch_size,)
    steps = distr.shape[1]
    return start, end, steps

class LatentShiftImpl:
    def __init__(self, shift_step, shift_axis):
        self.shift_step = shift_step
        self.shift_axis = shift_axis
        self.call_count = 0
    def __call__(self, data: dict):
        self.call_count += 1
        print("LatentShift called, call count:", self.call_count)
        # retrieve linspace parameters
        # TODO: distribution of specified direction
        target_direction = 2
        samples = data["img_ids"][:, :, target_direction]
        start, end, steps = retrieve_linspace_parameters(samples)
        stepsize = (end - start) / (steps - 1)
        # calculate shift amount
        shift_amount = self.shift_step * stepsize * self.call_count  # (batch_size,)
        shifted_samples = torch.fmod(samples + (shift_amount - start)[:, None], (end - start + stepsize)[:, None]) + start[:, None]
        new_img_ids = data["img_ids"].clone()
        new_img_ids[:, :, target_direction] = shifted_samples
        new_data = data.copy()
        new_data["img_ids"] = new_img_ids
        print("LatentShift executed:", data["img_ids"][0,0,:], "->", new_data["img_ids"][0,0,:])
        return new_data
    def cleanup(self):
        self.call_count = 0

class LatentShift:
    """Latent Shift (https://arxiv.org/abs/2502.20307) for looped structure."""
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model to apply the LatentShift to."}),
                "shift_step": ("INT", {
                    "default": 6,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "display": "number",
                    "tooltip": "The number of latent tokens to shift for each timestep."
                }),
                "shift_axis": (["x", "y", "t"], {"tooltip": "The axis along which to shift the latent tokens."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "execute"
    CATEGORY = "model_patches"

    def execute(self, model: io.Model.Type, shift_step: int, shift_axis: str):
        model = model.clone()
        patch_obj = LatentShiftImpl(shift_step, shift_axis)
        model.set_model_post_input_patch(patch_obj)
        return (model,)

class VAEDecodeCircular:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "The latent to be decoded."}),
                "vae": ("VAE", {"tooltip": "The VAE model used for decoding the latent."}),
                "overlap_x": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32, "tooltip": "The amount of overlap in pixels for the x-axis when generating a looped structure."}),
                "overlap_y": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32, "tooltip": "The amount of overlap in pixels for the y-axis when generating a looped structure."}),
                "overlap_t": ("INT", {"default": 8, "min": 0, "max": 4096, "step": 4, "tooltip": "The amount of overlap in pixels for the temporal-axis when generating a looped structure."}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("The decoded image.",)
    FUNCTION = "decode"

    CATEGORY = "latent"
    DESCRIPTION = "Decodes latent images back into pixel space images."
    SEARCH_ALIASES = ["decode", "decode latent", "latent to image", "render latent"]

    def decode(self, vae: io.Vae.Type, samples: io.Latent.Type, overlap_x: int = 64, overlap_y: int = 64, overlap_t: int = 8):
        temporal_compression = vae.temporal_compression_decode()
        if temporal_compression is not None:
            latent_overlap_t = max(0, overlap_t // temporal_compression)
        else:
            latent_overlap_t = 0
        compression = vae.spacial_compression_decode()
        latent_overlap_x = max(0, overlap_x // compression)
        latent_overlap_y = max(0, overlap_y // compression)
        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]
        # latent shape: (batch_size, channels, T?, H?, W)
        dims = latent.ndim - 2
        if dims == 1:
            pads = [latent_overlap_x]
            pads_px = [overlap_x]
        elif dims == 2:
            pads = [latent_overlap_y, latent_overlap_x]
            pads_px = [overlap_y, overlap_x]
        elif dims == 3:
            pads = [latent_overlap_t, latent_overlap_y, latent_overlap_x]
            pads_px = [overlap_t, overlap_y, overlap_x]
        else:
            pads = []
            pads_px = []
        for i, pad in enumerate(pads):
            if pad <= 0:
                continue
            dim_size = latent.shape[2 + i]
            left_glue = latent.narrow(2 + i, dim_size - pad, pad)
            right_glue = latent.narrow(2 + i, 0, pad)
            latent = torch.cat([left_glue, latent, right_glue], dim=2+i)

        images = vae.decode(latent)
        if len(images.shape) == 5: #Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        
        print("image shape:", images.shape)
        # remove pads
        for i, pad in enumerate(pads_px):
            if pad <= 0:
                continue
            dim_size = images.shape[1 + i]
            images = images.narrow(1 + i, pad, dim_size - 2 * pad)
        return (images, )

class GluedAttentionImpl:
    def __init__(self, loop_x: bool, loop_y: bool):
        self.loop_x = loop_x
        self.loop_y = loop_y

    def replicate_rope(self, data: dict):
        img_ids: Tensor = data["img_ids"]  # (batch_size, num_tokens, axes_dim)
        img_len = img_ids.shape[1]
        txt_len: int = data["txt_ids"].shape[1]
        valid_segment: List[Tuple[int, int]] = []  # valid distance of ids
        for loop, target_direction in [(self.loop_x, 2), (self.loop_y, 1)]:
            start, end, steps = retrieve_linspace_parameters(img_ids[:, :, target_direction])
            width = end - start
            if not loop:
                valid_segment.append((-width, width))  # full attention
                continue
            stepsize = (end - start) / (steps - 1)
            left_max = start + stepsize * (steps // 2)
            clone_img_ids = img_ids.clone()
            clone_img_ids[:, :, target_direction] = torch.where(
                img_ids[:, :, target_direction] <= left_max[:, None],
                img_ids[:, :, target_direction] + (stepsize * steps)[:, None],
                img_ids[:, :, target_direction] - (stepsize * steps)[:, None]
            )
            valid_segment.append((-width//2, width//2))  # only allow attention within half loop TODO: boundary case
            # valid_segment.append((-width//4, width//4))
            # update img_ids
            img_ids = torch.cat([img_ids, clone_img_ids], dim=1)
        out = data.copy()
        out["img_ids"] = img_ids
        mask_modifier = torch.ones((1, txt_len + img_len, txt_len + img_ids.shape[1]), device=img_ids.device, dtype=torch.bool)
        mask_modifier[:, :txt_len, txt_len + img_len:] = False  # prevent text tokens from attending to the glued image tokens
        # apply distance constraint
        for loop, target_direction, (left_bound, right_bound) in zip([self.loop_x, self.loop_y], [2, 1], valid_segment):
            if not loop:
                continue
            img_ids_axis = img_ids[:, :, target_direction]
            diff = img_ids_axis[:, None, :] - img_ids_axis[:, :img_len, None]
            mask_modifier[:, txt_len:, txt_len:] &= (left_bound <= diff) & (diff <= right_bound)
        self.mask_modifier = mask_modifier
        return out
    
    def attn_pre(self, q, k, v, pe=None, attn_mask=None, extra_options=None):
        """
        q: (batch_size, num_heads, L, dim)
        k: (batch_size, num_heads, S, dim)
        v: (batch_size, num_heads, S, dim)
        """
        block_type: Literal["single", "double"] = extra_options["block_type"]
        img_slice: List[int] = extra_options["img_slice"]
        """
        Note: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        L: query length, S: key/value length, N: batch size
        attn_mask (optional Tensor): Attention mask; shape must be broadcastable to the shape of attention weights, which is (N,...,L,S).
        Two types of masks are supported. A boolean mask where a value of True indicates that the element should take part in attention.
        A float mask of the same type as query, key, value that is added to the attention score.
        """
        attn_mask = attn_mask.clone() if attn_mask is not None else torch.ones((1, q.shape[2], k.shape[2]), device=q.device, dtype=torch.bool)
        assert attn_mask.shape == (1, q.shape[2], k.shape[2]), f"Expected attn_mask shape (1, {q.shape[2]}, {k.shape[2]}), got {attn_mask.shape}"
        k_glues = []
        v_glues = []
        for loop in [self.loop_x, self.loop_y]:
            if not loop:
                continue
            k_img = k[:, :, img_slice[0]:img_slice[1], :]
            v_img = v[:, :, img_slice[0]:img_slice[1], :]
            k_glues.append(k_img)
            v_glues.append(v_img)
            attn_mask = torch.cat([attn_mask, attn_mask[:, :, img_slice[0]:img_slice[1]]], dim=2)

        out = {
            "q": q,
            "k": torch.cat([k, *k_glues], dim=2),
            "v": torch.cat([v, *v_glues], dim=2),
            "pe": pe,
            "attn_mask": attn_mask & self.mask_modifier,
        }
        return out


class GluedAttention:
    """Glued Attention for looped structure."""
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model to apply the Glued Attention to."}),
                "loop_x": ("BOOLEAN", {"default": True}),
                "loop_y": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "execute"
    CATEGORY = "model_patches"

    def execute(self, model: io.Model.Type, loop_x: bool, loop_y: bool):
        model = model.clone()
        patch_obj = GluedAttentionImpl(loop_x, loop_y)
        model.set_model_post_input_patch(patch_obj.replicate_rope)
        model.set_model_attn1_patch(patch_obj.attn_pre)
        return (model,)



# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LatentShift": LatentShift,
    "VAEDecodeCircular": VAEDecodeCircular,
    "GluedAttention": GluedAttention,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentShift": "Latent Shift Node",
    "VAEDecodeCircular": "VAE Decode Circular Node",
    "GluedAttention": "Glued Attention Node"
}
