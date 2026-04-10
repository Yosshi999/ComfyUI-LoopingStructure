from inspect import cleandoc
from comfy_api.latest import io
import torch

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
        distr = torch.unique(samples, dim=1)  # (batch_size, num_unique == steps_w)
        start = samples.min(1).values  # (batch_size,)
        end = samples.max(1).values  # (batch_size,)
        steps = distr.shape[1]
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

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LatentShift": LatentShift,
    "VAEDecodeCircular": VAEDecodeCircular,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentShift": "Latent Shift Node",
    "VAEDecodeCircular": "VAE Decode Circular Node"
}
