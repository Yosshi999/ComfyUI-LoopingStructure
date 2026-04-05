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


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LatentShift": LatentShift
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentShift": "Latent Shift Node"
}
