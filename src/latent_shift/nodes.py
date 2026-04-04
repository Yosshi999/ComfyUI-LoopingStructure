from inspect import cleandoc
from comfy_api.latest import io

class LatentShiftImpl:
    def __init__(self, shift_step, shift_axis):
        self.shift_step = shift_step
        self.shift_axis = shift_axis
        self.call_count = 0
    def __call__(self, q, k, v, extra_options, **kwargs):
        return {"q": q, "k": k, "v": v}
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
