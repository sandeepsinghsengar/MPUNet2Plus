from .unet import UNet
from .unet3D import UNet3D
from .fusion_model import FusionModel
from .model_init import model_initializer
from .multitask_unet2d import MultiTaskUNet2D
from .unet3plus import UNet3Plus
from .unet2plus import UNet2Plus
from .unet2plus1 import UNet2Plus1
from .unet2plus2 import UNet2Plus2
from .unet2plus_up import UNet2Plus_up
from .unet2plus_up_deep import UNet2Plus_up_deep
from .unet3plus_deep import UNet3Plus_deep

# Prepare a dictionary mapping from model names to data prep. functions
from mpunet.preprocessing import data_preparation_funcs as dpf

PREPARATION_FUNCS = {
    "UNet": dpf.prepare_for_multi_view_unet,
    "UNet3Plus": dpf.prepare_for_multi_view_unet,
    "UNet3Plus_deep": dpf.prepare_for_multi_view_unet,
    "UNet3D": dpf.prepare_for_3d_unet,
    "UNet2Plus": dpf.prepare_for_multi_view_unet,
    "UNet2Plus_up": dpf.prepare_for_multi_view_unet,
    "UNet2Plus_up_deep": dpf.prepare_for_multi_view_unet,
    "UNet2Plus1": dpf.prepare_for_multi_view_unet,
    "UNet2Plus2": dpf.prepare_for_multi_view_unet,
    "MultiTaskUNet2D": dpf.prepare_for_multi_task_2d
}
