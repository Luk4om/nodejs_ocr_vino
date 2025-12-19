import torch
import openvino as ov
import sys
import os
import types
from collections import OrderedDict

# =============================================================================
# MOCKING DEPENDENCIES
# The segmentation model definition relies on 'mmseg' and 'mmcv', which are 
# complex to install. We mock them since we only need the class definition.
# =============================================================================

# Mock mmcv_custom
mmcv_custom = types.ModuleType("mmcv_custom")
mmcv_custom.load_checkpoint = lambda *args, **kwargs: None
sys.modules["mmcv_custom"] = mmcv_custom

# Mock mmseg
mmseg = types.ModuleType("mmseg")
mmseg.utils = types.ModuleType("mmseg.utils")
mmseg.utils.get_root_logger = lambda *args, **kwargs: None
mmseg.models = types.ModuleType("mmseg.models")
mmseg.models.builder = types.ModuleType("mmseg.models.builder")

class MockRegistry:
    def register_module(self):
        def decorator(cls):
            return cls
        return decorator
mmseg.models.builder.BACKBONES = MockRegistry()

sys.modules["mmseg"] = mmseg
sys.modules["mmseg.utils"] = mmseg.utils
sys.modules["mmseg.models"] = mmseg.models
sys.modules["mmseg.models.builder"] = mmseg.models.builder

# =============================================================================

# Add path for UniFormer segmentation definitions
current_dir = os.path.dirname(os.path.abspath(__file__))
# Note: uniformer_light.py is in semantic_segmentation/fpn_seg/
uniformer_seg_path = os.path.join(current_dir, "UniFormer", "semantic_segmentation", "fpn_seg")
if uniformer_seg_path not in sys.path:
    sys.path.append(uniformer_seg_path)

try:
    # Importing from the file we found in semantic_segmentation/fpn_seg/uniformer_light.py
    from uniformer_light import uniformer_xxs
except ImportError as e:
    print("Error Importing UniFormer (Segmentation version):", e)
    sys.exit(1)


def convert_uniformer():
    model_path = "basemodel/fpn_xxs_uniformer.pth"
    output_path = "modelvino/model_uniformer.xml"
    
    print("Instantiating UniFormer-XXS model (Segmentation Backbone)...")
    # segmentation version might accept pretrained via kwargs or arguments?
    # Checking definition: def uniformer_xxs(**kwargs) -> calls UniFormer_Light
    # UniFormer_Light.__init__ args: depth, in_chans... pretrained_path=None
    model = uniformer_xxs()
    
    # Load weights
    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Handle state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Clean keys (remove 'backbone.' prefix if present)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            name = k[9:] # remove `backbone.`
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v

    # Load state dict
    # strict=False allows ignoring head/neck keys if they exist in weights but not backbone
    msg = model.load_state_dict(new_state_dict, strict=False)
    print("Load weights result:", msg)
    
    model.eval()

    # Create dummy input
    # Adjust H, W to your desired input size (e.g., 512x512 common for seg)
    dummy_input = torch.randn(1, 3, 512, 512)

    # Convert to OpenVINO
    print("Converting to OpenVINO...")
    ov_model = ov.convert_model(model, example_input=dummy_input)
    
    # Save
    ov.save_model(ov_model, output_path)
    print(f"Saved to {output_path}")

    # Export to ONNX
    onnx_path = output_path.replace(".xml", ".onnx")
    print(f"Exporting to ONNX: {onnx_path}...")
    torch.onnx.export(model,               # model being run
                      dummy_input,         # model input (or a tuple for multiple inputs)
                      onnx_path,           # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=17,    # Use a newer opset version
                      do_constant_folding=True,
                      input_names = ['input'],
                      output_names = ['output_s1', 'output_s2', 'output_s3', 'output_s4'])
    print(f"ONNX Export complete.")

if __name__ == "__main__":
    convert_uniformer()
