import cv2
import numpy as np
import openvino as ov
import time
import os

def preprocess(image_path, input_size=(512, 512)):
    """
    Preprocess image for UniFormer:
    - Resize
    - Normalize (ImageNet mean/std)
    - HWC -> NCHW
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, input_size)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std =  np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    
    # HWC -> CHW
    img = img.transpose(2, 0, 1)
    
    # Add batch dimension -> NCHW
    img = np.expand_dims(img, axis=0)
    return img

def main():
    model_path = "modelvino/model_uniformer.xml"
    image_path = "image/car.jpg"  # Or image_test.png
    
    # 1. Load Model
    print(f"Loading model: {model_path}")
    core = ov.Core()
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, "CPU")
    infer_request = compiled_model.create_infer_request()
    
    # Check inputs/outputs
    input_layer = compiled_model.input(0)
    # Check inputs/outputs
    input_layer = compiled_model.input(0)
    print(f"Input partial shape: {input_layer.partial_shape}")
    
    # 2. Preprocess
    print(f"Reading image: {image_path}")
    input_data = preprocess(image_path)
    
    # 3. Inference
    print("Running inference...")
    t0 = time.time()
    infer_request.set_input_tensor(0, ov.Tensor(input_data))
    infer_request.infer()
    print(f"Inference time: {(time.time() - t0):.3f}s")
    
    # 4. Result Processing
    # The model outputs a tuple of features (Backbone features)
    # We will visualize the last feature map (Stage 4)
    
    # Get all outputs
    outputs = []
    for output in compiled_model.outputs:
        outputs.append(infer_request.get_tensor(output).data)
        
    print(f"Model returned {len(outputs)} feature maps.")
    
    for i, out in enumerate(outputs):
        print(f"Feature Map {i}: shape={out.shape}")
        
    # Visualize the last feature map (deepest features)
    last_feature = outputs[-1] # shape [1, 448, 16, 16] for 512x512 input
    
    # Average across channels to create a heatmap
    heatmap = np.mean(last_feature[0], axis=0) # [H, W]
    
    # Normalize to 0-255
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = np.uint8(heatmap)
    
    # Resize to original image size for overlay
    # For demo, just save the heatmap
    heatmap = cv2.resize(heatmap, (512, 512), interpolation=cv2.INTER_NEAREST)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    output_filename = "result_uniformer_features.jpg"
    cv2.imwrite(output_filename, heatmap_color)
    print(f"Saved feature heatmap to {output_filename}")
    print("Note: This model is a Backbone only. The output represents feature activations, not semantic classes.")

if __name__ == "__main__":
    main()
