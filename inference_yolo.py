import cv2
import numpy as np
import openvino as ov
import time

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    """
    Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def main():
    model_path = "modelvino/model_yolo.xml"
    image_path = "image/car.jpg"
    conf_thres = 0.25
    iou_thres = 0.45
    
    # 1. Load Model
    print(f"Loading model: {model_path}")
    core = ov.Core()
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, "CPU")
    infer_request = compiled_model.create_infer_request()

    # 2. Preprocessing
    print(f"Reading image: {image_path}")
    img0 = cv2.imread(image_path)
    if img0 is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Resize and Pad
    img_size = 640
    img, ratio, (dw, dh) = letterbox(img0, new_shape=img_size, auto=False)
    
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    
    img = img.astype(np.float32)
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    # 3. Inference
    print("Running inference...")
    t0 = time.time()
    input_tensor = ov.Tensor(img)
    infer_request.set_input_tensor(input_tensor)
    infer_request.infer()
    output = infer_request.get_output_tensor(0).data
    print(f"Inference time: {(time.time() - t0):.3f}s")
    
    # Output shape is usually [1, 84, 8400] for YOLOv8/v11
    # We need [1, 8400, 84]
    output = output.transpose(0, 2, 1)
    
    # 4. Postprocessing
    # Predictions: [x_center, y_center, w, h, class1_conf, class2_conf, ...]
    # For extraction, we only have one class (usually?) or multiple.
    # Check shape
    bs, num_proposals, num_features = output.shape
    print(f"Output shape: {output.shape}")

    # Parse results
    boxes = []
    scores = []
    class_ids = []

    # Iterate through proposals
    # Optimizing: vectorized approach is better but loop is clearer for demo
    predictions = output[0] # Take first batch
    
    # Filter by confidence
    # Assuming first 4 are box, rest are classes
    # If standard YOLO: [x, y, w, h, score] ?? No, v8/v11 is [x,y,w,h, cls1, cls2...]
    
    # Extract max class score
    classes_scores = predictions[:, 4:]
    max_scores = np.max(classes_scores, axis=1)
    argmax_classes = np.argmax(classes_scores, axis=1)
    
    # Filter
    mask = max_scores > conf_thres
    filtered_preds = predictions[mask]
    filtered_scores = max_scores[mask]
    filtered_classes = argmax_classes[mask]
    
    for i, pred in enumerate(filtered_preds):
        x, y, w, h = pred[0], pred[1], pred[2], pred[3]
        score = filtered_scores[i]
        cls_id = filtered_classes[i]
        
        # Convert to Top-Left Coordinate
        # x, y is center. w, h is width height
        left = int((x - 0.5 * w))
        top = int((y - 0.5 * h))
        width = int(w)
        height = int(h)
        
        boxes.append([left, top, width, height])
        scores.append(float(score))
        class_ids.append(cls_id)

    # NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    if len(indices) > 0:
        indices = indices.flatten()
    
    print(f"Detected {len(indices)} objects.")
    
    # 5. Draw
    for i in indices:
        box = boxes[i]
        score = scores[i]
        cls_id = class_ids[i]
        
        x, y, w, h = box
        
        # Rescale boxes to original image
        # New coordinates relative to resized image (640x640)
        # We need to undo the letterbox padding
        
        # Remove padding
        x -= dw
        y -= dh
        
        # Scale back
        x /= ratio[0]
        y /= ratio[1]
        w /= ratio[0]
        h /= ratio[1]
        
        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w)
        y2 = int(y + h)
        
        # Draw
        color = (0, 255, 0)
        cv2.rectangle(img0, (x1, y1), (x2, y2), color, 2)
        label = f"Class {cls_id} {score:.2f}"
        cv2.putText(img0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    output_path = "result_yolo.jpg"
    cv2.imwrite(output_path, img0)
    print(f"Saved result to {output_path}")

if __name__ == "__main__":
    main()
