from ultralytics import YOLO
import shutil
import os

def convert_and_organize():
    # Configuration
    model_path = "basemodel/yolo11n.pt"
    output_dir = "modelvino"
    temp_export_dir = "basemodel/yolo11n_openvino_model"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Export Model
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    
    print("Exporting to OpenVINO format...")
    # This automatically creates the temp_export_dir
    export_path = model.export(format="openvino")
    print(f"Export completed at: {export_path}")

    # 2. Organize Files
    print("Organizing files...")
    files_to_move = {
        "yolo11n.xml": "model_yolo.xml",
        "yolo11n.bin": "model_yolo.bin",
        "metadata.yaml": "metadata.yaml"
    }

    for src_name, dst_name in files_to_move.items():
        src_file = os.path.join(temp_export_dir, src_name)
        dst_file = os.path.join(output_dir, dst_name)
        
        if os.path.exists(src_file):
            print(f"Moving {src_name} -> {dst_file}")
            shutil.move(src_file, dst_file)
        else:
            print(f"Warning: {src_name} not found in export output.")

    # 3. Cleanup
    if os.path.exists(temp_export_dir):
        print(f"Removing temporary directory: {temp_export_dir}")
        shutil.rmtree(temp_export_dir)

    print("\nconversion and setup complete!")
    print(f"Files are ready in '{output_dir}/'")

if __name__ == "__main__":
    convert_and_organize()
