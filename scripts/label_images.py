from ultralytics import YOLO
import os

def label_images(image_dir, label_dir):
    model = YOLO("yolov8n.pt")  # Load YOLOv8 model
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    
    # Ensure necessary directories exist
    os.makedirs("runs/detect/predict/labels", exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for img in image_files:
        img_path = os.path.join(image_dir, img)
        
        if not os.path.exists(img_path):
            print(f"Skipping {img} - image not found")
            continue
        
        try:
            # Run YOLOv8 model prediction
            results = model.predict(img_path, save_txt=True, conf=0.05)

            # Define paths for labels
            label_file = img.replace(".jpg", ".txt")
            src_path = os.path.join("runs/detect/predict/labels", label_file)
            dest_path = os.path.join(label_dir, label_file)

            # Move label file if detections exist
            if os.path.exists(src_path) and os.path.getsize(src_path) > 0:
                os.rename(src_path, dest_path)
                print(f"Labeled {img} with detections")
            else:
                # Assign a default label based on filename keywords
                if "handgun" in img.lower():
                    class_id = 0
                elif "knife" in img.lower():
                    class_id = 1
                elif "sharp" in img.lower():
                    class_id = 2
                elif "masked" in img.lower():
                    class_id = 3
                elif "violence" in img.lower():
                    class_id = 4
                else:
                    class_id = 5  # Default class for normal cases

                # Write label file manually
                with open(dest_path, "w") as f:
                    f.write(f"{class_id} 0.5 0.5 0.2 0.2\n")
                
                print(f"No detections for {img} - assigned class {class_id} from filename")
        
        except Exception as e:
            print(f"Error processing {img}: {e}")

    # Clean up temporary prediction files
    os.system("rm -rf runs/detect/predict")

if __name__ == "__main__":
    label_images("data/images", "data/labels")
