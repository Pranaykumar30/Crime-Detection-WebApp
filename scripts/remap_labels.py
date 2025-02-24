import os

def remap_labels(label_dir):
    class_map = {
        46: 0,  # COCO 'gun' -> handguns
        43: 1,  # COCO 'knife' -> knives
        0: 3    # COCO 'person' -> masked-intruders (default)
    }

    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        if not os.path.isfile(label_path):
            continue

        with open(label_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            if "handgun" in label_file.lower():
                new_class = 0
            elif "knife" in label_file.lower():
                new_class = 1
            elif "sharp" in label_file.lower():
                new_class = 2
            elif "masked" in label_file.lower():
                new_class = 3
            elif "violence" in label_file.lower():
                new_class = 4
            else:
                new_class = 5  # Default class for normal cases

            with open(label_path, "w") as f:
                f.write(f"{new_class} 0.5 0.5 0.2 0.2\n")

            print(f"Empty label for {label_file} - assigned class {new_class}")
            continue

        with open(label_path, "w") as f:
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split()
                old_class = int(parts[0])

                if old_class == 0:
                    if "masked" in label_file.lower():
                        new_class = 3
                    elif "violence" in label_file.lower():
                        new_class = 4
                    else:
                        new_class = 5
                elif old_class == 43:
                    new_class = 2 if "sharp" in label_file.lower() else 1
                else:
                    new_class = class_map.get(old_class, old_class)

                f.write(f"{new_class} {' '.join(parts[1:])}\n")

if __name__ == "__main__":
    remap_labels("data/labels")
