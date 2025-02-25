import os

def remap_labels(label_dir):
    class_map = {
        46: 0,  # COCO 'gun' -> handguns
        43: 1,  # COCO 'knife' -> knives
        17: 2,  # COCO 'sharp-edged-weapons'
        0: 3    # COCO 'person' -> masked-intruders (default)
    }

    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        if not os.path.isfile(label_path):
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        if not lines:
            # Assign class based on filename keywords if the label file is empty
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
                new_class = 5  # Default class for normal behavior

            with open(label_path, "w") as f:
                f.write(f"{new_class} 0.5 0.5 0.2 0.2\n")

            print(f"Empty label for {label_file} - assigned class {new_class}")
            continue

        updated_lines = []
        for line in lines:
            parts = line.split()
            if not parts:
                continue

            old_class = int(parts[0])

            # Maintain correct class mappings
            if old_class in class_map:
                new_class = class_map[old_class]
            else:
                # Infer class from filename if not mapped
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

            updated_lines.append(f"{new_class} {' '.join(parts[1:])}\n")

        # Write the updated label mappings back to the file
        with open(label_path, "w") as f:
            f.writelines(updated_lines)

        print(f"Updated labels for {label_file}")

if __name__ == "__main__":
    remap_labels("data/labels")
