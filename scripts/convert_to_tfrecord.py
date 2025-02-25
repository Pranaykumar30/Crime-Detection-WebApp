import tensorflow as tf
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2

def int64_feature(value):
    """Returns a TFRecord int64 feature."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    """Returns a TFRecord bytes feature."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def float_list_feature(value):
    """Returns a TFRecord float feature list."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_list_feature(value):
    """Returns a TFRecord bytes feature list."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v.encode('utf8') for v in value]))

def int64_list_feature(value):
    """Returns a TFRecord int64 feature list."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def create_tf_example(img_path, label_path, class_map):
    """Convert an image and its label into a TFRecord example."""
    with open(img_path, "rb") as img_file:
        encoded_image = img_file.read()

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Invalid image: {img_path}")
    
    height, width = img.shape[:2]
    filename = os.path.basename(img_path).encode("utf8")

    xmins, xmaxs, ymins, ymaxs, classes_text, classes = [], [], [], [], [], []

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.split()
                class_id = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:])
                x_min = (x_center - w / 2) * width
                x_max = (x_center + w / 2) * width
                y_min = (y_center - h / 2) * height
                y_max = (y_center + h / 2) * height

                xmins.append(x_min / width)
                xmaxs.append(x_max / width)
                ymins.append(y_min / height)
                ymaxs.append(y_max / height)
                classes_text.append(class_map[class_id])
                classes.append(class_id)

    return tf.train.Example(features=tf.train.Features(feature={
        "image/height": int64_feature(height),
        "image/width": int64_feature(width),
        "image/filename": bytes_feature(filename),
        "image/source_id": bytes_feature(filename),
        "image/encoded": bytes_feature(encoded_image),
        "image/format": bytes_feature(b"jpeg"),
        "image/object/bbox/xmin": float_list_feature(xmins),
        "image/object/bbox/xmax": float_list_feature(xmaxs),
        "image/object/bbox/ymin": float_list_feature(ymins),
        "image/object/bbox/ymax": float_list_feature(ymaxs),
        "image/object/class/text": bytes_list_feature(classes_text),
        "image/object/class/label": int64_list_feature(classes)
    }))

def convert_to_tfrecord(image_dir, label_dir, output_path, class_map):
    """Convert a dataset into a TFRecord file."""
    writer = tf.io.TFRecordWriter(output_path)

    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        label_file = img_file.replace(".jpg", ".txt")
        label_path = os.path.join(label_dir, label_file)

        try:
            tf_example = create_tf_example(img_path, label_path, class_map)
            writer.write(tf_example.SerializeToString())
            print(f"Converted {img_file} to TFRecord")
        except Exception as e:
            print(f"Error converting {img_file}: {e}")

    writer.close()

if __name__ == "__main__":
    class_map = {
        0: "handguns",
        1: "knives",
        2: "sharp-edged-weapons",
        3: "masked-intruders",
        4: "violence",
        5: "normal-behavior"
    }

    for split in ["train", "val", "test"]:
        image_dir = f"data/split/{split}/images"
        label_dir = f"data/split/{split}/labels"
        output_path = f"data/{split}.tfrecord"
        convert_to_tfrecord(image_dir, label_dir, output_path, class_map)
