def find_missing_labels(image_folder, label_folder):
    image_files = set([f.split('.')[0] for f in os.listdir(image_folder) if f.lower().endswith('.jpg')])
    label_files = set([f.split('.')[0] for f in os.listdir(label_folder) if f.lower().endswith('.txt')])
    
    missing_labels = image_files - label_files

    return missing_labels

if __name__ == "__main__":
    image_folder = "C:/Users/kwanghuijacksonng/Downloads/labs and projects/capstone/data/train/manual_train/images"
    label_folder = "C:/Users/kwanghuijacksonng/Downloads/labs and projects/capstone/data/train/manual_train/labels"

    missing_labels = find_missing_labels(image_folder, label_folder)

    if missing_labels:
        print("Missing labels for the following images:")
        for image_name in missing_labels:
            print(image_name)
    else:
        print("All images have corresponding labels.")