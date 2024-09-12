import os
import shutil
import random
import json

# Функція для створення директорій, якщо вони не існують
def create_folders(base_dir):
    for folder in ['train', 'test', 'val']:
        img_folder = os.path.join(base_dir, folder, 'img')
        ann_folder = os.path.join(base_dir, folder, 'ann')
        os.makedirs(img_folder, exist_ok=True)
        os.makedirs(ann_folder, exist_ok=True)

# Функція для створення анотаційного JSON-файлу
def generate_annotation(image_name, save_path):
    image_name = os.path.splitext(image_name)[0]
    annotation = {
        "state_id": 2,
        "region_id": 1,
        "name": image_name,
        "count_lines": 3,
        "skip_region": 1,
        "orientation": 0 
    }
    json_file = os.path.join(save_path, f"{image_name}.json")
    with open(json_file, 'w') as f:
        json.dump(annotation, f, indent=4)

# Основна функція для генерації датасету
def generate_dataset(input_dir, output_dir, split_ratio=(0.9, 0.05, 0.05)):
    # Створюємо потрібні папки
    create_folders(output_dir)

    # Отримуємо список всіх зображень
    images = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Перемішуємо зображення
    random.shuffle(images)

    # Рахуємо кількість для кожної папки
    total_images = len(images)
    train_count = int(total_images * split_ratio[0])
    test_count = int(total_images * split_ratio[1])

    train_images = images[:train_count]
    test_images = images[train_count:train_count + test_count]
    val_images = images[train_count + test_count:]

    dataset_splits = {
        'train': train_images,
        'test': test_images,
        'val': val_images
    }

    # Копіюємо зображення та генеруємо JSON файли
    for split, img_list in dataset_splits.items():
        img_folder = os.path.join(output_dir, split, 'img')
        ann_folder = os.path.join(output_dir, split, 'ann')

        for img in img_list:
            img_src = os.path.join(input_dir, img)
            img_dst = os.path.join(img_folder, img)
            shutil.copy(img_src, img_dst)

            generate_annotation(img, ann_folder)

    print("Датасет успішно згенеровано!")

# Використання
#input_directory = "./1lines_classification"
input_directory = "./ua_3lines_cropped"
output_directory = "./dataset_3lines"
generate_dataset(input_directory, output_directory)
