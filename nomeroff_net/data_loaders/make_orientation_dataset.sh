#!/bin/bash

# Перевірка наявності аргументів
if [ "$#" -ne 4 ]; then
  echo "Використання: $0 <директорія_зображень> <директорія_для_датасету> <тренувальний_відсоток> <тестовий_відсоток>"
  exit 1
fi

# Вхідні аргументи
IMAGES_DIR=$1
DATASET_DIR=$2
TRAIN_PERCENT=$3
TEST_PERCENT=$4

# Валідаційний відсоток обчислюється як залишок
VAL_PERCENT=$((100 - TRAIN_PERCENT - TEST_PERCENT))

# Функція для створення папок
create_directories() {
  local base_dir=$1
  mkdir -p "$base_dir/0" "$base_dir/90" "$base_dir/180"
}

# Створення папок для датасету
create_directories "$DATASET_DIR/train"
create_directories "$DATASET_DIR/test"
create_directories "$DATASET_DIR/val"

# Функція для створення повернених копій зображень
create_rotated_images() {
  local src=$1
  local base_dest=$2
  local filename=$(basename "$src")

  convert -rotate 0 "$src" "$base_dest/0/$filename"
  convert -rotate 180 "$src" "$base_dest/180/$filename"

  # Випадковий вибір між 90 і 270 градусами
  if (( RANDOM % 2 )); then
    convert -rotate 90 "$src" "$base_dest/90/$filename"
  else
    convert -rotate 270 "$src" "$base_dest/90/$filename"
  fi
}

# Отримання списку зображень і розподіл за наборами
images=("$IMAGES_DIR"/*)
total_images=${#images[@]}
train_count=$((total_images * TRAIN_PERCENT / 100))
test_count=$((total_images * TEST_PERCENT / 100))
val_count=$((total_images - train_count - test_count))

# Розподіл зображень по наборах
for i in "${!images[@]}"; do
  if [ "$i" -lt "$train_count" ]; then
    create_rotated_images "${images[$i]}" "$DATASET_DIR/train"
  elif [ "$i" -lt "$((train_count + test_count))" ]; then
    create_rotated_images "${images[$i]}" "$DATASET_DIR/test"
  else
    create_rotated_images "${images[$i]}" "$DATASET_DIR/val"
  fi
done

echo "Датасет створено в директорії $DATASET_DIR"
echo "Тренувальний набір: $train_count зображень"
echo "Тестовий набір: $test_count зображень"
echo "Валідаційний набір: $val_count зображень"
