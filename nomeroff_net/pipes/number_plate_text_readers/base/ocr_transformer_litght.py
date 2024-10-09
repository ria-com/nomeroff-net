"""
python3.9 -m nomeroff_net.pipes.number_plate_text_readers.base.ocr_transformer_litght -f ./nomeroff_net/pipes/number_plate_text_readers/base/ocr_transformer_litght.py
"""
import os
import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from torchvision.models import efficientnet_b2


transform = transforms.Compose([
    transforms.Resize((100, 300)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class LicensePlateDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.char_to_idx = self.create_char_dict()
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

    def create_char_dict(self):
        chars = set()
        for label in self.labels:
            chars.update(list(label))
        char_list = sorted(list(chars))
        char_to_idx = {char: idx + 1 for idx, char in enumerate(char_list)}  # Індексація з 1
        char_to_idx['<blank>'] = 0  # CTC-порожній символ
        return char_to_idx

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')  # Конвертація в RGB
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        label_encoded = torch.tensor([self.char_to_idx[char] for char in label], dtype=torch.long)
        return image, label_encoded


def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    labels = torch.cat(labels)
    return images, labels, label_lengths


class OCRModel(pl.LightningModule):
    def __init__(self, num_classes):
        super(OCRModel, self).__init__()
        self.num_classes = num_classes

        # CNN
        conv_nn = efficientnet_b2(pretrained=True)
        conv_modules = list(conv_nn.children())[:-2]
        self.cnn = nn.Sequential(*conv_modules)
        _, backbone_c, backbone_h, backbone_w = self.cnn(torch.rand((1, 3, 100, 300))).shape

        # Проекція ознак
        self.fc = nn.Linear(backbone_c*backbone_h, 512)  # Налаштуйте відповідно до виходу CNN

        # Позиційне кодування
        self.pos_encoder = PositionalEncoding(d_model=512)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Класифікатор
        self.classifier = nn.Linear(512, self.num_classes)

        # Функція втрат
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn(x)  # (B, C, H, W)
        x = x.permute(3, 0, 1, 2)  # (W, B, C, H)
        x = x.view(x.size(0), batch_size, -1)  # (SeqLen, B, C*H)
        x = self.fc(x)  # (SeqLen, B, 256)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (SeqLen, B, 256)
        x = self.classifier(x)  # (SeqLen, B, num_classes)
        x = F.log_softmax(x, dim=2)
        return x

    def training_step(self, batch, batch_idx):
        images, labels, label_lengths = batch
        batch_size = images.size(0)
        logits = self(images)  # (SeqLen, B, num_classes)

        input_lengths = torch.full(size=(batch_size,), fill_value=logits.size(0), dtype=torch.long)
        loss = self.ctc_loss(logits, labels, input_lengths, label_lengths)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, label_lengths = batch
        batch_size = images.size(0)
        logits = self(images)
        input_lengths = torch.full(size=(batch_size,), fill_value=logits.size(0), dtype=torch.long)
        loss = self.ctc_loss(logits, labels, input_lengths, label_lengths)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()

        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(max_len, 1, d_model)  # (max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # Парні індекси
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # Непарні індекси

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x


def decode_predictions(logits, idx_to_char):
    # Greedy decoding
    logits = logits.permute(1, 0, 2)  # (B, SeqLen, num_classes)
    predicted_indices = torch.argmax(logits, dim=2)  # (B, SeqLen)
    predicted_indices = predicted_indices.cpu().numpy()

    decoded_texts = []
    for indices in predicted_indices:
        chars = []
        prev_idx = -1
        for idx in indices:
            if idx != prev_idx and idx != 0:
                chars.append(idx_to_char.get(idx, ''))
            prev_idx = idx
        decoded_texts.append(''.join(chars))
    return decoded_texts


def test(model, image_paths, labels, transform, idx_to_char):
    model.eval()
    predicted_texts = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)  # Додати розмір пакету
        with torch.no_grad():
            logits = model(image)
        predicted_texts.append(decode_predictions(logits, idx_to_char)[0])
    acc = 0
    for pred, label in zip(predicted_texts, labels):
        if pred == label:
            acc += 1
    if acc == 0:
        acc = 0
    else:
        acc = acc / len(predicted_texts)
    return acc, predicted_texts


def collect_sub_dataset(dataset_sub_directory):
    image_paths = []
    labels = []
    ann_dir = os.path.join(dataset_sub_directory, "ann")
    img_dir = os.path.join(dataset_sub_directory, "img")
    for file_name in tqdm(os.listdir(img_dir)):
        name, ext = os.path.splitext(file_name)
        if ext == '.png':
            json_filepath = os.path.join(ann_dir, name + '.json')
            if not os.path.exists(json_filepath):
                continue
            image_paths.append(os.path.join(img_dir, file_name))
            description = json.load(open(json_filepath, 'r'))['description']
            labels.append(description)
    print(dataset_sub_directory,len(image_paths), len(labels))
    return image_paths, labels


def collect_dataset(dataset_directory):
    train_dir = os.path.join(dataset_directory, "train")
    val_dir = os.path.join(dataset_directory, "val")
    test_dir = os.path.join(dataset_directory, "test")
    train_image_paths, train_labels = collect_sub_dataset(train_dir)
    val_image_paths, val_labels = collect_sub_dataset(val_dir)
    test_image_paths, test_labels = collect_sub_dataset(test_dir)
    return train_image_paths, train_labels, val_image_paths, val_labels, test_image_paths, test_labels


if __name__ == '__main__':
    dataset_dir = os.path.join(os.path.join(os.getcwd(), "./data/dataset/TextDetector/ocr_example"))

    # Припустимо, що у вас є списки шляхів до зображень та відповідних міток
    (train_image_paths, train_labels,
     val_image_paths, val_labels,
     test_image_paths, test_labels) = collect_dataset(dataset_dir)

    train_dataset = LicensePlateDataset(train_image_paths, train_labels, transform=transform)
    val_dataset = LicensePlateDataset(val_image_paths, val_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn)

    num_classes = len(train_dataset.char_to_idx)
    model = OCRModel(num_classes=num_classes)

    trainer = pl.Trainer(max_epochs=10, gpus=1 if torch.cuda.is_available() else 0)
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint("last.ckpt", weights_only=False)

    acc, predicted_texts = test(model, test_image_paths, test_labels, transform, train_dataset.idx_to_char)
    print("Accuracy", acc)

