import os
import glob
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True

import pywt  # Wavelet dönüşümü için
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T

# timm: EfficientNet, Vision Transformer vb. modelleri kullanmak için
import timm

# Scikit-learn metrikleri için:
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -----------------------------------------------------------------------------
# 1) Wavelet Dönüşümü Yardımcı Fonksiyonu
# -----------------------------------------------------------------------------
def wavelet_transform_grayscale(pil_img, wavelet='db1'):
    """
    Bir PIL Image (grayscale) üzerinde tek seviyeli 2D Discrete Wavelet Transform uygular.
    Dönüşüm sonucunda 4 adet alt bant elde edilir: (cA, cH, cV, cD).
    Örnek olarak cA, cH, cV bantlarını 3 kanala yerleştirip geri döndürüyoruz.
    """
    img_np = np.array(pil_img)
    cA, (cH, cV, cD) = pywt.dwt2(img_np, wavelet)
    cA = (cA - cA.min()) / (cA.max() - cA.min() + 1e-8)
    cH = (cH - cH.min()) / (cH.max() - cH.min() + 1e-8)
    cV = (cV - cV.min()) / (cV.max() - cV.min() + 1e-8)
    # 3 kanalı birleştiriyoruz. (cD atılıyor)
    wavelet_3ch = np.stack([cA, cH, cV], axis=0).astype(np.float32)
    wavelet_tensor = torch.from_numpy(wavelet_3ch)
    return wavelet_tensor

# -----------------------------------------------------------------------------
# 2) Özelleştirilmiş Dataset Sınıfı
# -----------------------------------------------------------------------------
class WaveletFaceDataset(Dataset):
    def __init__(self, mode='train', transform=None, wavelet='db1'):
        """
        mode: 'train' or 'test' modunu belirler
        transform: wavelet dönüşümü sonrası yapılacak ek transformlar (ör. Resize)
        wavelet: kullanılacak wavelet tipi (örn. 'db1', 'haar' vs.)
        """
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        train_fake = os.path.join(current_dir,'ciFake','train', 'fake')
        train_real = os.path.join(current_dir,'ciFake','train', 'real')
        test_fake = os.path.join(current_dir,'ciFake','test', 'fake')
        test_real = os.path.join(current_dir,'ciFake','test', 'real')

        max_images_per_class_train_fake = 50000
        max_images_per_class_train_real = 50000
        max_images_per_class_test_fake = 10000
        max_images_per_class_test_real = 10000
        
        # Initialize image paths and labels as class attributes
        self.image_paths = []
        self.labels = []
        
        # Get image lists for both train and test
        real_images_train = sorted(glob.glob(os.path.join(train_real, '**', '*.*'), recursive=True))
        fake_images_train = sorted(glob.glob(os.path.join(train_fake, '**', '*.*'), recursive=True))
        real_images_test = sorted(glob.glob(os.path.join(test_real, '**', '*.*'), recursive=True))
        fake_images_test = sorted(glob.glob(os.path.join(test_fake, '**', '*.*'), recursive=True))

        # Limit the number of images per class if specified
        if max_images_per_class_train_real is not None:
            real_images_train = real_images_train[:max_images_per_class_train_real]
            fake_images_train = fake_images_train[:max_images_per_class_train_fake]
            real_images_test = real_images_test[:max_images_per_class_test_real]
            fake_images_test = fake_images_test[:max_images_per_class_test_fake]

        if mode == 'train':
            # Gerçek eğitim görüntüleri (label 0)
            for path in real_images_train:
                self.image_paths.append(path)
                self.labels.append(0)

            # Sahte eğitim görüntüleri (label 1)
            for path in fake_images_train:
                self.image_paths.append(path)
                self.labels.append(1)
        else:  # mode == 'test'
            # Gerçek test görüntüleri (label 0)
            for path in real_images_test:
                self.image_paths.append(path)
                self.labels.append(0)

            # Sahte test görüntüleri (label 1)
            for path in fake_images_test:
                self.image_paths.append(path)
                self.labels.append(1)

        self.transform = transform
        self.wavelet = wavelet

    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            pil_img = Image.open(img_path).convert('L')
        except Exception as e:
            print(f"Hata: {e} | Dosya: {img_path}")
            return None  # Hatalı dosyalar atlanıyor
        
        wavelet_tensor = wavelet_transform_grayscale(pil_img, wavelet=self.wavelet)
        if self.transform is not None:
            wavelet_tensor = self.transform(wavelet_tensor)
        return wavelet_tensor, torch.tensor(label, dtype=torch.long)


def custom_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# -----------------------------------------------------------------------------
# 3) Hibrit Model: EfficientNet-B0 (CNN) + Transformer
# -----------------------------------------------------------------------------
class HybridModel(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridModel, self).__init__()
        # CNN kısmı: EfficientNet-B0, classifier katmanı kaldırılmış
        self.cnn = timm.create_model(
            'efficientnet_b1',
            pretrained=True,
            in_chans=3,
            num_classes=0,
            global_pool=''
        )
        # Transformer Encoder: 6 katman, 8 head
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1280,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        # Positional Embedding: 49 patch (7x7) için
        self.pos_embed = nn.Parameter(torch.zeros(1, 49, 1280))
        # Sınıflandırma katmanı
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        # CNN ile özellik haritası çıkarma: [B, 1280, 7, 7] varsayılıyor
        feat_map = self.cnn(x)
        B, C, Hc, Wc = feat_map.shape
        # Patch/Sequence haline getir: [B, 1280, 49] -> [B, 49, 1280]
        feat_map = feat_map.view(B, C, Hc * Wc).permute(0, 2, 1)
        # Positional embedding ekle
        feat_map = feat_map + self.pos_embed
        # Transformer Encoder: girişin boyutu (S, B, E)
        feat_map = feat_map.permute(1, 0, 2)
        transformer_out = self.transformer_encoder(feat_map)
        transformer_out = transformer_out.permute(1, 0, 2)  # [B, 49, 1280]
        out_pool = transformer_out.mean(dim=1)             # Global average pooling
        logits = self.fc(out_pool)
        return logits

# -----------------------------------------------------------------------------
# 4) Eğitim Döngüsü (train/test)
# -----------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, device):
    print("Başladı - train one epoch ...")
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = total_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, device):
    print("Başladı - evaluate ...")
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = F.cross_entropy(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = total_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# -----------------------------------------------------------------------------
# 5) Ana Kısım
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Hyper-parametreler
    batch_size = 32
    lr = 1e-5
    epochs = 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # StopLoss için parametreler
    patience = 5
    min_delta = 0.001
    best_val_loss = float('inf')
    counter = 0

    # Transform: 3 kanallı wavelet çıktısını 224x224'e indirger.
    custom_transform = T.Compose([
    T.Resize((224, 224)), # Boyutlandırmayı önce yapabiliriz
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1)), # İsteğe bağlı
    # T.ColorJitter(brightness=0.2), # Wavelet üzerinde dikkatli olun
    ])

    # Eğitim veri setini oluştur
    full_train_dataset = WaveletFaceDataset(
        mode='train',
        transform=custom_transform,
        wavelet='db1',
    )
    
    # Test veri setini oluştur
    test_dataset = WaveletFaceDataset(
        mode='test',
        transform=custom_transform,
        wavelet='db1',
    )
    
    # Eğitim setini eğitim ve doğrulama olarak böl (%90 eğitim, %10 doğrulama)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    print("Toplam eğitim veri seti boyutu:", len(full_train_dataset))
    print("Eğitim bölümü boyutu:", len(train_dataset))
    print("Doğrulama bölümü boyutu:", len(val_dataset))
    print("Test veri seti boyutu:", len(test_dataset))

    #--------------------------------
    # Eğitim seti için sınıf dengesini kontrol edelim:
    #--------------------------------
    num_real = sum(1 for idx in range(len(train_dataset)) if full_train_dataset.labels[train_dataset.indices[idx]] == 0)
    num_fake = sum(1 for idx in range(len(train_dataset)) if full_train_dataset.labels[train_dataset.indices[idx]] == 1)
    print(f"Train dataset içindeki gerçek görüntü sayısı: {num_real}")
    print(f"Train dataset içindeki sahte görüntü sayısı: {num_fake}")

    #--------------------------------
    # Doğrulama seti için sınıf dengesini kontrol edelim:
    #--------------------------------
    num_real_val = sum(1 for idx in range(len(val_dataset)) if full_train_dataset.labels[val_dataset.indices[idx]] == 0)
    num_fake_val = sum(1 for idx in range(len(val_dataset)) if full_train_dataset.labels[val_dataset.indices[idx]] == 1)
    print(f"Validation dataset içindeki gerçek görüntü sayısı: {num_real_val}")
    print(f"Validation dataset içindeki sahte görüntü sayısı: {num_fake_val}")

    #--------------------------------
    # Test seti için sınıf dengesini kontrol edelim:
    #--------------------------------
    num_real_test = sum(1 for label in test_dataset.labels if label == 0)
    num_fake_test = sum(1 for label in test_dataset.labels if label == 1)
    print(f"Test dataset içindeki gerçek görüntü sayısı: {num_real_test}")
    print(f"Test dataset içindeki sahte görüntü sayısı: {num_fake_test}")
    
    # Sınıf ağırlıkları belirleniyor
    class_counts = [num_real, num_fake]
    total = sum(class_counts)
    class_weights = [total / c for c in class_counts]
    weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    #--------------------------------
    # DataLoader'lar
    #--------------------------------
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)

    #--------------------------------
    # Model, optimizer 
    #--------------------------------
    model = HybridModel(num_classes=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    #--------------------------------
    # Eğitim Döngüsü
    #--------------------------------
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch [{epoch}/{epochs}] Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
        
        # Validation loss için evaluate fonksiyonunu kullan
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Epoch [{epoch}/{epochs}] Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")
        
        # Early stopping kontrolü
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            counter = 0
            # En iyi modeli kaydet
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Model improved, saving checkpoint at epoch {epoch}")
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")
            if counter >= patience:
                print("Early stopping")
                # En iyi modeli yükle
                model.load_state_dict(torch.load('best_model.pth'))
                break

    #--------------------------------
    # Eğitim sonrası, harici test seti üzerindeki performans değerlendirmesi
    #--------------------------------
    model.eval()
    all_test_preds = []
    all_test_labels = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_test_preds.append(preds.cpu())
            all_test_labels.append(labels.cpu())
    all_test_preds = torch.cat(all_test_preds).numpy()
    all_test_labels = torch.cat(all_test_labels).numpy()

    test_acc = accuracy_score(all_test_labels, all_test_preds)
    test_prec = precision_score(all_test_labels, all_test_preds, average='binary')
    test_rec = recall_score(all_test_labels, all_test_preds, average='binary')
    test_f1 = f1_score(all_test_labels, all_test_preds, average='binary')
    test_cm = confusion_matrix(all_test_labels, all_test_preds)

    print("------ TEST METRİKLERİ ------")
    print(f"Accuracy  : {test_acc:.4f}")
    print(f"Precision : {test_prec:.4f}")
    print(f"Recall    : {test_rec:.4f}")
    print(f"F1 Score  : {test_f1:.4f}")
    print("Confusion Matrix:")
    print(test_cm)