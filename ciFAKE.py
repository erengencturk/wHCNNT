import os
import glob
import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True

import pywt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import timm

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

train_real = os.path.join('/content/train', 'REAL')
train_fake = os.path.join('/content/train', 'FAKE')
test_real  = os.path.join('/content/test',  'REAL')
test_fake  = os.path.join('/content/test',  'FAKE')

def wavelet_transform_grayscale(pil_img, wavelet='db1'):
    img_np = np.array(pil_img)
    cA, (cH, cV, cD) = pywt.dwt2(img_np, wavelet)
    def _norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)
    wavelet_3ch = np.stack([_norm(cA), _norm(cH), _norm(cV)], axis=0).astype(np.float32)
    return torch.from_numpy(wavelet_3ch)  # CHW, [0,1]

class WaveletFaceDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None, wavelet='db1', limit_per_class=None):
        print(f"[Dataset] Initializing from:\n  REAL: {real_dir}\n  FAKE: {fake_dir}")
        if not os.path.isdir(real_dir):
            raise FileNotFoundError(f"REAL dizini bulunamadı: {real_dir}")
        if not os.path.isdir(fake_dir):
            raise FileNotFoundError(f"FAKE dizini bulunamadı: {fake_dir}")

        real_imgs = sorted(glob.glob(os.path.join(real_dir, '**', '*.*'), recursive=True))
        fake_imgs = sorted(glob.glob(os.path.join(fake_dir, '**', '*.*'), recursive=True))

        if limit_per_class is not None:
            real_imgs = real_imgs[:limit_per_class]
            fake_imgs = fake_imgs[:limit_per_class]

        self.image_paths = real_imgs + fake_imgs
        self.labels      = [0]*len(real_imgs) + [1]*len(fake_imgs)
        self.transform   = transform
        self.wavelet     = wavelet

        print(f"[Dataset] Loaded images: real={len(real_imgs)}, fake={len(fake_imgs)}, total={len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path  = self.image_paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert('L')  # grayscale
        except Exception as e:
            print(f"[Dataset] Error loading {path}: {e}")
            return None

        wv = wavelet_transform_grayscale(img, self.wavelet)  # CHW tensor
        if self.transform:
            wv = self.transform(wv)
        return wv, torch.tensor(label, dtype=torch.long)

def custom_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.dataloader.default_collate(batch) if batch else None

class HybridModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        print("[Model] Building HybridModel (EfficientNet-B1 + Transformer)...")
        self.cnn = timm.create_model('efficientnet_b1', pretrained=True,
                                     in_chans=3, num_classes=0, global_pool='')
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1280, nhead=8, dim_feedforward=2048,
            dropout=0.1, activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.pos_embed = nn.Parameter(torch.zeros(1, 49, 1280))
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        feat = self.cnn(x)                        # [B,1280,7,7]
        B,C,H,W = feat.shape
        feat = feat.view(B, C, H*W).permute(0,2,1) # [B,49,1280]
        feat = feat + self.pos_embed
        feat = feat.permute(1,0,2)                # [49,B,1280]
        out = self.transformer(feat)
        out = out.permute(1,0,2)                  # [B,49,1280]
        pooled = out.mean(dim=1)                  # [B,1280]
        return self.fc(pooled)

def train_one_epoch(model, loader, optimizer, criterion, device):
    print(">> Training epoch...")
    model.train()
    total_loss = correct = total = 0
    for batch in loader:
        if batch is None:
            continue
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds==labels).sum().item()
        total += imgs.size(0)
    return total_loss/total, correct/total

def evaluate(model, loader, criterion, device):
    print(">> Evaluating...")
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds==labels).sum().item()
            total += imgs.size(0)
    return total_loss/total, correct/total

if __name__ == "__main__":
    print("[Main] Starting script...")
    batch_size, lr, epochs = 32, 1e-5, 75
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")

    print("[Main] Preparing transforms (ImageNet Normalize)...")
    custom_transform = T.Compose([
        T.Resize((224,224)),
        T.RandomHorizontalFlip(0.5),
        T.RandomRotation(10),
        T.RandomAffine(degrees=0, translate=(0.1,0.1)),
        # ImageNet mean/std (EfficientNet ön-eğitimle uyumlu)
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])

    full_train = WaveletFaceDataset(
        real_dir=train_real,
        fake_dir=train_fake,
        transform=custom_transform,
        wavelet='db1',
        limit_per_class=None
    )
    test_ds = WaveletFaceDataset(
        real_dir=test_real,
        fake_dir=test_fake,
        transform=custom_transform,
        wavelet='db1',
        limit_per_class=None
    )

    print("[Main] Splitting train/validation...")
    n_train = int(0.9 * len(full_train))
    n_val   = len(full_train) - n_train
    train_ds, val_ds = random_split(full_train, [n_train, n_val])

    def counts(subset, labels):
        idxs = subset.indices if hasattr(subset, 'indices') else range(len(subset))
        real = sum(1 for i in idxs if labels[i] == 0)
        fake = sum(1 for i in idxs if labels[i] == 1)
        return real, fake

    r_tr, f_tr = counts(train_ds, full_train.labels)
    r_val, f_val = counts(val_ds, full_train.labels)
    r_test = sum(1 for l in test_ds.labels if l == 0)
    f_test = sum(1 for l in test_ds.labels if l == 1)
    print(f"  Train real/fake: {r_tr}/{f_tr}")
    print(f"  Val   real/fake: {r_val}/{f_val}")
    print(f"  Test  real/fake: {r_test}/{f_test}")

    print("[Main] Creating DataLoaders...")
    train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                              num_workers=2, collate_fn=custom_collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size, shuffle=False,
                              num_workers=2, collate_fn=custom_collate_fn, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size, shuffle=False,
                              num_workers=2, collate_fn=custom_collate_fn, pin_memory=True)

    print("[Main] Computing class weights...")
    total_tr = r_tr + f_tr
    w_real = total_tr / max(r_tr, 1)
    w_fake = total_tr / max(f_tr, 1)
    weights = torch.tensor([w_real, w_fake], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = HybridModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    patience   = 15
    min_delta  = 1e-4
    best_val   = float('inf')
    patience_c = 0
    best_state = None
    best_epoch = 0

    print("[Main] Beginning training loop with Early Stopping...")
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    for epoch in range(1, epochs+1):
        print(f"[Epoch {epoch}/{epochs}]")
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl, va = evaluate(model, val_loader,   criterion, device)
        train_losses.append(tl); train_accs.append(ta)
        val_losses.append(vl);   val_accs.append(va)
        print(f"  Train loss: {tl:.4f}, acc: {ta:.4f}")
        print(f"  Val   loss: {vl:.4f}, acc: {va:.4f}")

        if vl < best_val - min_delta:
            best_val   = vl
            best_epoch = epoch
            patience_c = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  >> New best val loss: {best_val:.6f} (epoch {best_epoch})")
        else:
            patience_c += 1
            print(f"  >> No improvement. Patience: {patience_c}/{patience}")
            if patience_c >= patience:
                print(f"[EarlyStopping] Stopping early at epoch {epoch}. Best epoch was {best_epoch} with val loss {best_val:.6f}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[Main] Loaded best model from epoch {best_epoch} (val loss {best_val:.6f}).")

    epochs_ran = len(train_losses)

    print("[Main] Plotting loss curves...")
    plt.figure()
    plt.plot(range(1, epochs_ran+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs_ran+1), val_losses,   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss Curve')
    plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("[Main] Plotting accuracy curves...")
    plt.figure()
    plt.plot(range(1, epochs_ran+1), train_accs, label='Train Acc')
    plt.plot(range(1, epochs_ran+1), val_accs,   label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy Curve')
    plt.savefig('acc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("[Main] Running final test evaluation...")
    all_preds, all_labels, all_probs = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            imgs, labels = batch
            imgs = imgs.to(device)
            logits = model(imgs)
            probs  = F.softmax(logits, dim=1)[:,1]
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs)

    print("[Main] Computing metrics...")
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec  = recall_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds)
    cm   = confusion_matrix(all_labels, all_preds)
    print("------ TEST METRICS ------")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("Confusion Matrix:\n", cm)

    print("[Main] Plotting ROC curve...")
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0,1], [0,1], '--', label='Random')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve'); plt.legend()
    plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("[Main] Script complete.")
