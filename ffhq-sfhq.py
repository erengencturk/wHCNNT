import os
import glob
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True

import pywt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as T

import timm

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

def wavelet_transform_grayscale(pil_img, wavelet='db1'):
    img_np = np.array(pil_img)
    cA, (cH, cV, cD) = pywt.dwt2(img_np, wavelet)

    def _norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    wavelet_3ch = np.stack([_norm(cA), _norm(cH), _norm(cV)], axis=0).astype(np.float32)
    return torch.from_numpy(wavelet_3ch)

class WaveletFaceDataset(Dataset):
    def __init__(self, transform=None, wavelet='db1'):
        print("[Dataset] Initializing dataset...")
        real_dir = '/content/ffhq-dataset/images1024x1024'
        fake_dir = '/content/data/sfhq/images/images'

        max_real = 70000
        max_fake = 122726

        self.real_images = sorted(glob.glob(os.path.join(real_dir, '**', '*.png'), recursive=True))[:max_real]
        self.fake_images = sorted(glob.glob(os.path.join(fake_dir, '**', '*.jpg'), recursive=True))[:max_fake]

        self.image_paths = self.real_images + self.fake_images
        self.labels = [0]*len(self.real_images) + [1]*len(self.fake_images)
        print(f"[Dataset] Loaded {len(self.real_images)} real and {len(self.fake_images)} fake images.")

        self.transform = transform
        self.wavelet = wavelet

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert('L')
        except Exception as e:
            print(f"[Dataset] Error loading {path}: {e}")
            return None

        wv = wavelet_transform_grayscale(img, self.wavelet)  # Tensor [3,H,W], float32
        if self.transform:
            wv = self.transform(wv)
        return wv, torch.tensor(label, dtype=torch.long)

def custom_collate_fn(batch):
    
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

class HybridModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        print("[Model] Building HybridModel (EfficientNet-B1 + Transformer)...")
        self.cnn = timm.create_model('efficientnet_b1', pretrained=True, in_chans=3, num_classes=0, global_pool='')
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1280, nhead=8, dim_feedforward=2048, dropout=0.1, activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.pos_embed = nn.Parameter(torch.zeros(1, 49, 1280))  # 7x7 token
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        feat = self.cnn(x)                        # [B,1280,7,7]
        B, C, H, W = feat.shape
        feat = feat.view(B, C, H*W).permute(0, 2, 1)  # [B,49,1280]
        feat = feat + self.pos_embed
        feat = feat.permute(1, 0, 2)                 # [49,B,1280]
        out = self.transformer(feat)
        out = out.permute(1, 0, 2)                   # [B,49,1280]
        pooled = out.mean(dim=1)                     # [B,1280]
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
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    avg_loss = total_loss/total if total > 0 else float('inf')
    avg_acc  = correct/total if total > 0 else 0.0
    return avg_loss, avg_acc

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
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    avg_loss = total_loss/total if total > 0 else float('inf')
    avg_acc  = correct/total if total > 0 else 0.0
    return avg_loss, avg_acc

def resolve_indices(obj):
    """
    Subset içinde Subset olsa bile, orijinal dataset indekslerini döndür.
    """
    if isinstance(obj, Subset):
        parent = resolve_indices(obj.dataset)
        return [parent[i] for i in obj.indices]
    else:
        return list(range(len(obj)))
    
if __name__ == "__main__":

    import torch

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True   # A100 matmul için TF32
    torch.backends.cudnn.allow_tf32 = True         # conv için TF32
    torch.set_float32_matmul_precision('high')     # PyTorch 2.x


    print("[Main] Starting script...")
    batch_size, lr, epochs = 32, 1e-5, 75
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")

    print("[Main] Preparing transforms...")
    custom_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])

    dataset = WaveletFaceDataset(transform=custom_transform, wavelet='db1')
    print(f"[Main] Full dataset size: {len(dataset)}")

    print("[Main] Splitting dataset (80% train+val, 20% test)...")
    n_train_val = int(0.8 * len(dataset))
    n_test = len(dataset) - n_train_val
    train_val_ds, test_ds = random_split(
        dataset, [n_train_val, n_test], generator=torch.Generator().manual_seed(42)
    )

    print("[Main] Splitting train+val into train (90%) and val (10%)...")
    n_val = int(0.1 * n_train_val)
    n_train = n_train_val - n_val
    train_ds, val_ds = random_split(
        train_val_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    print(f"[Main] Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    print("[Main] Computing class weights...")
    def counts(subset):
        idxs = resolve_indices(subset)
        real = sum(1 for i in idxs if dataset.labels[i] == 0)
        fake = sum(1 for i in idxs if dataset.labels[i] == 1)
        return real, fake

    r_tr, f_tr = counts(train_ds)
    total_tr = r_tr + f_tr
    eps = 1e-6
    weights = torch.tensor([total_tr/(r_tr+eps), total_tr/(f_tr+eps)], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    print(f"  Train real/fake: {r_tr}/{f_tr}")
    r_val, f_val = counts(val_ds)
    print(f"  Val   real/fake: {r_val}/{f_val}")
    r_test, f_test = counts(test_ds)
    print(f"  Test  real/fake: {r_test}/{f_test}")

    print("[Main] Creating DataLoaders...")
    train_loader = DataLoader(train_ds, batch_size, shuffle=True,  num_workers=2, collate_fn=custom_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)

    model = HybridModel(num_classes=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    print("[Main] Beginning training loop...")
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    patience = 15
    min_delta = 0.0001
    best_val_loss = float('inf')
    counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        print(f"[Epoch {epoch}/{epochs}]")
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl, va = evaluate(model, val_loader, criterion, device)

        train_losses.append(tl); train_accs.append(ta)
        val_losses.append(vl);   val_accs.append(va)

        print(f"  Train loss: {tl:.4f}, acc: {ta:.4f}")
        print(f"  Val   loss: {vl:.4f}, acc: {va:.4f}")

        # Early stopping
        if best_val_loss - vl > min_delta:
            best_val_loss = vl
            counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"[EarlyStopping] Improvement detected. best_val_loss={best_val_loss:.6f}")
        else:
            counter += 1
            print(f"[EarlyStopping] No improvement in {counter}/{patience} epochs.")
            if counter >= patience:
                print("[EarlyStopping] Patience reached. Stopping training.")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        torch.save(model.state_dict(), "best_model.pth")
        print("[Main] Best model restored and saved to best_model.pth")

    epochs_run = len(train_losses)
    x_axis = list(range(1, epochs_run + 1))

    print("[Main] Plotting loss curve...")
    plt.figure()
    plt.plot(x_axis, train_losses, label='Train Loss')
    plt.plot(x_axis, val_losses,   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss Curve')
    plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("[Main] Plotting accuracy curve...")
    plt.figure()
    plt.plot(x_axis, train_accs, label='Train Acc')
    plt.plot(x_axis, val_accs,   label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy Curve')
    plt.savefig('acc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("[Main] Evaluating on test set...")
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            imgs, labels = batch
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds); all_labels.append(labels); all_probs.append(probs.cpu())

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs  = torch.cat(all_probs).numpy()

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec  = recall_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds)
    cm   = confusion_matrix(all_labels, all_preds)
    print("------ TEST METRICS ------")
    print(f"Accuracy : {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print("Confusion Matrix:\n", cm)

    print("[Main] Plotting ROC curve...")
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], '--', label='Random')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("[Main] Script complete.")
