import os
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ── 允许加载损坏的图像 ─────────────────────────────────────────────────────────
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Efficient Channel Attention ──────────────────────────────────────────────
class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv     = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size//2, bias=False)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        # x: [N, C, H, W]
        y = self.avg_pool(x)                                # [N, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)                 # [N, 1, C]
        y = self.conv(y)                                    # [N, 1, C]
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # [N, C, 1, 1]
        return x * y.expand_as(x)                           # [N, C, H, W]

# ── MCP-X 模块 ─────────────────────────────────────────────────────────────
class MCPX(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, num_paths):
        super().__init__()
        # 多路分支：num_paths 个 3×3 Conv→BN→ReLU
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, hid_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(hid_ch),
                nn.ReLU(inplace=True)
            ) for _ in range(num_paths)
        ])
        # Expert Routing：先做全局平均池化，再用 1×1 Conv 得到 num_paths 个权重，接 Softmax
        self.attn_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, num_paths, kernel_size=1),
            nn.Softmax(dim=1)
        )
        # 将所有分支拼接后用 1×1 Conv→BN→ReLU 还原到 out_ch
        self.merge_conv = nn.Sequential(
            nn.Conv2d(hid_ch * num_paths, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        # 通道注意力
        self.eca      = ECABlock(out_ch)
        # 残差快捷：将 in_ch→out_ch
        self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        # x: [N, in_ch, H, W]
        w = self.attn_weights(x)  # [N, num_paths, 1, 1]
        outs = []
        for i, p in enumerate(self.paths):
            o  = p(x)                    # [N, hid_ch, H, W]
            wi = w[:, i:i+1, :, :].expand_as(o)
            outs.append(o * wi)          # 带权重的分支输出
        cat    = torch.cat(outs, dim=1)  # [N, hid_ch * num_paths, H, W]
        merged = self.merge_conv(cat)    # [N, out_ch, H, W]
        attn   = self.eca(merged)        # [N, out_ch, H, W]
        return attn + self.residual(x)   # 残差相加

# ── 去掉 ECA 的 MCP-X（仅做多分支+路由+残差） ─────────────────────────────
class MCPX_NoECA(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, num_paths):
        super().__init__()
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, hid_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(hid_ch),
                nn.ReLU(inplace=True)
            ) for _ in range(num_paths)
        ])
        self.attn_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, num_paths, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.merge_conv = nn.Sequential(
            nn.Conv2d(hid_ch * num_paths, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        w = self.attn_weights(x)
        outs = []
        for i, p in enumerate(self.paths):
            o = p(x)
            wi = w[:, i:i+1, :, :].expand_as(o)
            outs.append(o * wi)
        cat    = torch.cat(outs, dim=1)
        merged = self.merge_conv(cat)
        # 直接残差相加，不做 ECA
        return merged + self.residual(x)

# ── MetaCortexNet（完整版） ─────────────────────────────────────────────────
class MetaCortexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Stage 1: 浅层编码
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        # Stage 2: MCP-X 模块
        self.mcpx  = MCPX(in_ch=64, hid_ch=16, out_ch=64, num_paths=4)

        # Stage 3: BRSE 解码
        self.enc1  = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(2, 2)
        self.enc2  = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.deconv= nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.relu5 = nn.ReLU(inplace=True)
        self.outc  = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)

        # Stage 4: Meta-Causal Attention
        self.attn_pool = nn.AdaptiveAvgPool2d((32, 32))
        self.attn      = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

        # Stage 5: 分类头
        self.gap       = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_final  = nn.Linear(64, num_classes)

    def forward(self, x):
        N = x.size(0)
        # Stage 1
        x = self.relu1(self.conv1(x))  # [N, 32, 224, 224]
        x = self.relu2(self.conv2(x))  # [N, 64, 112, 112]
        # Stage 2
        x = self.mcpx(x)               # [N, 64, 112, 112]
        # Stage 3 (BRSE)
        x = self.relu3(self.enc1(x))   # [N, 64, 112, 112]
        x = self.pool(x)               # [N, 64, 56, 56]
        x = self.relu4(self.enc2(x))   # [N, 128, 56, 56]
        x = self.relu5(self.deconv(x)) # [N, 64, 112, 112]
        x = self.relu6(self.outc(x))   # [N, 64, 112, 112]
        # Stage 4 (Attention)
        x = self.attn_pool(x)          # [N, 64, 32, 32]
        C, H, W = x.shape[1:]
        x = x.view(N, C, H*W).permute(0, 2, 1)  # [N, H*W, C]
        x, _ = self.attn(x, x, x)      # [N, H*W, C]
        x = x.permute(0, 2, 1).view(N, C, H, W) # [N, 64, 32, 32]
        # Stage 5 (Classification)
        x = self.gap(x).view(N, -1)    # [N, 64]
        return self.fc_final(x)        # [N, num_classes]

# ── 基线模型：去掉 MCP-X，只保留浅层 + BRSE + Attention ─────────────────────
class BaselineNet(nn.Module):
    """
    Baseline：跳过 MCP-X 模块
    """
    def __init__(self, num_classes):
        super().__init__()
        # Stage 1: 浅层编码
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        # 跳过 Stage 2 (MCP-X)

        # Stage 3: BRSE
        self.enc1  = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(2, 2)
        self.enc2  = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.deconv = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.relu5  = nn.ReLU(inplace=True)
        self.outc   = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu6  = nn.ReLU(inplace=True)

        # Stage 4: Attention
        self.attn_pool = nn.AdaptiveAvgPool2d((32, 32))
        self.attn      = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

        # Stage 5: 分类头
        self.gap      = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_final = nn.Linear(64, num_classes)

    def forward(self, x):
        N = x.size(0)
        x = self.relu1(self.conv1(x))  # [N, 32, 224, 224]
        x = self.relu2(self.conv2(x))  # [N, 64, 112, 112]
        # Stage 3 (BRSE)
        x = self.relu3(self.enc1(x))   # [N, 64, 112, 112]
        x = self.pool(x)               # [N, 64, 56, 56]
        x = self.relu4(self.enc2(x))   # [N, 128, 56, 56]
        x = self.relu5(self.deconv(x)) # [N, 64, 112, 112]
        x = self.relu6(self.outc(x))   # [N, 64, 112, 112]
        # Stage 4 (Attention)
        x = self.attn_pool(x)          # [N, 64, 32, 32]
        C, H, W = x.shape[1:]
        x = x.view(N, C, H*W).permute(0, 2, 1)  # [N, H*W, C]
        x, _ = self.attn(x, x, x)      # [N, H*W, C]
        x = x.permute(0, 2, 1).view(N, C, H, W) # [N, 64, 32, 32]
        # Stage 5 (Classification)
        x = self.gap(x).view(N, -1)    # [N, 64]
        return self.fc_final(x)        # [N, num_classes]

# ── 去掉 Attention 的 MetaCortexNet ────────────────────────────────────────
class MetaCortexNet_NoAttn(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        # MCP-X（可以替换成 MCPX_NoECA 或 MCPX，根据需要）
        self.mcpx  = MCPX(in_ch=64, hid_ch=16, out_ch=64, num_paths=4)

        self.enc1  = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(2,2)
        self.enc2  = nn.Conv2d(64,128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.deconv= nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.relu5 = nn.ReLU(inplace=True)
        self.outc  = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)

        # 去掉了 MultiheadAttention
        # self.attn_pool = nn.AdaptiveAvgPool2d((32,32))
        # self.attn      = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

        self.gap      = nn.AdaptiveAvgPool2d((1,1))
        self.fc_final = nn.Linear(64, num_classes)

    def forward(self, x):
        N = x.size(0)
        x = self.relu1(self.conv1(x))  # [N, 32, 224,224]
        x = self.relu2(self.conv2(x))  # [N, 64, 112,112]
        x = self.mcpx(x)               # [N, 64, 112,112]
        x = self.relu3(self.enc1(x))   # [N, 64, 112,112]
        x = self.pool(x)               # [N, 64, 56, 56]
        x = self.relu4(self.enc2(x))   # [N,128, 56, 56]
        x = self.relu5(self.deconv(x)) # [N, 64,112,112]
        x = self.relu6(self.outc(x))   # [N, 64,112,112]
        # Stage4: 不做 Attention
        x = self.gap(x).view(N, -1)    # [N,64]
        return self.fc_final(x)        # [N, num_classes]

# ── 数据加载：支持可配置的 train/val/test 划分比例 ───────────────────────────
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def get_loaders(root, batch_size=16, random_state=42, split_ratio=(0.8, 0.1, 0.1)):
    """
    返回 train/val/test DataLoader，划分比例由 split_ratio 决定。
    split_ratio: (train_frac, val_frac, test_frac)，总和应为 1.0。
    """
    # 1) 载入整个 ImageFolder，不指定 transform
    base = ImageFolder(root, transform=None)

    # 2) 过滤掉损坏的图像
    good = []
    for p, lbl in base.samples:
        try:
            Image.open(p).verify()
            good.append((p, lbl))
        except Exception:
            print(f"⚠️ Skipping corrupted image: {p}")
    base.samples = good
    base.imgs = good

    total = len(good)
    labels = [lbl for _, lbl in good]
    idxs = list(range(total))

    train_frac, val_frac, test_frac = split_ratio
    tmp_frac = val_frac + test_frac

    # 3) 先划分 train vs tmp
    tr_idx, tmp_idx = train_test_split(
        idxs, test_size=tmp_frac, stratify=labels, random_state=random_state
    )

    # 4) 在 tmp_idx 中再划分 val vs test
    tmp_labels = [labels[i] for i in tmp_idx]
    val_ratio_over_tmp = val_frac / tmp_frac
    val_idx, tst_idx = train_test_split(
        tmp_idx, test_size=(test_frac / tmp_frac),
        stratify=tmp_labels, random_state=random_state
    )

    def make_loader(idxs, tfm, shuffle):
        ds = copy.deepcopy(base)
        ds.samples = [base.samples[i] for i in idxs]
        ds.imgs    = ds.samples
        ds.transform = tfm
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(tr_idx, train_transforms, True)
    val_loader   = make_loader(val_idx,   test_transforms,  False)
    test_loader  = make_loader(tst_idx,   test_transforms,  False)
    return train_loader, val_loader, test_loader

# ── 训练与评估函数 ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    cnt = 0
    true_labels = []
    pred_labels = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)               # [batch_size, num_classes]
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        cnt += x.size(0)

        true_labels.extend(y.cpu().tolist())
        pred_labels.extend(preds.cpu().tolist())

    avg_loss = total_loss / cnt
    accuracy = correct / cnt
    precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    recall    = recall_score   (true_labels, pred_labels, average='macro', zero_division=0)
    f1        = f1_score       (true_labels, pred_labels, average='macro', zero_division=0)
    return avg_loss, accuracy, precision, recall, f1

def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    cnt = 0
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            cnt += x.size(0)

            true_labels.extend(y.cpu().tolist())
            pred_labels.extend(preds.cpu().tolist())

    avg_loss = total_loss / cnt
    accuracy = correct / cnt
    precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    recall    = recall_score   (true_labels, pred_labels, average='macro', zero_division=0)
    f1        = f1_score       (true_labels, pred_labels, average='macro', zero_division=0)
    return avg_loss, accuracy, precision, recall, f1

# ── 计算模型参数量 ─────────────────────────────────────────────────────────────
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

# ── 主函数 ─────────────────────────────────────────────────────────────────────
def main():
    # ① 用户可在此修改：选择运行哪一种模型版本
    #    可选值："fulmcl"、"no_mcp"、"no_eca"、"no_attn"
    MODEL_TYPE = "no_attn"  # "full" for MetaCortexNet, "no_mcp" for BaselineNet, "no_eca" for MCPX_NoECA, "no_attn" for MetaCortexNet_NoAtt
    # ② 可在此修改：是否对比不同数据划分比例。若只想跑单一划分，直接注释 for 循环部分即可。
    split_list = [(0.8, 0.1, 0.1)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    dataset_root = r"C:\Users\wku\Documents\GitHub\riceleaftestdataset\archive4\plantvillage dataset\color" # ← 将此处改为你的 PlantVillage 数据集根目录

    results = []  # 用于记录不同划分下的最终表现

    for split_ratio in split_list:
        print(f"\n==================== Split: {split_ratio} ====================")
        # 加载数据
        train_loader, val_loader, test_loader = get_loaders(
            dataset_root, batch_size=16, random_state=42, split_ratio=split_ratio
        )
        num_classes = len(train_loader.dataset.classes)
        print(f"📦 类别数: {num_classes}")
        print(f"🚆 train batches: {len(train_loader)}, val batches: {len(val_loader)}, test batches: {len(test_loader)}")

        # 根据 MODEL_TYPE 实例化模型
        if MODEL_TYPE == "full":
            model = MetaCortexNet(num_classes=num_classes)
        elif MODEL_TYPE == "no_mcp":
            model = BaselineNet(num_classes=num_classes)
        elif MODEL_TYPE == "no_eca":
            # 创建一个临时子类，将 MCPX 替换为 MCPX_NoECA
            class MetaCortexNet_NoECA(MetaCortexNet):
                def __init__(self, num_classes):
                    super().__init__(num_classes)
                    self.mcpx = MCPX_NoECA(in_ch=64, hid_ch=16, out_ch=64, num_paths=4)
            model = MetaCortexNet_NoECA(num_classes=num_classes)
        elif MODEL_TYPE == "no_attn":
            model = MetaCortexNet_NoAttn(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

        model = model.to(device)

        # 输出参数量
        total_params, trainable_params = count_parameters(model)
        print(f"📝 Model params: total={total_params:,}, trainable={trainable_params:,}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        best_val_f1 = 0.0
        best_state = None

        # 训练循环
        for epoch in range(1, 101):  # 最多 50 轮
            t_loss, t_acc, t_prec, t_rec, t_f1 = train_epoch(model, train_loader, device, optimizer, criterion)
            v_loss, v_acc, v_prec, v_rec, v_f1 = evaluate(model, val_loader, device, criterion)

            print(f"[Split {split_ratio}, Epoch {epoch:02d}] "
                  f"Train → Loss: {t_loss:.4f}, Acc: {t_acc:.2%}, Prec: {t_prec:.2%}, Rec: {t_rec:.2%}, F1: {t_f1:.2%} | "
                  f"Val   → Loss: {v_loss:.4f}, Acc: {v_acc:.2%}, Prec: {v_prec:.2%}, Rec: {v_rec:.2%}, F1: {v_f1:.2%}")

            # 早停：如果验证 F1 更好，则保存当前参数
            if v_f1 > best_val_f1:
                best_val_f1 = v_f1
                best_state = copy.deepcopy(model.state_dict())
                # 保存模型，文件名可自定义
                model_save_name = f"best_model_no_attn_{int(split_ratio[0]*100)}_{int(split_ratio[1]*100)}_{int(split_ratio[2]*100)}_{MODEL_TYPE}_{epoch}.pth"
                torch.save(best_state, model_save_name)
                print(f"✅ New best model saved on validation F1: {model_save_name}")

        # 加载最佳参数后在测试集上评估
        model.load_state_dict(best_state)
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, device, criterion)
        print(f"\n🎯 Final Test → Loss: {test_loss:.4f}, Acc: {test_acc:.2%}, Prec: {test_prec:.2%}, Rec: {test_rec:.2%}, F1: {test_f1:.2%}\n")

        # 生成并保存混淆矩阵图像
        all_true, all_pred = [], []
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                preds = out.argmax(dim=1)
                all_true.extend(y.cpu().tolist())
                all_pred.extend(preds.cpu().tolist())
        cm = confusion_matrix(all_true, all_pred)
        # 打印混淆矩阵数值（完整显示，不省略）
        np.set_printoptions(threshold=np.inf, linewidth=200)
        print("混淆矩阵数值:")
        print(cm.tolist())
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix Split {split_ratio}')
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, train_loader.dataset.classes, rotation=45)
        plt.yticks(tick_marks, train_loader.dataset.classes)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        cm_filename = f"confusion_matrix_{int(split_ratio[0]*100)}_{int(split_ratio[1]*100)}_{int(split_ratio[2]*100)}.png"
        plt.savefig(cm_filename)
        print(f"🎨 Confusion matrix saved to {cm_filename}\n")
        
        results.append({
            "split": split_ratio,
            "test_acc": test_acc,
            "test_prec": test_prec,
            "test_rec": test_rec,
            "test_f1": test_f1,
            "params": total_params
        })

    # 输出所有结果汇总
    print("\n==================== Summary ====================")
    for res in results:
        print(f"Split {res['split']} | "
              f"Test Acc={res['test_acc']:.2%} | "
              f"Test Prec={res['test_prec']:.2%} | "
              f"Test Rec={res['test_rec']:.2%} | "
              f"Test F1={res['test_f1']:.2%} | "
              f"Params={res['params']:,}")

if __name__ == "__main__":
    main()
