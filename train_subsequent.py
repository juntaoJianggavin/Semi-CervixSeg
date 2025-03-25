import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from torchvision import models
from sklearn.model_selection import KFold
import random
from rwkv_unet import RWKV_UNet
from pvt import EMCADNet

class SemiSupervisedImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None, supervised=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.supervised = supervised
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.supervised: 
            mask_path = os.path.join(self.mask_dir, self.images[index])
            mask = cv2.imread(mask_path, 0)
        else: 
            mask = None

        if self.transform is not None:
            if mask is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]
            else:
                augmentations = self.transform(image=image)
                image = augmentations["image"]

        return image, mask

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        batch_size, height, width = targets.size()
        targets_flat = targets.view(-1).long()  # 确保为 long 类型
        targets_one_hot = torch.eye(num_classes)[targets_flat].to(inputs.device)
        targets_one_hot = targets_one_hot.view(batch_size, height, width, num_classes).permute(0, 3, 1, 2)
        inputs = torch.softmax(inputs, dim=1)
        intersection = torch.sum(inputs * targets_one_hot, dim=(2, 3))
        cardinality = torch.sum(inputs + targets_one_hot, dim=(2, 3))
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1 - dice_score.mean()
        return dice_loss

def calculate_mean_dice(outputs, targets):
    num_classes = outputs.size(1)
    batch_size, height, width = targets.size()
    targets_flat = targets.view(-1).long()
    targets_one_hot = torch.eye(num_classes)[targets_flat].to(outputs.device)
    targets_one_hot = targets_one_hot.view(batch_size, height, width, num_classes).permute(0, 3, 1, 2)
    outputs = torch.softmax(outputs, dim=1)
    intersection = torch.sum(outputs * targets_one_hot, dim=(2, 3))
    cardinality = torch.sum(outputs + targets_one_hot, dim=(2, 3))
    dice_score = (2. * intersection + 1e-5) / (cardinality + 1e-5)
    dice_score_without_background = dice_score[:, 1:]
    mean_dice = dice_score_without_background.mean()
    return mean_dice

class SupervisedLoss(nn.Module):
    def __init__(self):
        super(SupervisedLoss, self).__init__()
        self.dice_loss = DiceLoss(smooth=1e-5)
        self.cross_entropy_loss = nn.CrossEntropyLoss() 
    def forward(self, outputs, masks):
        supervised_loss = self.dice_loss(outputs, masks) +self.cross_entropy_loss(outputs, masks.long())
        return supervised_loss
supervised_loss=SupervisedLoss()
train_transform = A.Compose(
    [
        A.Resize(height=384, width=384),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
        A.Resize(height=384, width=384),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

# 数据加载器
train_supervised_dataset = SemiSupervisedImageDataset(
    image_dir="train/labeled_data/images",
    mask_dir="train/labeled_data/labels",
    transform=train_transform,
    supervised=True
)
train_unsupervised_dataset = SemiSupervisedImageDataset(
    image_dir="train/unlabeled_data/images",
    transform=train_transform,
    supervised=False
)

# 自定义数据集合并器
def custom_collate(batch):
    images = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    valid_masks = [mask for mask in masks if mask is not None]
    if valid_masks:
        masks = torch.stack(valid_masks)
    else:
        masks = None
    images = torch.stack(images)
    return images, masks
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(42)
num_epochs_phase = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_model = EMCADNet().cuda()
trained_model.load_state_dict(torch.load("emcad.pth"))  # 加载已经训练好的权重

model= EMCADNet().cuda()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs_phase)

def generate_pseudo_labels(model,unsupervised_loader, device):
    pseudo_labels = []
    images_list = []

    model.eval()
    with torch.no_grad():
        for images, _ in unsupervised_loader:
            images = images.to(device)
            outputs1,outputs2,outputs3,outputs4 = model(images)
            outputs=outputs1+outputs2+outputs3+outputs4
            pseudo_label = torch.argmax(outputs, dim=1)  # 获取最大概率的类别作为伪标签
        
            pseudo_labels.append(pseudo_label.cpu())
            images_list.append(images.cpu())

    # 合并结果
    pseudo_labels = torch.cat(pseudo_labels, dim=0)
    images_list = torch.cat(images_list, dim=0)
    
    return images_list, pseudo_labels
train_loader = DataLoader(train_supervised_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate)
train_unsupervised_loader = DataLoader(train_unsupervised_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate)
# 获取伪标签
unsupervised_images, pseudo_labels = generate_pseudo_labels(trained_model, train_unsupervised_loader, device)

def merge_datasets(supervised_loader, unsupervised_images, pseudo_labels):
    merged_images = []
    merged_labels = []
    for images, masks in supervised_loader:
        merged_images.append(images)
        merged_labels.append(masks)
    merged_images = torch.cat(merged_images, dim=0)
    merged_labels = torch.cat(merged_labels, dim=0)

    merged_images = torch.cat([merged_images, unsupervised_images], dim=0)
    merged_labels = torch.cat([merged_labels, pseudo_labels], dim=0)

    return merged_images, merged_labels


num_epochs_phase = 300
kf = KFold(n_splits=10, shuffle=True, random_state=42)
for fold, (train_index, val_index) in enumerate(kf.split(train_supervised_dataset)):
    if fold > 0:
        break
    train_subset = torch.utils.data.Subset(train_supervised_dataset, train_index)
    val_subset = torch.utils.data.Subset(train_supervised_dataset, val_index)
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, collate_fn=custom_collate)
    merged_images, merged_labels = merge_datasets(train_loader, unsupervised_images, pseudo_labels)
    merged_dataset = torch.utils.data.TensorDataset(merged_images, merged_labels)
    merged_loader = DataLoader(merged_dataset, batch_size=16, shuffle=True)
    print(f"Starting Fold {fold + 1}")
    fold_best_mean_dice = 0
    fold_model_path = f"phase2_best_model_fold{fold + 1}.pth" 
    fold_last_model_path = f"phase2_last_model_fold{fold + 1}.pth"
    for epoch in range(num_epochs_phase):
        model.train()
        total_loss = 0
        for images, masks in merged_loader:
            images = images.to(device)
            if masks is not None:
                masks = masks.to(device)
            if masks is not None:
                
                outputs1,outputs2,outputs3,outputs4 = model(images)
                loss = supervised_loss(outputs1, masks) + supervised_loss(outputs2, masks)+ supervised_loss(outputs3, masks)+ supervised_loss(outputs4, masks)
            else:
                loss = 0 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch [{epoch + 1}/{num_epochs_phase}], Loss: {total_loss / len(train_loader):.4f}")
        model.eval()
        total_val_loss = 0
        total_mean_dice = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                if masks is not None:
                    masks = masks.to(device)

                outputs1,outputs2,outputs3,outputs4  = model(images)
                outputs =outputs1+outputs2+outputs3+outputs4
                #outputs  = model(images)
                val_loss = supervised_loss(outputs, masks)  # 计算验证集上的损失
                total_val_loss += val_loss.item()
                mean_dice = calculate_mean_dice(outputs, masks)  # 计算 Mean Dice
                total_mean_dice += mean_dice.item()

        print(f"Validation Loss: {total_val_loss / len(val_loader):.4f}, Mean Dice: {total_mean_dice / len(val_loader):.4f}")
        if total_mean_dice / len(val_loader) > fold_best_mean_dice:
            fold_best_mean_dice = total_mean_dice / len(val_loader)
            torch.save(model.state_dict(), fold_model_path)
            print(f"Best model saved for fold {fold + 1} at epoch {epoch + 1}")
        torch.save(model.state_dict(), fold_last_model_path)



       
