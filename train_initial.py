import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from sklearn.model_selection import KFold
import random
from rwkv_unet import RWKV_UNet


# 数据集类
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

# 半监督损失函数
class SupervisedLoss(nn.Module):
    def __init__(self, supervised_loss_weight=1.0, smooth=1e-5):
        super(SupervisedLoss, self).__init__()
        self.supervised_loss_weight = supervised_loss_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.cross_entropy_loss = nn.CrossEntropyLoss() 
    def forward(self, outputs, masks):
        total_loss = self.dice_loss(outputs, masks) +self.cross_entropy_loss(outputs, masks.long())
        return total_loss


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
kf = KFold(n_splits=10, shuffle=True, random_state=42)
# 模型、优化器和学习率调度器
model = RWKV_UNet().cuda()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs_phase)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# 保存最佳模型路径
best_mean_dice = 0


# 随机增强方法
def get_random_augmentation(images):
    augmentations = []
    
    # 随机选择增强操作
    if random.random() > 0.5:
        # 水平翻转
        images = torch.flip(images, dims=[3])  # 水平翻转
        augmentations.append("flip_horizontal")
    
    if random.random() > 0.5:
        # 垂直翻转
        images = torch.flip(images, dims=[2])  # 垂直翻转
        augmentations.append("flip_vertical") 
    if random.random() > 0.5:
        images = torch.rot90(images, k=1, dims=[2, 3])  # 逆时针旋转90度
        augmentations.append("rotate1")
    if random.random() > 0.5:
        images = torch.rot90(images, k=3, dims=[2, 3])  # 逆时针旋转90度
        augmentations.append("rotate2")
    if random.random() > 0.5:
        images = torch.rot90(images, k=2, dims=[2, 3])  # 逆时针旋转90度
        augmentations.append("rotate3")
    return images, augmentations

# 逆变换方法
def inverse_augmentation(augmented_images, augmentations):
    for augment_type in augmentations:
        if augment_type == "flip_horizontal":
            augmented_images = torch.flip(augmented_images, dims=[3])  # 恢复水平翻转
        elif augment_type == "flip_vertical":
            augmented_images = torch.flip(augmented_images, dims=[2])  # 恢复垂直翻转
        elif augment_type == "rotate1":
            augmented_images = torch.rot90(augmented_images, k=3, dims=[2, 3])
        elif augment_type == "rotate2":
            augmented_images = torch.rot90(augmented_images, k=1, dims=[2, 3]) 
        elif augment_type == "rotate3":
            augmented_images = torch.rot90(augmented_images, k=2, dims=[2, 3]) 
    return augmented_images

supervised_loss = SupervisedLoss().cuda()
unsupervised_loss = SupervisedLoss().cuda()

for fold, (train_index, val_index) in enumerate(kf.split(train_supervised_dataset)):
    if fold > 0:
        break
    train_files = [
        os.path.join(train_supervised_dataset.image_dir, train_supervised_dataset.images[i]) 
        for i in train_index
    ]
    val_files = [
        os.path.join(train_supervised_dataset.image_dir, train_supervised_dataset.images[i]) 
        for i in val_index
    ]

    train_subset = torch.utils.data.Subset(train_supervised_dataset, train_index)
    val_subset = torch.utils.data.Subset(train_supervised_dataset, val_index)

    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, collate_fn=custom_collate)
    train_unsupervised_loader = DataLoader(train_unsupervised_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate)
    print(f"Starting Fold {fold + 1}")
    fold_best_mean_dice = 0
    fold_model_path = f"best_fold{fold + 1}.pth" 
    fold_last_model_path = f"last_fold{fold + 1}.pth"
    for epoch in range(num_epochs_phase):
        model.train()
        total_loss = 0
        for images, masks in train_loader:
            images = images.to(device)
            if masks is not None:
                masks = masks.to(device)
            if masks is not None:
                outputs = model(images)
                loss = supervised_loss(outputs, masks)
            else:
                loss = 0 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # 处理无标签数据
        model.eval()
        unsupervised_loss = 0
        for images, _ in train_unsupervised_loader:
            images = images.to(device)

            with torch.no_grad():
                augmented_images1, augmentations1 = get_random_augmentation(images)  # 第一次增强
                augmented_images2, augmentations2 = get_random_augmentation(images)
                outputs_1 = model(augmented_images1)
                outputs_1_restored = inverse_augmentation(outputs_1, augmentations1)
                outputs_2= model(augmented_images2)
                outputs_2_restored = inverse_augmentation(outputs_2, augmentations2)
                consistency_loss =  torch.mean((outputs_1_restored - outputs_2_restored) ** 2)
            unsupervised_loss += consistency_loss.item()

        # 将有标签和无标签数据的损失加权合并
        total_loss = total_loss + unsupervised_loss

        print(f"Epoch [{epoch + 1}/{num_epochs_phase}], Loss: {total_loss / (len(train_loader)+len(train_unsupervised_loader)):.4f}")

        # 验证步骤
        model.eval()
        total_val_loss = 0
        total_mean_dice = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                if masks is not None:
                    masks = masks.to(device)

                output1,output2,output3,output4= model(images)
                outputs=output1+output2+output3+output4
                val_loss = supervised_loss(outputs, masks)  # 计算验证集上的损失
                total_val_loss += val_loss.item()

                mean_dice = calculate_mean_dice(outputs, masks)  # 计算 Mean Dice
                total_mean_dice += mean_dice.item()
                print(f"Validation Loss: {total_val_loss / len(val_loader):.4f}, Mean Dice: {total_mean_dice / len(val_loader):.4f}")
                # 保存验证集上表现最好的模型
                if total_mean_dice / len(val_loader) > fold_best_mean_dice:
                    fold_best_mean_dice = total_mean_dice / len(val_loader)
                    torch.save(model.state_dict(), fold_model_path)
                    print(f"Best model saved for fold {fold + 1} at epoch {epoch + 1}")
                    torch.save(model.state_dict(), fold_last_model_path)



       
