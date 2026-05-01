import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
from pathlib import Path

# 1. إعداد المسارات (تعديل المسار ليكون ديناميكي فقط)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_BASE_PATH = BASE_DIR / "Data" / "real-vs-fake" 
SAVE_DIR = BASE_DIR / "Data" / "Face_Models"
os.makedirs(SAVE_DIR, exist_ok=True)

# 2. التحويلات (نفس معايير الدقة 97%)
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. تحميل الأقسام الثلاثة (Train, Valid, Test)
print("⏳ Loading local datasets...")
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(DATA_BASE_PATH, 'train'), data_transforms),
    'valid': datasets.ImageFolder(os.path.join(DATA_BASE_PATH, 'valid'), data_transforms),
    'test': datasets.ImageFolder(os.path.join(DATA_BASE_PATH, 'test'), data_transforms)
}

# إنشاء الـ Dataloaders
dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=(x == 'train'))
    for x in ['train', 'valid', 'test']
}

print(f"✅ Loaded: {len(image_datasets['train'])} Train | {len(image_datasets['valid'])} Valid")

# 4. بناء الموديل (ResNet18)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 5. إعدادات التدريب
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"🚀 Training starting on {device}...")

# حلقة التدريب (مثال بسيط لـ 1 Epoch - زود الـ Epochs للإنتاج)
for epoch in range(1):
    # مرحلة التدريب
    model.train()
    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # مرحلة التحقق (Validation)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['valid']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Epoch {epoch + 1}: Validation Accuracy: {100 * correct / total}%')

# 6. حفظ الموديل النهائي في فولدر الـ data
save_path = SAVE_DIR / "face_resnet_97.pth"
torch.save(model.state_dict(), save_path)
print(f"✅ Final Model Saved at: {save_path}")
