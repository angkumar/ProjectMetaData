import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import h5py

class H5CancerDataset(Dataset):
    def __init__(self, h5_path, csv_labels, transform=None, metadata_cols=None):
        """
        h5_path: .h5 file containing images under key 'x'
        csv_labels: CSV with labels + optional metadata
        """

        self.h5_path = h5_path
        self.transform = transform
        self.metadata_cols = metadata_cols

        self.labels_df = pd.read_csv(csv_labels)

        self.h5_file = h5py.File(self.h5_path, "r")
        self.images = self.h5_file["x"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        img = torch.tensor(img, dtype=torch.float32) / 255.0
        img = img.permute(2, 0, 1)

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(
            self.labels_df.iloc[idx]["label"],
            dtype=torch.long
        )

        if self.metadata_cols:
            meta = torch.tensor(
                self.labels_df.iloc[idx][self.metadata_cols].values,
                dtype=torch.float32
            )
            return img, meta, label

        return img, label


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])

class CancerCNN(nn.Module):
    def __init__(self, num_classes, metadata_features=0):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.3)

        self.flattened_size = 128 * 12 * 12

        if metadata_features > 0:
            self.meta_fc = nn.Sequential(
                nn.Linear(metadata_features, 32),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            combined_size = self.flattened_size + 32
        else:
            combined_size = self.flattened_size

        self.fc = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self.metadata_features = metadata_features

    def forward(self, x, meta=None):

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        if self.metadata_features and meta is not None:
            meta_features = self.meta_fc(meta)
            x = torch.cat([x, meta_features], dim=1)

        out = self.fc(x)
        return out

def train_model(model, train_loader, val_loader=None, num_epochs=50, lr=1e-3, device="mps"):

    device = torch.device(device if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0

        for batch in train_loader:

            if len(batch) == 3:
                images, meta, labels = batch
                images, meta, labels = images.to(device), meta.to(device), labels.to(device)
                outputs = model(images, meta)
            else:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

        if val_loader:
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 3:
                        images, meta, labels = batch
                        images, meta, labels = images.to(device), meta.to(device), labels.to(device)
                        outputs = model(images, meta)
                    else:
                        images, labels = batch
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_acc = 100 * correct / total
            print(f"Validation Accuracy: {val_acc:.2f}%")

train_dataset = H5CancerDataset(
    h5_path="train.h5",
    csv_labels="train_labels.csv",
    transform=transform,
    metadata_cols=["location"]
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

model = CancerCNN(
    num_classes=5,
    metadata_features=1
)

train_model(
    model,
    train_loader,
    val_loader=None,
    num_epochs=50,
    lr=1e-3,
    device="mps"
)
