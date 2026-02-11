import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import h5py

class H5CancerDataset(Dataset):
    def __init__(
        self,
        images_h5_path: str,
        labels_h5_path: str,
        transform=None,
        image_key: str = "x",
        label_key: str = "y",
    ):
        """
        images_h5_path: .h5 file containing images under key `image_key` (default: 'x')
        labels_h5_path: .h5 file containing labels under key `label_key` (default: 'y')
        """

        self.transform = transform
        self.images_h5_path = images_h5_path
        self.labels_h5_path = labels_h5_path
        self.image_key = image_key
        self.label_key = label_key

        # Lazy-open per process/worker (safer with DataLoader workers).
        self._images_h5 = None
        self._labels_h5 = None
        self.images = None
        self.labels = None

        # Cache length + validate alignment.
        with h5py.File(self.images_h5_path, "r") as f_img:
            self._length = int(f_img[self.image_key].shape[0])
        with h5py.File(self.labels_h5_path, "r") as f_lbl:
            lbl_len = int(f_lbl[self.label_key].shape[0])
        if lbl_len != self._length:
            raise ValueError(
                f"Image/label length mismatch: images={self._length}, labels={lbl_len}"
            )

    def _ensure_open(self):
        if self._images_h5 is None:
            self._images_h5 = h5py.File(self.images_h5_path, "r")
            self.images = self._images_h5[self.image_key]
        if self._labels_h5 is None:
            self._labels_h5 = h5py.File(self.labels_h5_path, "r")
            self.labels = self._labels_h5[self.label_key]

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        self._ensure_open()
        # (96,96,3) uint8
        img = self.images[idx]

        # Convert to float tensor + normalize
        img = torch.tensor(img, dtype=torch.float32) / 255.0
        img = img.permute(2, 0, 1)  # HWC -> CHW

        if self.transform:
            img = self.transform(img)

        # Labels are stored as (1,1,1) uint8; convert to scalar int64 class id.
        label_value = int(self.labels[idx].squeeze())
        label = torch.tensor(label_value, dtype=torch.long)

        return img, label

    def close(self):
        # Defensive close: during interpreter shutdown, h5py may already be partially torn down.
        for attr in ("_images_h5", "_labels_h5"):
            f = getattr(self, attr, None)
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
            setattr(self, attr, None)

    def __del__(self):
        self.close()


# -------------------------
# 2. Lightweight Augmentation
# -------------------------
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])


# -------------------------
# 3. CNN Model (FROM SCRATCH)
# Input size = 96x96
# After 3 pools -> 12x12 feature map
# -------------------------
class CancerCNN(nn.Module):
    def __init__(self, num_classes, metadata_features=0):
        super().__init__()

        # Image branch
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.3)

        # 96 -> 48 -> 24 -> 12
        self.flattened_size = 128 * 12 * 12

        # Metadata branch
        if metadata_features > 0:
            self.meta_fc = nn.Sequential(
                nn.Linear(metadata_features, 32),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            combined_size = self.flattened_size + 32
        else:
            combined_size = self.flattened_size

        # Final classifier
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


# -------------------------
# 4. Training Loop
# -------------------------
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

        # Validation
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


# -------------------------
# 5. Example Usage
# -------------------------
if __name__ == "__main__":
    test_dataset = H5CancerDataset(
        images_h5_path="Training_Data/pcam/test_split.h5",
        labels_h5_path="Training_Data/Labels/camelyonpatch_level_2_split_test_y.h5",
        transform=transform,
    )

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    images, labels = next(iter(test_loader))
    print("Batch images:", images.shape, images.dtype)
    print("Batch labels:", labels.shape, labels.dtype, "unique:", labels.unique().tolist())

    # Example model (binary classification for PCam labels 0/1)
    model = CancerCNN(num_classes=2, metadata_features=0)
    # train_model(model, train_loader=test_loader, val_loader=None, num_epochs=1, lr=1e-3, device="mps")


