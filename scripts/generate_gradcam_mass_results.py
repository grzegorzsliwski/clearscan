import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt


# Ścieżki w projekcie
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "images"
SPLIT_CSV = PROJECT_ROOT / "analysis" / "data_splits" / "test_split.csv"
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results"
GRADCAM_DIR = RESULTS_DIR / "gradcam"
CONFIG_PATH = RESULTS_DIR / "config.json"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model_stage3 (1).pth"
OUTPUT_PATH = GRADCAM_DIR / "gradcam_mass_examples.png"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    model_cfg = cfg.get("model", {})
    image_size = int(model_cfg.get("image_size", 512))
    dropout = float(model_cfg.get("dropout", 0.4))
    spatial_dropout = float(model_cfg.get("spatial_dropout", 0.2))
    return image_size, dropout, spatial_dropout


class ChestXrayDataset(Dataset):
    """Zbiór danych dla binarnej detekcji masy (Mass vs No Mass).

    Odtwarza logikę z complete_training_and_evaluation_COLABv2:
    - wczytywanie z katalogów images_XXX/images
    - CLAHE
    - etykieta binarna: 1 jeśli 'Mass' w Finding Labels, inaczej 0
    """

    def __init__(self, dataframe: pd.DataFrame, data_dir: Path, transform=None, target_class: str = "Mass"):
        self.dataframe = dataframe.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_class = target_class

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_name = row["Image Index"]

        # Szukaj obrazu w images_001..images_012
        image_path = None
        for i in range(1, 13):
            potential_path = self.data_dir / f"images_{i:03d}" / "images" / image_name
            if potential_path.exists():
                image_path = potential_path
                break

        if image_path is None:
            raise FileNotFoundError(f"Nie znaleziono obrazu: {image_name}")

        # Wczytanie w skali szarości
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Nie udało się wczytać obrazu: {image_path}")

        # CLAHE jak w notebooku
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        finding_labels = row["Finding Labels"]
        target = 1.0 if self.target_class in finding_labels else 0.0
        target = torch.tensor([target], dtype=torch.float32)

        return image, target


class MassDetectionModel(nn.Module):
    def __init__(self, pretrained: bool = False, dropout: float = 0.4, spatial_dropout_p: float = 0.2):
        super().__init__()
        import torchvision.models as models

        self.backbone = models.densenet121(pretrained=pretrained)

        # Modyfikacja pierwszej warstwy na 1 kanał (skala szarości)
        original_conv = self.backbone.features.conv0
        self.backbone.features.conv0 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            self.backbone.features.conv0.weight = nn.Parameter(
                original_conv.weight.mean(dim=1, keepdim=True)
            )

        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()

        self.spatial_dropout = nn.Dropout2d(p=spatial_dropout_p)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 1),
        )

        nn.init.kaiming_normal_(self.head[3].weight, mode="fan_out")
        nn.init.zeros_(self.head[3].bias)

    def forward(self, x):
        features = self.backbone.features(x)
        if self.training:
            features = self.spatial_dropout(features)
        output = self.head(features)
        return output


def create_val_transform(image_size: int):
    """Transformacje zgodne z COLABv2: LongestMaxSize + PadIfNeeded + Normalize(0.5, 0.5)."""
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=1.0,
        ),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])


def load_model(device: torch.device):
    image_size, dropout, spatial_dropout = load_config()

    model = MassDetectionModel(
        pretrained=False,
        dropout=dropout,
        spatial_dropout_p=spatial_dropout,
    ).to(device)

    state = torch.load(MODEL_PATH, map_location=device)

    # Obsługa ewentualnego 'module.' w nazwach warstw
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(k.startswith("module.") for k in state.keys()):
        new_state = {}
        for k, v in state.items():
            new_k = k[len("module.") :] if k.startswith("module.") else k
            new_state[new_k] = v
        state = new_state

    model.load_state_dict(state)
    model.eval()

    return model, image_size


def generate_gradcam_mass(model: nn.Module, image_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """Generuje mapę Grad-CAM dla klasy pozytywnej (Mass).

    Zakłada wyjście modelu o kształcie [1, 1] (logit dla Mass).
    """
    model.eval()

    image_tensor = image_tensor.unsqueeze(0).to(device)
    image_tensor.requires_grad = True

    features = None
    gradients = None

    def forward_hook(module, inp, output):
        nonlocal features
        features = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    target_layer = model.backbone.features.denseblock4
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    model.zero_grad()

    # Target: Mass (logit pozytywnej klasy)
    class_loss = output[0, 0]
    class_loss.backward()

    forward_handle.remove()
    backward_handle.remove()

    pooled_grads = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(features.shape[1]):
        features[:, i, :, :] *= pooled_grads[i]

    heatmap = features.mean(dim=1).squeeze().cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    return heatmap


def main(num_examples: int = 6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")

    # Wczytaj model i konfigurację
    model, image_size = load_model(device)
    print("Model załadowany z:", MODEL_PATH)

    # Wczytaj zbiór testowy
    test_df = pd.read_csv(SPLIT_CSV)
    print(f"Liczba obrazów w zbiorze testowym: {len(test_df)}")

    val_transform = create_val_transform(image_size)
    test_dataset = ChestXrayDataset(test_df, DATA_DIR, transform=val_transform)

    # Pobierz kilka przykładów (pierwsze num_examples z testu)
    n = min(num_examples, len(test_dataset))
    print(f"Generuję Grad-CAM dla {n} przykładów (target: Mass)")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i in range(n):
        image_tensor, target = test_dataset[i]
        heatmap = generate_gradcam_mass(model, image_tensor, device=device)

        # Dopasuj rozmiar heatmapy do wejścia
        heatmap_resized = cv2.resize(heatmap, (image_size, image_size))

        # Obraz do podglądu (po normalizacji, jak w notebooku)
        img = image_tensor.cpu().squeeze().numpy()

        axes[i].imshow(img, cmap="gray")
        axes[i].imshow(heatmap_resized, cmap="jet", alpha=0.4)

        label_str = "Mass" if float(target.item()) == 1.0 else "No Finding"
        axes[i].set_title(f"Etykieta: {label_str} | target: Mass", fontsize=9)
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Grad-CAM – obszary uwagi modelu (target: Mass)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Zapisano wizualizację Grad-CAM do:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
