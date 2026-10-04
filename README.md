# QKAN-Med

A reproducible CUDA implementation of QKAN-Med for nine-class colorectal histopathology classification on NCT-CRC-HE-100K and CRC-VAL-HE-7K.

The model combines an ImageNet-pretrained ResNet18 with a 16-qubit quantum head. Quantum computation uses exact batched state-vector simulation and first-order adjoint gradients implemented in PyTorch. No quantum hardware or external quantum simulator is required.

## Model

```text
64 × 64 RGB image
  → ResNet18 → 512 features
  → Linear(512, 16) → sigmoid
  → RY(πz) encoding on 16 qubits
  → RZ–RY–RZ rotations → linear CNOT chain
  → RZ–RY–RZ rotations → even then odd nearest-neighbor CNOT pairs
  → 16 Pauli-Z expectations
  → Dropout(0.5) → Linear(16, 9)
```

The quantum head has 96 trainable rotation parameters. The complete classification head contains 8,457 parameters, including the adapter and readout. The circuit uses 30 CNOT gates and 112 single-qubit rotations, including encoding. Only ResNet18 `layer3` and `layer4` are fine-tuned; earlier layers and their BatchNorm statistics remain frozen.

This release makes the encoding and regularization explicit: adapter outputs pass through sigmoid, and dropout is applied **after the quantum output**. It implements the manuscript's textual even/odd entanglement pattern. These choices define the implementation provided here; it should not be treated as a byte-for-byte reconstruction of the manuscript's original software.

## CUDA installation

Use Linux, Python 3.12, and an NVIDIA GPU with a driver compatible with CUDA 12.8. The reference configuration ran on an NVIDIA L40S. State-vector memory grows with batch size; reduce `--batch-size` if necessary.

From this repository directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

ResNet18 ImageNet weights download automatically on the first training run.

## Dataset layout

Download and extract **NCT-CRC-HE-100K** and **CRC-VAL-HE-7K**. Use the standard NCT-CRC-HE-100K training dataset, not the separately distributed `NCT-CRC-HE-100K-NONORM` variant.

Pass the directories that directly contain the nine class folders:

```text
/path/to/data/
├── NCT-CRC-HE-100K/
│   ├── ADI/
│   ├── BACK/
│   ├── DEB/
│   ├── LYM/
│   ├── MUC/
│   ├── MUS/
│   ├── NORM/
│   ├── STR/
│   └── TUM/
└── CRC-VAL-HE-7K/
    └── [the same nine class folders]
```

The trainer checks for 100,000 training images and 7,180 test images. It resizes images to 64 × 64 with bilinear interpolation and antialiasing, then applies ImageNet normalization. No image augmentation is used.

## Train on CUDA

```bash
source .venv/bin/activate
python train.py \
  --train-dir /path/to/data/NCT-CRC-HE-100K \
  --test-dir /path/to/data/CRC-VAL-HE-7K \
  --output runs/qkan-med-seed42 \
  --epochs 10 \
  --batch-size 128 \
  --workers 4 \
  --seed 42
```

Training uses AdamW with encoder learning rate `1e-4`, head learning rate `1e-3`, and weight decay `1e-4`. Cosine annealing reduces both learning rates toward `1e-6`. The quantum simulation uses complex64 states and float32 parameters without mixed precision or TF32.

Each epoch uses a deterministic permutation derived from the seed. The test dataset is evaluated once after the final epoch and is not used for checkpoint selection. Training writes:

- `config.json`: configuration, dataset path/label hashes, source hashes, and runtime information.
- `history.json`: training loss and accuracy for each epoch.
- `results.json`: final test accuracy, loss, confusion matrix, and per-class recall (`per_class_accuracy`).
- `model.pt`: final model weights, configuration, and class order.

Use a new output directory for every run. This minimal trainer saves the model after final evaluation; it does not resume interrupted training. A matching ten-epoch reference run took approximately 46 minutes on one L40S; this is an estimate, not a runtime guarantee for other GPUs.

## Reproducibility

A reference run of this model and training configuration achieved **93.5515% test accuracy** on 7,180 images with seed 42 after ten epochs. This is one measured run, not a guaranteed score. The standalone trainer is a simplified packaging of that configuration; this repository packaging has not itself undergone a second full CUDA training run.

The reference data manifests are SHA-256 hashes of sorted relative image paths and class indices, not image-content checksums:

```text
NCT-CRC-HE-100K: 01f0ea6c6a496cfa884af83ffe8222ea305b943a6c7738add3a7abb308e5bfb2
CRC-VAL-HE-7K:   5fe950ba7b71d1ccb25eedf419d2fd2430a5de36d39201876aa2b0b0e8da8e2e
```

Keep the dataset, package versions, batch size, epoch count, and seed fixed when comparing repeated runs. Floating-point differences across hardware and software versions can change training outcomes.

## Load trained weights

```python
import torch
from qkan_med import QKANMed

checkpoint = torch.load("runs/qkan-med-seed42/model.pt", map_location="cpu", weights_only=True)
model = QKANMed(pretrained=False).cuda()
model.load_state_dict(checkpoint["state_dict"])
model.eval()
# Pass a CUDA tensor of images using the same preprocessing as train.py.
```
