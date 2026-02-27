# Project Context: Automated Image Colorization (Master's Thesis)

## 1. Project Overview & Objectives
This repository contains the practical implementation for a Master's Thesis titled: *"Research and Comparative Analysis of Methods for Automatic Image Colorization"*. 
The primary goal is to develop, train, and comprehensively compare multiple Deep Learning architectures for image colorization, tracing the evolution from a basic Convolutional Neural Network (CNN) to an advanced Ensemble/Fusion Generative Adversarial Network (GAN) with global semantic priors.

**Core Tech Stack:**
* **Language:** Python 3
* **Deep Learning Framework:** PyTorch
* **Computer Vision Libraries:** `scikit-image` (crucial for color space conversions), `torchvision`, `Pillow`, `OpenCV`
* **Hardware Acceleration:** Apple Silicon MPS (`torch.backends.mps`) as the primary target, with fallback to CUDA/CPU.

## 2. Color Space Representation & Data Processing
The project strictly operates in the **CIE L\*a\*b\*** color space, separating luminosity from chrominance. Models do **not** output RGB directly.

* **Input (X):** `L` channel (Lightness/Grayscale). 
    * *Shape:* `(Batch, 1, 256, 256)`
    * *Normalization:* Scaled from `[0, 100]` to `[0, 1]` (divided by 100.0).
* **Target/Output (Y):** `ab` channels (Color components).
    * *Shape:* `(Batch, 2, 256, 256)`
    * *Normalization:* Scaled from `[-128, 127]` to approximately `[-1, 1]` (divided by 128.0).
* **Post-processing:** The model predicts the `ab` channels. These are denormalized, concatenated with the original `L` input, and converted back to RGB using `skimage.color.lab2rgb` for visual evaluation.
* **Dataset:** COCO val2017 (for rapid prototyping and benchmarking).

## 3. Directory & Architecture Structure
The codebase follows a strict modular Python package structure to ensure clean separation of concerns and reproducibility.

    ImageColorizationAnsamble/
    ├── checkpoints/            # Baseline CNN weights (.pth) — saved here by train_baseline.py
    ├── data/                   # Raw datasets (COCO) and custom test samples
    │   ├── coco/val2017/       # COCO val2017 images (5000 JPEGs)
    │   └── test_samples/       # Hand-picked images for visual thesis comparisons
    ├── outputs/
    │   ├── checkpoints/        # U-Net, GAN, and Fusion weights (.pth)
    │   └── images/             # Visual evaluation results (to be created by evaluate.py)
    ├── scripts/                # Execution entry points
    │   ├── trains/             # Training scripts sub-package
    │   │   ├── train_baseline.py   # Stage 1: Trains the simple Baseline CNN
    │   │   ├── train_unet.py       # Stage 2: Trains the standalone U-Net
    │   │   ├── train_gan.py        # Stage 3: Trains the Pix2Pix GAN
    │   │   ├── train_fusion.py     # Stage 4: Trains the Ensemble/Fusion GAN
    │   │   └── train.py            # Unified trainer (baseline + unet via --model flag)
    │   └── evaluate.py         # Universal inference and visualization script
    ├── src/                    # Core Python package
    │   ├── losses/
    │   │   └── __init__.py     # Reusable loss classes (GANLoss — to be moved here)
    │   ├── models/
    │   │   ├── baseline_cnn.py # Stage 1: Simple CNN Encoder-Decoder
    │   │   ├── u_net.py        # Stage 2 & 3: U-Net with skip connections (GAN Generator)
    │   │   ├── unet_fusion.py  # Stage 4: U-Net modified to accept global context vectors
    │   │   ├── global_hints.py # Stage 4: Frozen ResNet-18 semantic feature extractor
    │   │   └── gan/
    │   │       └── discriminator.py  # PatchGAN discriminator (Stages 3 & 4)
    │   └── utils/
    │       ├── dataset.py      # PyTorch Dataset class (ColorizationDataset)
    │       ├── common.py       # Shared helpers: get_device(), lab_to_rgb() — to be implemented
    │       └── metrics.py      # PSNR, SSIM, LPIPS — to be implemented
    └── tests/                  # Pytest unit tests (tensor shapes, data loading)

## 4. Model Architectures & Evolution Strategy
The thesis analyzes the progression of colorization quality through four specific architectural stages:

### Stage 1: Baseline CNN (`src/models/baseline_cnn.py`)
* **Concept:** A basic Encoder-Decoder architecture with a spatial bottleneck.
* **Loss Function:** Mean Squared Error (`MSELoss`).
* **Expected Result:** Produces a "sepia" or desaturated effect. MSE heavily penalizes bold color guesses, forcing the network to predict safe, averaged grayish-brown tones. It serves as the baseline to demonstrate why regression loss alone is insufficient.

### Stage 2: U-Net Architecture (`src/models/u_net.py`)
* **Concept:** Encoder-Decoder with Skip Connections.
* **Loss Function:** Mean Absolute Error (`L1Loss`).
* **Improvement:** Skip connections preserve spatial high-frequency details (edges, boundaries) that are normally lost in the bottleneck. L1 loss provides slightly sharper and more vibrant colors than MSE, but still lacks true photorealism.

### Stage 3: Pix2Pix GAN (`scripts/trains/train_gan.py`)
* **Concept:** Generative Adversarial Network based on the Pix2Pix framework.
* **Generator (G):** The U-Net from Stage 2 (`src/models/u_net.py`).
* **Discriminator (D):** A `PatchGAN` (`src/models/gan/discriminator.py`) that classifies 70x70 overlapping patches of the image as real or fake, rather than the whole image at once. This forces the Generator to produce high-frequency, realistic local textures (like grass or fur).
* **Loss Function:** Combined Objective.
    * L_GAN = `BCEWithLogitsLoss(D(x, G(x)), True)`
    * L_L1 = `L1Loss(G(x), y) * lambda` (where lambda = 100)
* **Improvement:** Overcomes the "safe averaging" problem, producing vibrant and realistic colors.

### Stage 4: Ensemble / Fusion Architecture (The Primary Contribution)
* **Concept:** Conditional GAN with Global Semantic Priors. PatchGAN struggles with global context (e.g., it might color daytime sky with sunset colors because the local patch looks similar). We fix this by fusing local and global networks.
* **Global Feature Extractor (`src/models/global_hints.py`):** A pre-trained `ResNet-18` (trained on ImageNet). We remove the classification head and freeze the weights. It outputs a 512-dimensional semantic context vector representing the "vibe/scene" of the image.
* **Fusion Generator (`src/models/unet_fusion.py`):** * The `L` channel passes down the U-Net encoder to a bottleneck of shape `(Batch, 512, 1, 1)`.
    * The bottleneck is flattened and concatenated with the 512-dim ResNet vector.
    * A Linear layer mixes them back to 512 dimensions, reshaping it to `(Batch, 512, 1, 1)` before passing it up the decoder.
* **Improvement:** The Generator now makes colorization decisions based on both precise local edges (U-Net) and global scene understanding (ResNet).

## 5. Training Methodology
* **Optimizers:** `Adam` optimizer is used across all models.
* **Hyperparameters:**
    * *Baseline/U-Net:* Learning rate = `1e-3` to `2e-4`.
    * *GANs:* Learning rate = `2e-4` with betas `(0.5, 0.999)` to ensure stable adversarial training. Batch size typically 8-16 depending on VRAM.
* **Training Loop Logic:** For GANs, the Discriminator is updated first using a batch of real images and a batch of detached fake images. Then, the Generator is updated to fool the Discriminator while simultaneously minimizing the L1 distance to the ground truth.
* **Checkpointing:** Weights (`.pth`) are saved every 5 epochs and at the final epoch. **Note on actual checkpoint locations:** Baseline CNN weights are saved to `./checkpoints/` (project root). U-Net, GAN, and Fusion weights are saved to `./outputs/checkpoints/`.

## 6. Evaluation & Benchmarking (`evaluate.py`)
To fulfill the "Comparative Analysis" requirement of the thesis, the models must be evaluated both qualitatively and quantitatively.

**Qualitative (Visual) Evaluation:**
* `evaluate.py` generates concatenated comparison strips: `[Grayscale Input | Model Prediction | Ground Truth]`.
* Saved to `./outputs/images/` for direct inclusion in the thesis document.

**Quantitative Metrics (to be implemented — `src/utils/metrics.py` does not yet exist):**
1.  **PSNR (Peak Signal-to-Noise Ratio):** Measures absolute pixel-level accuracy.
2.  **SSIM (Structural Similarity Index):** Measures perceived change in structural information.
3.  **LPIPS (Learned Perceptual Image Patch Similarity):** A neural-network-based metric that aligns better with human perception of color realism.

## 7. Strict Directives for the AI Agent
When generating, refactoring, or reviewing code for this project, the AI Agent MUST:
1.  **Preserve the Lab Color Space workflow:** Never suggest rewriting the core pipeline to predict RGB directly.
2.  **Respect Modularity:** Place new helper functions in `src/utils/`, new losses in `src/losses/`, and keep training scripts in `scripts/trains/` and the evaluation script in `scripts/`. Always use `sys.path.append` in scripts to resolve the `src` module.
3.  **Hardware Agnosticism:** Always include logic to detect and utilize Apple Silicon `mps` alongside `cuda` and `cpu` (e.g., `device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")`).
4.  **No Arbitrary Refactoring:** Do not alter the normalization values (`/ 100.0` and `/ 128.0`) in `dataset.py` without explicit user instruction, as the models rely heavily on this specific scale.