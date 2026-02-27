# AI Agent Master Instructions: Full-Stack Image Colorization Platform

## 1. Project Identity & Context
This repository contains the codebase for a Master's Thesis project focused on "Research and comparative analysis of methods for automatic image colorization and color restoration." 

This is a collaborative project developed by a team of three people. The codebase is maintained on GitHub, so all generated code must be highly readable, modular, well-documented, and avoid merge conflicts where possible to facilitate smooth teamwork. The development environment spans across macOS and Windows, requiring strict cross-platform compatibility for all machine learning scripts and backend services.

The project has evolved from a pure PyTorch Machine Learning pipeline into a full-stack web application. It allows users to upload black-and-white images, select a specific colorization model, and view/download the restored colorized results.

## 2. Technology Stack
* **Machine Learning Core:** Python 3, PyTorch, `scikit-image` (crucial for color space conversions), `torchvision`, `Pillow`.
* **Backend API:** Python 3, Flask, `flask-cors`, `Werkzeug` (for secure file handling).
* **Frontend UI:** Vue.js, Tailwind CSS, Vite. **Strict Language Rule:** Use standard JavaScript (ES6+) only. Do not generate TypeScript code or `.ts` files.
* **Hardware Acceleration:** Agnostic support is mandatory (`torch.backends.mps` for Macbooks, `cuda` for Windows machines, and a `cpu` fallback).

## 3. Model Architectures & Evolution Strategy
The core of the thesis analyzes the progression of color restoration quality through four specific architectural stages. The AI agent must understand the purpose of each:

1.  **Baseline CNN:** A simple Encoder-Decoder with a spatial bottleneck. Trained with `MSELoss`. Produces a "sepia" or desaturated effect because MSE heavily penalizes bold color guesses, forcing the network to predict safe, averaged grayish-brown tones.
2.  **U-Net Architecture:** Encoder-Decoder with Skip Connections. Trained with `L1Loss`. Skip connections preserve spatial high-frequency details (edges, boundaries) lost in the bottleneck.
3.  **Pix2Pix GAN:** * *Generator:* The U-Net architecture.
    * *Discriminator:* A `PatchGAN` that classifies overlapping patches of the image as real or fake.
    * *Objective:* Forces the Generator to produce high-frequency, realistic local textures, overcoming the "safe averaging" problem of standard regression losses.
4.  **Ensemble / Fusion Architecture (Primary Contribution):** Conditional GAN with Global Semantic Priors.
    * *Global Feature Extractor:* A pre-trained `ResNet-18` (weights frozen, classification head removed) outputs a 512-dimensional semantic context vector representing the "vibe/scene" of the image.
    * *Fusion Generator:* The L channel passes down the U-Net. At the bottleneck, it is flattened, concatenated with the ResNet vector, passed through a Linear layer, reshaped, and passed up the decoder. This provides both precise local edges and global scene understanding.

## 4. Color Space & Data Processing Logic (CRITICAL)
The Machine Learning pipeline operates strictly in the **CIE L\*a\*b\*** color space. **Never** suggest rewriting the core ML pipeline to predict RGB directly.
* **Input (X):** `L` channel (Lightness/Grayscale). Shape: `(Batch, 1, 256, 256)`. Normalized from `[0, 100]` to `[0, 1]` (divided by 100.0).
* **Target/Output (Y):** `ab` channels (Color components). Shape: `(Batch, 2, 256, 256)`. Normalized from `[-128, 127]` to approximately `[-1, 1]` (divided by 128.0).
* **Post-processing:** The model predicts the `ab` channels. These are denormalized, concatenated with the original `L` input, and converted back to RGB using `skimage.color.lab2rgb` before being sent to the frontend.

## 5. Feature Implementation Roadmap & Integration
When tasked with building or updating features, follow this logic:

* **ML Pipeline:** Ensure all PyTorch models are trainable via standalone scripts. Evaluation scripts must calculate quantitative metrics: PSNR, SSIM, and LPIPS.
* **Flask API:** Expose the inference scripts via RESTful endpoints (e.g., `/upload`, `/colorize`). Securely handle `multipart/form-data` for image uploads. Implement a global caching mechanism in Flask to keep the large `.pth` models in memory after the first load to prevent massive latency on every request. Wrap inference calls in `with torch.no_grad():` and `model.eval()`.
* **Vue.js UI:** Build a responsive interface using Tailwind CSS utility classes. Core components include a drag-and-drop upload zone, a dropdown for model selection (Baseline, U-Net, GAN, Fusion), a loading state indicator, and an interactive Before/After image comparison tool.
* **End-to-End:** Connect the Vue frontend to the Flask backend using standard fetch/Axios calls. Ensure CORS is configured correctly in Flask. Handle API errors (4xx, 5xx) gracefully in the UI with toast notifications.

## 6. How to Process User Prompts
When the user asks you to implement a feature or fix a bug:
1.  **Acknowledge the Context:** Briefly confirm which part of the stack (ML, API, UI) you are modifying.
2.  **Review Dependencies:** Check if UI changes require API route updates, or if API updates require ML refactoring.
3.  **Provide Complete Code:** Output complete code blocks or clearly define where snippets should be inserted to avoid breaking the team's shared codebase. Maintain the strict JavaScript requirement for the frontend.