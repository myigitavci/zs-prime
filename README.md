

# **ZS-PRIME: Zero-Shot Self-Supervised Distortion-Free Diffusion MRI Reconstruction**

> **Multi-Shot Distortion-Free Diffusion-Weighted MRI Reconstruction with Zero-Shot Self-Supervised Learning**

Welcome to the official repository for **ZS-PRIME**, a novel framework for zero-shot self-supervised diffusion MRI reconstruction that operates directly on undersampled k-space. This repository includes code for training and inference to help researchers and practitioners replicate and apply the methodology.

---

## **Key Features**
- **Zero-Shot Learning**: Works without external reference data, enabling single-subject training.
- **Hybrid-Space Regularization**: Incorporates network-based denoisers in both k-space and image space.
- **Advanced Field Mapping**: Generates high-fidelity field maps with no extra scan time.
- **Improved Reconstruction**: Produces distortion-free multi-shot diffusion-weighted MRI images.

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/username/ZS-PRIME.git
   cd ZS-PRIME
   ```

2. Install spesific tensorflow version and other necessary libraries:
   ```bash
   pip install tensorflow==2.8.0
   ```


---

## **Usage**

### **1. Training**
To train the model on your diffusion MRI dataset, use the `zsprime_all_directions_all_slices.py` script.

---

### **2. Inference**
To perform inference using the trained model, use the script `zsprime_all_directions_all_slices_inference.py`.







