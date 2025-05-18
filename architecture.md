# System Architecture Overview

This project leverages Generative Adversarial Networks (GANs) for synthesizing medical images, focusing on brain MRIs and other radiological modalities. The system is modular, scalable, and user-friendly, integrating model training and an interactive web application.

## 1. High-Level Architecture Diagram
```

+-------------------+       +---------------------+       +---------------------+
|                   |       |                     |       |                     |
|  Medical Image    +------>+  GAN Training       +------>+  Trained GAN Models |
|  Datasets         |       |  (Custom/Pretrained)|       |  (Weights/Checkpts) |
|                   |       |                     |       |                     |
+-------------------+       +----------+----------+       +----------+----------+
                                         |                             |
                                         v                             v
                              +---------------------+        +----------------------+
                              |                     |        |                      |
                              | Streamlit Web App   +<-------+  medigan Pretrained  |
                              | (User Interface)    |        |  GAN Collection      |
                              |                     |        |                      |
                              +----------+----------+        +----------------------+
                                         |
                                         v
                              +----------------------+
                              |                      |
                              | Synthetic Image      |
                              | Generation & Display |
                              |                      |
                              +----------------------+
```

## 2. Components

- **Medical Image Datasets:**  
  Public datasets (e.g., BraTS for MRI) serve as sources for training and evaluation.

- **GAN Training Module:**  
  - *Custom GAN:* Dual-network system (generator and discriminator) for adversarial training.  
  - *Pretrained GANs:* Integration with Medigan library for pretrained models.

- **Model Storage:**  
  Stores trained weights and checkpoints.

- **Streamlit Web Application:**  
  User interface for model selection, image generation, and visualization.

- **Synthetic Image Generation:**  
  Produces new medical images for download or downstream tasks.

## 3. GAN Architecture Details

- **Generator:**  
  CNN with upsampling layers to synthesize images from noise or conditions.

- **Discriminator:**  
  CNN to distinguish real vs generated images, providing adversarial feedback.

- **Training Loop:**  
  Adversarial training where generator tries to fool discriminator.

## 4. Data Flow

1. Input medical images are preprocessed for training.  
2. Trained models are loaded by the Streamlit app.  
3. Users generate and visualize synthetic images.  
4. Images can be downloaded or used for further training/evaluation.
