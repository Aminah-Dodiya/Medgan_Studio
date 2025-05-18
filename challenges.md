# 🚧 Technical Hurdles, Design Considerations & Future Work

This document outlines the key challenges faced during the development of the GAN-based Medical Image Generator, the solutions implemented, and areas identified for future improvement.

---

## 1. 🧬 Data Complexity and Diversity

- **Challenge:**  
  High variation across modalities (MRI, CT, MMG, etc.), anatomy types, and image resolutions make it hard to generalize models.

- **Solution:**  
  Modular integration of multiple pretrained models (via `medigan`); standardized input/output processing and consistent metadata handling.

---

## 2. ⚠️ Training Stability in GANs

- **Challenge:**  
  GANs are inherently unstable — prone to mode collapse, vanishing gradients, and convergence failure.

- **Solution:**  
  Use pretrained and validated models (e.g., DCGAN, ProGAN). Incorporate models trained on public datasets with robust architectures.

---

## 3. 🎯 Evaluation of Synthetic Image Quality

- **Challenge:**  
  Metrics like PSNR or FID may not fully capture the clinical value or realism of generated medical images.

- **Solution:**  
  Combine quantitative metrics with qualitative expert reviews, and where applicable, validate images via downstream classification or segmentation performance.

---

## 4. 🧠 Integration of Pretrained Models (Medigan)

- **Challenge:**  
  Inconsistent configuration schema, complex metadata, and loading issues when using third-party pretrained models.

- **Solution:**  
  Abstracted model metadata and inference logic in `helpers.py`. Parsed configs via `get_config_by_id()` to extract clean `description`, `modality`, `organ`.

---

## 5. 🧪 Dynamic Model Metadata Display

- **Challenge:**  
  Streamlit’s reactive state system caused stale model metadata when the user switched models too quickly.

- **Solution:**  
  Managed state via `st.session_state` and triggered forced refresh (`st.rerun()`) when model selection changed. Cached metadata using `st.cache_data()` for performance.

---

## 6. 🖼️ Grid Layout Rendering

- **Challenge:**  
  Default Streamlit rendering displayed all generated images in one row, regardless of grid layout value.

- **Solution:**  
  Created dynamic layout with `st.columns()` based on user-defined grid size. Capped image generation at 16 for memory and usability control.

---

## 7. 💻 Computational Resource Constraints

- **Challenge:**  
  GAN inference (and especially training) is compute-heavy, often requiring GPU acceleration.

- **Solution:**  
  Offloaded training entirely. Focused on lightweight inference using small sample sizes and cached models. Streamlit UI supports CPU usage for demo purposes.

---

## 8. 🧾 User Experience and Interface

- **Challenge:**  
  Users need to select models, control layout, understand metadata, and download outputs — all intuitively.

- **Solution:**  
  Minimalist UI with:
  - Controlled dropdowns/sliders
  - Rich metadata display using `st.expander()`
  - Download as `.zip`
  - Help tooltips for image count and layout settings

---

## 9. 🔐 Ethical and Regulatory Considerations

- **Challenge:**  
  Generated synthetic images should not be misconstrued as clinically validated data.

- **Solution:**  
  Added persistent warnings and disclaimers. Models and images are for **educational and research use only**.

---

## 🔮 Future Work

| Area                  | Suggested Action                                                                 |
|-----------------------|----------------------------------------------------------------------------------|
| Metadata Enrichment   | Add fields like resolution, dataset, model source                               |
| Evaluation Metrics    | Include SSIM, FID, or CLIP similarity                                            |
| Model Upload          | Allow users to upload custom GAN checkpoints or weights                         |
| Security              | Add user authentication if hosted online                                        |
| Model Comparison      | Enable side-by-side comparison of different models                               |
| Annotation Tools      | Add interactive drawing/annotation features for synthetic image validation      |

---

## ✅ Summary

This project successfully integrates cutting-edge GAN models through a polished and extensible Streamlit interface. Despite challenges with model integration and layout rendering, the app is now stable, visually refined, and ready for research usage. Continued focus on model interpretability, validation, and ethical safeguards will drive its evolution forward.
