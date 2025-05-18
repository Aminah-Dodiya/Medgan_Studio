import io
import os
import zipfile
from functools import lru_cache
import torch
from medigan import Generators
from torchvision.transforms.functional import to_pil_image
import streamlit as st

@st.cache_resource(show_spinner="Initializing Medigan Engine...")
def get_generator():
    """Initialize and cache Medigan generator instance"""
    return Generators()

@st.cache_data(ttl=3600)
def img_to_bytes(img):
    """Convert PIL image to PNG byte format for download"""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

@st.cache_resource(ttl=86400)
def get_model_registry():
    generators = get_generator()
    model_registry = {}
    
    for model_id in generators.list_models():
        try:
            model_config = generators.get_config_by_id(model_id=model_id)
            model_info = {
                "name": model_id.replace("_", " ").title(),
                "description": model_config.get("description", {}).get("title", "No description provided."),
                "modality": model_config.get("selection", {}).get("modality", "Unknown"),
                "anatomy": model_config.get("selection", {}).get("organ", "Unknown"),
            }
            model_registry[model_id] = model_info
        except Exception as e:
            model_registry[model_id] = {
                "name": model_id.replace("_", " ").title(),
                "description": "Error fetching config",
                "modality": "Unknown",
                "anatomy": "Unknown",
            }
            st.warning(f"Could not fetch config for {model_id}: {e}")

    return model_registry

def generate_image_batch(model_id, num_images):
    """Generate a batch of images for a selected model"""
    try:
        generators = get_generator()
        dataloader = generators.get_as_torch_dataloader(
            model_id=model_id,
            install_dependencies=True,
            num_samples=num_images,
            prefetch_factor=None,
        )
        
        batch = []
        for _, data in enumerate(dataloader):
            batch.extend(process_images(data))
            if len(batch) >= num_images:
                break
                
        return batch[:num_images]
    
    except Exception as e:
        st.error(f"Image generation failed: {str(e)}")
        return []

def process_images(data_dict):
    """Convert model output tensors to PIL images"""
    images = []
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            img = to_pil_image(value.squeeze())
            images.append(img)
        elif isinstance(value, list) and all(isinstance(v, torch.Tensor) for v in value):
            for tensor in value:
                img = to_pil_image(tensor.squeeze())
                images.append(img)
    return images

def create_zip(images, prefix="medigan"):
    """Bundle all images into a ZIP archive for download"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for idx, img in enumerate(images):
            img_bytes = img_to_bytes(img)
            zf.writestr(f"{prefix}_{idx+1}.png", img_bytes)
    zip_buffer.seek(0)
    return zip_buffer