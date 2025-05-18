import streamlit as st
import time
import psutil
import os
import plotly.express as px
from streamlit_extras.stylable_container import stylable_container
from utils import get_model_registry, generate_image_batch, img_to_bytes, create_zip

PAGE_CONFIG = {
    "page_title": "Medigan Studio",
    "page_icon": "",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

st.set_page_config(**PAGE_CONFIG)

def init_session():
    """
    Initialize session state variables.
    """
    if 'generated_images' not in st.session_state:
        st.session_state.generated_images = []
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = list(get_model_registry().keys())[0]
    if 'grid_size' not in st.session_state:
        st.session_state.grid_size = 4
    if 'latency_history' not in st.session_state:
        st.session_state.latency_history = []

def model_selection_panel():
    """
    Renders model selection panel in sidebar.
    """
    with st.sidebar:
        with st.container(border=True):
            st.header("⚙️ Configuration")
            model_registry = get_model_registry()
            selected_model = st.session_state.selected_model
            model_data = model_registry[selected_model]

            selected_model = st.selectbox(
                "Select GAN Model",
                options=list(model_registry.keys()),
                index=list(model_registry.keys()).index(st.session_state.selected_model),
                format_func=lambda x: model_registry[x]['name'],
                help="Choose from pretrained medical imaging models"
            )

            if selected_model != st.session_state.selected_model:
                st.session_state.selected_model = selected_model
                st.session_state.generated_images = []
                st.rerun()

            with st.expander("ℹ️ Model Details", expanded=False):
                modality = model_data["modality"]
                anatomy = model_data["anatomy"]

                modality_html = (
                    "<ul style='margin-top: 0.2rem; margin-bottom: 0.5rem; padding-left: 1.2rem;'>"
                    + "".join(f"<li>{m}</li>" for m in modality) +
                    "</ul>" if isinstance(modality, list) else f"<span>{modality}</span>"
                )

                anatomy_html = (
                    "<ul style='margin-top: 0.2rem; margin-bottom: 0.5rem; padding-left: 1.2rem;'>"
                    + "".join(f"<li>{a}</li>" for a in anatomy) +
                    "</ul>" if isinstance(anatomy, list) else f"<span>{anatomy}</span>"
                )

                st.markdown(f"""
                <div style='line-height: 1.6; font-size: 0.95rem;'>
                    <p><strong>🆔 Model ID:</strong> <code>{selected_model}</code></p>
                    <p><strong>📝 Description:</strong><br>{model_data['description']}</p>
                    <p><strong>🧪 Modality:</strong>{modality_html}</p>
                    <p><strong>🧍 Anatomy / Region:</strong>{anatomy_html}</p>
                </div>
                """, unsafe_allow_html=True)

            # Image controls
            col1, col2 = st.columns(2)
            with col1:
                num_images = st.slider(
                    "Image Count",
                    1, 16, 4,
                    help="Max 16 images per generation"
                )
            with col2:
                grid_size = st.select_slider(
                    "Grid Layout",
                    options=[2, 3, 4],
                    value=4,
                    help="Images per row"
                )

            st.session_state.grid_size = grid_size

            if st.button("✨ Generate", type="primary", use_container_width=True):
                start_time = time.perf_counter()
                st.session_state.generated_images = generate_image_batch(
                    st.session_state.selected_model,
                    num_images
                )
                end_time = time.perf_counter()

                latency = (end_time - start_time) / num_images
                st.session_state.latency_history.append(round(latency, 4))

            if st.button("Reset Session"):
                for key in ["generated_images", "selected_model", "latency_history"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

def display_metrics():
    """
    Displays dynamic system and inference performance metrics.
    """
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        gpu_util = f"{gpus[0].load * 100:.0f}%" if gpus else "N/A"
    except:
        gpu_util = "N/A"

    process = psutil.Process(os.getpid())
    memory_used = process.memory_info().rss / 1e9
    memory_total = psutil.virtual_memory().total / 1e9
    memory_display = f"{memory_used:.1f}/{memory_total:.0f}GB"

    with st.expander("📈 Performance Dashboard", expanded=True):
        col1, col2, col3 = st.columns(3)

        if st.session_state.latency_history:
            last = st.session_state.latency_history[-1]
            avg = sum(st.session_state.latency_history) / len(st.session_state.latency_history)
            stdev = (
                sum([(x - avg) ** 2 for x in st.session_state.latency_history]) /
                len(st.session_state.latency_history)
            ) ** 0.5
            col1.metric("Inference Time", f"{last:.2f}s/img", f"±{stdev:.2f}s")
        else:
            col1.metric("Inference Time", "–")

        col2.metric("GPU Utilization", gpu_util)
        col3.metric("Memory Usage", memory_display)

        latency = st.session_state.get("latency_history", [])

        if latency:
            st.plotly_chart(px.line(
                x=[f"Batch {i+1}" for i in range(len(latency))],
                y=latency,
                title="Generation Latency Trends",
                labels={"x": "Batch", "y": "Seconds/Image"}
            ), use_container_width=True)
        else:
            st.info("No latency data yet. Generate images to populate latency trends.")
        
def image_grid():
    """
    Renders a grid of generated images with download buttons.
    """
    if not st.session_state.generated_images:
        return

    st.header("🎨 Generated Images")
    cols = st.columns(st.session_state.get('grid_size', 4))

    for idx, img in enumerate(st.session_state.generated_images):
        with cols[idx % len(cols)]:
            with stylable_container(
                key=f"img_{idx}",
                css_styles="""
                {
                border: 1px solid rgba(138, 141, 147, 0.2);
                border-radius: 12px;
                padding: 1rem;
                transition: background-color 0.2s;
                }
                :hover {
                background-color: rgba(255, 255, 255, 0.05);
                }"""
            ):
                st.image(img, use_container_width=True)
                st.download_button(
                    label=f"Download #{idx+1}",
                    data=img_to_bytes(img),
                    file_name=f"medigan_{st.session_state.selected_model}_{idx+1}.png",
                    mime="image/png",
                    key=f"dl_{idx}",
                    use_container_width=True
                )

    with st.container():
        zip_buffer = create_zip(st.session_state.generated_images, st.session_state.selected_model)
        st.download_button(
            label="⬇️ Download All as ZIP",
            data=zip_buffer,
            file_name=f"{st.session_state.selected_model}_images.zip",
            mime="application/zip",
            use_container_width=True
        )

def main():
    """
    Main function that controls the full UI rendering and state lifecycle.
    """
    try:
        init_session()

        st.title("Medigan Synthetic Imaging Studio")
        st.markdown("""
        **Clinical-grade synthetic medical imaging powered by**  
        [![Medigan](https://img.shields.io/badge/Medigan-1.0.0-blue)](https://github.com/richardobi/medigan) 
        [![Streamlit](https://img.shields.io/badge/Streamlit-1.45.1-FF4B4B)](https://streamlit.io)
        """)

        model_selection_panel()
        image_grid()
        display_metrics()

        st.markdown("---")
        st.caption("HIPAA-compliant synthetic imaging pipeline | Powered by Medigan v1.0.0")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()