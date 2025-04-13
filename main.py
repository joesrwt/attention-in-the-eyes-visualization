# main.py
import streamlit as st
import os
from gaze_hull_utils import process_video_with_gaze_and_hulls, generate_area_dataframe_and_plot

st.set_page_config(page_title="Gaze & Hull Visualizer", layout="wide")
st.title("👁️ Gaze + Hull Analysis Viewer")
st.markdown("""
Upload your gaze `.mat` files and the corresponding video to visualize gaze points, convex and concave hulls frame-by-frame. 
Also see how the hull areas evolve over time.
""")

with st.sidebar:
    st.header("Input Files")
    mat_folder = st.text_input("📁 Path to folder containing .mat gaze files")
    video_file = st.file_uploader("🎥 Upload video file (MP4 preferred)", type=["mp4", "mov"])
    alpha = st.slider("Alpha for concave hull", min_value=0.01, max_value=0.1, step=0.01, value=0.03)
    plot_window = st.slider("Rolling average window size (frames)", 5, 100, 20)
    run_btn = st.button("🚀 Run Visualization")

if run_btn and mat_folder and video_file:
    with st.spinner("Processing video and computing hulls..."):
        video_path = os.path.join("temp_vid.mp4")
        with open(video_path, 'wb') as f:
            f.write(video_file.read())

        frame_nums, convex_areas, concave_areas, out_path = process_video_with_gaze_and_hulls(
            mat_folder, video_path, alpha=alpha
        )

    if out_path:
        st.subheader("📽️ Video with Gaze & Hulls")
        st.video(out_path)

    if frame_nums:
        st.subheader("📊 Hull Area Analysis")
        generate_area_dataframe_and_plot(frame_nums, convex_areas, concave_areas, window_size=plot_window)
else:
    st.info("Please select input files and click Run Visualization.")
