import streamlit as st
import os
import tempfile
import shutil
from gaze_hull_utils import process_video_with_gaze_and_hulls, generate_area_dataframe_and_plot

st.title("📤 Gaze and Hull Analysis Viewer")
st.markdown("Upload your gaze `.mat` files and a video to visualize convex and concave hulls.")

# Upload gaze data (.mat files)
mat_files = st.file_uploader("Upload gaze data files (.mat)", type="mat", accept_multiple_files=True)

# Upload video file
video_file = st.file_uploader("Upload video file", type=["mp4", "mov", "avi"])

if mat_files and video_file:
    with tempfile.TemporaryDirectory() as temp_dir:
        mat_dir = os.path.join(temp_dir, "mat_data")
        os.makedirs(mat_dir, exist_ok=True)

        # Save uploaded .mat files
        for uploaded_file in mat_files:
            file_path = os.path.join(mat_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

        # Save uploaded video file
        video_path = os.path.join(temp_dir, video_file.name)
        with open(video_path, "wb") as f:
            f.write(video_file.read())

        # Buttons to run visualization or plot
        if st.button("▶️ Run Gaze + Hull Video"):
            st.info("Processing video...")
            output_path = process_video_with_gaze_and_hulls(mat_dir, video_path)

            if os.path.exists(output_path):
                st.video(output_path)
            else:
                st.error("Processed video not found.")

        if st.button("📈 Show Hull Area Analysis"):
            st.info("Generating area plot...")
            frame_nums, convex_areas, concave_areas = process_video_with_gaze_and_hulls(mat_dir, video_path)
            df = generate_area_dataframe_and_plot(frame_nums, convex_areas, concave_areas)
            st.dataframe(df)
else:
    st.warning("Please upload both gaze data (.mat files) and a video file to begin.")
