# main.py
import streamlit as st
import os
import shutil
from gaze_hull_utils import process_video_with_gaze_and_hulls, generate_area_dataframe_and_plot

# Page setup
st.set_page_config(page_title="Gaze & Hull Visualizer", layout="wide")
st.title("👁️ Gaze and Hull Analysis Viewer")
st.markdown("""
Upload a folder containing `.mat` gaze data and a matching video clip to visualize gaze hulls and analyze area changes.
""")

# Temp storage paths
UPLOAD_FOLDER = "uploaded_data"
VIDEO_FOLDER = os.path.join(UPLOAD_FOLDER, "videos")
GAZE_FOLDER = os.path.join(UPLOAD_FOLDER, "gaze")

# Make sure upload directories exist
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(GAZE_FOLDER, exist_ok=True)

# Upload video
video_file = st.file_uploader("📹 Upload a video file (.mov)", type=["mov"])
# Upload .mat files (allow multiple)
gaze_files = st.file_uploader("📁 Upload gaze data files (.mat)", type=["mat"], accept_multiple_files=True)

# Run after uploads
if video_file and gaze_files:
    # Save video
    video_path = os.path.join(VIDEO_FOLDER, video_file.name)
    with open(video_path, 'wb') as f:
        f.write(video_file.read())

    # Create a unique subfolder for gaze files
    mat_folder = os.path.join(GAZE_FOLDER, os.path.splitext(video_file.name)[0])
    os.makedirs(mat_folder, exist_ok=True)

    # Save .mat files
    for file in gaze_files:
        file_path = os.path.join(mat_folder, file.name)
        with open(file_path, 'wb') as f:
            f.write(file.read())

    st.success("Files uploaded successfully!")

    # Run visualization
    if st.button("▶️ Run Gaze + Hull Video"):
        st.info("Processing video...")
        output_path = process_video_with_gaze_and_hulls(mat_folder, video_path)

        if output_path and os.path.exists(output_path):
            st.video(output_path)
        else:
            st.error("Processed video not found. Check if gaze data and video match correctly.")

    # Show area plot
    if st.button("📈 Show Hull Area Analysis"):
        st.info("Generating plot...")
        result = process_video_with_gaze_and_hulls(mat_folder, video_path)

        if result:
            frame_nums, convex_areas, concave_areas, _ = result
            df = generate_area_dataframe_and_plot(frame_nums, convex_areas, concave_areas)
            st.dataframe(df)
        else:
            st.error("Failed to generate area analysis plot.")
else:
    st.warning("Please upload both a video and gaze data files to continue.")
