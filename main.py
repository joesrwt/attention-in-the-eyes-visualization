import streamlit as st
from gaze_hull_utils import process_video_with_gaze_and_hulls, generate_area_dataframe_and_plot
import os

# Display app title and description
st.title("🎥 Gaze and Hull Analysis Viewer")
st.markdown("""
This app visualizes gaze data over a video by plotting convex and concave hulls per frame,
and shows area analysis over time.
""")

# Folder options for gaze data
folders = [f for f in os.listdir('./') if os.path.isdir(f) and not f.startswith('.')]
selected_folder = st.selectbox("Select a folder with gaze data", folders)

# Paths for video and gaze data
base_path = "./APPAL_2a"
video_path = "./raw clip/APPAL_2a_c.mov"

# Show video with gaze and hulls
if st.button("▶️ Run Gaze + Hull Video"):
    st.info("Processing video. This may take a moment...")
    output_path = process_video_with_gaze_and_hulls(base_path, video_path)
    st.video(output_path)

# Show hull area analysis plot
if st.button("📈 Show Hull Area Analysis"):
    st.info("Generating plot...")
    frame_nums, convex_areas, concave_areas = process_video_with_gaze_and_hulls(base_path, video_path)
    df = generate_area_dataframe_and_plot(frame_nums, convex_areas, concave_areas)
    st.dataframe(df)
