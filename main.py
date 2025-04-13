import streamlit as st
import os
from gaze_hull_utils import process_video_with_gaze_and_hulls, generate_area_dataframe_and_plot

# Display app title and description
st.title("🎥 Gaze and Hull Analysis Viewer")
st.markdown("""
This app visualizes gaze data over a video by plotting convex and concave hulls per frame,
and shows area analysis over time.
""")

# Folder options for gaze data (exclude hidden folders)
folders = [f for f in os.listdir('./') if os.path.isdir(f) and not f.startswith('.')]
selected_folder = st.selectbox("Select a folder with gaze data", folders)

# Define paths for video and gaze data
base_path = f"./{selected_folder}"
video_path = f"./raw clip/{selected_folder}_c.mov"

# Check if paths exist before processing
if not os.path.exists(base_path):
    st.error(f"Folder {base_path} does not exist. Please check the folder.")
elif not os.path.exists(video_path):
    st.error(f"Video file {video_path} does not exist. Please check the video file.")
else:
    # Show video with gaze and hulls
    if st.button("▶️ Run Gaze + Hull Video"):
        st.info("Processing video. This may take a moment...")
        output_path = process_video_with_gaze_and_hulls(base_path, video_path)

        # Check if output_path is a valid file
        if os.path.exists(output_path):
            st.video(output_path)
        else:
            st.error(f"Processed video was not saved correctly. Output path: {output_path}")

    # Show hull area analysis plot
    if st.button("📈 Show Hull Area Analysis"):
        st.info("Generating plot...")
        frame_nums, convex_areas, concave_areas = process_video_with_gaze_and_hulls(base_path, video_path)
        df = generate_area_dataframe_and_plot(frame_nums, convex_areas, concave_areas)
        st.dataframe(df)
