import streamlit as st
import os
from gaze_hull_utils import process_video_with_gaze_and_hulls, generate_area_dataframe_and_plot

# Display app title and description
st.title("🎥 Gaze and Hull Analysis Viewer")
st.markdown("""
This app visualizes gaze data over a video by plotting convex and concave hulls per frame,
and shows area analysis over time.
""")

# Function to list folders with corresponding .mat files
def get_folders_with_mat_files(directory):
    return [f for f in os.listdir(directory) if os.path.isdir(f) and not f.startswith('.') and any(f.endswith('.mat') for f in os.listdir(os.path.join(directory, f)))]

# Get folders that contain .mat files
base_directory = "./"  # Replace with the correct directory where your data folders are stored
folders = get_folders_with_mat_files(base_directory)

# Folder selection dropdown
selected_folder = st.selectbox("Select a folder with gaze data", folders)

# Construct the paths based on the selected folder
base_path = os.path.join(base_directory, selected_folder)
video_path = f"./raw clip/{selected_folder}_c.mov"

# Show video with gaze and hulls
if st.button("▶️ Run Gaze + Hull Video"):
    st.info("Processing video. This may take a moment...")
    output_path = process_video_with_gaze_and_hulls(base_path, video_path)
    
    # Check if output video is successfully created
    if os.path.exists(output_path):
        st.video(output_path)
    else:
        st.error("Video processing failed. The output video file does not exist.")

# Show hull area analysis plot
if st.button("📈 Show Hull Area Analysis"):
    st.info("Generating plot...")
    frame_nums, convex_areas, concave_areas = process_video_with_gaze_and_hulls(base_path, video_path)
    df = generate_area_dataframe_and_plot(frame_nums, convex_areas, concave_areas)
    st.dataframe(df)
