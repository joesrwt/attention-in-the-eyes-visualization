import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint
import alphashape

# Function to load gaze data from .mat files
@st.cache_data
def load_gaze_data(mat_dir):
    gaze_data_per_viewer = []
    for mat_file in os.listdir(mat_dir):
        if mat_file.endswith(".mat"):
            # Load gaze data from .mat file (use your custom loading function)
            mat_path = os.path.join(mat_dir, mat_file)
            mat = scipy.io.loadmat(mat_path)
            eyetrack = mat['eyetrackRecord']
            gaze_x = eyetrack['x'][0, 0].flatten()
            gaze_y = eyetrack['y'][0, 0].flatten()
            timestamps = eyetrack['t'][0, 0].flatten()
            valid = (gaze_x != -32768) & (gaze_y != -32768)
            gaze_x = gaze_x[valid]
            gaze_y = gaze_y[valid]
            timestamps = timestamps[valid] - timestamps[0]
            gaze_data_per_viewer.append((gaze_x, gaze_y, timestamps))
    return gaze_data_per_viewer

# Function to run hull analysis and plot the result
def run_hull_analysis_plot(mat_dir, video_path):
    gaze_data_per_viewer = load_gaze_data(mat_dir)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("❌ Cannot open video.")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_numbers = []
    video_frames = []

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gaze_points = []
        for gaze_x_norm, gaze_y_norm, timestamps in gaze_data_per_viewer:
            frame_indices = (timestamps / 1000 * fps).astype(int)
            if frame_num in frame_indices:
                idx = np.where(frame_indices == frame_num)[0]
                for i in idx:
                    gx = int(np.clip(gaze_x_norm[i], 0, 1) * (frame.shape[1] - 1))
                    gy = int(np.clip(gaze_y_norm[i], 0, 1) * (frame.shape[0] - 1))
                    gaze_points.append((gx, gy))

        if len(gaze_points) >= 3:
            points = np.array(gaze_points)
            try:
                # Compute convex hull
                hull = ConvexHull(points)
                hull_pts = points[hull.vertices].reshape((-1, 1, 2))
                cv2.polylines(frame, [hull_pts], isClosed=True, color=(0, 255, 0), thickness=2)
            except:
                pass

            try:
                # Compute concave hull (Alpha shape)
                concave = alphashape.alphashape(points, alpha=0.007)
                if concave and concave.geom_type == 'Polygon':
                    exterior = np.array(concave.exterior.coords).astype(np.int32)
                    cv2.polylines(frame, [exterior.reshape((-1, 1, 2))], isClosed=True, color=(0, 0, 255), thickness=2)
            except:
                pass

            # Mark gaze points with red circles
            for (gx, gy) in gaze_points:
                cv2.circle(frame, (gx, gy), radius=3, color=(255, 0, 0), thickness=-1)

        frame_numbers.append(frame_num)
        video_frames.append(frame)
        frame_num += 1

    cap.release()

    return pd.DataFrame({
        'Frame': frame_numbers,
        'Video Frames': video_frames
    }).set_index('Frame')

# === Streamlit App ===
st.title("🎯 Gaze & Hull Analysis Tool")

uploaded_files = st.file_uploader("Upload .mat files and one .mov file", type=['mat', 'mov'], accept_multiple_files=True)
if uploaded_files:
    with tempfile.TemporaryDirectory() as tmpdir:
        mat_dir = os.path.join(tmpdir, "mat_data")
        os.makedirs(mat_dir, exist_ok=True)
        video_path = None
        for uploaded in uploaded_files:
            file_path = os.path.join(mat_dir, uploaded.name)
            with open(file_path, "wb") as f:
                f.write(uploaded.getbuffer())
            if uploaded.name.endswith(".mov"):
                video_path = file_path
        if video_path:
            st.success("✅ Files uploaded successfully.")
            df = run_hull_analysis_plot(mat_dir, video_path)
            if df is not None:
                frame_slider = st.slider("Select Frame", min_value=df.index.min(), max_value=df.index.max(), value=df.index.min(), step=1)
                selected_frame = df.loc[frame_slider]
                frame_rgb = cv2.cvtColor(selected_frame['Video Frames'], cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption=f"Frame {frame_slider}", use_column_width=True)
