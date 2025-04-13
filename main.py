import os
import math
import cv2
import numpy as np
import pandas as pd
import scipy.io
import streamlit as st
import tempfile
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint, LineString, MultiLineString
from shapely.ops import polygonize, unary_union


# === Alpha Shape Function ===
def alpha_shape(points, alpha):
    if len(points) < 4:
        return MultiPoint(list(points)).convex_hull
    tri = Delaunay(points)
    edges = set()
    for ia, ib, ic in tri.simplices:
        pa, pb, pc = points[ia], points[ib], points[ic]
        a, b, c = np.linalg.norm(pb - pa), np.linalg.norm(pc - pb), np.linalg.norm(pa - pc)
        s = (a + b + c) / 2.0
        area = math.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))
        if area == 0:
            continue
        circum_r = a * b * c / (4.0 * area)
        if circum_r < 1.0 / alpha:
            edges.update([(ia, ib), (ib, ic), (ic, ia)])
    edge_points = [LineString([points[i], points[j]]) for i, j in edges]
    m = MultiLineString(edge_points)
    return unary_union(list(polygonize(m)))


# === Load Gaze Data ===
def load_gaze_data(base_path):
    gaze_data_per_viewer = []
    mat_files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.mat')]
    for mat_file in mat_files:
        mat = scipy.io.loadmat(mat_file)
        eyetrack = mat['eyetrackRecord']
        gaze_x = eyetrack['x'][0, 0].flatten()
        gaze_y = eyetrack['y'][0, 0].flatten()
        timestamps = eyetrack['t'][0, 0].flatten()
        valid = (gaze_x != -32768) & (gaze_y != -32768)
        gaze_x = gaze_x[valid]
        gaze_y = gaze_y[valid]
        timestamps = timestamps[valid] - timestamps[0]
        gaze_x_norm = gaze_x / np.max(gaze_x)
        gaze_y_norm = gaze_y / np.max(gaze_y)
        gaze_data_per_viewer.append((gaze_x_norm, gaze_y_norm, timestamps))
    return gaze_data_per_viewer


# === Preprocess Video and Generate Frames with Hulls ===
def preprocess_video_and_generate_frames(base_path, video_path, alpha=0.007):
    gaze_data_per_viewer = load_gaze_data(base_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_num = 0
    processed_frames = []
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
                    gx = int(np.clip(gaze_x_norm[i], 0, 1) * (w - 1))
                    gy = int(np.clip(gaze_y_norm[i], 0, 1) * (h - 1))
                    gaze_points.append((gx, gy))

        if len(gaze_points) >= 3:
            points = np.array(gaze_points)
            try:
                hull = ConvexHull(points)
                hull_pts = points[hull.vertices].reshape((-1, 1, 2))
                cv2.polylines(frame, [hull_pts], isClosed=True, color=(0, 255, 0), thickness=2)
            except:
                pass
            try:
                concave = alpha_shape(points, alpha)
                if concave and concave.geom_type == 'Polygon':
                    exterior = np.array(concave.exterior.coords).astype(np.int32)
                    cv2.polylines(frame, [exterior.reshape((-1, 1, 2))], isClosed=True, color=(255, 215, 0), thickness=2)
            except:
                pass

        # Save the frame as an image to list
        processed_frames.append(frame)
        frame_num += 1

    cap.release()
    return processed_frames


# === Streamlit App ===
st.title("Gaze & Hull Analysis Tool")

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
            processed_frames = preprocess_video_and_generate_frames(mat_dir, video_path)

            if processed_frames is not None:
                # Create Slider for Frame Selection
                num_frames = len(processed_frames)
                frame_slider = st.slider("Select Frame", min_value=0, max_value=num_frames-1, value=0, step=1)

                # Display the selected frame as an image
                st.image(processed_frames[frame_slider], caption=f"Frame {frame_slider}", use_container_width=True)

                # Display Score (based on your analysis, adjust as needed)
                st.write(f"**Frame {frame_slider}**")
                # You can calculate score as before, if needed:
                # score = calculate_score(frame_slider) 
                # st.write(f"Score: {score:.2f}")
                
                # Optional: Display a chart with convex vs concave area analysis (if available)
                # Example: 
                # chart_data = pd.DataFrame(...)
                # st.line_chart(chart_data)
