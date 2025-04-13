import os
import math
import cv2
import numpy as np
import pandas as pd
import scipy.io
import streamlit as st
import altair as alt
from scipy.spatial import ConvexHull, Delaunay
from shapely.geometry import MultiPoint, LineString, MultiLineString
from shapely.ops import polygonize, unary_union


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


def load_gaze_data(mat_files):
    gaze_data_per_viewer = []
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


def process_video_analysis(gaze_data_per_viewer, video_path, alpha=0.03, window_size=20):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Cannot open video.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_numbers = []
    convex_areas = []
    concave_areas = []

    frame_num = 0
    while cap.isOpened():
        ret, _ = cap.read()
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
                convex_area = ConvexHull(points).volume
            except:
                convex_area = 0
            try:
                concave = alpha_shape(points, alpha)
                concave_area = concave.area if concave and concave.geom_type == 'Polygon' else 0
            except:
                concave_area = 0

            frame_numbers.append(frame_num)
            convex_areas.append(convex_area)
            concave_areas.append(concave_area)

        frame_num += 1

    cap.release()

    df = pd.DataFrame({
        'Frame': frame_numbers,
        'Convex Area': convex_areas,
        'Concave Area': concave_areas
    })
    df.set_index('Frame', inplace=True)
    df['Convex Area (Rolling Avg)'] = df['Convex Area'].rolling(window=window_size, min_periods=1).mean()
    df['Concave Area (Rolling Avg)'] = df['Concave Area'].rolling(window=window_size, min_periods=1).mean()
    df['Score'] = (df['Convex Area (Rolling Avg)'] - df['Concave Area (Rolling Avg)']) / df['Convex Area (Rolling Avg)']
    df['Score'] = df['Score'].fillna(0)
    return df


# ========== Streamlit UI ==========

st.title("🎯 Gaze & Hull Analysis Tool")

uploaded_files = st.file_uploader("Upload your `.mat` gaze data and a `.mov` video", accept_multiple_files=True)

if uploaded_files:
    mat_files = [f for f in uploaded_files if f.name.endswith('.mat')]
    mov_files = [f for f in uploaded_files if f.name.endswith('.mov')]

    if not mat_files or not mov_files:
        st.warning("Please upload at least one `.mat` file and one `.mov` video.")
    else:
        st.success(f"✅ Loaded {len(mat_files)} .mat files and 1 video.")

        # Save uploaded files temporarily
        temp_dir = "temp_data"
        os.makedirs(temp_dir, exist_ok=True)

        mat_paths = []
        for file in mat_files:
            path = os.path.join(temp_dir, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            mat_paths.append(path)

        video_file = mov_files[0]
        video_path = os.path.join(temp_dir, video_file.name)
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())

        # Processing
        with st.spinner("Processing gaze data and computing hull areas..."):
            gaze_data = load_gaze_data(mat_paths)
            df = process_video_analysis(gaze_data, video_path)

        if df is not None and not df.empty:
            st.subheader("📊 Convex vs Concave Hull Area Over Time")

            frame_slider = st.slider("Select Frame", int(df.index.min()), int(df.index.max()), int(df.index.min()))

            df_melt = df.reset_index().melt(id_vars='Frame', value_vars=[
                'Convex Area', 'Concave Area', 
                'Convex Area (Rolling Avg)', 'Concave Area (Rolling Avg)'
            ], var_name='Metric', value_name='Area')

            chart = alt.Chart(df_melt).mark_line().encode(
                x='Frame',
                y='Area',
                color='Metric'
            )

            rule = alt.Chart(pd.DataFrame({'Frame': [frame_slider]})).mark_rule(color='red').encode(x='Frame')

            st.altair_chart(chart + rule, use_container_width=True)

            st.metric("Score at Selected Frame", f"{df.loc[frame_slider, 'Score']:.3f}")
