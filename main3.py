import os
import math
import cv2
import numpy as np
import pandas as pd
import scipy.io
import streamlit as st
import altair as alt
import requests
from io import BytesIO
from scipy.spatial import ConvexHull
import alphashape
from shapely.geometry import MultiPoint

st.set_page_config(page_title="Gaze Hull Visualizer", layout="wide")

# ----------------------------
# SECTION: CONFIG
# ----------------------------
video_files = {
    "APPAL_2a": "APPAL_2a_hull_area.mp4",
    "FOODI_2a": "FOODI_2a_hull_area.mp4",
    "MARCH_12a": "MARCH_12a_hull_area.mp4",
    "NANN_3a": "NANN_3a_hull_area.mp4",
    "SHREK_3a": "SHREK_3a_hull_area.mp4",
    "SIMPS_19a": "SIMPS_19a_hull_area.mp4",
    "SIMPS_9a": "SIMPS_9a_hull_area.mp4",
    "SUND_36a_POR": "SUND_36a_POR_hull_area.mp4",
}

base_video_url = "https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/processed%20hull%20area%20overlay/"
user = "nutteerabn"
repo = "InfoVisual"
clips_folder = "clips_folder"

# ----------------------------
# SECTION: HELPERS
# ----------------------------
def list_mat_files_from_github_repo(user, repo, folder):
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents/{folder}"
    response = requests.get(api_url)
    if response.status_code != 200:
        st.error(f"Failed to list files from: {folder}")
        return []
    files = response.json()
    return [f["name"] for f in files if f["name"].endswith(".mat")]

def load_gaze_data_from_github(user, repo, folder):
    mat_files = list_mat_files_from_github_repo(user, repo, folder)
    gaze_data = []
    for filename in mat_files:
        raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/main/{folder}/{filename}"
        res = requests.get(raw_url)
        if res.status_code == 200:
            mat = scipy.io.loadmat(BytesIO(res.content))
            record = mat['eyetrackRecord']
            x = record['x'][0, 0].flatten()
            y = record['y'][0, 0].flatten()
            t = record['t'][0, 0].flatten()
            valid = (x != -32768) & (y != -32768)
            gaze_data.append({
                'x': x[valid] / np.max(x[valid]),
                'y': y[valid] / np.max(y[valid]),
                't': t[valid] - t[valid][0]
            })
        else:
            st.warning(f"Could not load {filename}")
    return [(d['x'], d['y'], d['t']) for d in gaze_data]

def process_video_analysis(gaze_data_per_viewer, video_path, alpha=0.007, window_size=20):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Cannot open video.")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_numbers, convex_areas, concave_areas, video_frames = [], [], [], []

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
                concave = alphashape.alphashape(points, alpha)
                concave_area = concave.area if concave.geom_type == 'Polygon' else 0
            except:
                concave_area = 0
        else:
            convex_area, concave_area = 0, 0

        frame_numbers.append(frame_num)
        convex_areas.append(convex_area)
        concave_areas.append(concave_area)
        video_frames.append(frame)
        frame_num += 1

    cap.release()

    df = pd.DataFrame({
        'Frame': frame_numbers,
        'Convex Area': convex_areas,
        'Concave Area': concave_areas
    }).set_index('Frame')

    df['Convex Area (Rolling Avg)'] = df['Convex Area'].rolling(window=window_size, min_periods=1).mean()
    df['Concave Area (Rolling Avg)'] = df['Concave Area'].rolling(window=window_size, min_periods=1).mean()
    df['F-C score'] = 1 - (df['Convex Area (Rolling Avg)'] - df['Concave Area (Rolling Avg)']) / df['Convex Area (Rolling Avg)']
    df['F-C score'] = df['F-C score'].fillna(0)

    return df, video_frames

# ----------------------------
# SECTION: UI
# ----------------------------
st.title("🎬 Gaze Hull Visualization")

selected_video = st.selectbox("🎥 เลือกวิดีโอ", list(video_files.keys()))
video_url = base_video_url + video_files[selected_video]
st.video(video_url)

if st.button("📊 Run Gaze Analysis"):
    with st.spinner("🔍 Downloading and processing gaze data..."):
        folder_path = f"{clips_folder}/{selected_video}"
        gaze_data = load_gaze_data_from_github(user, repo, folder_path)

        # Download video temporarily
        video_temp_path = f"{selected_video}_temp.mp4"
        if not os.path.exists(video_temp_path):
            video_response = requests.get(video_url)
            with open(video_temp_path, "wb") as f:
                f.write(video_response.content)

        df, video_frames = process_video_analysis(gaze_data, video_temp_path)
        if df is not None:
            st.session_state.df = df
            st.session_state.frames = video_frames
            st.session_state.frame_index = int(df.index.min())
            st.success("✅ Gaze analysis complete!")

# ----------------------------
# SECTION: DISPLAY RESULTS
# ----------------------------
if "df" in st.session_state and "frames" in st.session_state:
    df = st.session_state.df
    video_frames = st.session_state.frames
    frame_idx = st.slider("🎞️ Select Frame", int(df.index.min()), int(df.index.max()), st.session_state.frame_index)
    st.session_state.frame_index = frame_idx

    col1, col2 = st.columns([2, 1])

    # Chart
    with col1:
        df_melt = df.reset_index().melt(id_vars='Frame', value_vars=[
            'Convex Area (Rolling Avg)', 'Concave Area (Rolling Avg)'
        ], var_name='Metric', value_name='Area')
        chart = alt.Chart(df_melt).mark_line().encode(
            x='Frame',
            y='Area',
            color=alt.Color('Metric:N', scale=alt.Scale(range=['green', 'deepskyblue']))
        ).properties(width=600, height=300)
        rule = alt.Chart(pd.DataFrame({'Frame': [frame_idx]})).mark_rule(color='red').encode(x='Frame')
        st.altair_chart(chart + rule, use_container_width=True)

    # Frame & Metric
    with col2:
        rgb_frame = cv2.cvtColor(video_frames[frame_idx], cv2.COLOR_BGR2RGB)
        st.image(rgb_frame, caption=f"Frame {frame_idx}", use_container_width=True)
        st.metric("Focus-Concentration Score", f"{df.loc[frame_idx, 'F-C score']:.3f}")
