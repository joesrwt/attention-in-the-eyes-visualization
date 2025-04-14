import os
import math
import cv2
import numpy as np
import pandas as pd
import scipy.io
import streamlit as st
import altair as alt
from scipy.spatial import ConvexHull
import alphashape
from shapely.geometry import MultiPoint

# Helper function to load gaze data
@st.cache_data
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

@st.cache_resource
def process_video_analysis(gaze_data_per_viewer, video_path, alpha=0.007, window_size=20):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Cannot open video.")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_numbers = []
    convex_areas = []
    concave_areas = []
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
    })
    df.set_index('Frame', inplace=True)
    df['Convex Area (Rolling Avg)'] = df['Convex Area'].rolling(window=window_size, min_periods=1).mean()
    df['Concave Area (Rolling Avg)'] = df['Concave Area'].rolling(window=window_size, min_periods=1).mean()
    df['F-C score'] = 1 - (df['Convex Area (Rolling Avg)'] - df['Concave Area (Rolling Avg)']) / df['Convex Area (Rolling Avg)']
    df['F-C score'] = df['F-C score'].fillna(0)

    return df, video_frames

# Streamlit UI
st.title("🎯 Gaze & Hull Analysis Tool")

# Show uploaded video (if available)
if 'video_path' in st.session_state:
    st.video(st.session_state['video_path'])

if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0

# File upload form
with st.form(key='file_upload_form'):
    uploaded_files = st.file_uploader("Upload your `.mat` gaze data and a `.mp4` video", accept_multiple_files=True)
    submit_button = st.form_submit_button("Submit Files")

if submit_button:
    if uploaded_files:
        mat_files = [f for f in uploaded_files if f.name.endswith('.mat')]
        mp4_files = [f for f in uploaded_files if f.name.endswith('.mp4')]

        if not mat_files or not mp4_files:
            st.warning("Please upload at least one `.mat` file and one `.mp4` video.")
        else:
            st.success(f"✅ Loaded {len(mat_files)} .mat files and 1 video.")

            temp_dir = "temp_data"
            os.makedirs(temp_dir, exist_ok=True)

            mat_paths = []
            for file in mat_files:
                path = os.path.join(temp_dir, file.name)
                with open(path, "wb") as f:
                    f.write(file.getbuffer())
                mat_paths.append(path)

            video_file = mp4_files[0]
            video_path = os.path.join(temp_dir, video_file.name)
            with open(video_path, "wb") as f:
                f.write(video_file.getbuffer())

            # Save video path in session state
            st.session_state['video_path'] = video_path

            with st.spinner("Processing gaze data and computing hull areas..."):
                gaze_data = load_gaze_data(mat_paths)
                df, video_frames = process_video_analysis(gaze_data, video_path)

                if df is not None:
                    st.session_state.df = df
                    st.session_state.video_frames = video_frames
                    st.session_state.csv_path = os.path.join(temp_dir, "analysis.csv")
                    df.to_csv(st.session_state.csv_path)
                    st.session_state.data_processed = True
                    st.session_state.current_frame = int(df.index.min())
                    st.success("✅ Data processing completed successfully!")

# Display analysis
if st.session_state.data_processed:
    csv_path = st.session_state.get('csv_path')
    if csv_path and os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        df = pd.read_csv(csv_path, index_col='Frame')
    else:
        st.error("❌ Could not load the data. Please upload files and run the analysis again.")
        st.stop()

    video_frames = st.session_state.video_frames
    current_frame = st.session_state.current_frame
    min_frame, max_frame = int(df.index.min()), int(df.index.max())
    frame_increment = 10

    st.subheader("📊 Convex vs Concave Hull Area Over Time")

    # Frame slider
    new_frame = st.slider("Select Frame", min_frame, max_frame, current_frame)
    st.session_state.current_frame = new_frame

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.button("Previous <10"):
            st.session_state.current_frame = max(min_frame, st.session_state.current_frame - frame_increment)
    with col3:
        if st.button("Next >10"):
            st.session_state.current_frame = min(max_frame, st.session_state.current_frame + frame_increment)

    current_frame = st.session_state.current_frame

    # Prepare data for Altair chart
    df_melt = df.reset_index().melt(id_vars='Frame', value_vars=[
        'Convex Area (Rolling Avg)', 'Concave Area (Rolling Avg)'
    ], var_name='Metric', value_name='Area')

    chart = alt.Chart(df_melt).mark_line().encode(
        x='Frame',
        y='Area',
        color=alt.Color(
            'Metric:N',
            scale=alt.Scale(
                domain=['Convex Area (Rolling Avg)', 'Concave Area (Rolling Avg)'],
                range=['rgb(0, 210, 0)', 'rgb(0, 200, 255)']
            ),
            legend=alt.Legend(orient='bottom', title='Hull Type')
        )
    ).properties(
        width=500,
        height=300
    )

    rule = alt.Chart(pd.DataFrame({'Frame': [current_frame]})).mark_rule(color='red').encode(x='Frame')

    col_chart, col_right = st.columns([2, 1])

    with col_chart:
        st.altair_chart(chart + rule, use_container_width=True)

    with col_right:
        frame_rgb = cv2.cvtColor(video_frames[current_frame], cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption=f"Frame {current_frame}", use_container_width=True)
        st.metric("Focus-Concentration Score", f"{df.loc[current_frame, 'F-C score']:.3f}")
