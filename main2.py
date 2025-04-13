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
import alphashape


# Helper functions to calculate convex and concave areas (same as before)

@st.cache_data  # Use st.cache_data for loading gaze data
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
    
@st.cache_resource  # Use st.cache_resource for video processing and handling large files
# Modify the code for concave hull calculation
def process_video_analysis(gaze_data_per_viewer, video_path, alpha=0.007, window_size=20):
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
    video_frames = []  # Store frames

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
                # Use alphashape to get the concave hull
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
    df['Score'] = (df['Convex Area (Rolling Avg)'] - df['Concave Area (Rolling Avg)']) / df['Convex Area (Rolling Avg)']
    df['Score'] = df['Score'].fillna(0)

    return df, video_frames

# Streamlit UI

st.title("🎯 Gaze & Hull Analysis Tool")

# If data has already been processed, load it from session state
if st.session_state.data_processed:
    df = st.session_state.df
    video_frames = st.session_state.video_frames

    st.subheader("📊 Convex vs Concave Hull Area Over Time")

    # Create a slider for selecting frames
    frame_slider = st.slider("Select Frame", int(df.index.min()), int(df.index.max()), int(df.index.min()))

    # Melt the DataFrame to only include rolling averages
    df_melt = df.reset_index().melt(id_vars='Frame', value_vars=[
        'Convex Area (Rolling Avg)', 'Concave Area (Rolling Avg)'
    ], var_name='Metric', value_name='Area')

    # Set dynamic width for the chart based on the range of frames
    chart_width = 1200  # Default width

    # Create the Altair chart with specific colors for the rolling averages
    chart = alt.Chart(df_melt).mark_line().encode(
        x='Frame',
        y='Area',
        color=alt.Color('Metric:N', scale=alt.Scale(domain=['Convex Area (Rolling Avg)', 'Concave Area (Rolling Avg)'], range=['green', 'blue']))
    ).properties(width=chart_width)

    # Add a vertical line for the selected frame (accurately placed)
    rule = alt.Chart(pd.DataFrame({'Frame': [frame_slider]})).mark_rule(color='red').encode(x='Frame')

    # Display the chart with the vertical rule and dynamic width
    st.altair_chart(chart + rule, use_container_width=True)

    # Display the score for the selected frame
    st.metric("Score at Selected Frame", f"{df.loc[frame_slider, 'Score']:.3f}")

    # Display the selected video frame with a fixed width
    frame_rgb = cv2.cvtColor(video_frames[frame_slider], cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, caption=f"Frame {frame_slider}", width=700)  # Adjust width as needed
