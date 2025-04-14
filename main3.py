import streamlit as st
import os
import cv2
import numpy as np
import scipy.io
import pandas as pd
from scipy.spatial import ConvexHull
import alphashape
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

st.set_page_config(layout="wide")
st.title("Gaze & Hull Analysis Tool")

# Upload section
video_file = st.file_uploader("Upload a .mov video file", type=["mov"])
data_files = st.file_uploader("Upload one or more .mat data files", type=["mat"], accept_multiple_files=True)

if video_file and data_files:
    # Save uploaded video to disk
    video_path = os.path.join("temp_video.mov")
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    # Load video
    cap = cv2.VideoCapture(video_path)
    video_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        video_frames.append(frame)
    cap.release()

    total_frames = len(video_frames)
    h, w, _ = video_frames[0].shape
    fps = 30  # Adjust if needed

    # Process gaze data
    gaze_data = []
    for file in data_files:
        mat = scipy.io.loadmat(file)
        gaze_x = mat['gaze_x'][0]
        gaze_y = mat['gaze_y'][0]
        timestamps = mat['timestamps'][0]

        # Normalize gaze
        norm_x = np.clip(gaze_x, 0, 1)
        norm_y = np.clip(gaze_y, 0, 1)

        gaze_data.append((norm_x, norm_y, timestamps))

    # Calculate hull areas and F-C score per frame
    convex_areas, concave_areas, scores = [], [], []
    for frame_idx in range(total_frames):
        gaze_points = []
        for gaze_x_norm, gaze_y_norm, timestamps in gaze_data:
            frame_indices = (timestamps / 1000 * fps).astype(int)
            if frame_idx in frame_indices:
                idx = np.where(frame_indices == frame_idx)[0]
                for i in idx:
                    gx = int(np.clip(gaze_x_norm[i], 0, 1) * (w - 1))
                    gy = int(np.clip(gaze_y_norm[i], 0, 1) * (h - 1))
                    gaze_points.append((gx, gy))

        if len(gaze_points) >= 3:
            points = np.array(gaze_points)
            try:
                convex = ConvexHull(points)
                convex_area = convex.area
            except:
                convex_area = 0
            try:
                concave = alphashape.alphashape(points, 0.007)
                concave_area = concave.area if concave else 0
            except:
                concave_area = 0
        else:
            convex_area = 0
            concave_area = 0

        score = concave_area - convex_area
        convex_areas.append(convex_area)
        concave_areas.append(concave_area)
        scores.append(score)

    df = pd.DataFrame({
        "Convex Area": convex_areas,
        "Concave Area": concave_areas,
        "F-C score": scores
    })

    st.session_state.video_frames = video_frames
    st.session_state.gaze_data = gaze_data
    st.session_state.df = df
    st.session_state.data_processed = True

if "data_processed" in st.session_state and st.session_state.data_processed:
    video_frames = st.session_state.video_frames
    gaze_data = st.session_state.gaze_data
    df = st.session_state.df
    total_frames = len(video_frames)
    fps = 30

    # Frame slider
    current_frame = st.slider("Select Frame", 0, total_frames - 1, 0)

    # Draw overlays like run_video_with_gaze_and_hulls
    frame = video_frames[current_frame].copy()
    h, w, _ = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX

    gaze_points = []
    for gaze_x_norm, gaze_y_norm, timestamps in gaze_data:
        frame_indices = (timestamps / 1000 * fps).astype(int)
        if current_frame in frame_indices:
            idx = np.where(frame_indices == current_frame)[0]
            for i in idx:
                gx = int(np.clip(gaze_x_norm[i], 0, 1) * (w - 1))
                gy = int(np.clip(gaze_y_norm[i], 0, 1) * (h - 1))
                gaze_points.append((gx, gy))
                cv2.circle(frame, (gx, gy), 4, (0, 0, 255), -1)

    if len(gaze_points) >= 3:
        points = np.array(gaze_points)
        centroid = np.mean(points, axis=0).astype(int)
        cv2.circle(frame, tuple(centroid), 6, (255, 0, 255), -1)

        try:
            hull = ConvexHull(points)
            hull_pts = points[hull.vertices].reshape((-1, 1, 2))
            cv2.polylines(frame, [hull_pts], isClosed=True, color=(0, 255, 0), thickness=2)
        except:
            pass

        try:
            concave = alphashape.alphashape(points, 0.007)
            if concave and concave.geom_type == 'Polygon':
                exterior = np.array(concave.exterior.coords).astype(np.int32)
                cv2.polylines(frame, [exterior.reshape((-1, 1, 2))], isClosed=True, color=(255, 215, 0), thickness=2)
        except:
            pass

    score = df.loc[current_frame, 'F-C score']

    # Legend (top right)
    legend_x, legend_y = w - 120, 10
    legend_w, legend_h = 160, 60
    font_scale = 0.38
    line_height = 14

    cv2.rectangle(frame, (legend_x, legend_y), (legend_x + legend_w, legend_y + legend_h), (255, 255, 255), -1)
    cv2.circle(frame, (legend_x + 10, legend_y + 15), 4, (0, 0, 255), -1)
    cv2.putText(frame, "Gaze Point", (legend_x + 20, legend_y + 19), font, font_scale, (0, 0, 0), 1)
    cv2.circle(frame, (legend_x + 10, legend_y + 15 + line_height), 5, (255, 0, 255), -1)
    cv2.putText(frame, "Centroid", (legend_x + 20, legend_y + 19 + line_height), font, font_scale, (0, 0, 0), 1)
    cv2.putText(frame, f"F-C score: {score:.2f}", (legend_x + 10, legend_y + 19 + 2 * line_height), font, font_scale, (0, 0, 0), 1)

    # Bottom-left legend
    bottom_legend_x, bottom_legend_y = 10, h - 50
    cv2.rectangle(frame, (bottom_legend_x, bottom_legend_y), (bottom_legend_x + 160, bottom_legend_y + 40), (255, 255, 255), -1)
    cv2.rectangle(frame, (bottom_legend_x, bottom_legend_y), (bottom_legend_x + 160, bottom_legend_y + 40), (255, 255, 255), 1)
    cv2.putText(frame, "Convex Hull Area", (bottom_legend_x + 25, bottom_legend_y + 15), font, font_scale, (0, 0, 0), 1)
    cv2.putText(frame, "Concave Hull Area", (bottom_legend_x + 25, bottom_legend_y + 30), font, font_scale, (0, 0, 0), 1)
    cv2.line(frame, (bottom_legend_x + 10, bottom_legend_y + 12), (bottom_legend_x + 20, bottom_legend_y + 12), (0, 255, 0), 2)
    cv2.line(frame, (bottom_legend_x + 10, bottom_legend_y + 27), (bottom_legend_x + 20, bottom_legend_y + 27), (255, 215, 0), 2)

    # Show video frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, caption=f"Frame {current_frame}", use_container_width=True)
    st.metric("Focus-Concentration Score", f"{score:.3f}")

    # Plot area graph
    fig, ax = plt.subplots()
    ax.plot(df.index, df['Convex Area'], label='Convex Area', color='green')
    ax.plot(df.index, df['Concave Area'], label='Concave Area', color='gold')
    ax.axvline(x=current_frame, color='blue', linestyle='--')
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Area')
    ax.set_title('Convex vs Concave Hull Area Over Time')
    ax.legend()
    st.pyplot(fig)
