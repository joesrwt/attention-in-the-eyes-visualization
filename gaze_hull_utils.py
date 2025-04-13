# gaze_hull_utils.py
import os
import cv2
import math
import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
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

def process_video_with_gaze_and_hulls(base_path, video_path, alpha=0.03):
    mat_files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.mat')]
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

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], [], [], None

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = "output_with_hulls.mp4"
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_numbers, convex_areas, concave_areas = [], [], []
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gaze_points_in_frame = []
        for gaze_x_norm, gaze_y_norm, timestamps in gaze_data_per_viewer:
            frame_indices = (timestamps / 1000 * fps).astype(int)
            if frame_num in frame_indices:
                idx = np.where(frame_indices == frame_num)[0]
                for i in idx:
                    gx = int(np.clip(gaze_x_norm[i], 0, 1) * (w - 1))
                    gy = int(np.clip(gaze_y_norm[i], 0, 1) * (h - 1))
                    gaze_points_in_frame.append((gx, gy))

        for x, y in gaze_points_in_frame:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        if len(gaze_points_in_frame) >= 3:
            points = np.array(gaze_points_in_frame)
            try:
                hull = ConvexHull(points)
                hull_pts = points[hull.vertices].reshape((-1, 1, 2))
                cv2.polylines(frame, [hull_pts], isClosed=True, color=(0, 255, 0), thickness=2)
                convex_area = hull.volume
            except:
                convex_area = 0
            try:
                concave = alpha_shape(points, alpha=alpha)
                if concave and concave.geom_type == 'Polygon':
                    exterior = np.array(concave.exterior.coords).astype(np.int32)
                    cv2.polylines(frame, [exterior.reshape((-1, 1, 2))], isClosed=True, color=(255, 215, 0), thickness=2)
                    concave_area = concave.area
                else:
                    concave_area = 0
            except:
                concave_area = 0
            frame_numbers.append(frame_num)
            convex_areas.append(convex_area)
            concave_areas.append(concave_area)

        out.write(frame)
        frame_num += 1

    cap.release()
    out.release()
    return frame_numbers, convex_areas, concave_areas, out_path

def generate_area_dataframe_and_plot(frame_numbers, convex_areas, concave_areas, window_size=20):
    df = pd.DataFrame({
        'Frame': frame_numbers,
        'Convex Area': convex_areas,
        'Concave Area': concave_areas
    })
    df.set_index('Frame', inplace=True)
    df['Convex Area (Rolling Avg)'] = df['Convex Area'].rolling(window=window_size, min_periods=1).mean()
    df['Concave Area (Rolling Avg)'] = df['Concave Area'].rolling(window=window_size, min_periods=1).mean()
    df['Score'] = (df['Convex Area (Rolling Avg)'] - df['Concave Area (Rolling Avg)']) / df['Convex Area (Rolling Avg)']

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['Convex Area'], alpha=0.3, label='Convex Area (Raw)', color='green')
    ax.plot(df.index, df['Concave Area'], alpha=0.3, label='Concave Area (Raw)', color='blue')
    ax.plot(df.index, df['Convex Area (Rolling Avg)'], label=f'Convex Area (Avg)', color='darkgreen', linewidth=2)
    ax.plot(df.index, df['Concave Area (Rolling Avg)'], label=f'Concave Area (Avg)', color='navy', linewidth=2)
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Area (px²)")
    ax.set_title("Convex vs Concave Hull Area Over Time")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
