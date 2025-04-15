import os
import scipy.io
import numpy as np
import cv2
from scipy.spatial import ConvexHull, Delaunay
from shapely.geometry import MultiPoint, LineString, MultiLineString
from shapely.ops import unary_union, polygonize

# โหลด gaze data
def load_gaze_data_from_folder(folder_path):
    gaze_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".mat"):
            data = scipy.io.loadmat(os.path.join(folder_path, filename))
            record = data['eyetrackRecord']
            x = record['x'][0, 0].flatten()
            y = record['y'][0, 0].flatten()
            t = record['t'][0, 0].flatten()
            valid = (x != -32768) & (y != -32768)
            gaze_data.append({
                'x': x[valid] / np.max(x[valid]),
                'y': y[valid] / np.max(y[valid]),
                't': t[valid] - t[valid][0]
            })
    return gaze_data

# ฟังก์ชันสร้าง concave hull
def alpha_shape(points, alpha=0.03):
    if len(points) < 4:
        return MultiPoint(points).convex_hull

    try:
        tri = Delaunay(points, qhull_options='QJ')
    except:
        return MultiPoint(points).convex_hull

    edges = set()
    edge_points = []

    for ia, ib, ic in tri.simplices:
        pa, pb, pc = points[ia], points[ib], points[ic]
        a, b, c = np.linalg.norm(pb - pa), np.linalg.norm(pc - pb), np.linalg.norm(pa - pc)
        s = (a + b + c) / 2.0
        area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))
        if area == 0:
            continue
        circum_r = a * b * c / (4.0 * area)
        if circum_r < 1.0 / alpha:
            edges.update([(ia, ib), (ib, ic), (ic, ia)])

    for i, j in edges:
        edge_points.append(LineString([points[i], points[j]]))

    mls = MultiLineString(edge_points)
    return unary_union(polygonize(mls))

# วาด convex และ concave hull ลงบน frame
def draw_hulls_on_frame(frame, gaze_points, alpha=0.03):
    if len(gaze_points) < 3:
        return frame  # ไม่พอวาด

    points = np.array(gaze_points)
    points = np.unique(points, axis=0)

    # convex
    try:
        hull = ConvexHull(points)
        hull_pts = points[hull.vertices].reshape((-1, 1, 2))
        cv2.polylines(frame, [hull_pts], isClosed=True, color=(0, 0, 255), thickness=2)  # 🔴 Red
    except:
        pass

    # concave
    try:
        concave = alpha_shape(points, alpha=alpha)
        if concave and concave.geom_type == 'Polygon':
            coords = np.array(concave.exterior.coords).astype(np.int32)
            cv2.polylines(frame, [coords.reshape((-1, 1, 2))], isClosed=True, color=(255, 0, 0), thickness=2)  # 🔵 Blue
    except:
        pass

    return frame
