import os
import cv2
import altair as alt
import streamlit as st
from utils import load_gaze_data, download_video, analyze_gaze
import pandas as pd

st.set_page_config(page_title="Gaze Hull Visualizer", layout="wide")

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Concept Visualization", "📊 Interactive Analysis"])

# -------------------- PAGE 1: Conceptual Visualization --------------------
if page == "🏠 Concept Visualization":
    st.markdown("### Conceptual Visualization")
    st.markdown("To explore the conceptual visualization, click the link below:")
    st.markdown("[Go to Conceptual Visualization](https://infovisual-cbf7u8kncgbgspg2n7e52d.streamlit.app/)")

# -------------------- PAGE 2: Interactive Analysis --------------------

@st.cache_data
def get_gaze_and_analysis(user, repo, folder, video_url, video_filename):
    gaze = load_gaze_data(user, repo, folder)
    if not os.path.exists(video_filename):
        download_video(video_url, video_filename)
    return analyze_gaze(gaze, video_filename)

if page == "📊 Interactive Analysis":
    st.title("🎯 Stay Focused or Float Away? : Focus-Concentration Analysis")

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

    selected_video = st.selectbox("🎬 Select a video", list(video_files.keys()))

    if selected_video:
        st.video(base_video_url + video_files[selected_video])
        folder = f"{clips_folder}/{selected_video}"
        video_filename = f"{selected_video}.mp4"

        with st.spinner("Running analysis..."):
            df, frames = get_gaze_and_analysis(
                user, repo, folder, base_video_url + video_files[selected_video], video_filename
            )
            st.session_state.df = df
            st.session_state.frames = frames
            st.session_state.frame_min = int(df.index.min())
            st.session_state.frame_max = int(df.index.max())
            st.session_state.chart_data = df.reset_index().melt(
                id_vars="Frame", 
                value_vars=["Convex Area (Rolling)", "Concave Area (Rolling)"], 
                var_name="Metric", 
                value_name="Area"
            )

    if "df" in st.session_state:
        frame = st.slider(
            "🎞️ Select Frame", 
            st.session_state.frame_min, 
            st.session_state.frame_max, 
            st.session_state.frame_min
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            base_chart = alt.Chart(st.session_state.chart_data).mark_line().encode(
                x="Frame:Q", y="Area:Q", color="Metric:N"
            ).properties(width=600, height=300)
            rule = alt.Chart(pd.DataFrame({'Frame': [frame]})).mark_rule(color='red').encode(x='Frame:Q')
            st.altair_chart(base_chart + rule, use_container_width=True)

        with col2:
            rgb = cv2.cvtColor(st.session_state.frames[frame], cv2.COLOR_BGR2RGB)
            st.image(rgb, caption=f"Frame {frame}", use_container_width=True)
            st.metric("F-C Score", f"{st.session_state.df.loc[frame, 'F-C score']:.3f}")
