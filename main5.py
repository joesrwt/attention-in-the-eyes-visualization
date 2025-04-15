# ----------------------------
# Helper with Caching
# ----------------------------

@st.cache_resource(show_spinner=False)
def get_analysis(user, repo, folder, video_url, local_filename):
    if not os.path.exists(local_filename):
        download_video(video_url, local_filename)
    gaze = load_gaze_data(user, repo, folder)
    return analyze_gaze(gaze, local_filename)

# ----------------------------
# UI
# ----------------------------
st.title("🎯 Stay Focused or Float Away? : Focus-Concentration Analysis")

selected_video = st.selectbox("🎬 Select a video", list(video_files.keys()))

if selected_video:
    st.video(base_video_url + video_files[selected_video])

    folder = f"{clips_folder}/{selected_video}"
    video_filename = f"{selected_video}.mp4"
    video_url = base_video_url + video_files[selected_video]

    with st.spinner("Running analysis..."):
        df, frames = get_analysis(user, repo, folder, video_url, video_filename)
        st.session_state.df = df
        st.session_state.frames = frames
        st.session_state.frame = st.session_state.get("frame", int(df.index.min()))

# ----------------------------
# Results
# ----------------------------
if "df" in st.session_state:
    df = st.session_state.df
    frames = st.session_state.frames

    frame = st.slider(
        "🎞️ Select Frame", 
        int(df.index.min()), 
        int(df.index.max()), 
        st.session_state.frame,
        key="frame"  # persist selection in session state
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        data = df.reset_index().melt(id_vars="Frame", value_vars=[
            "Convex Area (Rolling)", "Concave Area (Rolling)"
        ], var_name="Metric", value_name="Area")
        chart = alt.Chart(data).mark_line().encode(
            x="Frame:Q", y="Area:Q", color="Metric:N"
        ).properties(width=600, height=300)
        rule = alt.Chart(pd.DataFrame({'Frame': [frame]})).mark_rule(color='red').encode(x='Frame')
        st.altair_chart(chart + rule, use_container_width=True)

    with col2:
        rgb = cv2.cvtColor(frames[frame], cv2.COLOR_BGR2RGB)
        st.image(rgb, caption=f"Frame {frame}", use_container_width=True)
        st.metric("F-C Score", f"{df.loc[frame, 'F-C score']:.3f}")
