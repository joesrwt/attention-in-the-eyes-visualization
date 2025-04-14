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

    # === 🧼 UPPER HALF: Video Player ===
    st.subheader("🎥 Uploaded Video")
    with open(video_path, 'rb') as video_file:
        video_bytes = video_file.read()
        st.video(video_bytes)

    st.markdown("---")

    # === 📊 LOWER HALF: Frame-wise Analysis ===
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
