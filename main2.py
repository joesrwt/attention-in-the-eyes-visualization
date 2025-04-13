# Streamlit UI

st.title("🎯 Gaze & Hull Analysis Tool")

# Check if the data is already in session_state
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False

# Form for uploading files
with st.form(key='file_upload_form'):
    uploaded_files = st.file_uploader("Upload your `.mat` gaze data and a `.mov` video", accept_multiple_files=True)
    submit_button = st.form_submit_button("Submit Files")

if submit_button:
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

            # Processing and caching
            with st.spinner("Processing gaze data and computing hull areas..."):
                gaze_data = load_gaze_data(mat_paths)
                df, video_frames = process_video_analysis(gaze_data, video_path)

                # Store processed data in session state
                st.session_state.df = df
                st.session_state.video_frames = video_frames
                st.session_state.data_processed = True

            st.success("✅ Data processing completed successfully!")

# If data has already been processed, load it from session state
if st.session_state.data_processed:
    df = st.session_state.df
    video_frames = st.session_state.video_frames

    st.subheader("📊 Convex vs Concave Hull Area Over Time")

    frame_slider = st.slider("Select Frame", int(df.index.min()), int(df.index.max()), int(df.index.min()))

    # Melt the DataFrame to only include rolling averages
    df_melt = df.reset_index().melt(id_vars='Frame', value_vars=[
        'Convex Area (Rolling Avg)', 'Concave Area (Rolling Avg)'
    ], var_name='Metric', value_name='Area')

    # Create the Altair chart with specific colors for the rolling averages
    chart = alt.Chart(df_melt).mark_line().encode(
        x='Frame',
        y='Area',
        color=alt.Color('Metric:N', scale=alt.Scale(domain=['Convex Area (Rolling Avg)', 'Concave Area (Rolling Avg)'], range=['green', 'blue']))
    )

    # Add a vertical line for the selected frame
    rule = alt.Chart(pd.DataFrame({'Frame': [frame_slider]})).mark_rule(color='red').encode(x='Frame')

    # Display the chart with the vertical rule
    st.altair_chart(chart + rule, use_container_width=True)

    # Display the score for the selected frame
    st.metric("Score at Selected Frame", f"{df.loc[frame_slider, 'Score']:.3f}")

    # Display the selected video frame
    st.image(video_frames[frame_slider], caption=f"Frame {frame_slider}", use_column_width=True)
