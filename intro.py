import streamlit as st

st.title("🎯 Understanding Viewer Focus Through Gaze Visualization")

with st.expander("📌 Goal of This Visualization", expanded=True):
    st.markdown("""
    Is the viewer’s attention firmly focused on key moments, or does it float, drifting between different scenes in search of something new?

    The goal of this visualization is to understand how viewers engage with a video by examining **where and how they focus their attention**. By comparing the areas where viewers look (represented by **convex and concave hulls**), the visualization highlights whether their attention stays focused on a specific part of the video or shifts around.

    Ultimately, this helps us uncover **patterns of focus and exploration**, providing insights into how viewers interact with different elements of the video.
    """)

with st.expander("📐 Explain Convex and Concave Concept"):
    st.markdown("""
    To analyze visual attention, we enclose gaze points with geometric boundaries:

    - **Convex Hull** wraps around all gaze points to show the overall extent of where viewers looked.
    - **Concave Hull** creates a tighter boundary that closely follows the actual shape of the gaze pattern, adapting to gaps and contours in the data.

    👉 **The difference in area between them reveals how dispersed or concentrated the viewers’ gaze is.**
    """)

with st.expander("📊 Focus-Concentration (F-C) Score"):
    st.markdown("""
    The **Focus Concentration Score (FCS)** quantifies how focused or scattered a viewer’s attention is during the video:

    - A score **close to 1** → gaze is tightly grouped → **high concentration**.
    - A score **much lower than 1** → gaze is more spread out → **lower concentration / more exploration**.

    It helps to measure whether attention is **locked onto a specific spot** or **wandering across the frame**.
    """)

with st.expander("🎥 Example: High vs Low F-C Score"):
    st.markdown("""
    - **High F-C Score**: The viewer’s gaze remains focused in one tight area, suggesting strong interest or attention.
    - **Low F-C Score**: The gaze is scattered, moving across many regions of the screen, indicating exploration or distraction.

    You can observe this difference visually in the graph and video overlays as you explore different frames.
    """)
