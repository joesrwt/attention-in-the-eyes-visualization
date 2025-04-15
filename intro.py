import streamlit as st

st.set_page_config(layout="wide")

st.title("🎯 Understanding Viewer Focus Through Gaze Visualization")

# Apply background color for Sections 1-2 (Light Cyan)
st.markdown(
    """
    <style>
    .section-1-2 {
        background-color: #e0f7fa;  /* Light Cyan */
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# SECTION 1: Hook
st.markdown('<div class="section-1-2">', unsafe_allow_html=True)
st.markdown("## 📌 What Captures Attention?")
st.markdown("""
Is the viewer’s attention firmly focused on key moments, or does it float, drifting between different scenes in search of something new?

This visualization explores how viewers engage with a video by examining **where and how they focus their attention**.
""")
st.markdown("---")
st.markdown('</div>', unsafe_allow_html=True)

# SECTION 2: Hull Concepts
st.markdown('<div class="section-1-2">', unsafe_allow_html=True)
st.markdown("## 📐 How Do We Measure Focus?")
st.markdown("""
We use geometric shapes to visualize how tightly the viewer’s gaze is grouped:

- **Convex Hull**: Encloses all gaze points loosely.
- **Concave Hull**: Follows the actual shape of gaze, revealing true focus.

👉 The **difference in area** between the two tells us how spread out or concentrated the gaze is.
""")

col1, col2 = st.columns(2)

with col1:
    st.image(
        "https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/convex_concave_image.jpg",
        caption="📊 Diagram: Convex vs Concave Hulls"
    )

with col2:
    st.image(
        "https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/convex_concave_SIMPS_9a.gif",
        caption="🎥 Real Example: Gaze Boundaries Over Time"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Apply background color for Sections 3-4 (Light Peach)
st.markdown(
    """
    <style>
    .section-3-4 {
        background-color: #fff3e0;  /* Light Peach */
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# SECTION 3: F-C Score
st.markdown('<div class="section-3-4">', unsafe_allow_html=True)
st.markdown("## 📊 Focus-Concentration (F-C) Score")

st.image(
    "https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/formula_image.jpeg",
    caption="🧮 Area calculation using a rolling average across the last 20 frames", width=900
)

st.markdown("""
The **F-C Score** helps quantify gaze behavior:

- **Close to 1** → tight gaze cluster → **high concentration**.
- **Much lower than 1** → scattered gaze → **low concentration**.

This metric reveals whether attention is **locked in** or **wandering**.
""")

st.markdown("---")
st.markdown('</div>', unsafe_allow_html=True)

# SECTION 4: Visual Examples
st.markdown('<div class="section-3-4">', unsafe_allow_html=True)
st.markdown("## 🎥 Visual Examples of Focus")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### High F-C Score")
    st.image("https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/FOODI_2a_high_F-C_score.gif")
    st.caption("Gaze remains tightly grouped in one region.")

with col2:
    st.markdown("### Low F-C Score")
    st.image("https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/FOODI_2a_low_F-C_score.gif")
    st.caption("Gaze jumps around, showing exploration or distraction.")

st.markdown("""
You’ll see this visualized dynamically in the graph and overlays as you explore different segments of the video.
""")
st.markdown('</div>', unsafe_allow_html=True)
