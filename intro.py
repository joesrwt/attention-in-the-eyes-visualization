import streamlit as st

st.set_page_config(layout="wide")
st.title("🎯 Understanding Viewer Focus Through Gaze Visualization")

# กำหนดโทนสีแบบจับคู่
COLOR_GROUP1 = "#ECF0F1"   
COLOR_GROUP2 = "#F3EAE0"   

# SECTION 1: Hook
st.markdown(f"""
<div style="background-color: {COLOR_GROUP1}; padding: 20px; border-radius: 10px;">
    <h3>📌 What Captures Attention?</h3>
    <p>
    Is the viewer’s attention firmly focused on key moments, or does it float, drifting between different scenes in search of something new?
    </p>
    <p>
    This visualization explores how viewers engage with a video by examining <strong>where and how they focus their attention</strong>.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# SECTION 2: Hull Concepts
st.markdown(f"""
<div style="background-color: {COLOR_GROUP1}; padding: 20px; border-radius: 10px;">
    <h3>📐 How Do We Measure Focus?</h3>
    <p>
    We use geometric shapes to visualize how tightly the viewer’s gaze is grouped:
    </p>
    <ul>
        <li><strong>Convex Hull</strong>: Encloses all gaze points loosely.</li>
        <li><strong>Concave Hull</strong>: Follows the actual shape of gaze, revealing true focus.</li>
    </ul>
    <p>👉 The <strong>difference in area</strong> between the two tells us how spread out or concentrated the gaze is.</p>
</div>
""", unsafe_allow_html=True)

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

# SECTION 3: F-C Score
st.markdown(f"""
<div style="background-color: {COLOR_GROUP2}; padding: 20px; border-radius: 10px;">
    <h3>📊 Focus-Concentration (F-C) Score</h3>
</div>
""", unsafe_allow_html=True)

st.image(
    "https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/formula_image.jpeg",
    caption="🧮 Area calculation using a rolling average across the last 20 frames", width=900
)

st.markdown(f"""
<div style="background-color: #F3EAE0; padding: 15px; border-left: 5px solid #FFE4B5; margin-top: 10px;">
    <ul>
        <li><strong>Close to 1</strong> → tight gaze cluster → <span style="color:#2e7d32;"><strong>high concentration</strong></span></li>
        <li><strong>Much lower than 1</strong> → scattered gaze → <span style="color:#d32f2f;"><strong>low concentration</strong></span></li>
    </ul>
    <p>This metric reveals whether attention is <strong>locked in</strong> or <strong>wandering</strong>.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# SECTION 4: Visual Examples
st.markdown(f"""
<div style="background-color: {COLOR_GROUP2}; padding: 20px; border-radius: 10px;">
    <h3>🎥 Visual Examples of Focus</h3>
</div>
""", unsafe_allow_html=True)

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
<p style="margin-top: 1em;">
You’ll see this visualized dynamically in the graph and overlays as you explore different segments of the video.
</p>
""", unsafe_allow_html=True)
