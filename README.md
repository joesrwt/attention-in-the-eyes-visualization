```python
import streamlit as st

st.set_page_config(layout="wide")
st.title("🎯 Understanding Viewer Focus Through Gaze Visualization")

COLOR_GROUP1 = "#ECF0F1"   
COLOR_GROUP2 = "#F8F3EF"   

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
        caption="📊 Diagram: Convex vs Concave Hulls",width =320
    )
with col2:
    st.image(
        "https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/convex_concave_SIMPS_9a.gif",
        caption="🎥 Real Example: Gaze Boundaries Over Time"
    )
st.markdown(f"""
<div style="background-color: {COLOR_GROUP1}; padding: 20px; border-radius: 10px;">
</div>
""", unsafe_allow_html=True)

# SECTION 3: F-C Score
st.markdown(f"""
<div style="background-color: {COLOR_GROUP2}; padding: 20px; border-radius: 10px;">
    <h3>📊 Focus-Concentration (F-C) Score</h3>
</div>
""", unsafe_allow_html=True)

# Use HTML to control image height
st.markdown(f"""
<div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/formula_image.jpeg" 
         alt="🧮 Area calculation using a rolling average across the last 20 frames"
         style="height: 100px; border-radius: 10px;"/>
    <p><em>🧮 Area calculation using a rolling average across the last 20 frames</em></p>
</div>
""", unsafe_allow_html=True)


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

st.markdown(f"""
<div style="background-color: {COLOR_GROUP2}; padding: 20px; border-radius: 10px; margin-top: 1em;">
    <p>You’ll see this visualized dynamically in the graph and overlays as you explore different segments of the video.</p>
</div>
""", unsafe_allow_html=True)
