import streamlit as st

st.set_page_config(layout="wide")

# -------------------- 🎯 TITLE ---------------------
st.image("conclip/Image.jpeg", use_container_width=True)

# -------------------- 📌 SECTION 1 ---------------------
st.markdown("""
<div style="background-color: #DCEEFF; padding: 25px; border-radius: 10px; margin-top: 30px;">

<blockquote style="
    font-size: 1.1em;
    text-align: center;
    font-weight: bold;
    font-style: italic;
    border-left: 6px solid #95A5A6;
    background-color: #ECF0F1;
    padding: 1em;
    margin: 1.5em 0;
    border-radius: 6px;">
    “Is the viewer’s attention firmly focused on key moments,<br>
    or does it float, drifting between different scenes in search of something new?”
</blockquote>

<p style="font-size: 1.05em;">
    This visualization explores how viewers engage with a video by examining where and how they focus their attention.
</p>

</div>
""", unsafe_allow_html=True)

# -------------------- 📐 SECTION 2 ---------------------
st.markdown("""
<div style="background-color: #DCEEFF; padding: 25px; border-radius: 10px; margin-top: 30px;">

<h3>📐 How Do We Measure Focus?</h3>

<p style="font-size: 1.05em;">We use geometric shapes to visualize how tightly the viewer’s gaze is grouped:</p>

<ul style="font-size: 1.05em;">
    <li><b>Convex Hull</b>: Encloses all gaze points loosely.</li>
    <li><b>Concave Hull</b>: Follows the actual shape of gaze, revealing true focus.</li>
</ul>

<p style="font-size: 1.05em;">👉 The <b>difference in area</b> between the two tells us how spread out or concentrated the gaze is.</p>

<div style="display: flex; gap: 20px; justify-content: space-between;">
    <div style="width: 48%;">
        <img src="https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/convex_concave_image.jpg"
             style="width: 100%; border-radius: 8px;" alt="Diagram of Convex and Concave Hulls">
        <p style="font-size: 0.95em; text-align: center; color: #6c757d; margin-top: 8px;">
            📊 Diagram: Convex vs Concave Hulls
        </p>
    </div>
    <div style="width: 48%;">
        <img src="https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/convex_concave_SIMPS_9a.gif"
             style="width: 100%; border-radius: 8px;" alt="Gaze Boundaries Example">
        <p style="font-size: 0.95em; text-align: center; color: #6c757d; margin-top: 8px;">
            🎥 Real Example: Gaze Boundaries Over Time
        </p>
    </div>
</div>

</div>
""", unsafe_allow_html=True)
# -------------------- 📊 SECTION 3 ---------------------
st.markdown("""
<div style="background-color:#f3e5f5; padding: 25px; border-radius: 10px; margin-top: 30px;">

<h3>📊 Focus-Concentration (F-C) Score</h3>

<img src="https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/formula_image.jpeg"
     style="width: 100%; border-radius: 8px;" alt="Formula Example">

<p style="font-size: 0.95em; text-align: center; color: #6c757d; font-style: italic; margin-top: 8px;">
    🧮 Area calculation using a rolling average across the last 20 frames
</p>

<p style="font-size: 1.05em;">
    The <b>F-C Score</b> helps quantify gaze behavior:
</p>
<ul style="font-size: 1.05em;">
    <li><b>Close to 1</b> → tight gaze cluster → <span style="color:#2e7d32;"><b>high concentration</b></span>.</li>
    <li><b>Much lower than 1</b> → scattered gaze → <span style="color:#d32f2f;"><b>low concentration / exploration</b></span>.</li>
</ul>

<p style="font-size: 1.05em;">
    This metric reveals whether attention is <b>locked in</b> or <b>wandering</b>.
</p>

</div>
""", unsafe_allow_html=True)

# -------------------- 🎥 SECTION 4 ---------------------
st.markdown("""
<div style="background-color: #f3e5f5; padding: 25px; border-radius: 10px; margin-top: 30px;">

<div style="display: flex; gap: 20px;">
    <div style="width: 50%;">
        <h4>High F-C Score</h4>
        <img src="https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/FOODI_2a_high_F-C_score.gif"
             style="width: 100%; border-radius: 8px;">
        <p style="font-size: 0.95em; text-align: center; color: #6c757d; font-style: italic;">
            Gaze remains tightly grouped in one region.
        </p>
    </div>
    <div style="width: 50%;">
        <h4>Low F-C Score</h4>
        <img src="https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/FOODI_2a_low_F-C_score.gif"
             style="width: 100%; border-radius: 8px;">
        <p style="font-size: 0.95em; text-align: center; color: #6c757d; font-style: italic;">
            Gaze jumps around, showing exploration or distraction.
        </p>
    </div>
</div>

<p style="font-size: 1.05em; margin-top: 1.5em;">
    You’ll see this visualized dynamically in the graph and overlays as you explore different segments of the video.
</p>

</div>
""", unsafe_allow_html=True)
