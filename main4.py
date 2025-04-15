import streamlit as st

# Page config
st.set_page_config(layout="wide", page_title="Focus Visualization")

# Sidebar Navigation with HTML anchor links
st.sidebar.title("🧭 Jump to Section")
st.sidebar.markdown("""
- [📌 Introduction](#introduction)
- [📐 Hull Concepts](#hull-concepts)
- [📊 F-C Score](#f-c-score)
- [🎥 Visual Examples](#visual-examples)
- [🎯 Interactive Analysis](#interactive-analysis)
""", unsafe_allow_html=True)

# Section 1: Introduction
st.markdown('<h2 id="introduction">📌 What Captures Attention?</h2>', unsafe_allow_html=True)
st.markdown("""
<div style="background-color: #ECF0F1; padding: 20px; border-radius: 10px;">
    Is the viewer’s attention firmly focused on key moments, or does it float, drifting between different scenes?
    This visualization explores how viewers engage with video by examining where and how they focus their attention.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Section 2: Hull Concepts
st.markdown('<h2 id="hull-concepts">📐 How Do We Measure Focus?</h2>', unsafe_allow_html=True)
st.markdown("""
<div style="background-color: #ECF0F1; padding: 20px; border-radius: 10px;">
    We use two geometric shapes:
    <ul>
        <li><strong>Convex Hull</strong>: Loosely wraps all gaze points.</li>
        <li><strong>Concave Hull</strong>: Closely follows actual shape of gaze data.</li>
    </ul>
    👉 The difference in area indicates how focused or scattered the viewer is.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.image("https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/convex_concave_image.jpg", caption="Convex vs Concave", width=320)
with col2:
    st.image("https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/convex_concave_SIMPS_9a.gif", caption="Real Example")

st.markdown("---")

# Section 3: F-C Score
st.markdown('<h2 id="f-c-score">📊 Focus-Concentration (F-C) Score</h2>', unsafe_allow_html=True)
st.markdown("""
<div style="background-color: #F8F3EF; padding: 20px; border-radius: 10px;">
    F-C Score = Convex Area / Concave Area (rolling avg over last 20 frames)
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/formula_image.jpeg" 
         style="height: 100px; border-radius: 10px;"/>
    <p><em>Rolling area calculation for smoother interpretation</em></p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Section 4: Visual Examples
st.markdown('<h2 id="visual-examples">🎥 Visual Examples of Focus</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### High F-C Score")
    st.image("https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/FOODI_2a_high_F-C_score.gif")
    st.caption("Gaze remains tightly grouped.")

with col2:
    st.markdown("### Low F-C Score")
    st.image("https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/FOODI_2a_low_F-C_score.gif")
    st.caption("Gaze shifts frequently between regions.")

st.markdown("---")

# Section 5: Interactive Analysis
st.markdown('<h2 id="interactive-analysis">🎯 Interactive Analysis</h2>', unsafe_allow_html=True)

# 👉 You can now reuse your interactive analysis logic here (from your original code block)
# e.g., video selection, loading gaze data, and visualization.
# You could wrap it in a function called show_interactive_analysis() to keep it modular.
