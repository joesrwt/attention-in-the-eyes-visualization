# 🎯 Understanding Viewer Focus Through Gaze Visualization

---

## 📌 What Captures Attention?
![Gaze Boundaries Over Time](https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/convex_concave_SIMPS_9a.gif)
Is the viewer’s attention firmly focused on key moments, or does it float, drifting between different scenes in search of something new?

This visualization explores how viewers engage with a video by examining **where and how they focus their attention**.

---

## 📐 How Do We Measure Focus?

We use geometric shapes to visualize how tightly the viewer’s gaze is grouped:

- **Convex Hull**: Encloses all gaze points loosely.  
- **Concave Hull**: Follows the actual shape of gaze, revealing true focus.

👉 The **difference in area** between the two tells us how spread out or concentrated the gaze is.

| Diagram: Convex vs Concave Hulls | Real Example: Gaze Boundaries Over Time |
|---------------------------------|-----------------------------------------|
| ![Convex vs Concave Hulls](https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/convex_concave_image.jpg) | ![Gaze Boundaries Over Time](https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/convex_concave_SIMPS_9a.gif) |

---

## 📊 Focus-Concentration (F-C) Score

<div align="center">

![Formula](https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/formula_image.jpeg)  
*🧮 Area calculation using a rolling average across the last 20 frames*

</div>

The **F-C Score** helps quantify gaze behavior:

- **Close to 1** → tight gaze cluster → 🟢 **high concentration**  
- **Much lower than 1** → scattered gaze → 🔴 **low concentration / exploration**

This metric reveals whether attention is **locked in** or **wandering**.

---

## 🎥 Visual Examples of Focus (Folder : FOODI_2a)

<table>
  <tr>
    <td align="center" width="50%">
      <img src="https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/FOODI_2a_high_F-C_score.gif" alt="High F-C Score" style="max-width:100%; border-radius:8px;">
      <p style="font-style: italic; color: #555;">High F-C : Gaze tightly grouped in one region.</p>
    </td>
    <td align="center" width="50%">
      <img src="https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/FOODI_2a_low_F-C_score.gif" alt="Low F-C Score" style="max-width:100%; border-radius:8px;">
      <p style="font-style: italic; color: #555;">Low F-C : Gaze jumps around, showing exploration or distraction.</p>
    </td>
  </tr>
</table>

You’ll see this visualized dynamically in the graph and overlays as you explore different segments of the video.

---

*Made with passion for understanding human attention through eye-tracking and visualization.*
