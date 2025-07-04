# 🎯 Attention in the Eyes: Gaze Visualization Toolkit

## Overview

This repository presents a toolkit and example dataset for visualizing and analyzing human visual attention through eye-tracking data. Using geometric hull analysis and a novel Focus-Concentration (F-C) Score, it helps researchers and practitioners understand how viewers focus or explore within video scenes.

The project includes a Streamlit app with two main components:
- **Concept Visualization**: Introduces hull concepts and the F-C Score with illustrative examples.
- **Interactive Analysis**: Allows frame-by-frame exploration of gaze data overlaid on video clips with dynamic metrics.

---

## Motivation

> "Is the viewer’s attention firmly focused on key moments, or does it float, drifting between different scenes in search of something new?"

Traditional gaze heatmaps show where attention lands but not how concentrated or scattered it is. This toolkit addresses that gap by quantifying spatial gaze dispersion via hull areas and the Focus-Concentration metric.

---

## Core Concepts

- **Convex Hull**: A loose boundary enclosing all gaze points.
- **Concave Hull**: A tighter boundary following the actual shape of the gaze cluster.
- **Focus-Concentration (F-C) Score**:  
  \[
  \text{F-C Score} = \frac{\text{Area of Concave Hull}}{\text{Area of Convex Hull}}
  \]  
  - Score close to 1 → tightly focused gaze  
  - Score much less than 1 → scattered or exploratory gaze
---

## Dataset & Example Clips

Sample gaze datasets and video overlays included:

- folder : `FOODI_2a`, `SIMPS_9a`, `SHREK_3a`, and others   
- Gaze points, hull area computations, and processed videos available in `clips_folder/` and `processed_hull_area_overlay/`
