# AR-VIZ-Project-Registration-and-Evaluation-code

This repository provides the Python scripts used in the study   **“\[Usability Study Impact of Augmented Reality Visualization Modalities on Localization Accuracy in the Head and Neck: Randomized Crossover Trial]”**, which investigated superimposed vs. adjacent holography in head and neck localization tasks.



\## Contents

\- `RANSAC ICP registration.py` – RANSAC + ICP registration of phantom scans to the planned head phantoms



\- `Planned Point Nearest Projection.py` – is to project the planned NEPs to the nearest points of head (optional), in this study we used blender to achieve this

\- `Planned Nerve Course Extraction.py` – is to extract curve from the planned nerve course

\- `Planned Salivary Gland Curve Extraction.py` – is to extract curve from the planned salivary glands



\- `Euclidean distance Calculation.py` – is to calculate the primary endpoint:Euclidean distance between the planned and drawn nerve end points (NEP)s.

\- `ASD \& HD Calculation.py`  – is to calculate the secondary endpoints:average surface distance (ASD) and Hausdorff distance (HD) between the planned and drawn NEPs.

\- `Relative Distance Calculation.py` -  is to calculate the secondary endpoint: relative distance between the planned and drawn NEPs

&nbsp; 

\- `NC GLD Soft tissue Thickness.py`– is to calculate the soft tissue thickness of the nerve courses and salivary glands

\- `NEP Soft tissue thickness.py`– is to calculate the soft tissue thicknessof the NEPs

&nbsp; 

\-  `Reform the Excel.py` is to reform the excel for R statistical processing

\## Requirements

\- Python 3.10  

\- numpy, scipy, open3d, blender  



\## Usage



