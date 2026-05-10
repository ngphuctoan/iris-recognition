# Plan: Academic Report Generation

This document outlines the step-by-step plan to fill in the academic report for the Iris Recognition System based on the template found in `report.docx`.

---

## Objective
Complete all chapters of the academic report, leveraging the actual implementation details, test results, and visualizations generated during the project. For screenshots, placeholder captions are provided to guide placement.

---

## Execution Plan

### Phase 1: Foundation and Setup (Chapters 1 and 2)
- **Task 1: Fill Chapter 1: Executive Summary**
  - Write a concise summary of the project goals and achievements.
  - Explain the choice of Daugman's algorithm for iris recognition.
  - State the use of the CASIA-Iris-Thousand dataset subset.
  - Highlight the achievement of 90% visual audit success and fully functional SSR web interface.
  - Mention the successful implementation of the feature extraction and matching pipeline.

- **Task 2: Fill Chapter 2: Lab Environment and Tools Setup**
  - **2.1 Hardware / Virtual Machines**:
    - Document the host OS (Linux/Fedora).
    - Mention local execution without heavy cloud dependencies.
  - **2.2 Software and Libraries**:
    - List Python 3.14+ as the core runtime.
    - Document FastAPI for the web interface.
    - Document OpenCV for computer vision tasks (Hough, Canny, Blur).
    - Document SQLModel for database ORM.
    - Document the use of `uv` as the high-performance package manager.
  - **2.3 Dataset Description**:
    - Describe the CASIA-Iris-Thousand dataset.
    - Specify the use of the local subset (subjects 000-049) for development and testing.
    - Note the image resolution and characteristics (near-infrared lighting).
  - **2.4 Network Configuration**:
    - Document the local server setup on `localhost:8000`.
    - Mention that no external network calls are required for the core system.

### Phase 2: Core Implementation and Code Analysis (Chapter 3)
- **Task 3: Fill Chapter 3: Core Implementation**
  - **3.1 System Architecture and Workflow**:
    - Detail the processing pipeline: Image Loading -> Preprocessing -> Segmentation -> Normalization -> Encoding -> Matching.
  - **3.2 Image Acquisition and Preprocessing**:
    - Explain the use of CLAHE (Contrast Limited Adaptive Histogram Equalization) in the API to enhance image contrast.
    - Document the median blur step to reduce noise before circle detection.
  - **3.3 Iris Segmentation**:
    - Explain the use of Hough Circles to find the pupil and limbus.
    - Document the custom Canny edge approach for finding the iris boundary.
    - Detail the anatomical radius caps enforced (Iris radius 1.5x - 3.8x of pupil radius).
  - **3.4 Iris Feature Extraction**:
    - Document the Rubber Sheet model implementation in `normalization.py`.
    - Explain the coordinate transformation from Cartesian to Polar.
  - **3.5 Iris Code Generation**:
    - Explain the 2D Gabor filter application.
    - Detail the generation of the 2-bit IrisCode (real and imaginary parts).
    - Explain the shape of the code (64, 1024) and the inclusion of the mask.
  - **3.6 Iris Matching and Authentication**:
    - Explain the fractional Hamming Distance calculation.
    - Document the use of bitwise XOR and AND operations for fast matching.
  - **3.7 Key Code Snippets and Security Analysis**:
    - Insert code snippets for `detect_eyelines` and Gabor application.
    - Analyze the computational efficiency of bitwise operations.

### Phase 3: Demonstration Walkthrough (Chapter 4)
- **Task 4: Fill Chapter 4: Step-by-Step Walkthrough**
  - **4.1 Step 1 -- System Initialization**:
    - Document running the FastAPI server and database initialization.
    - *[SCREENSHOT CAPTION]*: `Figure 4.1: Terminal output showing FastAPI server startup and database initialization.`
  - **4.2 Step 2 -- Input Eye Image**:
    - Document the enrollment page with the live image preview feature.
    - *[SCREENSHOT CAPTION]*: `Figure 4.2: Web interface showing the image upload area with live preview and metadata.`
  - **4.3 Step 3 -- Iris Detection Process**:
    - Show and explain the visual evidence of segmentation (Teal and Royal Blue circles).
    - Mention the yellow lines added for eyelid boundary detection.
    - *[SCREENSHOT CAPTION]*: `Figure 4.3: Ocular capture evidence showing correct segmentation of pupil and iris with yellow eyelash boundaries.`
  - **4.4 Step 4 -- Iris Code Extraction**:
    - Show the visualization of the generated IrisCode bitmask.
    - *[SCREENSHOT CAPTION]*: `Figure 4.4: Visualization of the generated IrisCode bitmask.`
  - **4.5 Step 5 -- Matching Against Database**:
    - Explain the process of iterating through stored templates.
  - **4.6 Step 6 -- Authentication Result Verification**:
    - Show the Hamming Heatmap visualization (Green for match, Red for mismatch).
    - *[SCREENSHOT CAPTION]*: `Figure 4.5: Hamming Match result showing the bit comparison heatmap.`

### Phase 4: Vulnerability Patch and Defense Mechanism (Chapter 5)
- **Task 5: Fill Chapter 5: Vulnerability Patch / Defense Mechanism**
  - **5.1 Potential Security Threats**:
    - Discuss the risk of presentation attacks (showing a printed photo of an eye).
  - **5.2 Biometric Spoofing Attacks**:
    - Detail how static images bypass the current system.
  - **5.3 Liveness Detection Mechanism**:
    - Propose solutions like checking for pupil dilation under light or high-frequency texture analysis.
  - **5.4 Secure Storage of Iris Templates**:
    - Note that current storage is unencrypted hex in SQLite.
    - Propose using cryptographic hashing or Zero-Knowledge Proofs for template storage.
  - **5.5 Future Security Improvements**:
    - Suggest adding network layer security (HTTPS) and API authentication.

### Phase 5: System Evaluation and Testing (Chapters 6, 7, and 8)
- **Task 6: Fill Chapter 6: System Evaluation and Testing**
  - **6.1 Accuracy Testing**:
    - Report the 90% success rate on the 10-sample visual audit.
    - *[SCREENSHOT CAPTION]*: `Figure 6.1: Diagnostic audit grid showing 10 random samples with segmentation overlays and yellow eyelines.`
  - **6.2 and 6.3 FAR and FRR**:
    - Explain the concepts and how they relate to the chosen Hamming Distance threshold (0.32).
  - **6.4 Performance Analysis**:
    - Note the computation time per image (usually under 1 second).
  - **6.5 Limitations**:
    - Mention sensitivity to low contrast or over-exposed images.

- **Task 7: Fill Conclusion and References**
  - Summarize the project achievements.
  - List academic references including Daugman's original papers and libraries used.
