# 🤖 Face ID Lab — Biometric Face Recognition & Identification System

Face ID Lab is a web-based face recognition application built with Python, OpenCV, and Streamlit. The system allows users to register faces using webcam captures or uploaded images and identify registered individuals by comparing facial feature embeddings.

The application uses YuNet for face detection and SFace for generating 128-dimensional face embeddings. It also includes similarity-based matching and rejection rules to reduce incorrect identifications and classify uncertain cases as Unknown.

---
🚀 Live Demo

👉 Open Face ID Lab https://face-id-lab.streamlit.app

The application is deployed using Streamlit and can be accessed directly from a web browser.
## 🚀 How to Start

### Step 1: Install Dependencies
Open a terminal (PowerShell / Command Prompt) in this project folder:

```powershell
pip install -r requirements.txt
pip install -e .
```

### Step 2: Launch the Web Application
Double-click **`start.bat`**, or run:

```powershell
streamlit run app.py
```

The app opens automatically in your web browser at **[http://localhost:8501](http://localhost:8501)**.

---

## 🎯 How to Use the System

1. **Enroll Individuals**:
   - Navigate to **Enroll** in the sidebar.
   - Enter the subject's name (e.g. `Alice`).
   - Choose **📸 Webcam (Current Frame)** to snap a live photo, or **📁 Upload Image File** to pick a picture from your computer.
   - Click **Save Face to Database**.
   - *Tip*: Enrolling 2–3 photos of the same person under the same name (with slight head angle or lighting variations) improves recognition accuracy.
2. **Identify Faces**:
   - Navigate to **Identify** in the sidebar.
   - Snap a current frame from your webcam or upload a photo.
   - The system instantly detects the face, extracts its 128-D embedding, and compares it against your enrolled database:
     - **VERIFIED MATCH (Green)**: The person is identified with their name and cosine similarity score.
     - **ALERT: UNKNOWN (Red)**: Strangers, look-alikes, or poor-quality photos are safely rejected as **Unknown** with the exact security reason.
3. **Manage Enrolled People**:
   - Go to **People** to inspect registered identities and template counts, delete individuals, or clear the database.
4. **Tune Sensitivity Sliders**:
   - Adjust the **Cosine Threshold ($\tau$)** or **Ambiguity Margin ($\Delta$)** sliders in the sidebar with live effect.

---

## 🧬 System Architecture & Models

```
[Webcam / Upload]
       │
       ▼
[Stage 1: Face Detection (YuNet)] ──► Bounding Box (x, y, w, h) + 5 Key Landmarks
       │
       ▼
[Stage 2: Feature Alignment & Embedding (SFace)] ──► 128-D L2-Normalized Vector
       │
       ▼
[Stage 3: Cosine Similarity Matching] ──► 1:N Database Comparison
       │
       ▼
[Stage 4: 4-Stage Unknown Rejection] ──► Verified Identity OR Safe "Unknown"
```

| Component | Model / Method | Description |
| :--- | :--- | :--- |
| **Face Detector** | **YuNet** (`face_detection_yunet_2023mar.onnx`) | Lightweight CNN that detects faces and extracts 5 facial landmarks (eyes, nose, mouth corners). Automatically downloaded from OpenCV Zoo. |
| **Feature Embedder** | **SFace** (`face_recognition_sface_2021dec.onnx`) | Aligns face features via affine warp and produces a **128-dimensional unit vector** ($\|\mathbf{v}\|_2 = 1.0$). |
| **Similarity Metric** | **Cosine Similarity** ($[-1.0, 1.0]$) | Measures inner product between 128-D unit vectors in hyperspace. Higher scores mean greater similarity. |
| **Unknown Rejection** | **4-Stage Rule Engine** | Prevents false positives by enforcing security thresholds and ambiguity guardrails. |

---

## 🛡️ "Unknown" Rejection Rules & Thresholds

Rather than force-guessing an identity, the system safely triggers **Unknown** under 4 conditions:

1. **`below_threshold`**: Top cosine similarity is lower than cutoff ($\text{Score} < \tau = 0.363$). The person is an unregistered stranger.
2. **`ambiguous_margin`**: Top two matches are too close in score ($S_1 - S_2 < \Delta = 0.05$). Rejection prevents confusing look-alikes or twins.
3. **`empty_gallery`**: Triggered when no individuals have been enrolled yet.
4. **`no_face`**: Triggered when the detector finds zero faces in the image.

| Parameter | Default | Interpretation |
| :--- | :---: | :--- |
| **Threshold ($\tau$)** | **`0.363`** | Official OpenCV Zoo cutoff for SFace. Above 0.363 = genuine match; below 0.363 = stranger. |
| **Margin ($\Delta$)** | **`0.050`** | Minimum score separation between #1 and #2 candidate to rule out ambiguous pairs. |

---

## 📊 Essential Evaluation Results

Run the essential evaluation benchmark:

```powershell
python -m face_id evaluate
```
*(Or click **▶ Run Essential Evaluation** on the **Evaluation** page in the web app)*

### Benchmark Results Table

| Metric | Result | Target | Meaning |
| :--- | :---: | :---: | :--- |
| **Rank-1 Accuracy** | **100.0%** | $> 90\%$ | Enrolled individuals are identified correctly on genuine test probes. |
| **Unknown Rejection Rate** | **100.0%** | $> 98\%$ | 100% of strangers/unregistered probes are correctly rejected as **Unknown**. |
| **False Acceptance Rate (FAR)** | **0.00%** | $< 0.1\%$ | Zero impostors were mistakenly accepted as enrolled people. |
| **False Rejection Rate (FRR)** | **0.00%** | $< 5.0\%$ | Zero genuine probes were falsely rejected. |
| **Mean Genuine Score** | **0.4559** | $> 0.363$ | Comfortably above the acceptance threshold $\tau = 0.363$. |
| **Mean Impostor Score** | **0.0468** | $< 0.200$ | Safely separated from the threshold, ensuring strong security margins. |

---

## ⚠️ Real-World Failure Cases

1. **Extreme Head Angles**: Head turned $> 45^\circ$ or looking down causes landmarks to become occluded, distorting SFace alignment.
2. **Backlighting / Heavy Shadows**: Extreme backlight darkens facial contours and drops similarity scores by 0.10–0.20.
3. **Heavy Occlusions**: Sunglasses or masks prevent landmark detection.
4. **Look-Alikes & Identical Twins**: Two close relatives may both score $> 0.363$. The **Ambiguity Margin** rule safely marks them as **Unknown** rather than misidentifying.

---

## 🛠️ Practical Improvements

- **Multi-Angle Guided Enrollment**: Prompt user to enroll front, slight left, and slight right angles.
- **Quality Filtering**: Reject blurry or poorly lit images during enrollment using Laplacian blur variance.
- **Liveness Detection**: Add eye blink tracking or challenge-response head turns to defeat photo spoofing.
- **FAISS Vector Indexing**: For galleries with thousands of faces, accelerate nearest-neighbor retrieval with FAISS or HNSW.

---

## 🧪 Running Automated Tests

Run the core test suite:

```powershell
pytest -v
```

All 5 core tests verify:
- Gallery enrollment, NPZ compression, and disk reloading
- Genuine face matching above threshold
- Unknown stranger rejection below threshold
- Ambiguity margin look-alike rejection
- Biometric HUD landmark mesh rendering

---
