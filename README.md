# CellScope — Medical Computer Vision & Distributed Inference Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![RabbitMQ](https://img.shields.io/badge/RabbitMQ-3.12-FF6600.svg)](https://www.rabbitmq.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.0-47A248.svg)](https://www.mongodb.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com/)

**CellScope** is an end-to-end medical computer vision system for automated nuclei detection, instance segmentation, and cell-level morphometric quantification from multi-modal microscopy images. Built to bridge research-grade deep learning with clinical production systems, CellScope couples a PyTorch U-Net and classical marker-controlled watershed segmentation with an asynchronous **FastAPI + RabbitMQ + MongoDB** microservice and an interactive analytical web dashboard.

The repository is organized into two distinct, decoupled phases:
- **[Phase 1 (Core CV & Quantification)](#phase-1):** Standalone data processing, U-Net training with composite BCE + Soft Dice loss on Google Colab (Tesla T4 GPU), marker-controlled watershed segmentation, and cell-level morphometry.
- **[Phase 2 (Async Serving, UI & Microservice)](#phase-2):** Non-blocking FastAPI REST gateway, RabbitMQ task queue, MongoDB persistence, interactive analytics web dashboard, and Docker Compose orchestration.

---

## 🌟 Key Features

### 🔬 Core Computer Vision & Morphometry
- **Multi-Modal Histology Ingestion:** Preprocesses diverse microscopy modalities (fluorescence, brightfield, H&E stains) from the 2018 Data Science Bowl into unified single-channel 16-bit instance label masks.
- **Deep Segmentation (U-Net):** Vanilla encoder-decoder U-Net trained with a composite binary cross-entropy (BCE) and soft Dice loss function to handle severe foreground-background class imbalance.
- **Marker-Controlled Watershed Separation:** Computes Euclidean Distance Transforms (EDT) and regional morphological peaks to isolate seeds and cleanly resolve touching or overlapping nuclei boundaries.
- **Clinical Morphometric Phenotyping:** Extracts individual cell area ($\text{px}$ & $\mu\text{m}^2$), perimeter ($\mu\text{m}$), circularity ($4\pi A / P^2$), eccentricity via fitted ellipses, and spatial centroid coordinates.
- **Tissue-Level Spatial Density & Clustering:** Computes overall cellular density (cells/$\text{mm}^2$) and spatial Nearest Neighbor Distance (NND) distributions via k-d tree spatial partitioning (`scipy.spatial.cKDTree`).

### ⚡ Production & Microservice Capabilities
- **Non-Blocking Asynchronous Gateway:** Heavy image processing is decoupled from the HTTP cycle. Image uploads return immediately with an HTTP 202 status and a unique UUID job ID.
- **Worker Singleton Model Caching:** PyTorch U-Net weights are loaded into GPU/CPU memory strictly **once** at worker boot, completely eliminating multi-second per-request disk I/O and weight reconstruction.
- **Poison-Pill Immunity & Resilience:** Malformed inputs or corrupted images are caught by worker exception handlers, logged with tracebacks, atomically recorded as `"failed"` in MongoDB, and **guaranteed ACKed** to prevent broken tasks from stalling the queue in an infinite retry loop.
- **Dynamic Host Resolution:** Intelligent environment detection (`get_mongo_uri()`, `get_rabbitmq_uri()`) transparently switches between internal Docker DNS (`mongodb:27017`, `rabbitmq:5672`) and host localhost (`localhost:27017`, `localhost:5672`) without configuration overrides.
- **Interactive Web Dashboard :** Built-in zero-dependency single-page application served directly at `http://localhost:8000/` featuring drag-and-drop uploads, 1-click sample slide evaluation, real-time pipeline status tracking, side-by-side segmentation overlays, interactive canvas size distribution histograms, shape morphometry scatter plots with hover tooltips, and CSV export.

---

## 🏗️ System Architecture

```
                                      [ Client / Browser / Doctor ]
                                                    │
                                     1. POST /jobs (Microscopy Image)
                                     2. Open http://localhost:8000/ (Dashboard)
                                                    │
                                                    ▼
                                     ┌─────────────────────────────┐
                                     │      FastAPI Gateway        │
                                     │  - Static Web UI (Port 8000)│
                                     │  - REST Endpoints           │
                                     └──────┬───────────────┬──────┘
                                            │               │
                         Insert Job Record  │               │ Publish Persistent Message
                        (Status: "queued")  │               │ {job_id, image_path}
                                            ▼               ▼
                                   ┌──────────────┐   ┌───────────────────┐
                                   │   MongoDB    │   │  RabbitMQ Broker  │
                                   │ (Port 27017) │   │    (Port 5672)    │
                                   └──────▲───────┘   └─────────┬─────────┘
                                          │                     │
                                          │                     │ Consume Job Task
                                          │                     ▼
                                          │        ┌─────────────────────────────┐
                                          │        │   Inference Worker Engine   │
                                          │        │  - Singleton U-Net Preloaded│
                                          │        │  - Distance Transform & WS  │
                                          │        │  - Morphometry Extraction   │
                                          │        └────────────┬────────────────┘
                                          │                     │
                       Update Status to   │                     │ Save Overlay & Features
                     "done" with Summary  └─────────────────────┤
                     and Nuclei Features                        ▼
                                                   ┌─────────────────────────────┐
                                                   │    Shared Storage Volume    │
                                                   │  /data/uploads   (Raw Slide)│
                                                   │  /data/overlays  (Overlay)  │
                                                   └─────────────────────────────┘
```

---

## 📂 Repository Structure

```
cellscope/
├── phase1_cv/                                # Phase 1: Model Training & Evaluation
│   ├── cellscope_phase1_results/             # Phase 1 Model Deliverables & Metrics
│   │   ├── eval_metrics.json                 # Quantitative metrics on test slides
│   │   ├── train_log.csv                     # Epoch-by-epoch training history log
│   │   ├── training_curves.png               # Loss, Dice, and IoU progression plots
│   │   └── unet_nuclei_best.pt               # Trained U-Net model checkpoint 
│   ├── cv.ipynb                              # Google Colab training notebook
├──phase2_service/                            # Phase 2: Async Serving & Microservice
│   ├── app/                                  # Serving application package
│   │   ├── db.py                             # MongoDB async and sync (PyMongo) layer
│   │   ├── inference.py                      # Serving model lifecycle & singleton inference
│   │   ├── main.py                           # FastAPI application & UI router
│   │   ├── schemas.py                        # Pydantic data schemas 
│   │   └── worker.py                         # RabbitMQ consumer process
│   ├── cv/                                   # Standalone computer vision modules
│   │   ├── infer_single.py                   # Standalone inference runner & key adapter
│   │   ├── postprocess.py                    # Watershed segmentation & feature extraction
│   │   └── unet.py                           # Vanilla U-Net neural network architecture
│   ├── docker/                               # Containerization manifests
│   │   ├── Dockerfile.api                    # FastAPI container build
│   │   ├── Dockerfile.worker                 # Worker container (PyTorch CPU + OpenCV)
│   │   └── docker-compose.yml                # Multi-container orchestration
│   ├── frontend/                             # Single-page application web dashboard
│   │   └── index.html                        # Interactive dashboard with canvas charts
│   ├── models/                               # Local model artifact for serving
│       └── unet_nuclei_best.pt               # Model weights reference
|
├── models/                                   # Root model weight directory
│   └── unet_nuclei_best.pt                   # Serving model checkpoint
├── requirements.txt                          # Unified repository requirements
├── README.md                                 # Project documentation
└── .gitignore                     
```

<a id="phase-1"></a>
## 🔬 Phase 1 — Core Computer Vision & Results

Phase 1 establishes the mathematical image processing and deep learning pipeline. Training is orchestrated via the Google Colab notebook [`phase1_cv/cv.ipynb`](phase1_cv/cv.ipynb).

### 📓 Google Colab Training Notebook
- **Notebook Path:** [`phase1_cv/cv.ipynb`](phase1_cv/cv.ipynb)
- **Dataset:** data-science-bowl-2018 training data from `data-science-bowl-2018.zip` 
- **Reproducible Split:** Splits multi-modal microscopy images into **70% Train**, **15% Validation**, and **15% Held-Out Test** using seed `42`.
- **Optimization Strategy:** Vanilla U-Net trained with mixed-precision arithmetic, composite BCE + Soft Dice Loss, Adam optimizer ($1 \times 10^{-4}$), and `ReduceLROnPlateau` scheduler over 50 epochs.

### 📊 Evaluation Deliverables
All Phase 1 evaluation metrics, training logs, curves, and weights are in [`phase1_cv/cellscope_phase1_results/`](phase1_cv/cellscope_phase1_results/):
- **Metrics JSON:** [`phase1_cv/cellscope_phase1_results/eval_metrics.json`](phase1_cv/cellscope_phase1_results/eval_metrics.json)
- **Training Log:** [`phase1_cv/cellscope_phase1_results/train_log.csv`](phase1_cv/cellscope_phase1_results/train_log.csv)
- **Loss & Dice Curves:** [`phase1_cv/cellscope_phase1_results/training_curves.png`](phase1_cv/cellscope_phase1_results/training_curves.png)
- **Model Checkpoint:** [`phase1_cv/cellscope_phase1_results/unet_nuclei_best.pt`](phase1_cv/cellscope_phase1_results/unet_nuclei_best.pt)

### 📈 Quantitative Performance Summary

| Metric | Split      | Value Achieved | Clinical Relevance |
| :--- | :--- | :--- | :--- |
| **Best Validation Dice** | Validation (100 slides) | **`0.9073` (90.7%)** | High boundary conformity on unseen validation images |
| **Best Validation IoU** | Validation (100 slides) | **`0.8371` (83.7%)** | High overlap area ratio between prediction and ground truth |
| **Mean Test Pixel Dice** | Held-Out Test (102 slides) | **`0.8686` (86.9%)** | Generalizes across varied staining types and contrast levels |
| **Mean Test Pixel IoU** | Held-Out Test (102 slides) | **`0.7843` (78.4%)** | Strong semantic segmentation overlap |
| **Matched Instance $F_1$** | Held-Out Test (@ IoU $\ge 0.5$) | **`0.6820` (68.2%)** | Accurate discrete nucleus identification and counting |

---

<a id="phase-2"></a>
## 🖥️ Phase 2 — Service Architecture & Interactive UI

Phase 2 wraps the trained model in an asynchronous, containerized microservice and serves an interactive web dashboard.

### 🎨 Interactive Web Dashboard Features:
1. **Live Service Health Badges:** Real-time polling indicators for **API Gateway**, **RabbitMQ**, **MongoDB**, and **Worker U-Net**.
2. **Microscopy Input & 1-Click Testing:** Drag & drop zone supporting `.png`, `.jpg`, `.jpeg`, and `.tif`, plus a **"Load Sample Histology Slide"** button to immediately run `sample.png`.
3. **Animated Pipeline Stepper:** Step-by-step visual tracker (`Upload` $\rightarrow$ `Queued` $\rightarrow$ `Processing` $\rightarrow$ `Completed`) with a live elapsed execution timer.
4. **Diagnostic Comparison Viewer:** Toggle between **Side-by-Side** (original slide vs colorized watershed segmentation overlay), **Overlay Only**, and **Original Only**, with a 1-click **Download Overlay PNG** action.
5. **Quantitative Summary Cards:** Immediate display of Detected Nuclei Count, Cell Density (/ $\text{mm}^2$), Mean Area ($\mu\text{m}^2 \pm \text{std}$), Circularity Index, Eccentricity, and Mean Nearest Neighbor Distance.
6. **Interactive Canvas Visual Analytics:**
   - **Nuclear Area Distribution:** Binned frequency histogram with a vertical dashed marker highlighting the mean nuclear area and interactive hover tooltips.
   - **Shape Morphometry Scatter Plot:** Circularity vs Eccentricity scatter plot with coordinate axes and point hover tooltips showing individual cell IDs, area, and shape indices.
7. **Sortable Feature Table & CSV Export:** Searchable, sortable table of all detected cell instances with an **"Export CSV"** button.

### 🌐 REST API Endpoints

| Method | Endpoint | Description | Status Code |
| :--- | :--- | :--- | :--- |
| `GET` | `/` or `/ui` | Serves the interactive Single Page Application web dashboard | `200 OK` |
| `GET` | `/sample.png` | Serves bundled sample histology slide for 1-click testing | `200 OK` |
| `GET` | `/health` | Liveness and readiness probe for MongoDB, RabbitMQ, and Model | `200 OK` |
| `POST` | `/jobs` | Submits a microscopy slide for asynchronous processing | `202 Accepted` |
| `GET` | `/jobs/{id}` | Polls current status (`queued`, `processing`, `done`, `failed`) | `200 OK` |
| `GET` | `/jobs/{id}/result` | Retrieves morphological features and summary metrics (409 if active) | `200 OK` / `409` |
| `GET` | `/jobs/{id}/overlay`| Serves the colorized instance segmentation PNG overlay | `200 OK` |

---

## 🚀 Setup & Execution Guidelines

You can run CellScope using **Docker Compose (Recommended)** or directly in a **Local Python Virtual Environment**.

### Method 1: Run with Docker Compose (Recommended)

Docker Compose coordinates all 4 services (**FastAPI**, **RabbitMQ**, **MongoDB**, and **Worker**) with a single command.

#### Step 1: Clone 
```powershell
git clone https://github.com/AjayChikate/CellScopeCV
```

#### Step 2: Build and Launch Containers
```bash
docker compose -f phase2_service/docker/docker-compose.yml up --build
```
> **Tip:** Add `-d` to run in detached background mode:
> ```bash
> docker compose -f phase2_service/docker/docker-compose.yml up --build -d
> ```

#### Step 3: Access Running Services
- **Interactive Web Dashboard:** [http://localhost:8000/](http://localhost:8000/)
- **Swagger Interactive API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **RabbitMQ Management Portal:** [http://localhost:15672](http://localhost:15672) *(User: `guest`, Password: `guest`)*
- **MongoDB:** `localhost:27017`

#### Step 4: Stop Containers
```bash
docker compose -f phase2_service/docker/docker-compose.yml down
```

---

### Method 2: Run Locally (Python Environment)

If you prefer running Python directly on your host machine:

#### Step 1: Create and Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate in Windows PowerShell
.venv\Scripts\Activate.ps1

# Or in Windows CMD
.venv\Scripts\activate.bat
```

#### Step 2: Install Unified Requirements
```bash
pip install -r requirements.txt
```

#### Step 3: Start MongoDB & RabbitMQ Brokers
Run the message broker and database containers via Docker:
```bash
docker run -d --name cellscope-rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management
docker run -d --name cellscope-mongodb -p 27017:27017 mongo:7
```

#### Step 4: Start the FastAPI Web Server (Terminal 1)
```powershell
uvicorn phase2_service.app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Step 5: Start the Background Inference Worker (Terminal 2)
```powershell
python -m phase2_service.app.worker
```

#### Step 6: Open the Dashboard or Test via CLI
- Open your browser at **[http://localhost:8000/](http://localhost:8000/)**
- Or test via `curl.exe`:
  ```powershell
  # Check health
  curl.exe http://localhost:8000/health

  # Submit image
  curl.exe -X POST http://localhost:8000/jobs -F "file=@sample.png"

  # Fetch result (replace with returned job_id)
  curl.exe http://localhost:8000/jobs/<JOB_ID>/result
  ```

---

## 💡 Production Considerations & Engineering Highlights

1. **Decoupled Asynchronous Processing:** Medical images frequently exceed standard web timeouts. CellScope uses persistent AMQP task queuing so the client receives an instant confirmation while worker nodes process heavy segmentation workloads asynchronously.
2. **Zero Per-Request Model Reloading:** By preloading the PyTorch model once during worker boot, inference latency is reduced from seconds to milliseconds per image tile.
3. **Calibrated Microscopy Scaling:** Cell calculations default to an assumed scale of $0.25 \, \mu\text{m}/\text{pixel}$ (typical for $40\times$ objective lenses), transparently reporting both pixel-space and calibrated physical measurements ($\mu\text{m}$, $\mu\text{m}^2$, cells/$\text{mm}^2$).
4. **Poison-Pill Immunity:** Malformed inputs or corrupted files never crash or block the processing queue. Errors are safely caught, documented in MongoDB, and acknowledged to the message broker.

---

## ⚖️ Clinical Disclaimer & License

> **Notice:** CellScope is an open-source medical computer vision research and engineering demonstration pipeline. It is not approved as a medical device for primary diagnostic interpretation or clinical intervention without human-in-the-loop expert pathologist validation.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

⭐ If you find this project helpful, please consider giving it a star!
