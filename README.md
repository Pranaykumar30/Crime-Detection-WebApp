# Crime-Detection-WebApp
A web app for crime detection using YOLOv8 and MobileNet.

## Setup
1. Clone the repo: `git clone https://github.com/Pranaykumar30/Crime-Detection-WebApp.git`
2. Activate virtual env: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the app: `python app/main.py`

## 📌 Status  

### ✅ Day 1: Project Setup Completed in GitHub Codespaces  
- Initialized **Git repository** and **Codespaces** with Python 3.9.  
- Set up **virtual environment** with core dependencies:  
  - `torch`, `tensorflow`, `opencv`, `flask`  
- Created **Flask app skeleton** in `app/app.py`.  

### ✅ Day 2: Dataset collection and preprocessing completed
  - Created data structure with 180 images across 6 classes.
  - Labeled images with YOLOv8 and remapped to custom classes.
  - Split into train (70%), val (15%), test (15%) sets.
  - Preprocessed images with augmentation pipeline.
  - Converted to TFRecord for Faster-RCNN and SSD.

