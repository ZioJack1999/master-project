# My Master's Project: Real-Time Deep Learning Classifier for Multiclass Motor Imagery BCIs

This repository contains my complete codebase, analytical notebooks, reports, and presentation files for my Master's Project on multiclass Motor Imagery (MI) Brain-Computer Interfaces (BCIs). I implemented and evaluated deep learning convolutional structures alongside classical machine learning pipelines to decode multiclass EEG motor imagery signals.

## Focus: `MultiModel_BCI_Pipeline_v2.ipynb`
My latest working implementation is **`MultiModel_BCI_Pipeline_v2.ipynb`**. 
This notebook represents the consolidated real-time BCI pipeline. In it, I identified and successfully fixed five critical bugs present in standard real-time BCI processing:
1. **Streaming Normalization**: Replaced per-window z-scoring (which destroys amplitude info) with training mean/std normalization values loaded directly from checkpoint states.
2. **Filter Consistency**: Swapped out standard causal IIR filters (causing ~153% phase distortion) for zero-phase pre-filtering matching training conditions.
3. **Smoothing Window**: Adjusted window size from 5 steps (1.25s / 25% trial coverage) to 16 steps (4s / 80% trial coverage) for robust voting.
4. **Checkpoint Validation**: Transitioned from cross-subject Leave-One-Subject-Out (LOSO) checks to Leave-One-Run-Out (LORO) within-subject checkpoints to secure stable real-time streams.
5. **Stream Buffer Speed**: Replaced slow `np.roll` commands (O(n) latency per sample) with a custom O(1) ring buffer, accelerating data ingestion 6x.

---

## Timetable of My Project Progress

The chronological timeline of my master's project development:

* **February 18–19, 2026 (Project Initiation)**: I designed the initial project abstract and reviewed the BME707/708 syllabus constraints to outline goals.
* **February 24, 2026 (Literature & Design)**: I finalized my research slide deck and literature review on *Real-Time Deep Learning Classifier for Multiclass Motor Imagery BCIs*.
* **March 2–3, 2026 (Conceptual Brainstorming)**: I compiled my core algorithmic ideas (`Master project.txt`) and integrated concepts from my computational neuroscience labs.
* **April 2–3, 2026 (Modular Implementation)**: I wrote my core modular Python files including my dataset loader (`data_loader.py`), CNN modeling (`model.py`), training workflow (`train.py`), and EDF real-time simulation engine (`realtime_inference.py`).
* **April 5, 2026 (Model Training)**: I ran training scripts to produce optimized neural network checkpoints (`checkpoint_loso_sub01.pt` etc.).
* **April 6, 2026 (Classifier Testing & Drafts)**: I implemented and ran comparisons inside notebooks for multiple architectures (`CSP_LDA_MI.ipynb`, `ShallowConvNet_MI.ipynb`, `MI_EEGNet.ipynb`, `BandPower_SVM_MI.ipynb`), outputting real-time SVM charts and preparing first reports.
* **April 9, 2026 (Pipeline Integration)**: I resolved the 5 main BCI streaming bugs and integrated them into `MultiModel_BCI_Pipeline_v2.ipynb`, validating my final pipeline.
* **April 13, 2026 (SVM Metrics Finalization)**: I finalized my classical classifier analysis and SVM comparison summaries (`svm_summary.xls`).
* **April 19–21, 2026 (Final Project Completion)**: I compiled my final comprehensive report (`BME707 Final Report Albergo.pdf`), finished my graduation presentation (`BME707 Presentation Albergo.pptx`), and finalized my BME708 forward-looking project plan (`BME708_Plan_Albergo.pdf`).

---

## Setup & Running My Pipeline

1. Install my dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Running a quick test on the demo runs:
   ```bash
   python quickstart.py
   ```
3. Run within-subject Cross Validation:
   ```bash
   python train.py --root "/path/to/dataset" --mode within_subject
   ```
4. Run real-time inference simulation:
   ```bash
   python realtime_inference.py --edf sub-01_ses-01_task-MI_run-01_eeg.edf --events sub-01_ses-01_task-MI_run-01_events.tsv --ckpt checkpoint_sub01_ses01_run1.pt
   ```


