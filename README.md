#ReT-ONets Flow Field Prediction - Unsteady Flow Around Cylinder Case 

This project is used for the prediction of unsteady flow fields around a cylinder, employing a two-stage training pipeline:

1.	Stage 1: Reference Module Pre-training
Train a data-driven neural operator using observation/supervision data D_ref = {(u_i, s_i, g_i)}.
where the model learns the mapping from input conditions and spatio-temporal coordinates to flow variables.
2.	Stage 2: Physics-Guided Distillation 
The parameters obtained from Stage 1 are used as initialization. A frozen Teaching Module (TM) is introduced to provide distillation targets at spatiotemporal collocation points. The optimization objective in the second stage is:
         L_total = L_ref + β * L_distill
In this second stage, the PDE residual is not directly incorporated into the total loss; instead, the teacher model's output is used for distillation.



## 1. Directory Structure

├── data/                              # CFD datasets
├── data_generation.py                 # Data processing utilities
├── stage1_dataloader.py               # Stage 1 data loader
├── stage1_model.py                    # Reference Module definition
├── stage1_train.py                    # Stage 1 training script
├── stage2_dataloader1.py              # Collocation point loader
└── stage2_collaborative_train.py      # Stage 2 distillation training

## 2. Environment Requirements

The code is implemented with Python and PyTorch.
Main dependencies:
torch
numpy
pandas
matplotlib
scikit-learn

## 3. Data Preparation
The training data should be organized as:
supervision_data
├── data_U_0_xxx.npy
├── data_U_0_xxx.npy
└── ...
Each data file contains:
x, y, t, p, u, v
where:
•	x, y: spatial coordinates
•	t: time
•	p: pressure
•	u, v: velocity components
The flow conditions in the dataset should be consistent with the velocity settings in the training scripts.

## 4. Training Process
4.1. Train Reference Module
Run:
python stage1_train.py
This stage trains the data-driven neural operator and saves:
best_model.pth
normalizer.pth
The trained model will be used as the initial model for Stage 2.

4.2. Physics-guided Distillation Training
Before training, specify:
•	the Stage 1 pre-trained model path
•	the Teaching Module checkpoint path
•	the dataset path
Then run:
python stage2_collaborative_train.py
The final optimized model will be saved as:
best_model.pth

4.3. Flow Field Prediction
The final model from Stage 2 can be used for flow field reconstruction and prediction.
Load:
best_model.pth
normalizer.pth
and provide the required input conditions and spatio-temporal coordinates.

## 5. Training Pipeline
CFD Dataset
     |
Reference Module Pre-training
     |
Pre-trained RM
     |
Physics-guided Distillation with TM
     |
Final Neural Operator Model
     |
Transient Flow Prediction

## 6.Notes
•	The Teaching Module is used as a physics-guided teacher during Stage 2 training.
•	The final model maintains the same prediction structure as the Reference Module.
•	Dataset format and velocity conditions should match the configuration in the training scripts.

