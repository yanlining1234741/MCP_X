MCP-X: Ultra-Compact CNN for Rice Disease Classification
Overview
This repository contains the implementation of MCP-X, an ultra-compact and interpretable Convolutional Neural Network (CNN) designed for rice disease classification in resource-constrained environments. MCP-X achieves 98.9% accuracy with only 0.26M parameters, making it suitable for on-device deployment in agricultural settings. The model is trained and evaluated on the PlantVillage dataset, which includes images of rice leaves affected by bacterial and fungal diseases such as bacterial blight, brown spot, and leaf smut.
Highlights

High Accuracy, Low Complexity: MCP-X achieves 98.9% accuracy with only 0.26M parameters, enabling efficient on-device rice disease classification.
Enhanced Interpretability: Integrates expert routing and efficient channel attention to provide intrinsic saliency maps for better understanding of model decisions.
No Pretraining Required: Trained end-to-end from scratch, eliminating the need for large-scale pretraining.
Robustness: Designed for practical deployment in resource-constrained environments, with preliminary field tests showing resilience to real-world variability.

Abstract
Rice is a vital staple for over half the global population but is susceptible to diseases like bacterial blight, brown spot, and leaf smut, which cause significant yield losses. Traditional manual scouting methods are labor-intensive and often lead to delayed interventions and excessive chemical use. While deep learning models like CNNs offer high accuracy, their computational complexity and lack of interpretability limit their use in resource-constrained agricultural settings. MCP-X addresses these challenges with an ultra-compact CNN architecture featuring a shallow encoder, multi-branch expert routing, a bi-level recurrent simulation encoder–decoder, meta-causal attention, and a lightweight classification head. Achieving 98.9% validation accuracy on the PlantVillage dataset without external pretraining, MCP-X outperforms larger models like MobileNetV2 (3.5M parameters) and Inception-V3 (27.2M parameters) with significantly fewer resources. Ablation studies confirm the contributions of expert routing and attention mechanisms, and preliminary field tests suggest robustness to real-world conditions. Future work will focus on validating MCP-X in diverse field settings and extending it to pixel-level segmentation.
Dataset
The model is trained and evaluated on the PlantVillage dataset, accessible at:Dataset Link: https://www.kaggle.com/datasets/mohitsingh1804/plantvillageThe dataset contains images of rice leaves affected by various bacterial and fungal diseases, including:

Bacterial blight
Brown spot
Leaf smut

The dataset is publicly available on Kaggle and is widely used for plant disease classification research.
Model Architecture
MCP-X is designed for efficiency and interpretability, featuring:

Shallow Encoder: Reduces computational overhead while maintaining feature extraction capability.
Multi-Branch Expert Routing: Enhances model performance by dynamically selecting specialized pathways for different disease patterns.
Bi-Level Recurrent Simulation Encoder–Decoder: Captures temporal and spatial dependencies in disease features.
Meta-Causal Attention: Improves focus on critical image regions for accurate classification.
Lightweight Classification Head: Minimizes parameter count while preserving high accuracy.
Efficient Channel Attention: Provides intrinsic saliency maps for interpretable predictions.

Performance

Validation Accuracy: 98.9% on the PlantVillage dataset.
Parameter Count: 0.26M, significantly lower than MobileNetV2 (3.5M) and Inception-V3 (27.2M).
Comparison: Matches or exceeds the performance of larger models with a fraction of the computational resources.
Field Robustness: Preliminary tests indicate robustness to real-world variability in lighting, orientation, and environmental conditions.

