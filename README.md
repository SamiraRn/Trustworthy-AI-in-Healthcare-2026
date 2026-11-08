# Trustworthy-AI-in-Healthcare-2026
MAEF-XAI: Modality-Aware Evaluation Framework for XAI in Healthcare

🚧 Note: The accompanying research paper is currently under review / unpublished. 🚧

MAEF-XAI is a lightweight, reproducible framework for evaluating Explainable AI (XAI) methods—specifically SHAP, LIME, and Integrated Gradients—in clinical decision support systems.

Instead of relying on a single metric, this framework calculates a Trustworthiness Index (TI) that combines:

Fidelity: Does the explanation truly reflect the model's logic?

Stability: Does the explanation hold up against realistic data noise?

Plausibility: Does the explanation align with known ground-truth features?

🚀 Quick Start

The entire evaluation pipeline is designed to be highly reproducible, running on synthetic datasets with a fixed random seed.

Requirements

You only need standard data science libraries:

numpy

scipy

scikit-learn

Running the Experiment

The main script executes the full evaluation on both a tabular (EHR-style) dataset and a low-resolution imaging dataset.
It takes less than 3 minutes on a single CPU core.


📊 Key Findings

Our micro-experiment shows that XAI performance depends heavily on the data modality:

Tabular Data: SHAP is highly stable, but LIME edges out on the composite score.

Imaging Data: SHAP's stability collapses under noise; Integrated Gradients (IG) proves significantly more stable.

Takeaway: There is no "best" universal explainer. The choice must match the task's structure and clinical context.
