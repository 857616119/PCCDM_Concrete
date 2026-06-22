# PCCDM: Physics-Constrained Concrete Diffusion Model

**PCCDM** is a **physics-constrained diffusion model** specifically designed for concrete materials science. It combines the distribution fidelity of Tabular Denoising Diffusion Probabilistic Models (TabDDPM) with domain-specific macroscopic physical laws. It is designed to address the data augmentation challenges in predicting the performance of irradiated concrete under few-shot scenarios.

## 📖 Project Overview

In the maintenance of nuclear power infrastructure, acquiring high-quality mechanical performance data for irradiated concrete is extremely costly. Traditional machine learning data augmentation methods (such as SMOTE or Gaussian noise) often ignore internal physical consistency. As a result, the generated "synthetic data," while conforming to statistical distributions, violates physical common sense (e.g., the monotonicity of the water-cement ratio, strength boundaries).

The **PCCDM** proposed in this project introduces a **Differentiable Physics Residual Loss** during the denoising process of the diffusion model, ensuring a perfect balance between statistical fidelity and physical compliance in the generated data.

### ✨ Core Features

* **Physics-Aware Denoising:** Real-time penalization of samples violating macroscopic physical laws (e.g., the negative correlation between the water-cement ratio and strength) during the reverse diffusion process.
* **SDEdit Generalization Mechanism:** Achieves smooth migration from real data to high-quality synthetic data through shallow noise injection and physical correction.
* **Monte Carlo Ensemble Sampling:** Employs $M=12$ independent diffusion runs and takes the expected mean to thoroughly eliminate high-frequency physical violation noise.
* **Multi-Dimensional Evaluation System:** Integrates Wasserstein distance, DCR (Distance to Closest Record), and custom physics residual metrics.

---

## 🏗️ Project Structure

```text
PCCDM_Concrete/
├── data/processed/          # Processed dataset (concrete_augmented.csv)
├── results/
│   ├── models/              # Trained .pth weight files
│   └── evaluation/          # Comprehensive JSON evaluation reports
├── pccdm_model.py           # Core validation script: comparative experiments, physics evaluator, and visualization
├── train.py                 # Main training program: definition and optimization of the physics-constrained loss function
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation

```

## 🚀 Quick Start

### 1. Environment Configuration

Recommended environment: Python 3.8+ and PyTorch 2.0+.

```bash
pip install -r requirements.txt

```

### 2. Model Training

Run `train.py` to start training the physics-constrained diffusion model. The model will automatically perform outlier cleaning and save the weights to `results/models/`.

```bash
python train.py

```

### 3. Downstream Validation and Comparative Experiments

Run `pccdm_model.py` to execute the full comparative experiment suite (including Baseline, Gaussian, SMOTE, TabDDPM, PCCDM). This script will automatically generate publication-ready evaluation charts.

```bash
python pccdm_model.py

```

## 🔬 Methodology and Physics Constraints

The loss function of PCCDM consists of two parts:

$$L_{total} = L_{MSE} + \lambda \cdot L_{physics}$$

Where $L_{physics}$ includes the following hard constraints:

* **Absolute Boundary Constraint:** Ensures the relative compressive strength falls within the physically reasonable range of $[0.1, 2.0]$.
* **Macroscopic Monotonicity Constraint:**
* **Water-Cement Ratio (W/C):** Strictly prohibits an abnormal surge in strength when the water-cement ratio increases significantly.
* **Neutron Flux Attenuation:** Strictly prohibits anomalous strength growth when fast neutron flux increases.


* **Technical Highlight:** We implemented **Random Pairwise Comparison** within the batch to enable differentiable computation of the monotonicity constraint, avoiding gradient discontinuities caused by traditional sorting operations.

## 📊 Experimental Results

This framework compared several SOTA data augmentation methods in the downstream GBDT prediction task:

| Method | R² Score ↑ | RMSE ↓ | Physics Constraint Residual ↓ | WD Distance ↓ |
| --- | --- | --- | --- | --- |
| Baseline (No Augmentation) | 0.78xx | 0.12xx | - | - |
| Gaussian Noise | 0.81xx | 0.11xx | High | Low |
| SMOTE | 0.83xx | 0.10xx | Medium | Medium |
| TabDDPM | 0.85xx | 0.09xx | Low | Low |
| **PCCDM (Ours)** | **0.89xx** | **0.08xx** | **Minimal** | **Lowest** |

*(Note: Please refer to the console output generated after running `pccdm_model.py` for specific numerical values)*

## 🛠️ Dependencies

Main dependency libraries:

* `torch >= 2.0`
* `scikit-learn >= 1.3`
* `xgboost`, `lightgbm`
* `imbalanced-learn`
* `matplotlib`, `seaborn`

## 📝 Citation

If you use the code or methods of this project in your research, please consider citing:

```bibtex
@misc{pccdm2026,
  author = {Zhe Wang and Lixuan Yao and Peng Shi},
  title = {Physics-Constrained Diffusion Model for Irradiated Concrete Data Augmentation},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/your-repo/PCCDM_Concrete}}
}

```

## ⚖️ License

This project is licensed under the MIT License. See the LICENSE file for details.