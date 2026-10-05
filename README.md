# Breast Cancer Detection Using Machine Learning

Classifying breast tumors as **benign** or **malignant** with a Support Vector Machine (SVM) and a Decision Tree, using the Breast Cancer Wisconsin (Diagnostic) Dataset.

## Abstract

This project applies machine learning classification algorithms to breast cancer detection. Two supervised models, a **Support Vector Machine (SVM)** and a **Decision Tree Classifier**, classify tumors as benign or malignant using 30 diagnostic features derived from fine needle aspirate (FNA) images.

The work covers data cleaning, exploratory data analysis, feature scaling, model training, and evaluation with statistical metrics and visualizations. Both models performed strongly. The SVM generalized slightly better, while the Decision Tree offered interpretability through feature and tree visualizations.

## Introduction

Breast cancer is the most commonly diagnosed cancer among women and a leading cause of cancer-related deaths worldwide. Early diagnosis greatly improves outcomes. Conventional methods such as mammography, biopsy, and histopathology are effective but can be resource-intensive, slow, and subject to human interpretation error. Machine learning can support clinicians by automating pattern recognition in diagnostic data.

The goal here is to compare two complementary approaches: an SVM (a strong high-dimensional classifier) and a Decision Tree (an interpretable, rule-based model). Together they illustrate the balance between **accuracy** and **explainability** that matters in healthcare AI.

## Dataset

The [Breast Cancer Wisconsin (Diagnostic) dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data), developed by Dr. William H. Wolberg at the University of Wisconsin Hospitals, obtained from Kaggle.

Samples: 569 
Features: 30 numerical features 
Target: `diagnosis` (M = malignant, B = benign)

Features describe cell nucleus properties (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), each given as a mean, standard error, and worst value.

## Implementation

The project follows six stages of the classical ML workflow.

1. **Dataset description:** understand the source, size, and target variable.
2. **Data cleaning:**
   - Dropped non-predictive columns (`id`, `Unnamed: 32`)
   - Encoded the diagnosis as binary (M → 1, B → 0)
   - Checked for missing values (none found)
3. **Exploratory data analysis:**
   - Diagnosis distribution plot (class balance)
   - Correlation heatmap across all features (to spot redundancy)
   - Summary statistics via `df.describe()`
4. **Feature scaling and splitting:**
   - 80% training / 20% testing split
   - `StandardScaler` applied, since SVMs are sensitive to feature magnitude
5. **Model development:**
   - **SVM:** linear kernel, `C = 1.0`, probability estimation enabled for ROC-AUC
   - **Decision Tree:** `max_depth = 4` to limit overfitting, Gini criterion
6. **Evaluation and visualization:** confusion matrices, ROC curves, feature importance, model comparison charts, and a full decision tree diagram.

## Results

Evaluated on the 114-sample test set:

| Model | TN | FP | FN | TP | Accuracy | AUC |
| SVM (linear) | 68 | 3 | 2 | 41 | ~0.956 | ~0.99 to 1.00 |
| Decision Tree (depth 4) | 68 | 3 | 3 | 40 | ~0.947 | ~0.94 |

Malignant-class metrics derived from the confusion matrices:

| Model | Precision | Recall | F1 |
| SVM | 0.93 | 0.95 | 0.94 |
| Decision Tree | 0.93 | 0.93 | 0.93 |

**Feature importance**
- **Decision Tree:** `concave points_mean` dominates (about 0.70), followed by `concave points_worst`, `radius_worst`, and `perimeter_worst`.
- **SVM (absolute coefficients):** `concave points_mean`, `texture_worst`, `radius_se`, `symmetry_worst`, and `area_se` rank highest.

## Key Insights

- **SVM:** excellent in high-dimensional spaces, robust with proper regularization, and the more accurate model here.
- **Decision Tree:** transparent, rankable features, and explainable decision paths.
- **Trade-off:** the SVM gives better predictive power; the Decision Tree gives transparency and trust, which matters in medical settings.
- Preprocessing and feature scaling are crucial for performance.

## Applications

- **Computer-Aided Diagnosis (CAD):** assisting oncologists with early screening
- **Predictive health analytics:** patient risk profiling
- **Feature selection research:** identifying biologically relevant markers
- **Education and research:** a practical example of supervised learning in biomedicine

Such systems act as decision support tools that flag potential malignancies for expert review. They do not replace medical expertise.

## Conclusion and Future Work

Both algorithms effectively separate malignant from benign tumors using structured numerical data. The SVM outperformed the Decision Tree in accuracy and AUC, while the Decision Tree offered clearer clinical transparency.

Future directions:
- Ensemble methods (Random Forest, Gradient Boosting)
- Deep learning architectures
- Hyperparameter tuning with cross-validation
- Integration into clinical workflows to improve diagnostic precision and screening efficiency

## Disclaimer

This project is for educational and research purposes only. It is not a medical device and must not replace professional medical diagnosis.

## References

1. Breast Cancer Wisconsin Dataset: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
2. Scikit-learn documentation: https://scikit-learn.org/
3. W. H. Wolberg et al., "Breast Cancer Wisconsin (Diagnostic) Data Set", UCI Machine Learning Repository.
4. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297.
5. Quinlan, J. R. (1986). Induction of decision trees. *Machine Learning*, 1(1), 81-106.
