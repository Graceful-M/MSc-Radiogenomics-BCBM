## Author


 MSc Thesis Project: Predictive Models for Breast Cancer Brain Metastasis: An Integration of Neuroimaging and Genomics Data for Better Clinical Practice



  
## Aim

To develop statistical and predictive models that integrate genomics and neuroimaging data for the detection of brain tumors in Breast Cancer Brain Metastasis (BCBM).



## Specific Objectives

1. Develop a BCBM predictive model using MRI-derived brain imaging features to detect BCBM tumors.

2. Identify neuroimaging features that best explain BCBM using MRI brain data.

3. Identify differentially expressed genes (DEGs) associated with BCBM using gene expression data.

4. Identify major brain regions (Regions of Interest, ROIs) associated with BCBM through gene–neuroimaging correlations.

5. Develop an integrated BCBM predictive model targeting genomically identified ROIs for improved tumor detection.



## Project Overview

This project combines machine learning and statistical modeling to predict brain metastases in breast cancer patients using integrated neuroimaging (MRI) and genomics data. The workflow includes preprocessing, feature selection, model training, evaluation, and radiogenomic analysis to identify tumor-associated brain regions.



## Dataset

1.Radiomic Dataset: MRI Dataset of Metastatic Breast Cancer to the Brain with Expert-reviewed Segmentations and Tumor-derived Radiomic Features (BCBM-RadioGenomics).  
2. Genomic Dataset: Gene expression data from TCGA (The Cancer Genome Atlas).  



## Methodology

### 1. Data Preprocessing
- Data cleaning and handling missing values.
- Feature normalization and scaling.
- Train-test split for machine learning models.

### 2. Feature Selection
- Identification of informative neuroimaging features using LASSO and univariate analysis.

### 3. Machine Learning Models
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

### 4. Model Evaluation
- Performance metrics: Accuracy, Precision, Recall, F1-score, Specificity, AUC.
- 10-fold cross-validation with repeated sampling.
- ROC curve visualization for model comparison.

### 5. Radiogenomic Integration
- Identification of differentially expressed genes (DEGs) related to BCBM.
- Mapping DEGs to MRI brain regions (ROIs) to find radiogenomic associations.
- Development of a predictive model focused on genomically informed ROIs.


## Software and Tools
- R: Statistical modeling, preprocessing, feature selection, machine learning, visualization.  
- Python: Optional for advanced ML pipelines and data handling.  
- NeuroimaGene R package: For gene–neuroimaging associations.  
- Git & GitHub: Version control and repository hosting.


## Expected Outputs
- Preprocessed and balanced datasets ready for modeling.  
- Predictive models for BCBM tumor detection.  
- Feature importance rankings for each machine learning model.  
- ROC curves and performance metrics visualizations.  
- Gene–neuroimaging correlations identifying tumor-associated ROIs.  



