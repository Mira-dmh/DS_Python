# DS_Python

A comprehensive collection of Python notebooks, datasets, and scripts for data science study and practice. This repository covers fundamental to advanced topics in data science, including data exploration, visualization, machine learning algorithms, and real-world case studies.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Datasets](#datasets)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Usage Examples](#usage-examples)
- [Learning Outcomes](#learning-outcomes)
- [Contributing](#contributing)

## 🎯 Project Overview

This repository serves as a comprehensive learning resource for data science with Python. It includes hands-on tutorials, real-world datasets, and practical implementations of various machine learning algorithms and data analysis techniques.

## 📁 Project Structure

### Core Data Science Topics

- **Numpy/** - Numerical computing with arrays, mathematical operations, and functions
- **Pandas/** - Data manipulation, cleaning, and analysis with DataFrames
- **Matplotlib/** - Data visualization and plotting techniques
- **Dictionaries/** - Python dictionary operations and data structures
- **Functions/** - Function definitions, lambda functions, and modular programming

### Data Exploration & Visualization

- **boxplot_histogram/** - Exploratory data analysis using boxplots and histograms
  - `Chap3_BoxPlot_Histogram.ipynb` - Statistical visualization techniques
  - `Descriptive statistics in pandas.ipynb` - Statistical summaries
  - Datasets: `nhanes.csv`, `ratings.csv`

- **code 0331 DataExploration/** - Comprehensive data exploration with Palmer Penguins
  - `DataExploration_PalmerPenguins_01-04.ipynb` - Multi-part penguin species analysis
  - Dataset: `palmer_penguins.csv` (Antarctic penguin measurements)

### Machine Learning Applications

- **KNN/** - K-Nearest Neighbors classification
  - `BeanClassifier_KNN.ipynb` - Bean variety classification
  - Dataset: `Dry_Bean_Dataset.csv`

- **cervical cancer/** - Medical data analysis and risk prediction
  - `KNN.ipynb`, `code.ipynb` - Classification models for cancer risk
  - Dataset: `risk_factors_cervical_cancer.csv`
  - Comprehensive variable documentation in `expalin.txt`
  - Visualizations: confusion matrices, correlation heatmaps, feature importance

- **RegresssionTree/** - Decision tree regression analysis

### Data Processing & Manipulation

- **Data Manipulating/** - Advanced pandas operations
  - `1_Example_pivot_table.ipynb` - Pivot table creation and analysis
  - `2_Manipulating_data.ipynb` - Data transformation techniques

- **code data/** - Data cleaning and preprocessing
  - **Data cleaning/** - Missing value handling, data type conversion
  - **Feature scaling/** - Normalization and standardization techniques

### Specialized Applications

- **code0407/, code0409model/, code0411/** - Weather prediction with decision trees
  - `Weather_DecisionTree.ipynb` - Time series weather analysis
  - Dataset: `daily_weather.csv`

- **code0414/** - Diabetes prediction using KNN
  - `Diabetes_KNN_ModelTuning.ipynb` - Model optimization techniques
  - Dataset: `diabetes.csv`

- **lab/** - Structured laboratory exercises and assignments
  - Progressive learning modules from basic concepts to advanced applications

## 🚀 Key Features

- **Comprehensive Coverage**: From basic Python data structures to advanced ML algorithms
- **Real-World Datasets**: Medical, ecological, agricultural, and meteorological data
- **Hands-On Learning**: Interactive Jupyter notebooks with step-by-step explanations
- **Visualization**: Extensive use of matplotlib and seaborn for data storytelling
- **Model Evaluation**: Confusion matrices, feature importance, and performance metrics
- **Data Quality**: Data cleaning, missing value handling, and preprocessing techniques

## 📊 Datasets

| Dataset | Domain | Description | Use Case |
|---------|--------|-------------|----------|
| `palmer_penguins.csv` | Ecology | Antarctic penguin species measurements | Species classification, EDA |
| `risk_factors_cervical_cancer.csv` | Medical | Cervical cancer risk factors | Medical prediction, feature analysis |
| `Dry_Bean_Dataset.csv` | Agriculture | Bean variety characteristics | Multi-class classification |
| `daily_weather.csv` | Meteorology | Daily weather observations | Time series analysis, regression |
| `diabetes.csv` | Medical | Diabetes patient data | Binary classification, model tuning |
| `nhanes.csv` | Health Survey | National health survey data | Statistical analysis |

## 🛠️ Getting Started

### Prerequisites

- Python 3.7+ 
- Jupyter Notebook or VS Code with Python extension
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mira-dmh/DS_Python.git
   cd DS_Python
   ```

2. **Install required packages**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

## 📦 Requirements

```python
# Core libraries
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0

# Additional utilities
jupyter>=1.0.0
mlxtend>=0.19.0  # For advanced plotting
```

## 💡 Usage Examples

### Quick Start with Palmer Penguins Analysis
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the penguin dataset
penguins = pd.read_csv('code 0331 DataExploration/palmer_penguins.csv')

# Basic exploration
print(penguins.info())
print(penguins.describe())

# Visualization
sns.scatterplot(data=penguins, x='bill_length_mm', y='body_mass_g', hue='species')
plt.show()
```

### KNN Classification Example
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load and prepare data
# Follow examples in KNN/BeanClassifier_KNN.ipynb
```

## 🎓 Learning Outcomes

After working through this repository, you will be able to:

- **Data Manipulation**: Master pandas for data cleaning, transformation, and analysis
- **Visualization**: Create informative plots using matplotlib and seaborn
- **Machine Learning**: Implement KNN, decision trees, and evaluate model performance
- **Statistical Analysis**: Perform exploratory data analysis and interpret results
- **Real-World Applications**: Apply data science techniques to medical, ecological, and other domains
- **Best Practices**: Follow proper data science workflows and documentation

## 🤝 Contributing

This repository is primarily for educational purposes. Feel free to:
- Report issues or suggest improvements
- Share additional datasets or examples
- Contribute documentation or tutorials

## 📝 License

This project is for educational use. Please respect the original data sources and cite appropriately when using the datasets.

---

*Last updated: September 2025*