# 🚀 Software License Anomaly Detection Project

A comprehensive machine learning project for detecting suspicious software installations using multiple unsupervised anomaly detection algorithms.

## 📁 Project Structure

```
Software Anomaly Project/
├── 📊 data/
│   └── Software_License_Cleaned.csv          # Input dataset
├── 🔧 src/
│   ├── models/                              # Anomaly detection models
│   │   ├── software_if.py                   # Isolation Forest
│   │   ├── software_ocsvm.py                # One-Class SVM
│   │   ├── software_lof.py                  # Local Outlier Factor
│   │   ├── software_autoencoder.py          # Autoencoder
│   │   └── software_summary.py              # Master summary
│   └── utils/                               # Utility scripts
│       ├── image_analyzer.py                # Image analysis tool
│       └── create_html_viewer.py            # HTML viewer generator
├── 📈 output/
│   ├── results/                             # Analysis results
│   │   ├── Software_Anomalies_IF.csv
│   │   ├── Software_Anomalies_SVM.csv
│   │   ├── Software_Anomalies_LOF.csv
│   │   ├── Software_Anomalies_Autoencoder.csv
│   │   └── Software_Flagged_By_3_Models.csv
│   ├── plots/                               # Visualizations
│   │   ├── isolation_forest_*.png
│   │   ├── oneclass_svm_*.png
│   │   ├── lof_*.png
│   │   ├── autoencoder_*.png
│   │   └── comprehensive_analysis.png
│   └── reports/                             # Generated reports
│       └── visualizations.html
├── 📚 docs/                                 # Documentation
│   ├── README.md                           # This file
│   └── VISUALIZATION_GUIDE.md              # Visualization tools guide
├── 🚀 run_analysis.py                      # Main execution script
├── requirements.txt                         # Python dependencies
└── .gitignore                              # Git ignore file
```

## 🎯 Quick Start

### 1. Install Dependencies
```bash
pip3 install -r requirements.txt
```

### 2. Run Complete Analysis
```bash
python3 run_analysis.py
```

This will execute the entire pipeline:
- ✅ Isolation Forest Analysis
- ✅ One-Class SVM Analysis
- ✅ Local Outlier Factor Analysis
- ✅ Autoencoder Analysis
- ✅ Master Summary Analysis
- ✅ Visualization Generation

### 3. Run Individual Models
```bash
# From project root
python3 src/models/software_if.py
python3 src/models/software_ocsvm.py
python3 src/models/software_lof.py
python3 src/models/software_autoencoder.py
python3 src/models/software_summary.py
```

## 📊 Dataset Description

The project analyzes `Software_License_Cleaned.csv` containing software license audit data:

**Columns:**
- `Computer_Name`: Name of the computer
- `Last_Logged_User`: Last user who logged in
- `Publisher`: Software publisher/company
- `Product_Name`: Name of the software product
- `Product_Key`: Product license key
- `License_Code`: License code
- `License_Version`: Version of the license
- `Install_Date`: Installation date

## 🔍 Anomaly Detection Models

### 1. **Isolation Forest**
- **Algorithm**: Tree-based ensemble method
- **Parameters**: `n_estimators=100`, `contamination=0.05`
- **Output**: `Software_Anomalies_IF.csv`

### 2. **One-Class SVM**
- **Algorithm**: Support Vector Machine for novelty detection
- **Parameters**: `nu=0.05`, `kernel='rbf'`, `gamma='scale'`
- **Output**: `Software_Anomalies_SVM.csv`

### 3. **Local Outlier Factor (LOF)**
- **Algorithm**: Density-based outlier detection
- **Parameters**: `n_neighbors=20`, `contamination=0.05`
- **Output**: `Software_Anomalies_LOF.csv`

### 4. **Autoencoder**
- **Algorithm**: Neural network for reconstruction error
- **Architecture**: Encoder(32→16), Decoder(16→32→input_dim)
- **Parameters**: `epochs=50`, threshold at 95th percentile
- **Output**: `Software_Anomalies_Autoencoder.csv`

## 📈 Output Files

### Results (CSV Files)
- **Individual Model Results**: Each model generates a CSV with anomaly flags
- **Master Summary**: `Software_Flagged_By_3_Models.csv` - High-risk software flagged by ≥3 models

### Visualizations (PNG Files)
- **Scatter Plots**: Publisher vs Product Name with anomaly detection
- **Count Plots**: Normal vs anomalous record distributions
- **Publisher Analysis**: Top publishers with most anomalies
- **Model Comparisons**: Cross-model analysis and overlap
- **Comprehensive Dashboard**: Multi-panel analysis

### Reports (HTML Files)
- **Interactive Viewer**: `visualizations.html` - Browse all visualizations

## 🛠️ Key Features

### Data Preprocessing
- ✅ Duplicate removal
- ✅ Missing value handling
- ✅ Label encoding for categorical variables
- ✅ Feature scaling (MinMaxScaler for Autoencoder, StandardScaler for SVM/LOF)

### Model Training
- ✅ Consistent preprocessing across all models
- ✅ Reproducible results with random seeds
- ✅ Model-specific parameter optimization

### Visualization
- ✅ Comprehensive plotting with matplotlib/seaborn
- ✅ High-resolution PNG outputs
- ✅ Interactive HTML viewer
- ✅ Model comparison dashboards

### Analysis
- ✅ Individual model results
- ✅ Cross-model comparison
- ✅ High-risk software identification
- ✅ Publisher and product analysis

## 📋 Use Cases

### Software License Auditing
- Detect unauthorized software installations
- Identify suspicious license patterns
- Monitor software compliance

### Security Analysis
- Find potential security risks
- Identify unusual software combinations
- Track software installation patterns

### Compliance Monitoring
- Ensure software license compliance
- Identify over-licensed software
- Monitor software usage patterns

## 🔧 Technical Requirements

### Python Version
- Python 3.7 or higher

### Key Dependencies
- `pandas>=1.3.0` - Data manipulation
- `numpy>=1.21.0,<2.0.0` - Numerical operations
- `scikit-learn>=1.0.0` - Machine learning models
- `tensorflow>=2.8.0` - Autoencoder implementation
- `matplotlib>=3.4.0` - Plotting
- `seaborn>=0.11.0` - Statistical visualizations

## 📚 Documentation

- **Main Guide**: This README file
- **Visualization Guide**: `docs/VISUALIZATION_GUIDE.md`
- **Code Comments**: Extensive inline documentation in all scripts

## 🎯 Analysis Results

### High-Risk Software Identified
- **VMware Workstation**: 9 records flagged by all models
- **HPActiveSupport**: 7 records flagged by multiple models
- **PaperStream IP (TWAIN)**: 5 records flagged by all models

### Top Suspicious Publishers
- **VMware, Inc.**: 13 high-risk records
- **Autodesk**: 9 high-risk records
- **FUJITSU**: 5 high-risk records

### Model Performance
- **Isolation Forest**: 199 anomalies (4.59%)
- **One-Class SVM**: 203 anomalies (4.68%)
- **Local Outlier Factor**: 193 anomalies (4.45%)
- **Autoencoder**: 183 anomalies (4.22%)

## 🚀 Getting Started

1. **Clone/Download** the project
2. **Install dependencies**: `pip3 install -r requirements.txt`
3. **Run analysis**: `python3 run_analysis.py`
4. **View results**: Check `output/` directory
5. **Explore visualizations**: Open `output/reports/visualizations.html`

## 📞 Support

For questions or issues:
1. Check the documentation in `docs/`
2. Review the code comments
3. Examine the output files for insights

---

**Happy Anomaly Detection! 🚀** 