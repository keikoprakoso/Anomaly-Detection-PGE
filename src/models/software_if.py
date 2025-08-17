import os
import sys
#!/usr/bin/env python3
"""
Software License Anomaly Detection using Isolation Forest

This script performs anomaly detection on software license data using Isolation Forest algorithm.
It loads the cleaned data, preprocesses it, trains the model, and saves results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 80)
    print("SOFTWARE LICENSE ANOMALY DETECTION - ISOLATION FOREST")
    print("=" * 80)
    
    # Create output directories if they don't exist
    os.makedirs('output/results', exist_ok=True)
    os.makedirs('output/plots', exist_ok=True)
    
    # 1. Load Data
    print("\n1. Loading data...")
    try:
        df = pd.read_csv('data/Software_License_Cleaned.csv')
        print(f"   Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print("   Error: ../data/Software_License_Cleaned.csv not found in data/ directory")
        return
    
    # 2. Basic Data Information
    print("\n2. Basic data information:")
    print(f"   Dataset shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Data types:\n{df.dtypes}")
    print(f"   Missing values:\n{df.isnull().sum()}")
    
    # 3. Data Preprocessing
    print("\n3. Data preprocessing...")
    
    # Drop duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"   Dropped {initial_rows - len(df)} duplicate rows")
    
    # Drop rows with missing values in key columns
    df_clean = df.dropna(subset=['Product_Name', 'Publisher', 'License_Code'])
    print(f"   Dropped {len(df) - len(df_clean)} rows with missing values")
    
    # Select features for anomaly detection
    features = ['Publisher', 'Product_Name', 'License_Code']
    X = df_clean[features].copy()
    
    # Label encoding for categorical variables
    print("   Applying label encoding...")
    label_encoders = {}
    for feature in features:
        le = LabelEncoder()
        X[feature] = le.fit_transform(X[feature].astype(str))
        label_encoders[feature] = le
        print(f"     {feature}: {len(le.classes_)} unique values encoded")
    
    # 4. Model Training
    print("\n4. Training Isolation Forest model...")
    isolation_forest = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )
    
    # Fit the model
    isolation_forest.fit(X)
    
    # 5. Predictions
    print("\n5. Making predictions...")
    predictions = isolation_forest.predict(X)
    
    # Convert predictions: -1 for anomalies, 1 for normal
    # We'll convert to 1 for anomalies, 0 for normal for easier interpretation
    anomaly_flags = (predictions == -1).astype(int)
    
    # Add predictions to original dataframe
    df_clean['Anomaly_IF'] = anomaly_flags
    
    # 6. Results Analysis
    print("\n6. Results analysis:")
    anomaly_count = anomaly_flags.sum()
    normal_count = len(anomaly_flags) - anomaly_count
    print(f"   Total records: {len(anomaly_flags)}")
    print(f"   Normal records: {normal_count}")
    print(f"   Anomalous records: {anomaly_count}")
    print(f"   Anomaly percentage: {(anomaly_count/len(anomaly_flags)*100):.2f}%")
    
    # 7. Visualizations
    print("\n7. Creating visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Software License Anomaly Detection - Isolation Forest', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot: Publisher vs Product_Name with anomaly flag
    ax1 = axes[0, 0]
    scatter = ax1.scatter(X['Publisher'], X['Product_Name'], 
                         c=anomaly_flags, cmap='RdYlBu_r', alpha=0.6, s=30)
    ax1.set_xlabel('Publisher (Encoded)')
    ax1.set_ylabel('Product Name (Encoded)')
    ax1.set_title('Publisher vs Product Name with Anomaly Detection')
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    legend1 = ax1.legend(*scatter.legend_elements(), title="Anomaly Flag")
    ax1.add_artist(legend1)
    
    # 2. Count plot of anomalies
    ax2 = axes[0, 1]
    anomaly_counts = df_clean['Anomaly_IF'].value_counts()
    bars = ax2.bar(['Normal', 'Anomaly'], [anomaly_counts[0], anomaly_counts[1]], 
                   color=['lightblue', 'lightcoral'])
    ax2.set_title('Count of Normal vs Anomalous Records')
    ax2.set_ylabel('Count')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom')
    
    # 3. Top publishers with anomalies
    ax3 = axes[1, 0]
    publisher_anomalies = df_clean[df_clean['Anomaly_IF'] == 1]['Publisher'].value_counts().head(10)
    publisher_anomalies.plot(kind='bar', ax=ax3, color='lightcoral')
    ax3.set_title('Top 10 Publishers with Most Anomalies')
    ax3.set_xlabel('Publisher')
    ax3.set_ylabel('Number of Anomalies')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Anomaly distribution by product
    ax4 = axes[1, 1]
    product_anomalies = df_clean[df_clean['Anomaly_IF'] == 1]['Product_Name'].value_counts().head(10)
    product_anomalies.plot(kind='bar', ax=ax4, color='lightgreen')
    ax4.set_title('Top 10 Products with Most Anomalies')
    ax4.set_xlabel('Product Name')
    ax4.set_ylabel('Number of Anomalies')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('output/plots/.png', dpi=300, bbox_inches='tight')
    print("   Saved: isolation_forest_analysisoutput/plots/.png")
    
    # Create individual plots for better visibility
    # Scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X['Publisher'], X['Product_Name'], 
                         c=anomaly_flags, cmap='RdYlBu_r', alpha=0.6, s=30)
    plt.xlabel('Publisher (Encoded)')
    plt.ylabel('Product Name (Encoded)')
    plt.title('Isolation Forest: Publisher vs Product Name with Anomaly Detection')
    plt.grid(True, alpha=0.3)
    plt.legend(*scatter.legend_elements(), title="Anomaly Flag")
    plt.tight_layout()
    plt.savefig('output/plots/.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Count plot
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Normal', 'Anomaly'], [anomaly_counts[0], anomaly_counts[1]], 
                   color=['lightblue', 'lightcoral'])
    plt.title('Isolation Forest: Count of Normal vs Anomalous Records')
    plt.ylabel('Count')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('output/plots/.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Publisher analysis
    plt.figure(figsize=(12, 8))
    publisher_anomalies.plot(kind='bar', color='lightcoral')
    plt.title('Isolation Forest: Top 10 Publishers with Most Anomalies')
    plt.xlabel('Publisher')
    plt.ylabel('Number of Anomalies')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output/plots/.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Save Results
    print("\n8. Saving results...")
    
    # Save anomalies to CSV
    output_file = 'output/results/Software_Anomalies_IF.csv'
    df_clean.to_csv(output_file, index=False)
    print(f"   Anomalies saved to: {output_file}")
    
    # Save sample of anomalies for inspection
    anomalies_df = df_clean[df_clean['Anomaly_IF'] == 1]
    print(f"\n   Sample of detected anomalies:")
    print(anomalies_df[['Computer_Name', 'Publisher', 'Product_Name', 'License_Code']].head(10))
    
    print("\n" + "=" * 80)
    print("ISOLATION FOREST ANALYSIS COMPLETED!")
    print("=" * 80)

if __name__ == "__main__":
    main() 