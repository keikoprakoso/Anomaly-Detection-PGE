import os
import sys
#!/usr/bin/env python3
"""
Software License Anomaly Detection - Local Outlier Factor (LOF)

This script implements Local Outlier Factor for detecting suspicious software installations 
in the license audit dataset.
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

def main():
    print("=" * 60)
    print("SOFTWARE LICENSE ANOMALY DETECTION - LOCAL OUTLIER FACTOR")
    print("=" * 60)
    
    # 1. Load and Explore Data
    print("\n1. Loading and exploring data...")
    
    # Load the cleaned CSV file
    df = pd.read_csv('../data/Software_License_Cleaned.csv')
    
    print("Dataset Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Basic information about the dataset
    print("\nDataset Info:")
    print(f"Total records: {len(df)}")
    print(f"Total computers: {df['Computer_Name'].nunique()}")
    print(f"Total users: {df['Last_Logged_User'].nunique()}")
    print(f"Total publishers: {df['Publisher'].nunique()}")
    print(f"Total products: {df['Product_Name'].nunique()}")
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # 2. Data Preprocessing
    print("\n2. Preprocessing data...")
    
    # Create a copy for preprocessing
    df_processed = df.copy()
    
    print("Original shape:", df_processed.shape)
    
    # Drop duplicates
    df_processed = df_processed.drop_duplicates()
    print("After dropping duplicates:", df_processed.shape)
    
    # Drop rows with missing values in key columns
    df_processed = df_processed.dropna(subset=['Product_Name', 'Publisher', 'License_Code'])
    print("After dropping missing values:", df_processed.shape)
    
    # Label encode categorical columns
    le_publisher = LabelEncoder()
    le_product = LabelEncoder()
    le_license = LabelEncoder()
    
    # Fit and transform the categorical columns
    df_processed['Publisher_Encoded'] = le_publisher.fit_transform(df_processed['Publisher'])
    df_processed['Product_Name_Encoded'] = le_product.fit_transform(df_processed['Product_Name'])
    df_processed['License_Code_Encoded'] = le_license.fit_transform(df_processed['License_Code'])
    
    print("Label encoding completed!")
    print(f"Unique publishers: {len(le_publisher.classes_)}")
    print(f"Unique products: {len(le_product.classes_)}")
    print(f"Unique license codes: {len(le_license.classes_)}")
    
    # Prepare features for anomaly detection
    features = ['Publisher_Encoded', 'Product_Name_Encoded', 'License_Code_Encoded']
    X = df_processed[features]
    
    # Standardize features (important for LOF)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Feature matrix shape:", X_scaled.shape)
    print("\nFeature statistics after scaling:")
    print(pd.DataFrame(X_scaled, columns=features).describe())
    
    # 3. Local Outlier Factor Model
    print("\n3. Training Local Outlier Factor model...")
    
    # Initialize and train Local Outlier Factor
    lof_model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.05,
        metric='euclidean',
        novelty=False
    )
    
    # Fit and predict anomalies (-1 for anomalies, 1 for normal)
    predictions = lof_model.fit_predict(X_scaled)
    
    # Convert to boolean (True for anomalies, False for normal)
    anomalies = predictions == -1
    
    print("Model training completed!")
    print(f"Total predictions: {len(predictions)}")
    print(f"Anomalies detected: {anomalies.sum()}")
    print(f"Normal records: {(~anomalies).sum()}")
    print(f"Anomaly percentage: {anomalies.sum()/len(anomalies)*100:.2f}%")
    
    # Add anomaly column to the dataframe
    df_processed['Anomaly_LOF'] = anomalies
    
    # Display anomaly statistics
    print("\nAnomaly Detection Results:")
    print(f"Total records: {len(df_processed)}")
    print(f"Anomalies: {df_processed['Anomaly_LOF'].sum()}")
    print(f"Normal: {(~df_processed['Anomaly_LOF']).sum()}")
    
    # Show some anomalous records
    print("\nSample anomalous records:")
    anomalous_records = df_processed[df_processed['Anomaly_LOF'] == True]
    print(anomalous_records[['Computer_Name', 'Publisher', 'Product_Name', 'License_Code']].head(10))
    
    # Get LOF scores (negative scores indicate anomalies)
    lof_scores = lof_model.negative_outlier_factor_
    
    print("\nLOF Score Statistics:")
    print(f"Min score: {lof_scores.min():.4f}")
    print(f"Max score: {lof_scores.max():.4f}")
    print(f"Mean score: {lof_scores.mean():.4f}")
    print(f"Std score: {lof_scores.std():.4f}")
    
    # 4. Visualizations
    print("\n4. Creating visualizations...")
    
    # Create scatter plot: Publisher vs Product_Name with anomaly flag
    plt.figure(figsize=(14, 8))
    
    # Plot normal points
    normal_data = df_processed[df_processed['Anomaly_LOF'] == False]
    plt.scatter(normal_data['Publisher_Encoded'], normal_data['Product_Name_Encoded'], 
               alpha=0.6, s=30, c='blue', label='Normal', edgecolors='white', linewidth=0.5)
    
    # Plot anomalous points
    anomaly_data = df_processed[df_processed['Anomaly_LOF'] == True]
    plt.scatter(anomaly_data['Publisher_Encoded'], anomaly_data['Product_Name_Encoded'], 
               alpha=0.8, s=50, c='red', label='Anomaly', edgecolors='black', linewidth=1)
    
    plt.xlabel('Publisher (Encoded)')
    plt.ylabel('Product Name (Encoded)')
    plt.title('Local Outlier Factor: Publisher vs Product Name with Anomaly Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lof_scatteroutput/plots/.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot count of anomalies
    plt.figure(figsize=(10, 6))
    
    anomaly_counts = df_processed['Anomaly_LOF'].value_counts()
    colors = ['#2E8B57', '#DC143C']
    
    bars = plt.bar(['Normal', 'Anomaly'], anomaly_counts.values, color=colors, alpha=0.7)
    plt.title('Local Outlier Factor: Count of Normal vs Anomalous Records')
    plt.ylabel('Count')
    
    # Add value labels on bars
    for bar, count in zip(bars, anomaly_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                 str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lof_countsoutput/plots/.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Top publishers with anomalies
    publisher_anomalies = df_processed[df_processed['Anomaly_LOF'] == True]['Publisher'].value_counts().head(10)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(publisher_anomalies)), publisher_anomalies.values, color='coral', alpha=0.7)
    plt.title('Top 10 Publishers with Most Anomalies (Local Outlier Factor)')
    plt.xlabel('Publisher')
    plt.ylabel('Number of Anomalies')
    plt.xticks(range(len(publisher_anomalies)), publisher_anomalies.index, rotation=45, ha='right')
    
    # Add value labels
    for bar, count in zip(bars, publisher_anomalies.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lof_publishersoutput/plots/.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # LOF scores distribution
    plt.figure(figsize=(15, 5))
    
    # Plot histogram of LOF scores
    plt.subplot(1, 3, 1)
    plt.hist(lof_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('LOF Score (Negative Outlier Factor)')
    plt.ylabel('Frequency')
    plt.title('Distribution of LOF Scores')
    plt.grid(True, alpha=0.3)
    
    # Plot LOF scores by class
    plt.subplot(1, 3, 2)
    normal_scores = lof_scores[~anomalies]
    anomaly_scores = lof_scores[anomalies]
    
    plt.hist(normal_scores, bins=30, alpha=0.7, label='Normal', color='blue')
    plt.hist(anomaly_scores, bins=30, alpha=0.7, label='Anomaly', color='red')
    plt.xlabel('LOF Score')
    plt.ylabel('Frequency')
    plt.title('LOF Scores by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box plot of LOF scores by class
    plt.subplot(1, 3, 3)
    data_to_plot = [normal_scores, anomaly_scores]
    plt.boxplot(data_to_plot, labels=['Normal', 'Anomaly'])
    plt.ylabel('LOF Score')
    plt.title('LOF Scores Distribution by Class')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lof_scoresoutput/plots/.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Save Results
    print("\n5. Saving results...")
    
    # Save anomalies to CSV
    anomaly_output = df_processed[df_processed['Anomaly_LOF'] == True].copy()
    
    # Remove encoded columns for cleaner output
    columns_to_drop = ['Publisher_Encoded', 'Product_Name_Encoded', 'License_Code_Encoded']
    anomaly_output = anomaly_output.drop(columns=columns_to_drop)
    
    # Save to CSV
    anomaly_output.to_csv('output/results/Software_Anomalies_LOF.csv', index=False)
    
    print(f"Anomalies saved to 'output/results/Software_Anomalies_LOF.csv'")
    print(f"Total anomalies saved: {len(anomaly_output)}")
    print("\nColumns in output file:")
    print(anomaly_output.columns.tolist())
    
    # Display summary
    print("\nSummary of detected anomalies:")
    print(f"Total records processed: {len(df_processed)}")
    print(f"Anomalies detected: {len(anomaly_output)}")
    print(f"Anomaly rate: {len(anomaly_output)/len(df_processed)*100:.2f}%")
    
    # Show sample of saved anomalies
    print("\nSample of saved anomalies:")
    print(anomaly_output[['Computer_Name', 'Publisher', 'Product_Name', 'License_Code']].head())
    
    print("\n" + "=" * 60)
    print("LOCAL OUTLIER FACTOR ANALYSIS COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    main() 