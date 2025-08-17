import os
import sys
#!/usr/bin/env python3
"""
Software License Anomaly Detection - Autoencoder

This script implements Autoencoder for detecting suspicious software installations 
in the license audit dataset.
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

def main():
    print("=" * 60)
    print("SOFTWARE LICENSE ANOMALY DETECTION - AUTOENCODER")
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
    
    # Normalize features using MinMaxScaler (0-1 range)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Feature matrix shape:", X_scaled.shape)
    print("\nFeature statistics after scaling:")
    print(pd.DataFrame(X_scaled, columns=features).describe())
    
    # 3. Autoencoder Model
    print("\n3. Building and training Autoencoder model...")
    
    # Define the autoencoder architecture
    input_dim = X_scaled.shape[1]
    
    # Input layer
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoded = Dense(32, activation='relu')(input_layer)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(16, activation='relu')(encoded)
    
    # Decoder
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dropout(0.2)(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    
    # Create the autoencoder model
    autoencoder = Model(input_layer, decoded)
    
    # Compile the model
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    print("Autoencoder architecture:")
    autoencoder.summary()
    
    # Train the autoencoder
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    
    history = autoencoder.fit(
        X_scaled, X_scaled,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("Training completed!")
    print(f"Final training loss: {history.history['loss'][-1]:.6f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'][-20:], label='Training Loss (Last 20 epochs)')
    plt.plot(history.history['val_loss'][-20:], label='Validation Loss (Last 20 epochs)')
    plt.title('Training History (Last 20 Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('autoencoder_trainingoutput/plots/.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Reconstruct the data and calculate reconstruction error
    X_reconstructed = autoencoder.predict(X_scaled)
    
    # Calculate Mean Squared Error (MSE) for each sample
    mse_scores = np.mean(np.square(X_scaled - X_reconstructed), axis=1)
    
    print("\nReconstruction Error Statistics:")
    print(f"Min MSE: {mse_scores.min():.6f}")
    print(f"Max MSE: {mse_scores.max():.6f}")
    print(f"Mean MSE: {mse_scores.mean():.6f}")
    print(f"Std MSE: {mse_scores.std():.6f}")
    print(f"95th percentile MSE: {np.percentile(mse_scores, 95):.6f}")
    
    # Set threshold for anomaly detection (95th percentile)
    threshold = np.percentile(mse_scores, 95)
    anomalies = mse_scores > threshold
    
    print("\nAnomaly Detection Results:")
    print(f"Threshold (95th percentile): {threshold:.6f}")
    print(f"Total records: {len(df_processed)}")
    print(f"Anomalies detected: {anomalies.sum()}")
    print(f"Normal records: {(~anomalies).sum()}")
    print(f"Anomaly percentage: {anomalies.sum()/len(anomalies)*100:.2f}%")
    
    # Add anomaly column and MSE scores to the dataframe
    df_processed['Anomaly_Autoencoder'] = anomalies
    df_processed['MSE_Score'] = mse_scores
    
    # Show some anomalous records
    print("\nSample anomalous records:")
    anomalous_records = df_processed[df_processed['Anomaly_Autoencoder'] == True]
    print(anomalous_records[['Computer_Name', 'Publisher', 'Product_Name', 'License_Code', 'MSE_Score']].head(10))
    
    # 4. Visualizations
    print("\n4. Creating visualizations...")
    
    # Create scatter plot: Publisher vs Product_Name with anomaly flag
    plt.figure(figsize=(14, 8))
    
    # Plot normal points
    normal_data = df_processed[df_processed['Anomaly_Autoencoder'] == False]
    plt.scatter(normal_data['Publisher_Encoded'], normal_data['Product_Name_Encoded'], 
               alpha=0.6, s=30, c='blue', label='Normal', edgecolors='white', linewidth=0.5)
    
    # Plot anomalous points
    anomaly_data = df_processed[df_processed['Anomaly_Autoencoder'] == True]
    plt.scatter(anomaly_data['Publisher_Encoded'], anomaly_data['Product_Name_Encoded'], 
               alpha=0.8, s=50, c='red', label='Anomaly', edgecolors='black', linewidth=1)
    
    plt.xlabel('Publisher (Encoded)')
    plt.ylabel('Product Name (Encoded)')
    plt.title('Autoencoder: Publisher vs Product Name with Anomaly Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('autoencoder_scatteroutput/plots/.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot count of anomalies
    plt.figure(figsize=(10, 6))
    
    anomaly_counts = df_processed['Anomaly_Autoencoder'].value_counts()
    colors = ['#2E8B57', '#DC143C']
    
    bars = plt.bar(['Normal', 'Anomaly'], anomaly_counts.values, color=colors, alpha=0.7)
    plt.title('Autoencoder: Count of Normal vs Anomalous Records')
    plt.ylabel('Count')
    
    # Add value labels on bars
    for bar, count in zip(bars, anomaly_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                 str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('autoencoder_countsoutput/plots/.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # MSE Histogram and Distribution Analysis
    plt.figure(figsize=(15, 10))
    
    # Plot 1: MSE Histogram
    plt.subplot(2, 3, 1)
    plt.hist(mse_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
    plt.xlabel('MSE Score')
    plt.ylabel('Frequency')
    plt.title('MSE Scores Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: MSE by class
    plt.subplot(2, 3, 2)
    normal_mse = mse_scores[~anomalies]
    anomaly_mse = mse_scores[anomalies]
    
    plt.hist(normal_mse, bins=30, alpha=0.7, label='Normal', color='blue')
    plt.hist(anomaly_mse, bins=30, alpha=0.7, label='Anomaly', color='red')
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold')
    plt.xlabel('MSE Score')
    plt.ylabel('Frequency')
    plt.title('MSE Scores by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Box plot
    plt.subplot(2, 3, 3)
    data_to_plot = [normal_mse, anomaly_mse]
    plt.boxplot(data_to_plot, labels=['Normal', 'Anomaly'])
    plt.ylabel('MSE Score')
    plt.title('MSE Scores Distribution by Class')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Top publishers with anomalies
    plt.subplot(2, 3, 4)
    publisher_anomalies = df_processed[df_processed['Anomaly_Autoencoder'] == True]['Publisher'].value_counts().head(8)
    bars = plt.bar(range(len(publisher_anomalies)), publisher_anomalies.values, color='coral', alpha=0.7)
    plt.title('Top Publishers with Anomalies')
    plt.xlabel('Publisher')
    plt.ylabel('Number of Anomalies')
    plt.xticks(range(len(publisher_anomalies)), publisher_anomalies.index, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: MSE vs Publisher (box plot)
    plt.subplot(2, 3, 5)
    top_publishers = df_processed['Publisher'].value_counts().head(5).index
    publisher_mse_data = [df_processed[df_processed['Publisher'] == pub]['MSE_Score'].values for pub in top_publishers]
    plt.boxplot(publisher_mse_data, labels=top_publishers)
    plt.title('MSE Scores by Top Publishers')
    plt.ylabel('MSE Score')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Cumulative distribution
    plt.subplot(2, 3, 6)
    sorted_mse = np.sort(mse_scores)
    cumulative = np.arange(1, len(sorted_mse) + 1) / len(sorted_mse)
    plt.plot(sorted_mse, cumulative, linewidth=2)
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'95th percentile')
    plt.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='95% line')
    plt.xlabel('MSE Score')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution of MSE Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('autoencoder_analysisoutput/plots/.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Save Results
    print("\n5. Saving results...")
    
    # Save anomalies to CSV
    anomaly_output = df_processed[df_processed['Anomaly_Autoencoder'] == True].copy()
    
    # Remove encoded columns for cleaner output
    columns_to_drop = ['Publisher_Encoded', 'Product_Name_Encoded', 'License_Code_Encoded']
    anomaly_output = anomaly_output.drop(columns=columns_to_drop)
    
    # Save to CSV
    anomaly_output.to_csv('output/results/Software_Anomalies_Autoencoder.csv', index=False)
    
    print(f"Anomalies saved to 'output/results/Software_Anomalies_Autoencoder.csv'")
    print(f"Total anomalies saved: {len(anomaly_output)}")
    print("\nColumns in output file:")
    print(anomaly_output.columns.tolist())
    
    # Display summary
    print("\nSummary of detected anomalies:")
    print(f"Total records processed: {len(df_processed)}")
    print(f"Anomalies detected: {len(anomaly_output)}")
    print(f"Anomaly rate: {len(anomaly_output)/len(df_processed)*100:.2f}%")
    print(f"Threshold used: {threshold:.6f}")
    
    # Show sample of saved anomalies with MSE scores
    print("\nSample of saved anomalies:")
    print(anomaly_output[['Computer_Name', 'Publisher', 'Product_Name', 'License_Code', 'MSE_Score']].head())
    
    print("\n" + "=" * 60)
    print("AUTOENCODER ANALYSIS COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    main() 