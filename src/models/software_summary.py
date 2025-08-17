import os
import sys
#!/usr/bin/env python3
"""
Software License Anomaly Detection - Master Summary

This script provides a comprehensive summary of all anomaly detection methods and their results.
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)

def create_id(row):
    """Create unique identifier for each record"""
    return f"{row['Computer_Name']}_{row['Product_Name']}_{row['License_Code']}"

def main():
    print("=" * 60)
    print("SOFTWARE LICENSE ANOMALY DETECTION - MASTER SUMMARY")
    print("=" * 60)
    
    # 1. Load All Anomaly Detection Results
    print("\n1. Loading all anomaly detection results...")
    
    # Load all anomaly detection results
    try:
        df_if = pd.read_csv('output/results/Software_Anomalies_IF.csv')
        print(f"Isolation Forest anomalies loaded: {len(df_if)} records")
    except FileNotFoundError:
        print("Warning: output/results/Software_Anomalies_IF.csv not found")
        df_if = pd.DataFrame()
    
    try:
        df_svm = pd.read_csv('output/results/Software_Anomalies_SVM.csv')
        print(f"One-Class SVM anomalies loaded: {len(df_svm)} records")
    except FileNotFoundError:
        print("Warning: output/results/Software_Anomalies_SVM.csv not found")
        df_svm = pd.DataFrame()
    
    try:
        df_lof = pd.read_csv('output/results/Software_Anomalies_LOF.csv')
        print(f"Local Outlier Factor anomalies loaded: {len(df_lof)} records")
    except FileNotFoundError:
        print("Warning: output/results/Software_Anomalies_LOF.csv not found")
        df_lof = pd.DataFrame()
    
    try:
        df_auto = pd.read_csv('output/results/Software_Anomalies_Autoencoder.csv')
        print(f"Autoencoder anomalies loaded: {len(df_auto)} records")
    except FileNotFoundError:
        print("Warning: output/results/Software_Anomalies_Autoencoder.csv not found")
        df_auto = pd.DataFrame()
    
    # Create unique identifiers for each record
    if not df_if.empty:
        df_if['Record_ID'] = df_if.apply(create_id, axis=1)
        df_if['IF_Flag'] = True
    
    if not df_svm.empty:
        df_svm['Record_ID'] = df_svm.apply(create_id, axis=1)
        df_svm['SVM_Flag'] = True
    
    if not df_lof.empty:
        df_lof['Record_ID'] = df_lof.apply(create_id, axis=1)
        df_lof['LOF_Flag'] = True
    
    if not df_auto.empty:
        df_auto['Record_ID'] = df_auto.apply(create_id, axis=1)
        df_auto['Auto_Flag'] = True
    
    print("Record IDs created for all datasets!")
    
    # 2. Merge All Results
    print("\n2. Merging all results...")
    
    # Start with the original data
    df_original = pd.read_csv('../data/Software_License_Cleaned.csv')
    df_original['Record_ID'] = df_original.apply(create_id, axis=1)
    
    # Create a base dataframe with all records
    df_merged = df_original[['Record_ID', 'Computer_Name', 'Publisher', 'Product_Name', 'License_Code']].copy()
    
    # Merge with each anomaly detection result
    if not df_if.empty:
        df_merged = df_merged.merge(df_if[['Record_ID', 'IF_Flag']], on='Record_ID', how='left')
    else:
        df_merged['IF_Flag'] = False
    
    if not df_svm.empty:
        df_merged = df_merged.merge(df_svm[['Record_ID', 'SVM_Flag']], on='Record_ID', how='left')
    else:
        df_merged['SVM_Flag'] = False
    
    if not df_lof.empty:
        df_merged = df_merged.merge(df_lof[['Record_ID', 'LOF_Flag']], on='Record_ID', how='left')
    else:
        df_merged['LOF_Flag'] = False
    
    if not df_auto.empty:
        df_merged = df_merged.merge(df_auto[['Record_ID', 'Auto_Flag']], on='Record_ID', how='left')
    else:
        df_merged['Auto_Flag'] = False
    
    # Fill NaN values with False
    flag_columns = ['IF_Flag', 'SVM_Flag', 'LOF_Flag', 'Auto_Flag']
    for col in flag_columns:
        df_merged[col] = df_merged[col].fillna(False)
    
    print("Merged dataset shape:", df_merged.shape)
    print("\nColumns in merged dataset:")
    print(df_merged.columns.tolist())
    
    # Calculate total flags for each record
    df_merged['Total_Flags'] = df_merged[flag_columns].sum(axis=1)
    
    # Create summary statistics
    print("\nAnomaly Detection Summary:")
    print(f"Total records: {len(df_merged)}")
    print(f"Records flagged by Isolation Forest: {df_merged['IF_Flag'].sum()}")
    print(f"Records flagged by One-Class SVM: {df_merged['SVM_Flag'].sum()}")
    print(f"Records flagged by Local Outlier Factor: {df_merged['LOF_Flag'].sum()}")
    print(f"Records flagged by Autoencoder: {df_merged['Auto_Flag'].sum()}")
    
    print("\nFlag Distribution:")
    flag_distribution = df_merged['Total_Flags'].value_counts().sort_index()
    for flags, count in flag_distribution.items():
        print(f"Records flagged by {flags} model(s): {count}")
    
    # 3. High-Risk Software Analysis
    print("\n3. Analyzing high-risk software...")
    
    # Identify software flagged by 3 or more models
    high_risk_software = df_merged[df_merged['Total_Flags'] >= 3].copy()
    
    print(f"High-risk software (flagged by ≥3 models): {len(high_risk_software)} records")
    print(f"Percentage of total: {len(high_risk_software)/len(df_merged)*100:.2f}%")
    
    # Save high-risk software to CSV
    high_risk_software.to_csv('Software_Flagged_By_3_Models.csv', index=False)
    print("\nHigh-risk software saved to 'Software_Flagged_By_3_Models.csv'")
    
    # Show sample of high-risk software
    print("\nSample high-risk software:")
    print(high_risk_software[['Computer_Name', 'Publisher', 'Product_Name', 'License_Code', 'Total_Flags']].head(10))
    
    # Analyze high-risk software by publisher
    high_risk_publishers = high_risk_software['Publisher'].value_counts().head(10)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(high_risk_publishers)), high_risk_publishers.values, color='red', alpha=0.7)
    plt.title('Top 10 Publishers with High-Risk Software (Flagged by ≥3 Models)')
    plt.xlabel('Publisher')
    plt.ylabel('Number of High-Risk Records')
    plt.xticks(range(len(high_risk_publishers)), high_risk_publishers.index, rotation=45, ha='right')
    
    # Add value labels
    for bar, count in zip(bars, high_risk_publishers.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('high_risk_publishersoutput/plots/.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Model Comparison Visualizations
    print("\n4. Creating model comparison visualizations...")
    
    # Model comparison bar chart
    models = ['Isolation Forest', 'One-Class SVM', 'Local Outlier Factor', 'Autoencoder']
    anomaly_counts = [df_merged['IF_Flag'].sum(), df_merged['SVM_Flag'].sum(), 
                     df_merged['LOF_Flag'].sum(), df_merged['Auto_Flag'].sum()]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, anomaly_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.7)
    plt.title('Anomaly Detection Results by Model')
    plt.ylabel('Number of Anomalies Detected')
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar, count in zip(bars, anomaly_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                 str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_comparisonoutput/plots/.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Flag distribution pie chart
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    flag_distribution = df_merged['Total_Flags'].value_counts().sort_index()
    colors = ['#2E8B57', '#FFD700', '#FF8C00', '#DC143C', '#8B0000']
    plt.pie(flag_distribution.values, labels=[f'{i} Model(s)' for i in flag_distribution.index], 
            autopct='%1.1f%%', colors=colors[:len(flag_distribution)])
    plt.title('Distribution of Records by Number of Models Flagging Them')
    
    plt.subplot(1, 3, 2)
    high_risk_dist = high_risk_software['Total_Flags'].value_counts().sort_index()
    plt.bar(high_risk_dist.index, high_risk_dist.values, color='red', alpha=0.7)
    plt.title('High-Risk Software by Number of Models')
    plt.xlabel('Number of Models Flagging')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    model_overlap = df_merged[flag_columns].sum()
    plt.bar(model_overlap.index, model_overlap.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.7)
    plt.title('Total Anomalies Detected by Each Model')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('flag_distributionoutput/plots/.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Model overlap analysis
    plt.figure(figsize=(15, 10))
    
    # Create sets for overlap analysis
    if_set = set(df_merged[df_merged['IF_Flag']]['Record_ID'])
    svm_set = set(df_merged[df_merged['SVM_Flag']]['Record_ID'])
    lof_set = set(df_merged[df_merged['LOF_Flag']]['Record_ID'])
    auto_set = set(df_merged[df_merged['Auto_Flag']]['Record_ID'])
    
    # Plot overlap analysis (simplified version without venn diagram)
    plt.subplot(2, 3, 1)
    overlap_data = {
        'IF': len(if_set),
        'SVM': len(svm_set),
        'LOF': len(lof_set),
        'Auto': len(auto_set),
        'IF∩SVM': len(if_set & svm_set),
        'IF∩LOF': len(if_set & lof_set),
        'IF∩Auto': len(if_set & auto_set),
        'SVM∩LOF': len(svm_set & lof_set),
        'SVM∩Auto': len(svm_set & auto_set),
        'LOF∩Auto': len(lof_set & auto_set),
        'All': len(if_set & svm_set & lof_set & auto_set)
    }
    
    plt.bar(overlap_data.keys(), overlap_data.values(), color='lightblue', alpha=0.7)
    plt.title('Model Overlap Analysis')
    plt.ylabel('Number of Records')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Heatmap of model correlations
    plt.subplot(2, 3, 2)
    correlation_matrix = df_merged[flag_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Model Correlation Matrix')
    
    # Stacked bar chart of model combinations
    plt.subplot(2, 3, 3)
    model_combinations = df_merged[flag_columns].apply(lambda x: '_'.join(x.astype(str)), axis=1).value_counts().head(10)
    plt.bar(range(len(model_combinations)), model_combinations.values, color='lightcoral', alpha=0.7)
    plt.title('Top 10 Model Combinations')
    plt.xlabel('Model Combination')
    plt.ylabel('Count')
    plt.xticks(range(len(model_combinations)), model_combinations.index, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Publisher analysis for high-risk software
    plt.subplot(2, 3, 4)
    top_publishers_high_risk = high_risk_software['Publisher'].value_counts().head(8)
    bars = plt.bar(range(len(top_publishers_high_risk)), top_publishers_high_risk.values, color='red', alpha=0.7)
    plt.title('Top Publishers in High-Risk Software')
    plt.xlabel('Publisher')
    plt.ylabel('Count')
    plt.xticks(range(len(top_publishers_high_risk)), top_publishers_high_risk.index, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Product analysis for high-risk software
    plt.subplot(2, 3, 5)
    top_products_high_risk = high_risk_software['Product_Name'].value_counts().head(8)
    bars = plt.bar(range(len(top_products_high_risk)), top_products_high_risk.values, color='darkred', alpha=0.7)
    plt.title('Top Products in High-Risk Software')
    plt.xlabel('Product')
    plt.ylabel('Count')
    plt.xticks(range(len(top_products_high_risk)), top_products_high_risk.index, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Risk level distribution
    plt.subplot(2, 3, 6)
    risk_levels = ['Low (0-1)', 'Medium (2)', 'High (3)', 'Critical (4)']
    risk_counts = [
        len(df_merged[df_merged['Total_Flags'] <= 1]),
        len(df_merged[df_merged['Total_Flags'] == 2]),
        len(df_merged[df_merged['Total_Flags'] == 3]),
        len(df_merged[df_merged['Total_Flags'] == 4])
    ]
    colors = ['green', 'yellow', 'orange', 'red']
    plt.pie(risk_counts, labels=risk_levels, autopct='%1.1f%%', colors=colors)
    plt.title('Risk Level Distribution')
    
    plt.tight_layout()
    plt.savefig('comprehensive_analysisoutput/plots/.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Summary Statistics
    print("\n5. Generating comprehensive summary...")
    
    # Comprehensive summary
    print("=" * 60)
    print("SOFTWARE LICENSE ANOMALY DETECTION - COMPREHENSIVE SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   Total records analyzed: {len(df_merged):,}")
    print(f"   Unique computers: {df_merged['Computer_Name'].nunique():,}")
    print(f"   Unique publishers: {df_merged['Publisher'].nunique():,}")
    print(f"   Unique products: {df_merged['Product_Name'].nunique():,}")
    
    print(f"\n🔍 ANOMALY DETECTION RESULTS:")
    print(f"   Isolation Forest anomalies: {df_merged['IF_Flag'].sum():,} ({df_merged['IF_Flag'].sum()/len(df_merged)*100:.2f}%)")
    print(f"   One-Class SVM anomalies: {df_merged['SVM_Flag'].sum():,} ({df_merged['SVM_Flag'].sum()/len(df_merged)*100:.2f}%)")
    print(f"   Local Outlier Factor anomalies: {df_merged['LOF_Flag'].sum():,} ({df_merged['LOF_Flag'].sum()/len(df_merged)*100:.2f}%)")
    print(f"   Autoencoder anomalies: {df_merged['Auto_Flag'].sum():,} ({df_merged['Auto_Flag'].sum()/len(df_merged)*100:.2f}%)")
    
    print(f"\n⚠️  HIGH-RISK SOFTWARE:")
    print(f"   Records flagged by ≥3 models: {len(high_risk_software):,} ({len(high_risk_software)/len(df_merged)*100:.2f}%)")
    print(f"   Records flagged by all 4 models: {len(df_merged[df_merged['Total_Flags'] == 4]):,}")
    
    print(f"\n🏢 TOP HIGH-RISK PUBLISHERS:")
    top_3_publishers = high_risk_software['Publisher'].value_counts().head(3)
    for i, (publisher, count) in enumerate(top_3_publishers.items(), 1):
        print(f"   {i}. {publisher}: {count} records")
    
    print(f"\n📦 TOP HIGH-RISK PRODUCTS:")
    top_3_products = high_risk_software['Product_Name'].value_counts().head(3)
    for i, (product, count) in enumerate(top_3_products.items(), 1):
        print(f"   {i}. {product}: {count} records")
    
    print(f"\n📁 OUTPUT FILES GENERATED:")
    print(f"   • output/results/Software_Anomalies_IF.csv")
    print(f"   • output/results/Software_Anomalies_SVM.csv")
    print(f"   • output/results/Software_Anomalies_LOF.csv")
    print(f"   • output/results/Software_Anomalies_Autoencoder.csv")
    print(f"   • Software_Flagged_By_3_Models.csv")
    
    print("\n" + "=" * 60)
    
    # Show detailed breakdown of high-risk software
    print("\n🔍 DETAILED HIGH-RISK SOFTWARE BREAKDOWN:")
    print("=" * 50)
    
    for flags in [4, 3]:
        subset = high_risk_software[high_risk_software['Total_Flags'] == flags]
        if len(subset) > 0:
            print(f"\n📋 Software flagged by {flags} model(s) ({len(subset)} records):")
            for _, row in subset.head(5).iterrows():
                models_flagged = []
                if row['IF_Flag']: models_flagged.append('IF')
                if row['SVM_Flag']: models_flagged.append('SVM')
                if row['LOF_Flag']: models_flagged.append('LOF')
                if row['Auto_Flag']: models_flagged.append('Auto')
                
                print(f"   • {row['Product_Name']} ({row['Publisher']}) - Models: {', '.join(models_flagged)}")
            
            if len(subset) > 5:
                print(f"   ... and {len(subset) - 5} more records")
    
    print("\n" + "=" * 60)
    print("MASTER SUMMARY ANALYSIS COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    main() 