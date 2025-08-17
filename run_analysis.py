#!/usr/bin/env python3
"""
Software Anomaly Detection - Main Runner Script

This script orchestrates the complete anomaly detection pipeline:
1. Isolation Forest Analysis
2. One-Class SVM Analysis  
3. Local Outlier Factor Analysis
4. Autoencoder Analysis
5. Master Summary Analysis
6. Visualization Generation

Usage:
    python3 run_analysis.py
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"🚀 {title}")
    print("=" * 80)

def print_step(step_num, description):
    """Print a formatted step"""
    print(f"\n📋 Step {step_num}: {description}")
    print("-" * 60)

def run_script(script_path, description):
    """Run a Python script and handle errors"""
    print(f"   Running: {description}")
    print(f"   Script: {script_path}")
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"   ✅ {description} completed successfully!")
            if result.stdout:
                print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"   ❌ {description} failed!")
            print(f"   Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error running {description}: {str(e)}")
        return False
    
    return True

def check_dependencies():
    """Check if required directories and files exist"""
    print_step(0, "Checking Dependencies")
    
    # Check if data file exists
    data_file = "data/Software_License_Cleaned.csv"
    if not os.path.exists(data_file):
        print(f"   ❌ Data file not found: {data_file}")
        return False
    print(f"   ✅ Data file found: {data_file}")
    
    # Check if source directories exist
    required_dirs = ["src/models", "src/utils"]
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"   ❌ Directory not found: {dir_path}")
            return False
        print(f"   ✅ Directory found: {dir_path}")
    
    # Check if model scripts exist
    model_scripts = [
        "src/models/software_if.py",
        "src/models/software_ocsvm.py", 
        "src/models/software_lof.py",
        "src/models/software_autoencoder.py",
        "src/models/software_summary.py"
    ]
    
    for script in model_scripts:
        if not os.path.exists(script):
            print(f"   ❌ Model script not found: {script}")
            return False
        print(f"   ✅ Model script found: {script}")
    
    return True

def create_directories():
    """Create output directories if they don't exist"""
    print_step(0.5, "Creating Output Directories")
    
    directories = [
        "output/results",
        "output/plots", 
        "output/reports"
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"   ✅ Created/verified: {dir_path}")

def main():
    """Main execution function"""
    print_header("SOFTWARE LICENSE ANOMALY DETECTION PIPELINE")
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependencies check failed. Please ensure all required files are present.")
        return
    
    # Create directories
    create_directories()
    
    # Define analysis steps
    analysis_steps = [
        ("src/models/software_if.py", "Isolation Forest Analysis"),
        ("src/models/software_ocsvm.py", "One-Class SVM Analysis"),
        ("src/models/software_lof.py", "Local Outlier Factor Analysis"),
        ("src/models/software_autoencoder.py", "Autoencoder Analysis"),
        ("src/models/software_summary.py", "Master Summary Analysis")
    ]
    
    # Execute each step
    for i, (script_path, description) in enumerate(analysis_steps, 1):
        print_step(i, description)
        
        start_time = time.time()
        success = run_script(script_path, description)
        end_time = time.time()
        
        if success:
            print(f"   ⏱️  Execution time: {end_time - start_time:.2f} seconds")
        else:
            print(f"\n❌ Pipeline failed at step {i}: {description}")
            return
    
    # Generate visualizations
    print_step(6, "Generating Visualizations")
    
    # Run image analyzer
    if run_script("src/utils/image_analyzer.py", "Image Analysis"):
        print("   ✅ Image analysis completed")
    
    # Create HTML viewer
    if run_script("src/utils/create_html_viewer.py", "HTML Viewer Creation"):
        print("   ✅ HTML viewer created")
    
    # Final summary
    print_header("PIPELINE COMPLETED SUCCESSFULLY!")
    
    print("\n📊 Generated Output Files:")
    print("   📁 Results:")
    results_dir = "output/results"
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            if file.endswith('.csv'):
                print(f"      • {file}")
    
    print("   📁 Plots:")
    plots_dir = "output/plots"
    if os.path.exists(plots_dir):
        plot_count = len([f for f in os.listdir(plots_dir) if f.endswith('.png')])
        print(f"      • {plot_count} visualization files")
    
    print("   📁 Reports:")
    reports_dir = "output/reports"
    if os.path.exists(reports_dir):
        for file in os.listdir(reports_dir):
            if file.endswith('.html'):
                print(f"      • {file}")
    
    print("\n🎯 Next Steps:")
    print("   1. View results in output/results/")
    print("   2. Examine visualizations in output/plots/")
    print("   3. Open output/reports/visualizations.html in your browser")
    print("   4. Check docs/ for detailed documentation")
    
    print("\n🚀 Happy Analyzing!")

if __name__ == "__main__":
    main() 