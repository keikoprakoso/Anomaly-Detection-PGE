#!/usr/bin/env python3
"""
Create HTML Viewer for Software Anomaly Detection Visualizations
"""

import os

def create_html_viewer():
    html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>Software Anomaly Detection Visualizations</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .image-container { margin: 30px 0; text-align: center; }
        img { max-width: 100%; height: auto; border: 2px solid #ddd; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; border-bottom: 3px solid #007acc; padding-bottom: 10px; }
        h2 { color: #007acc; margin-top: 30px; }
        .stats { background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .category { background: #e3f2fd; padding: 10px; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🖼️ Software Anomaly Detection Visualizations</h1>
        
        <div class="stats">
            <h3>📊 Summary</h3>
            <p><strong>Total Visualizations:</strong> 19 files</p>
            <p><strong>Total Size:</strong> 4.15 MB</p>
            <p><strong>Categories:</strong> 5 (Isolation Forest, One-Class SVM, LOF, Autoencoder, Summary)</p>
        </div>
'''

    png_files = [f for f in os.listdir('.') if f.endswith('.png')]
    png_files.sort()

    # Group by category
    categories = {
        'Isolation Forest': [f for f in png_files if 'isolation_forest' in f],
        'One-Class SVM': [f for f in png_files if 'oneclass_svm' in f],
        'Local Outlier Factor': [f for f in png_files if 'lof' in f],
        'Autoencoder': [f for f in png_files if 'autoencoder' in f],
        'Summary & Analysis': [f for f in png_files if any(x in f for x in ['high_risk', 'model_comparison', 'flag_distribution', 'comprehensive_analysis'])]
    }

    for category, files in categories.items():
        if files:
            html_content += f'<div class="category"><h2>🎯 {category}</h2></div>'
            for file in files:
                title = file.replace('.png', '').replace('_', ' ').title()
                html_content += f'''
                <div class="image-container">
                    <h3>{title}</h3>
                    <img src="{file}" alt="{title}" onclick="window.open('{file}', '_blank')" style="cursor: pointer;">
                    <p><em>Click image to open in new tab</em></p>
                </div>
                '''

    html_content += '''
    </div>
    <script>
        // Add click handlers for better UX
        document.querySelectorAll('img').forEach(img => {
            img.addEventListener('click', function() {
                window.open(this.src, '_blank');
            });
        });
    </script>
</body>
</html>
'''

    with open('visualizations.html', 'w') as f:
        f.write(html_content)

    print('✅ Created visualizations.html - Open this file in your browser to view all visualizations!')

if __name__ == "__main__":
    create_html_viewer() 