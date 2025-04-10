# PinkCC - Medical Image Viewer

A comprehensive medical imaging application for viewing and analyzing CT scans and segmentation data, with support for 2D, Multi-Planar Reconstruction (MPR), and 3D visualization.

## Features

- **Data Management**: Load and browse patient CT data from different institutions (MSKCC, TCGA)
- **Visualization Modes**:
  - **2D Slice Viewer**: View axial, coronal, and sagittal slices with adjustable window level/width
  - **MPR View**: Synchronized multi-planar reconstruction with crosshairs
  - **3D Visualization**: Interactive 3D rendering with adjustable threshold and opacity
- **Segmentation Overlay**: Display tumor (red) and metastasis (green) segmentation overlays
- **Performance Optimization**: Image caching for improved responsiveness
- **Customization**: Adjustable window presets, quality settings, and display options

## Project Structure

- **local_view.py**: Main application with the medical image viewer interface
- **explore_dataset.py**: Helper functions for dataset exploration and patient ID management
- **assets/**: Contains CSS files for styling
- **DatasetChallenge/**: Dataset directory containing CT scans and segmentation files
  - **CT/**: CT scan files in NIfTI (.nii.gz) format
    - **MSKCC/**: Memorial Sloan Kettering Cancer Center dataset
    - **TCGA/**: The Cancer Genome Atlas dataset
  - **Segmentation/**: Segmentation mask files
    - **MSKCC/**: Segmentation files for MSKCC dataset
    - **TCGA/**: Segmentation files for TCGA dataset

## Requirements

The application requires the following Python packages:
- numpy
- nibabel
- matplotlib
- tkinter
- pillow
- plotly
- scikit-image
- pandas

Install the requirements using:
```
pip install -r requirements.txt
```

## Usage

1. Run the local viewer application:
   ```
   python local_view.py
   ```

2. The application will open with the following interface:
   - Select an institution and patient ID from the dropdown menus
   - Use the tabs to switch between 2D, MPR, and 3D visualization modes
   - Adjust window level/width using sliders or presets
   - Toggle segmentation overlay and other display options

3. Explore the dataset separately:
   ```
   python explore_dataset.py
   ```

## Implementation Details

### local_view.py

The main application file implementing the medical image viewer with a tkinter GUI. Key components:
- `MedicalImageViewer` class: Main application class
- Helper functions for loading and processing medical images
- 2D, MPR, and 3D visualization implementations
- Image windowing and segmentation overlay functionality

### explore_dataset.py

Utility module for exploring and managing dataset information:
- `scan_dataset()`: Scans dataset directory structure
- `get_patient_ids()`: Extracts unique patient IDs from available files
- `generate_dataset_report()`: Creates a summary of available images and patients

## License

This project is for educational and research purposes.