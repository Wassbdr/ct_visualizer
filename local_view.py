"""
CT Visualizer - Medical Image Viewer
===================================

This module implements an interactive medical image viewer for CT data with segmentation overlays.
The application provides multiple visualization modes including:
- 2D slice viewer with orientation options (axial, coronal, sagittal)
- Multi-Planar Reconstruction (MPR) view showing all three orientations simultaneously
- Interactive 3D volume rendering using Plotly

Features:
- CT data visualization with customizable window level/width
- Segmentation overlay with different colors for tumor (red) and metastasis (green)
- Proper anatomical orientation and labeling
- Multiple windowing presets (bone, soft tissue)
- Interactive slicing through volume data
- 3D volume rendering with adjustable threshold and quality settings
- Integrated dataset browser with institution and patient selection

Dependencies:
- nibabel: for NIFTI medical image file loading
- matplotlib: for 2D and MPR visualizations
- plotly: for interactive 3D visualizations
- tkinter: for the GUI framework
- skimage: for marching cubes surface extraction
- numpy: for numerical operations

Usage:
    Run the module directly to launch the application:
    ```
    python local_view.py
    ```

Author: PinkCC team
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from explore_dataset import get_patient_ids
import threading
from matplotlib.colors import ListedColormap
from PIL import Image, ImageTk
import plotly.graph_objects as go
from skimage import measure
import io

# Cache for storing loaded data to improve performance
image_cache = {}

# Helper functions
def get_voxel_spacing(affine):
    """Extract voxel spacing from affine matrix"""
    return np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))

# Function to load and cache CT data to improve performance
def load_ct_data(institution, patient_id):
    cache_key = f"{institution}_{patient_id}_ct"
    if cache_key in image_cache:
        return image_cache[cache_key]
    
    ct_path = f"DatasetChallenge/CT/{institution}/{patient_id}.nii.gz"
    if not os.path.exists(ct_path):
        return None, None
    
    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata()
    image_cache[cache_key] = (ct_data, ct_img.affine)
    return ct_data, ct_img.affine

# Function to load and cache segmentation data
def load_seg_data(institution, patient_id):
    cache_key = f"{institution}_{patient_id}_seg"
    if cache_key in image_cache:
        return image_cache[cache_key]
    
    seg_path = f"DatasetChallenge/Segmentation/{institution}/{patient_id}.nii.gz"
    if not os.path.exists(seg_path):
        return None
    
    seg_img = nib.load(seg_path)
    seg_data = seg_img.get_fdata()
    image_cache[cache_key] = seg_data
    return seg_data

def update_3d_view(patient_id, institution, threshold, opacity, quality, display_options):
    if not patient_id:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data available")
        return empty_fig
    
    # Load CT data from cache
    ct_data, ct_affine = load_ct_data(institution, patient_id)
    if ct_data is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="CT data not found")
        return empty_fig
    
    try:
        # Create a 3D figure
        fig = go.Figure()
        
        # Get the voxel spacing for proper scaling
        voxel_spacing = get_voxel_spacing(ct_affine)
        
        # Set downsampling factor based on quality setting
        if quality == 'low':
            downsample_factor = 8
        elif quality == 'high':
            downsample_factor = 3
        else:  # medium
            downsample_factor = 4
            
        
        # Downsample data for better performance
        downsampled_shape = np.array(ct_data.shape) // downsample_factor
        indices = np.round(np.linspace(0, ct_data.shape[0]-1, downsampled_shape[0])).astype(int)
        jndices = np.round(np.linspace(0, ct_data.shape[1]-1, downsampled_shape[1])).astype(int)
        kndices = np.round(np.linspace(0, ct_data.shape[2]-1, downsampled_shape[2])).astype(int)
        downsampled_data = ct_data[np.ix_(indices, jndices, kndices)]
        
        # Create isosurface for CT data
        try:
            # Use marching cubes to get surface vertices
            verts, faces, _, _ = measure.marching_cubes(downsampled_data, threshold)
            
            # Apply proper physical scaling based on voxel spacing and downsampling
            shape_ratio = np.array(ct_data.shape) / np.array(downsampled_shape)
            spacing_factors = voxel_spacing * shape_ratio
            
            # Scale each dimension appropriately
            verts[:, 0] = verts[:, 0] * spacing_factors[0]
            verts[:, 1] = verts[:, 1] * spacing_factors[1]
            verts[:, 2] = verts[:, 2] * spacing_factors[2]
            
            # Add CT surface using mesh3d
            x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
            i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
            
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                color='lightgray',
                opacity=opacity,
                name='CT Surface',
                lighting=dict(ambient=0.6, diffuse=0.8, specular=0.4, roughness=0.5),
                lightposition=dict(x=0, y=0, z=100000)
            ))
            
        except Exception as e:
            print(f"Error creating isosurface: {e}")
            fig.add_annotation(
                text=f"Error: Could not generate surface at threshold {threshold}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
        
        # Add segmentation if selected and available
        show_seg = display_options and 'show_seg' in display_options
        if show_seg:
            seg_data = load_seg_data(institution, patient_id)
            if seg_data is not None:
                # Downsample segmentation data
                downsampled_seg = seg_data[np.ix_(indices, jndices, kndices)]
                
                try:
                    # Create binary mask
                    seg_mask = downsampled_seg > 0
                    
                    if np.any(seg_mask):
                        # Use marching cubes to get segmentation surface
                        seg_verts, seg_faces, _, _ = measure.marching_cubes(seg_mask, 0.5)
                        
                        # Scale vertices with proper physical scaling
                        seg_verts[:, 0] = seg_verts[:, 0] * spacing_factors[0]
                        seg_verts[:, 1] = seg_verts[:, 1] * spacing_factors[1]
                        seg_verts[:, 2] = seg_verts[:, 2] * spacing_factors[2]
                        
                        # Add segmentation surface
                        x, y, z = seg_verts[:, 0], seg_verts[:, 1], seg_verts[:, 2]
                        i, j, k = seg_faces[:, 0], seg_faces[:, 1], seg_faces[:, 2]
                        
                        fig.add_trace(go.Mesh3d(
                            x=x, y=y, z=z,
                            i=i, j=j, k=k,
                            color='red',
                            opacity=1.0,  # Increased to full opacity for better visibility
                            name='Segmentation',
                            lighting=dict(ambient=0.8, diffuse=0.9)  # More ambient light for the segmentation
                        ))
                except Exception as e:
                    print(f"Error creating segmentation surface: {e}")
        
        # Set camera position for better initial view
        camera = dict(
            eye=dict(x=2, y=2, z=2)
        )
        
        # Show axes if selected
        show_axes = display_options and 'show_axes' in display_options
        
        # Show patient ID if selected
        show_id = display_options and 'show_id' in display_options
        
        title = "3D View"
        if show_id:
            title += f" - Patient {patient_id}"
        
        # Force equal scaling on all axes for proper proportions
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(visible=show_axes),
                yaxis=dict(visible=show_axes),
                zaxis=dict(visible=show_axes),
                aspectmode='data'
            ),
            scene_camera=camera,
            margin=dict(l=0, r=0, t=50, b=0),
            height=700
        )
        
        return fig
        
    except Exception as e:
        print(f"Error generating 3D visualization: {e}")
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title=f"Error generating 3D view: {str(e)}",
            height=700
        )
        return empty_fig

class MedicalImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Image Viewer")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.patient_id = None
        self.institution = None
        self.current_axis = 2  # Default to axial view
        self.current_slice = None
        self.ct_data = None
        self.ct_affine = None
        self.seg_data = None
        self.window_center = 40
        self.window_width = 400
        self.show_segmentation = True
        self.show_crosshairs = True
        self.show_orientation_labels = True
        
        # Window presets
        self.window_presets = {
            'bone': {'wc': 400, 'ww': 1500},
            'soft_tissue': {'wc': 40, 'ww': 400}
        }
        
        # Get patient IDs
        self.patient_ids = get_patient_ids()
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create patient selection controls
        self.create_patient_controls()
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create 2D view tab
        self.view_2d_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.view_2d_tab, text="2D View")
        self.create_2d_view()
        
        # Create MPR view tab
        self.mpr_view_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.mpr_view_tab, text="MPR View")
        self.create_mpr_view()
        
        # Create 3D view tab
        self.view_3d_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.view_3d_tab, text="3D View")
        self.create_3d_view()
        
        # Set up notebook tab change event
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
        # Initialize with first patient
        if 'MSKCC' in self.patient_ids and self.patient_ids['MSKCC']:
            self.institution_var.set('MSKCC')
            self.update_patient_dropdown()
            if self.patient_dropdown['values']:
                self.patient_dropdown.current(0)
                self.load_patient()
    
    def create_patient_controls(self):
        """Create patient selection controls"""
        control_frame = ttk.Frame(self.main_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Institution selection
        ttk.Label(control_frame, text="Institution:").pack(side=tk.LEFT, padx=5)
        self.institution_var = tk.StringVar()
        institutions = ['MSKCC', 'TCGA']
        self.institution_dropdown = ttk.Combobox(control_frame, textvariable=self.institution_var, 
                                                values=institutions, state="readonly", width=10)
        self.institution_dropdown.pack(side=tk.LEFT, padx=5)
        self.institution_dropdown.bind("<<ComboboxSelected>>", self.on_institution_changed)
        
        # Patient selection
        ttk.Label(control_frame, text="Patient ID:").pack(side=tk.LEFT, padx=5)
        self.patient_var = tk.StringVar()
        self.patient_dropdown = ttk.Combobox(control_frame, textvariable=self.patient_var, 
                                           width=15, state="readonly")
        self.patient_dropdown.pack(side=tk.LEFT, padx=5)
        self.patient_dropdown.bind("<<ComboboxSelected>>", self.on_patient_changed)
        
        # Window presets
        ttk.Label(control_frame, text="Window Preset:").pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Bone", command=self.set_bone_window).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Soft Tissue", command=self.set_soft_tissue_window).pack(side=tk.LEFT, padx=2)
    
    def set_bone_window(self):
        """Set bone window preset"""
        self.wc_var.set(self.window_presets['bone']['wc']) 
        self.ww_var.set(self.window_presets['bone']['ww'])
        self.on_window_changed()
    
    def set_soft_tissue_window(self):
        """Set soft tissue window preset"""
        self.wc_var.set(self.window_presets['soft_tissue']['wc'])
        self.ww_var.set(self.window_presets['soft_tissue']['ww'])
        self.on_window_changed()

    def on_institution_changed(self, event=None):
        """Handle institution change"""
        self.update_patient_dropdown()
    
    def update_patient_dropdown(self):
        """Update the patient dropdown based on selected institution"""
        institution = self.institution_var.get()
        self.patient_dropdown['values'] = self.patient_ids.get(institution, [])
        if self.patient_dropdown['values']:
            self.patient_dropdown.current(0)

    def on_patient_changed(self, event=None):
        """Handle patient selection change"""
        self.load_patient()

    def on_view_changed(self):
        """Handle view orientation change"""
        if self.ct_data is None:
            return
        
        self.current_axis = int(self.view_var.get())
        
        # Update slice slider range
        max_slice = self.ct_data.shape[self.current_axis] - 1
        self.slice_scale.configure(to=max_slice)
        self.slice_var.set(max_slice // 2)
        
        self.update_2d_view()

    def on_slice_changed(self, event=None):
        """Handle slice slider change"""
        if self.ct_data is None:
            return
        
        self.update_2d_view()

    def on_window_changed(self, event=None):
        """Handle window level/width change"""
        self.window_center = self.wc_var.get()
        self.window_width = self.ww_var.get()
        self.update_2d_view()
        if hasattr(self, 'update_mpr_view'):
            self.update_mpr_view()

    def on_mpr_slice_changed(self, event=None):
        """Handle MPR slice changes"""
        if self.ct_data is None:
            return
        
        self.update_mpr_view()

    def on_tab_changed(self, event):
        """Handle tab change events"""
        tab_id = self.notebook.index("current")
        
        # Update the relevant view when switching tabs
        if tab_id == 0:  # 2D View
            self.update_2d_view()
        elif tab_id == 1:  # MPR View
            self.update_mpr_view()
        elif tab_id == 2 and self.ct_data is not None:  # 3D View - don't auto-render, wait for button
            pass

    # Add remaining methods...
    def create_2d_view(self):
        """Create the 2D slice view tab"""
        # Split into left control panel and right image panel
        control_panel = ttk.Frame(self.view_2d_tab)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        image_panel = ttk.Frame(self.view_2d_tab)
        image_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Controls for 2D view
        ttk.Label(control_panel, text="View Controls").pack(anchor=tk.W)
        ttk.Separator(control_panel, orient='horizontal').pack(fill=tk.X, pady=5)
        
        # View orientation selection
        self.view_var = tk.StringVar(value="2")
        ttk.Radiobutton(control_panel, text="Axial", variable=self.view_var, value="2", 
                       command=self.on_view_changed).pack(anchor=tk.W)
        ttk.Radiobutton(control_panel, text="Coronal", variable=self.view_var, value="1", 
                       command=self.on_view_changed).pack(anchor=tk.W)
        ttk.Radiobutton(control_panel, text="Sagittal", variable=self.view_var, value="0", 
                       command=self.on_view_changed).pack(anchor=tk.W)
        
        # Slice slider
        ttk.Label(control_panel, text="Slice:").pack(anchor=tk.W, pady=(10,0))
        self.slice_var = tk.IntVar(value=0)
        self.slice_scale = ttk.Scale(control_panel, from_=0, to=100, orient=tk.HORIZONTAL,
                                   variable=self.slice_var, command=self.on_slice_changed)
        self.slice_scale.pack(fill=tk.X, pady=5)
        
        # Window level controls
        ttk.Label(control_panel, text="Window Controls").pack(anchor=tk.W, pady=(10,0))
        ttk.Separator(control_panel, orient='horizontal').pack(fill=tk.X, pady=5)
        
        ttk.Label(control_panel, text="Window Center:").pack(anchor=tk.W)
        self.wc_var = tk.IntVar(value=self.window_center)
        self.wc_scale = ttk.Scale(control_panel, from_=-400, to=400, orient=tk.HORIZONTAL,
                                variable=self.wc_var, command=self.on_window_changed)
        self.wc_scale.pack(fill=tk.X, pady=5)
        
        ttk.Label(control_panel, text="Window Width:").pack(anchor=tk.W)
        self.ww_var = tk.IntVar(value=self.window_width)
        self.ww_scale = ttk.Scale(control_panel, from_=1, to=2000, orient=tk.HORIZONTAL,
                                variable=self.ww_var, command=self.on_window_changed)
        self.ww_scale.pack(fill=tk.X, pady=5)
        
        # Display options
        ttk.Label(control_panel, text="Display Options").pack(anchor=tk.W, pady=(10,0))
        ttk.Separator(control_panel, orient='horizontal').pack(fill=tk.X, pady=5)
        
        self.show_seg_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_panel, text="Show Segmentation", variable=self.show_seg_var,
                       command=self.update_2d_view).pack(anchor=tk.W)
        
        self.show_labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_panel, text="Show Orientation Labels", variable=self.show_labels_var,
                       command=self.update_2d_view).pack(anchor=tk.W)
        
        # Create figure for 2D view with constrained layout
        self.fig_2d = Figure(figsize=(8, 8), dpi=100, constrained_layout=True)
        self.ax_2d = self.fig_2d.add_subplot(111)
        
        self.canvas_2d = FigureCanvasTkAgg(self.fig_2d, master=image_panel)
        self.canvas_2d.draw()
        self.canvas_2d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.canvas_2d, image_panel)
        toolbar.update()
    
    def create_mpr_view(self):
        """Create the multi-planar reconstruction view tab"""
        # Split into left control panel and right image panel
        control_panel = ttk.Frame(self.mpr_view_tab)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        image_panel = ttk.Frame(self.mpr_view_tab)
        image_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Controls for MPR view
        ttk.Label(control_panel, text="MPR Controls").pack(anchor=tk.W)
        ttk.Separator(control_panel, orient='horizontal').pack(fill=tk.X, pady=5)
        
        # Slice sliders
        ttk.Label(control_panel, text="Axial Slice:").pack(anchor=tk.W, pady=(10,0))
        self.axial_slice_var = tk.IntVar(value=0)
        self.axial_scale = ttk.Scale(control_panel, from_=0, to=100, orient=tk.HORIZONTAL,
                                   variable=self.axial_slice_var, command=self.on_mpr_slice_changed)
        self.axial_scale.pack(fill=tk.X, pady=5)
        
        ttk.Label(control_panel, text="Coronal Slice:").pack(anchor=tk.W, pady=(10,0))
        self.coronal_slice_var = tk.IntVar(value=0)
        self.coronal_scale = ttk.Scale(control_panel, from_=0, to=100, orient=tk.HORIZONTAL,
                                     variable=self.coronal_slice_var, command=self.on_mpr_slice_changed)
        self.coronal_scale.pack(fill=tk.X, pady=5)
        
        ttk.Label(control_panel, text="Sagittal Slice:").pack(anchor=tk.W, pady=(10,0))
        self.sagittal_slice_var = tk.IntVar(value=0)
        self.sagittal_scale = ttk.Scale(control_panel, from_=0, to=100, orient=tk.HORIZONTAL,
                                      variable=self.sagittal_slice_var, command=self.on_mpr_slice_changed)
        self.sagittal_scale.pack(fill=tk.X, pady=5)
        
        # MPR display options
        ttk.Label(control_panel, text="Display Options").pack(anchor=tk.W, pady=(10,0))
        ttk.Separator(control_panel, orient='horizontal').pack(fill=tk.X, pady=5)
        
        self.mpr_show_seg_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_panel, text="Show Segmentation", variable=self.mpr_show_seg_var,
                       command=self.update_mpr_view).pack(anchor=tk.W)
        
        self.mpr_show_labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_panel, text="Show Orientation Labels", variable=self.mpr_show_labels_var,
                       command=self.update_mpr_view).pack(anchor=tk.W)
        
        self.mpr_show_crosshairs_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_panel, text="Show Crosshairs", variable=self.mpr_show_crosshairs_var,
                       command=self.update_mpr_view).pack(anchor=tk.W)
        
        # Create figure for MPR view
        self.fig_mpr = Figure(figsize=(10, 5), dpi=100)
        gs = gridspec.GridSpec(1, 3)
        self.ax_sag = self.fig_mpr.add_subplot(gs[0, 0])
        self.ax_cor = self.fig_mpr.add_subplot(gs[0, 1])
        self.ax_ax = self.fig_mpr.add_subplot(gs[0, 2])
        
        self.canvas_mpr = FigureCanvasTkAgg(self.fig_mpr, master=image_panel)
        self.canvas_mpr.draw()
        self.canvas_mpr.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.canvas_mpr, image_panel)
        toolbar.update()
    
    def create_3d_view(self):
        """Create the 3D view tab with interactive Plotly rendering"""
        # Split into left control panel and right image panel
        control_panel = ttk.Frame(self.view_3d_tab)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        render_panel = ttk.Frame(self.view_3d_tab)
        render_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Controls for 3D view
        ttk.Label(control_panel, text="3D Visualization Controls").pack(anchor=tk.W)
        ttk.Separator(control_panel, orient='horizontal').pack(fill=tk.X, pady=5)
        
        ttk.Label(control_panel, text="Density Threshold:").pack(anchor=tk.W, pady=(10,0))
        self.threshold_var = tk.IntVar(value=-300)
        self.threshold_scale = ttk.Scale(control_panel, from_=-600, to=600, orient=tk.HORIZONTAL,
                                       variable=self.threshold_var)
        self.threshold_scale.pack(fill=tk.X, pady=5)
        
        ttk.Label(control_panel, text="Opacity:").pack(anchor=tk.W, pady=(10,0))
        self.opacity_var = tk.DoubleVar(value=0.8)
        self.opacity_scale = ttk.Scale(control_panel, from_=0.1, to=1.0, orient=tk.HORIZONTAL,
                                     variable=self.opacity_var)
        self.opacity_scale.pack(fill=tk.X, pady=5)
        
        # Quality settings
        ttk.Label(control_panel, text="Quality:").pack(anchor=tk.W, pady=(10,0))
        self.quality_var = tk.StringVar(value="high")
        ttk.Radiobutton(control_panel, text="Low", variable=self.quality_var, value="low").pack(anchor=tk.W)
        ttk.Radiobutton(control_panel, text="Medium", variable=self.quality_var, value="medium").pack(anchor=tk.W)
        ttk.Radiobutton(control_panel, text="High", variable=self.quality_var, value="high").pack(anchor=tk.W)
        
        # Color settings
        ttk.Label(control_panel, text="CT Color:").pack(anchor=tk.W, pady=(10,0))
        self.ct_color_var = tk.StringVar(value="lightgray")
        ttk.Radiobutton(control_panel, text="Bone", variable=self.ct_color_var, value="bone").pack(anchor=tk.W)
        ttk.Radiobutton(control_panel, text="White", variable=self.ct_color_var, value="white").pack(anchor=tk.W)
        ttk.Radiobutton(control_panel, text="Light Gray", variable=self.ct_color_var, value="lightgray").pack(anchor=tk.W)
        ttk.Radiobutton(control_panel, text="Tan", variable=self.ct_color_var, value="tan").pack(anchor=tk.W)
        
        # Display options
        ttk.Label(control_panel, text="Display Options").pack(anchor=tk.W, pady=(10,0))
        ttk.Separator(control_panel, orient='horizontal').pack(fill=tk.X, pady=5)
        
        self.show_tumor_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_panel, text="Show Tumor", variable=self.show_tumor_var).pack(anchor=tk.W)
        
        self.show_metastasis_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_panel, text="Show Metastasis", variable=self.show_metastasis_var).pack(anchor=tk.W)
        
        self.show_axes_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_panel, text="Show Axes", variable=self.show_axes_var).pack(anchor=tk.W)
        
        self.show_id_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_panel, text="Show Patient ID", variable=self.show_id_var).pack(anchor=tk.W)
        
        # Render button
        ttk.Button(control_panel, text="Render 3D View", command=self.render_3d_view).pack(anchor=tk.W, pady=(20,0))
        
        # Progress indicator
        self.progress_var = tk.StringVar(value="")
        ttk.Label(control_panel, textvariable=self.progress_var).pack(anchor=tk.W, pady=(10,0))
        
        # Create a frame for Plotly rendering
        self.plotly_frame = ttk.Frame(render_panel)
        self.plotly_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a label with instructions
        msg = ttk.Label(self.plotly_frame, text="Click 'Render 3D View' to generate interactive 3D visualization", 
                      font=("Arial", 12), foreground="gray")
        msg.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def render_3d_view(self):
        """Render the 3D visualization in a separate thread to keep UI responsive"""
        if self.ct_data is None:
            messagebox.showerror("Error", "No CT data loaded")
            return
            
        # Check if already rendering
        if hasattr(self, '_rendering_thread') and self._rendering_thread.is_alive():
            messagebox.showinfo("Info", "3D rendering is already in progress")
            return
            
        # Start progress indicator
        self.progress_var.set("Starting 3D rendering...")
        self.root.update_idletasks()
        self.update_progress()
            
        # Start rendering in a separate thread
        self._rendering_thread = threading.Thread(target=self._render_3d_view_thread)
        self._rendering_thread.daemon = True
        self._rendering_thread.start()

    def _render_3d_view_thread(self):
        """Thread function for 3D rendering with Plotly (more compatible)"""
        try:
            # Get parameters
            threshold = self.threshold_var.get()
            opacity = self.opacity_var.get()
            quality = self.quality_var.get()
            show_tumor = self.show_tumor_var.get()
            show_metastasis = self.show_metastasis_var.get()
            show_axes = self.show_axes_var.get()
            show_id = self.show_id_var.get()
            ct_color = self.ct_color_var.get()
            
            # Create display options dictionary
            display_options = {
                'show_tumor': show_tumor,
                'show_metastasis': show_metastasis,
                'show_axes': show_axes,
                'show_id': show_id
            }
            
            # Call the plotting function
            self.root.after(0, lambda: self.progress_var.set("Creating 3D model..."))
            fig = self.create_interactive_3d_plot(threshold, opacity, quality, ct_color, display_options)
            
            # Display the interactive figure in a web browser
            self.root.after(0, lambda: self.progress_var.set("Opening interactive 3D view..."))
            import plotly.io as pio
            pio.renderers.default = 'browser'
            fig.show()
            
            self.root.after(0, lambda: self.progress_var.set("Interactive 3D view opened in browser"))
            
        except Exception as e:
            print(f"3D rendering error: {str(e)}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("Error", f"3D rendering failed: {str(e)}"))
            self.root.after(0, lambda: self.progress_var.set(""))

    def create_interactive_3d_plot(self, threshold, opacity, quality, ct_color, display_options):
        """Create an interactive 3D plot using Plotly"""
        import plotly.graph_objects as go
        
        # Set downsampling factor based on quality setting
        if quality == 'low':
            downsample_factor = 8
        elif quality == 'high':
            downsample_factor = 3
        else:  # medium
            downsample_factor = 5
            
        # Downsample data for better performance
        self.progress_var.set("Downsampling data...")
        self.root.update_idletasks()
        
        downsampled_shape = np.array(self.ct_data.shape) // downsample_factor
        indices = np.round(np.linspace(0, self.ct_data.shape[0]-1, downsampled_shape[0])).astype(int)
        jndices = np.round(np.linspace(0, self.ct_data.shape[1]-1, downsampled_shape[1])).astype(int)
        kndices = np.round(np.linspace(0, self.ct_data.shape[2]-1, downsampled_shape[2])).astype(int)
        downsampled_data = self.ct_data[np.ix_(indices, jndices, kndices)]
        
        # Get voxel spacing
        spacing = self.get_voxel_spacing(self.ct_affine)
        
        # Create a 3D figure
        fig = go.Figure()
        
        # Create isosurface for CT data
        self.progress_var.set("Creating CT surface...")
        self.root.update_idletasks()
        try:
            # Use marching cubes to get surface vertices
            verts, faces, _, _ = measure.marching_cubes(downsampled_data, threshold)
            
            # Apply proper physical scaling based on voxel spacing and downsampling
            shape_ratio = np.array(self.ct_data.shape) / np.array(downsampled_shape)
            spacing_factors = spacing * shape_ratio
            
            # Scale each dimension appropriately
            verts[:, 0] = verts[:, 0] * spacing_factors[0]
            verts[:, 1] = verts[:, 1] * spacing_factors[1]
            verts[:, 2] = verts[:, 2] * spacing_factors[2]
            
            # Add CT surface using mesh3d
            x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
            i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
            
            # Get color based on selection
            color_dict = {
                "bone": "#E8E4D0",  # Light bone color
                "white": "#FFFFFF",  # White
                "lightgray": "#CCCCCC",  # Light gray
                "tan": "#D2B48C"  # Tan
            }
            ct_hex_color = color_dict.get(ct_color, "#E8E4D0")
            
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                color=ct_hex_color,
                opacity=opacity,
                name='CT Surface',
                lighting=dict(ambient=0.6, diffuse=0.8, specular=0.4, roughness=0.5),
                lightposition=dict(x=0, y=0, z=100000)
            ))
        except Exception as e:
            print(f"Error creating CT isosurface: {e}")
        
        # Add segmentation if available
        if self.seg_data is not None:
            # Downsample segmentation
            downsampled_seg = self.seg_data[np.ix_(indices, jndices, kndices)]
            
            # Show tumor (value 1)
            if display_options.get('show_tumor', True):
                self.progress_var.set("Creating tumor surface...")
                self.root.update_idletasks()
                tumor_mask = (downsampled_seg == 1).astype(float)
                
                if np.any(tumor_mask):
                    try:
                        # Use marching cubes to get segmentation surface
                        seg_verts, seg_faces, _, _ = measure.marching_cubes(tumor_mask, 0.5)
                        
                        # Scale vertices with proper physical scaling
                        seg_verts[:, 0] = seg_verts[:, 0] * spacing_factors[0]
                        seg_verts[:, 1] = seg_verts[:, 1] * spacing_factors[1]
                        seg_verts[:, 2] = seg_verts[:, 2] * spacing_factors[2]
                        
                        # Add tumor surface
                        x, y, z = seg_verts[:, 0], seg_verts[:, 1], seg_verts[:, 2]
                        i, j, k = seg_faces[:, 0], seg_faces[:, 1], seg_faces[:, 2]
                        
                        fig.add_trace(go.Mesh3d(
                            x=x, y=y, z=z,
                            i=i, j=j, k=k,
                            color='red',
                            opacity=0.85,
                            name='Tumor',
                            lighting=dict(ambient=0.8, diffuse=0.9)
                        ))
                    except Exception as e:
                        print(f"Error creating tumor surface: {e}")
            
            # Show metastasis (value 2)
            if display_options.get('show_metastasis', True):
                self.progress_var.set("Creating metastasis surface...")
                self.root.update_idletasks()
                meta_mask = (downsampled_seg == 2).astype(float)
                
                if np.any(meta_mask):
                    try:
                        # Use marching cubes to get segmentation surface
                        meta_verts, meta_faces, _, _ = measure.marching_cubes(meta_mask, 0.5)
                        
                        # Scale vertices with proper physical scaling
                        meta_verts[:, 0] = meta_verts[:, 0] * spacing_factors[0]
                        meta_verts[:, 1] = meta_verts[:, 1] * spacing_factors[1]
                        meta_verts[:, 2] = meta_verts[:, 2] * spacing_factors[2]
                        
                        # Add metastasis surface
                        x, y, z = meta_verts[:, 0], meta_verts[:, 1], meta_verts[:, 2]
                        i, j, k = meta_faces[:, 0], meta_faces[:, 1], meta_faces[:, 2]
                        
                        fig.add_trace(go.Mesh3d(
                            x=x, y=y, z=z,
                            i=i, j=j, k=k,
                            color='green',
                            opacity=0.85,
                            name='Metastasis',
                            lighting=dict(ambient=0.8, diffuse=0.9)
                        ))
                    except Exception as e:
                        print(f"Error creating metastasis surface: {e}")
        
        # Set camera position for better initial view
        camera = dict(eye=dict(x=2, y=2, z=2))
        
        # Show patient ID if selected
        title = "3D View"
        if display_options.get('show_id', True):
            title += f" - Patient {self.patient_id}"
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(visible=display_options.get('show_axes', True)),
                yaxis=dict(visible=display_options.get('show_axes', True)),
                zaxis=dict(visible=display_options.get('show_axes', True)),
                aspectmode='data'
            ),
            scene_camera=camera,
            margin=dict(l=0, r=0, t=50, b=0),
            height=700
        )
        
        return fig

    def update_progress(self):
        """Update the progress indicator with animation"""
        if hasattr(self, '_rendering_thread') and self._rendering_thread.is_alive():
            # Update progress message with animation
            current_text = self.progress_var.get()
            if current_text.endswith("..."):
                self.progress_var.set(current_text[:-3])
            else:
                self.progress_var.set(current_text + ".")
            
            # Schedule next update
            self.root.after(300, self.update_progress)
        else:
            # If thread is done, clear progress after a delay
            if not self.progress_var.get().startswith("Rendering complete"):
                self.progress_var.set("")

    def update_2d_view(self):
        """Update the 2D slice view with separate colors for tumor and metastasis"""
        if self.ct_data is None:
            return
        
        # Clear the axis
        self.ax_2d.clear()
        
        # Get current slice
        slice_idx = self.slice_var.get()
        
        # Get voxel spacing for aspect ratio calculation
        spacing = self.get_voxel_spacing(self.ct_affine)
        
        if self.current_axis == 0:  # Sagittal
            ct_slice = self.ct_data[slice_idx, :, :]
            orientation = "Sagittal"
            x_label = "Anterior → Posterior"
            y_label = "Inferior → Superior"
            # Apply a width adjustment factor to make sagittal view wider
            width_adjustment = 5  # Increase this value to make the view wider
            aspect_ratio = (spacing[2] / spacing[1]) / width_adjustment
            if self.seg_data is not None:
                seg_slice = self.seg_data[slice_idx, :, :]
        elif self.current_axis == 1:  # Coronal
            ct_slice = self.ct_data[:, slice_idx, :]
            orientation = "Coronal"
            x_label = "Left → Right"
            y_label = "Inferior → Superior"
            # Apply a width adjustment factor to make coronal view wider
            width_adjustment = 1.5  # Increase this value to make the view wider
            aspect_ratio = (spacing[2] / spacing[0]) / width_adjustment
            if self.seg_data is not None:
                seg_slice = self.seg_data[:, slice_idx, :]
        else:  # Axial
            ct_slice = self.ct_data[:, :, slice_idx]
            orientation = "Axial"
            x_label = "Left → Right"
            y_label = "Posterior → Anterior"
            # Keep axial view with anatomically correct aspect ratio
            aspect_ratio = spacing[1] / spacing[0]
            if self.seg_data is not None:
                seg_slice = self.seg_data[:, :, slice_idx]
        
        # Calculate image extent to maintain correct aspect ratio
        height, width = ct_slice.shape
        extent = [0, width, 0, height]
        
        # Apply window settings
        windowed_slice = self.apply_window(ct_slice, self.window_center, self.window_width)
        
        # Set aspect ratio BEFORE displaying any images
        self.ax_2d.set_aspect(aspect_ratio)
        
        # Display CT image with the correct extent
        self.ax_2d.imshow(windowed_slice.T, cmap='gray', origin='lower', extent=extent)
        
        # Display segmentation overlay if available and selected
        if self.show_seg_var.get() and self.seg_data is not None:
            # Create tumor mask (assuming tumor is label 1)
            tumor_mask = (seg_slice == 1).astype(float)
            if np.any(tumor_mask > 0):
                # Use the same extent as the CT image
                self.ax_2d.imshow(tumor_mask.T, cmap=self.create_color_cmap(color='red'), origin='lower', 
                                alpha=0.6, extent=extent)
                
            # Create metastasis mask (assuming metastasis is label 2)
            meta_mask = (seg_slice == 2).astype(float)
            if np.any(meta_mask > 0):
                # Use the same extent as the CT image
                self.ax_2d.imshow(meta_mask.T, cmap=self.create_color_cmap(color='green'), origin='lower', 
                                alpha=0.6, extent=extent)
        
        # Add orientation labels if selected
        if self.show_labels_var.get():
            self.ax_2d.set_xlabel(x_label)
            self.ax_2d.set_ylabel(y_label)
        
        # Set title
        self.ax_2d.set_title(f"{orientation} View - Patient {self.patient_id} - Slice {slice_idx} - Window: {self.window_center}/{self.window_width}")
        
        # Just draw the canvas directly
        self.canvas_2d.draw()

    def create_color_cmap(self, color='red', alpha=0.6):
        """Create a colormap with the specified color"""
        if color == 'red':
            return ListedColormap([(0, 0, 0, 0), (1, 0, 0, alpha)])
        elif color == 'green':
            return ListedColormap([(0, 0, 0, 0), (0, 1, 0, alpha)])
        elif color == 'blue':
            return ListedColormap([(0, 0, 0, 0), (0, 0, 1, alpha)])
        elif color == 'yellow':
            return ListedColormap([(0, 0, 0, 0), (1, 1, 0, alpha)])
        else:
            return ListedColormap([(0, 0, 0, 0), (1, 0, 0, alpha)])  # Default to red

    def update_mpr_view(self):
        """Update the multi-planar reconstruction view"""
        if self.ct_data is None:
            return
        
        # Get current slices
        axial_idx = self.axial_slice_var.get()
        coronal_idx = self.coronal_slice_var.get()
        sagittal_idx = self.sagittal_slice_var.get()
        
        # Get voxel spacing
        spacing = self.get_voxel_spacing(self.ct_affine)
        
        # Clear axes
        self.ax_sag.clear()
        self.ax_cor.clear()
        self.ax_ax.clear()
        
        # Extract slices and apply windowing
        sagittal_slice = self.apply_window(self.ct_data[sagittal_idx, :, :], self.window_center, self.window_width)
        coronal_slice = self.apply_window(self.ct_data[:, coronal_idx, :], self.window_center, self.window_width)
        axial_slice = self.apply_window(self.ct_data[:, :, axial_idx], self.window_center, self.window_width)
        
        # Calculate aspect ratios based on physical dimensions
        sag_aspect = spacing[2] / spacing[1]  # Z/Y for sagittal
        cor_aspect = spacing[2] / spacing[0]  # Z/X for coronal
        ax_aspect = spacing[1] / spacing[0]   # Y/X for axial
        
        # Display CT images with correct aspect ratios
        self.ax_sag.imshow(sagittal_slice.T, cmap='gray', origin='lower', aspect=sag_aspect)
        self.ax_cor.imshow(coronal_slice.T, cmap='gray', origin='lower', aspect=cor_aspect)
        self.ax_ax.imshow(axial_slice.T, cmap='gray', origin='lower', aspect=ax_aspect)
        
        # Display segmentation overlays if available and selected
        if self.mpr_show_seg_var.get() and self.seg_data is not None:
            seg_sagittal = self.seg_data[sagittal_idx, :, :]
            seg_coronal = self.seg_data[:, coronal_idx, :]
            seg_axial = self.seg_data[:, :, axial_idx]
            
            # Create tumor masks (label 1) and display in red
            if np.any(seg_sagittal == 1):
                tumor_sag_mask = (seg_sagittal == 1).astype(float)
                self.ax_sag.imshow(tumor_sag_mask.T, cmap=self.create_color_cmap('red'), origin='lower', aspect=sag_aspect, alpha=0.6)
            
            if np.any(seg_coronal == 1):
                tumor_cor_mask = (seg_coronal == 1).astype(float)
                self.ax_cor.imshow(tumor_cor_mask.T, cmap=self.create_color_cmap('red'), origin='lower', aspect=cor_aspect, alpha=0.6)
            
            if np.any(seg_axial == 1):
                tumor_ax_mask = (seg_axial == 1).astype(float)
                self.ax_ax.imshow(tumor_ax_mask.T, cmap=self.create_color_cmap('red'), origin='lower', aspect=ax_aspect, alpha=0.6)
            
            # Create metastasis masks (label 2) and display in green
            if np.any(seg_sagittal == 2):
                meta_sag_mask = (seg_sagittal == 2).astype(float)
                self.ax_sag.imshow(meta_sag_mask.T, cmap=self.create_color_cmap('green'), origin='lower', aspect=sag_aspect, alpha=0.6)
            
            if np.any(seg_coronal == 2):
                meta_cor_mask = (seg_coronal == 2).astype(float)
                self.ax_cor.imshow(meta_cor_mask.T, cmap=self.create_color_cmap('green'), origin='lower', aspect=cor_aspect, alpha=0.6)
            
            if np.any(seg_axial == 2):
                meta_ax_mask = (seg_axial == 2).astype(float)
                self.ax_ax.imshow(meta_ax_mask.T, cmap=self.create_color_cmap('green'), origin='lower', aspect=ax_aspect, alpha=0.6)
        
        # Add crosshairs if selected
        if self.mpr_show_crosshairs_var.get():
            # Sagittal view crosshairs
            self.ax_sag.axhline(y=axial_idx, color='blue', linestyle='--', alpha=0.7)
            self.ax_sag.axvline(x=coronal_idx, color='green', linestyle='--', alpha=0.7)
            
            # Coronal view crosshairs
            self.ax_cor.axhline(y=axial_idx, color='blue', linestyle='--', alpha=0.7)
            self.ax_cor.axvline(x=sagittal_idx, color='red', linestyle='--', alpha=0.7)
            
            # Axial view crosshairs
            self.ax_ax.axhline(y=coronal_idx, color='green', linestyle='--', alpha=0.7)
            self.ax_ax.axvline(x=sagittal_idx, color='red', linestyle='--', alpha=0.7)
        
        # Add orientation labels if selected
        if self.mpr_show_labels_var.get():
            self.ax_sag.set_xlabel("Anterior → Posterior")
            self.ax_sag.set_ylabel("Inferior → Superior")
            self.ax_cor.set_xlabel("Left → Right")
            self.ax_cor.set_ylabel("Inferior → Superior")
            self.ax_ax.set_xlabel("Left → Right")
            self.ax_ax.set_ylabel("Posterior → Anterior")
        
        # Set titles
        self.ax_sag.set_title("Sagittal")
        self.ax_cor.set_title("Coronal")
        self.ax_ax.set_title("Axial")
        self.fig_mpr.suptitle(f"Multi-Planar View - Patient {self.patient_id}")
        
        # Update the canvas
        self.canvas_mpr.draw()

    def get_voxel_spacing(self, affine):
        """Extract voxel spacing from affine matrix"""
        return np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))

    def apply_window(self, image_data, window_center, window_width):
        """Apply window level and width to image data"""
        min_value = window_center - window_width // 2
        max_value = window_center + window_width // 2
        windowed_image = np.clip(image_data, min_value, max_value)
        windowed_image = (windowed_image - min_value) / (max_value - min_value)
        return windowed_image

    def create_red_cmap(self, alpha=0.6):
        """Create a bright red colormap for segmentation overlay"""
        return ListedColormap([(0, 0, 0, 0), (1, 0, 0, alpha)])

    def load_ct_data(self, institution, patient_id):
        """Load and cache CT data"""
        cache_key = f"{institution}_{patient_id}_ct"
        if cache_key in image_cache:
            return image_cache[cache_key]
        
        ct_path = f"DatasetChallenge/CT/{institution}/{patient_id}.nii.gz"
        if not os.path.exists(ct_path):
            return None, None
        
        ct_img = nib.load(ct_path)
        ct_data = ct_img.get_fdata()
        image_cache[cache_key] = (ct_data, ct_img.affine)
        return ct_data, ct_img.affine
    
    def load_seg_data(self, institution, patient_id):
        """Load and cache segmentation data"""
        cache_key = f"{institution}_{patient_id}_seg"
        if cache_key in image_cache:
            return image_cache[cache_key]
        
        seg_path = f"DatasetChallenge/Segmentation/{institution}/{patient_id}.nii.gz"
        if not os.path.exists(seg_path):
            return None
        
        seg_img = nib.load(seg_path)
        seg_data = seg_img.get_fdata()
        image_cache[cache_key] = seg_data
        return seg_data

    def load_patient(self):
        """Load patient data"""
        self.root.config(cursor="watch")
        self.patient_id = self.patient_var.get()
        self.institution = self.institution_var.get()
        
        try:
            # Load CT and segmentation data
            self.ct_data, self.ct_affine = self.load_ct_data(self.institution, self.patient_id)
            self.seg_data = self.load_seg_data(self.institution, self.patient_id)
            
            # Update all view sliders
            if self.ct_data is not None:
                # Update 2D view slider
                axis = self.current_axis
                max_slice = self.ct_data.shape[axis] - 1
                self.slice_scale.configure(to=max_slice)
                self.slice_var.set(max_slice // 2)
                
                # Update MPR view sliders
                axial_max = self.ct_data.shape[2] - 1
                coronal_max = self.ct_data.shape[1] - 1
                sagittal_max = self.ct_data.shape[0] - 1
                
                self.axial_scale.configure(to=axial_max)
                self.axial_slice_var.set(axial_max // 2)
                
                self.coronal_scale.configure(to=coronal_max)
                self.coronal_slice_var.set(coronal_max // 2)
                
                self.sagittal_scale.configure(to=sagittal_max)
                self.sagittal_slice_var.set(sagittal_max // 2)
                
                # Update views
                self.update_2d_view()
                self.update_mpr_view()
            else:
                messagebox.showerror("Error", f"Could not load CT data for patient {self.patient_id}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error loading patient data: {str(e)}")
        
        finally:
            self.root.config(cursor="")

if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalImageViewer(root)
    root.mainloop()