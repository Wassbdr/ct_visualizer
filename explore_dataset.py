"""
Dataset Exploration Module for Medical Imaging Data

This module provides functions to explore, organize, and extract information from 
a medical imaging dataset containing CT scans and segmentation data from different 
institutions (MSKCC and TCGA).

The module includes functions for:
- Scanning the dataset directory structure
- Extracting patient IDs
- Generating summary reports of available data

Functions:
    scan_dataset(): Scan the dataset directory structure and return a dictionary of files
    get_patient_ids(): Extract unique patient IDs from the dataset
    generate_dataset_report(): Generate a summary report of dataset contents

Usage:
    import explore_dataset
    
    # Get dictionary of patient IDs by institution
    patient_ids = explore_dataset.get_patient_ids()
    
    # Generate and print a dataset summary report
    report = explore_dataset.generate_dataset_report()
"""

import os
import nibabel as nib
import pandas as pd
import numpy as np
from pathlib import Path

def scan_dataset(root_dir="DatasetChallenge"):
    """
    Scan the dataset directory and organize file information.
    
    This function traverses the dataset directory structure and collects
    information about CT scans and segmentation files organized by institution.
    
    Args:
        root_dir (str): Path to the root directory of the dataset. Defaults to "DatasetChallenge".
    
    Returns:
        dict: A nested dictionary with the following structure:
            {
                "CT": {
                    "MSKCC": [list of files],
                    "TCGA": [list of files]
                },
                "Segmentation": {
                    "MSKCC": [list of files],
                    "TCGA": [list of files]
                }
            }
    """
    dataset_info = {
        "CT": {"MSKCC": [], "TCGA": []},
        "Segmentation": {"MSKCC": [], "TCGA": []}
    }
    
    # Scan CT directory
    ct_dir = os.path.join(root_dir, "CT")
    for institution in ["MSKCC", "TCGA"]:
        inst_dir = os.path.join(ct_dir, institution)
        if os.path.exists(inst_dir):
            dataset_info["CT"][institution] = sorted(os.listdir(inst_dir))
    
    # Scan Segmentation directory
    seg_dir = os.path.join(root_dir, "Segmentation")
    for institution in ["MSKCC", "TCGA"]:
        inst_dir = os.path.join(seg_dir, institution)
        if os.path.exists(inst_dir):
            dataset_info["Segmentation"][institution] = sorted(os.listdir(inst_dir))
    
    return dataset_info

def get_patient_ids():
    """
    Extract unique patient IDs from the dataset.
    
    This function identifies all unique patient IDs by processing filenames
    from CT scans in the dataset, organized by institution.
    
    Returns:
        dict: A dictionary containing patient IDs by institution:
            {
                "MSKCC": [list of patient IDs],
                "TCGA": [list of patient IDs]
            }
    """
    dataset_info = scan_dataset()
    patient_ids = {}
    
    for institution in ["MSKCC", "TCGA"]:
        ct_files = dataset_info["CT"][institution]
        # Extract patient IDs by removing file extension
        ids = sorted(list(set([f.split('.')[0] for f in ct_files])))
        patient_ids[institution] = ids
        
    return patient_ids

def generate_dataset_report():
    """
    Generate a summary report of the dataset.
    
    This function analyzes the dataset structure and produces a summary report
    containing counts of CT images, segmentation files, and unique patient IDs
    for each institution.
    
    Returns:
        dict: A dictionary containing the report data with the following structure:
            {
                "CT Images Count": {"MSKCC": count, "TCGA": count},
                "Segmentation Files Count": {"MSKCC": count, "TCGA": count},
                "Unique Patient IDs": {"MSKCC": count, "TCGA": count}
            }
    """
    dataset_info = scan_dataset()
    patient_ids = get_patient_ids()
    
    report = {
        "CT Images Count": {
            "MSKCC": len(dataset_info["CT"]["MSKCC"]),
            "TCGA": len(dataset_info["CT"]["TCGA"])
        },
        "Segmentation Files Count": {
            "MSKCC": len(dataset_info["Segmentation"]["MSKCC"]),
            "TCGA": len(dataset_info["Segmentation"]["TCGA"])
        },
        "Unique Patient IDs": {
            "MSKCC": len(patient_ids["MSKCC"]),
            "TCGA": len(patient_ids["TCGA"])
        }
    }
    
    return report

if __name__ == "__main__":
    report = generate_dataset_report()
    print("Dataset Report:")
    print("===============")
    for category, values in report.items():
        print(f"\n{category}:")
        for institution, count in values.items():
            print(f"  {institution}: {count}")
            
    patient_ids = get_patient_ids()
    for institution in ["MSKCC", "TCGA"]:
        print(f"\nFirst 5 patient IDs from {institution}:")
        for pid in patient_ids[institution][:5]:
            print(f"  {pid}")