"""
ROI Selector for emotion-related brain regions
"""
import numpy as np
import pandas as pd

class EmotionROISelector:
    """
    Select emotion-related ROIs from FreeSurfer parcellation
    Based on emotion processing literature
    """

    # Emotion-related ROIs based on neuroimaging literature
    EMOTION_ROIS = [
        # Amygdala (bilateral)
        'Left-Amygdala',
        'Right-Amygdala',

        # Hippocampus (bilateral) - memory and emotion
        'Left-Hippocampus',
        'Right-Hippocampus',

        # Accumbens (reward processing)
        'Left-Accumbens-area',
        'Right-Accumbens-area',

        # Cingulate cortex (emotion regulation)
        'ctx-lh-caudalanteriorcingulate',
        'ctx-rh-caudalanteriorcingulate',
        'ctx-lh-rostralanteriorcingulate',
        'ctx-rh-rostralanteriorcingulate',
        'ctx-lh-posteriorcingulate',
        'ctx-rh-posteriorcingulate',
        'ctx-lh-isthmuscingulate',
        'ctx-rh-isthmuscingulate',

        # Orbitofrontal cortex (emotion and decision making)
        'ctx-lh-lateralorbitofrontal',
        'ctx-rh-lateralorbitofrontal',
        'ctx-lh-medialorbitofrontal',
        'ctx-rh-medialorbitofrontal',

        # Insula (interoception and emotion)
        'ctx-lh-insula',
        'ctx-rh-insula',

        # Temporal pole (social and emotional processing)
        'ctx-lh-temporalpole',
        'ctx-rh-temporalpole',

        # Superior temporal (social perception)
        'ctx-lh-superiortemporal',
        'ctx-rh-superiortemporal',

        # Inferior temporal
        'ctx-lh-inferiortemporal',
        'ctx-rh-inferiortemporal',

        # Fusiform (face processing)
        'ctx-lh-fusiform',
        'ctx-rh-fusiform',

        # Prefrontal regions (emotion regulation)
        'ctx-lh-superiorfrontal',
        'ctx-rh-superiorfrontal',
        'ctx-lh-rostralmiddlefrontal',
        'ctx-rh-rostralmiddlefrontal',
        'ctx-lh-caudalmiddlefrontal',
        'ctx-rh-caudalmiddlefrontal',
    ]

    def __init__(self, roi_list=None):
        """
        Initialize ROI selector

        Args:
            roi_list: Optional custom list of ROI names. If None, uses default emotion ROIs
        """
        self.roi_list = roi_list if roi_list is not None else self.EMOTION_ROIS

    def select_rois(self, df, strict=True):
        """
        Select emotion-related ROI columns from a DataFrame

        Args:
            df: pandas DataFrame with ROI timeseries (columns are ROI names)
            strict: If True, return None if any ROIs are missing. If False, return partial selection.

        Returns:
            DataFrame with only emotion-related ROIs, or None if strict=True and ROIs are missing
        """
        # Get all column names
        all_columns = df.columns.tolist()

        # Find matching ROI columns
        selected_columns = ['TR']  # Always keep TR column
        for roi_name in self.roi_list:
            if roi_name in all_columns:
                selected_columns.append(roi_name)

        # Check which ROIs were not found
        missing_rois = [roi for roi in self.roi_list if roi not in all_columns]
        if missing_rois:
            print(f"Warning: {len(missing_rois)} ROIs not found in data:")
            for roi in missing_rois[:5]:  # Print first 5
                print(f"  - {roi}")
            if len(missing_rois) > 5:
                print(f"  ... and {len(missing_rois) - 5} more")

            if strict:
                print(f"Strict mode: Skipping subject due to missing ROIs")
                return None

        print(f"Selected {len(selected_columns)-1} emotion-related ROIs from {len(all_columns)-1} total ROIs")

        return df[selected_columns]

    def get_roi_count(self):
        """Get number of ROIs in the selection"""
        return len(self.roi_list)

    def get_roi_names(self):
        """Get list of ROI names"""
        return self.roi_list.copy()


def load_roi_timeseries(csv_path, roi_selector=None):
    """
    Load ROI timeseries from CSV file and optionally select emotion-related ROIs

    Args:
        csv_path: Path to CSV file with ROI timeseries
        roi_selector: EmotionROISelector instance. If None, loads all ROIs

    Returns:
        DataFrame with ROI timeseries
    """
    df = pd.read_csv(csv_path)

    if roi_selector is not None:
        df = roi_selector.select_rois(df)

    return df
