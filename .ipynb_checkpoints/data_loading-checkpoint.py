import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

class WaferDataLoader:
    """
    A comprehensive class for loading, preprocessing, and visualizing the WM-811K wafer defect dataset.
    
    This class encapsulates all data loading and visualization functionality for the mixed-type 
    wafer defect dataset (38,015 samples with 38 unique defect classes).
    
    Attributes:
        X (np.ndarray): Wafer maps, shape (38015, 52, 52)
        y (np.ndarray): Class labels, shape (38015,)
        y_raw (np.ndarray): Raw binary patterns from dataset
        classes_dict (dict): Mapping from binary pattern to class index
        class_to_pattern (dict): Reverse mapping from class index to binary pattern
        num_classes (int): Number of unique defect classes (38)
        path (str): Path to the dataset directory
    """
    
    def __init__(self, dataset_name='mixedtype-wafer-defect-datasets'):
        """
        Initialize WaferDataLoader and load the dataset.
        
        Args:
            dataset_name (str): Name of the dataset in kagglehub cache
        """
        self.path = dataset_name
        self.X = None
        self.y = None
        self.y_raw = None
        self.classes_dict = self._create_classes_dict()
        self.class_to_pattern = None
        self.num_classes = len(self.classes_dict)
        
        # Load data
        self._load_data()
        self._convert_labels()
    
    def _create_classes_dict(self):
        """Create the mapping from binary pattern to class index."""
        return {
            "00000000": 0,
            "10000000": 1,
            "01000000": 2,
            "00100000": 3,
            "00010000": 4,
            "00001000": 5,
            "00000100": 6,
            "00000010": 7,
            "00000001": 8,
            "10100000": 9,
            "10010000": 10,
            "10001000": 11,
            "10000010": 12,
            "01100000": 13,
            "01010000": 14,
            "01001000": 15,
            "01000010": 16,
            "00101000": 17,
            "00100010": 18,
            "00011000": 19,
            "00010010": 20,
            "00001010": 21,
            "10101000": 22,
            "10100010": 23,
            "10011000": 24,
            "10010010": 25,
            "10001010": 26,
            "01101000": 27,
            "01100010": 28,
            "01011000": 29,
            "01010010": 30,
            "01001010": 31,
            "00101010": 32,
            "00011010": 33,
            "10101010": 34,
            "10011010": 35,
            "01101010": 36,
            "01011010": 37,
        }
    
    def _load_data(self):

        print("Loading dataset from kagglehub cache...")
        try:
            data = np.load(os.path.join(self.path, "Wafer_Map_Datasets.npz"))
            self.X = data['arr_0']  # Shape: (38015, 52, 52)
            self.y_raw = data['arr_1']  # Shape: (38015, 8) - binary pattern labels
            print(f"✓ Dataset loaded successfully")
            print(f"  Wafer maps shape: {self.X.shape}")
            print(f"  Raw labels shape: {self.y_raw.shape}")
        except FileNotFoundError:
            print(f"Error: Dataset not found at {self.path}")
            raise
    
    def _convert_labels(self):
        """Convert binary patterns to class indices."""
        self.y = np.zeros(len(self.y_raw), dtype=np.int64)
        for i in range(len(self.y_raw)):
            pattern_str = ''.join(map(str, self.y_raw[i].astype(int)))
            if pattern_str in self.classes_dict:
                self.y[i] = self.classes_dict[pattern_str]
            else:
                print(f"Warning: Unknown pattern {pattern_str} at index {i}")
        
        # Create reverse mapping
        self.class_to_pattern = {v: k for k, v in self.classes_dict.items()}
        print(f"✓ Labels converted to class indices")
        print(f"  Number of unique defect classes: {self.num_classes}")
    
    def print_class_distribution(self, num_classes_to_show=10):
        """
        Print the distribution of samples across classes.
        
        Args:
            num_classes_to_show (int): Number of classes to display in detail
        """
        unique, counts = np.unique(self.y, return_counts=True)
        print(f"\nClass Distribution (first {num_classes_to_show} classes):")
        print(f"{'Class':<8} {'Sample Count':<15} {'arr_1 Pattern (8 bits)':<30}")
        print("-" * 80)
        
        for class_idx in range(min(num_classes_to_show, self.num_classes)):
            pattern_str = self.class_to_pattern[class_idx]
            count = np.sum(self.y == class_idx)
            print(f"{class_idx:<8} {count:<15} {pattern_str:<30}")
        
        if self.num_classes > num_classes_to_show:
            print(f"... and {self.num_classes - num_classes_to_show} more classes")
    
    def plot_class_distribution(self):
        """Plot the distribution of wafer defects across all classes."""
        plt.figure(figsize=(14, 6))
        sns.countplot(x=self.y, palette='viridis')
        plt.title("Class Distribution of Wafer Defect Types", fontsize=14, fontweight='bold')
        plt.xlabel("Defect Class", fontsize=12)
        plt.ylabel("Number of Samples", fontsize=12)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def print_pattern_table(self):
        """Print a table of all class patterns."""
        print("\n" + "="*80)
        print("ARR_1 LABEL PATTERNS FOR EACH CLASS")
        print("="*80)
        print("\nEach class has a unique 8-column binary pattern representing defect type:")
        print(f"{'Class':<8} {'Sample Count':<15} {'arr_1 Pattern (8 bits)':<30}")
        print("-" * 80)
        
        for class_idx in range(self.num_classes):
            pattern_str = self.class_to_pattern[class_idx]
            count = np.sum(self.y == class_idx)
            print(f"{class_idx:<8} {count:<15} {pattern_str:<30}")
    
    def plot_pattern_heatmap(self):
        """Visualize all class patterns as a heatmap."""
        patterns_matrix = np.zeros((self.num_classes, 8), dtype=int)
        for class_idx in range(self.num_classes):
            pattern_str = self.class_to_pattern[class_idx]
            patterns_matrix[class_idx] = np.array([int(bit) for bit in pattern_str])
        
        fig, ax = plt.subplots(figsize=(10, 12))
        sns.heatmap(patterns_matrix, cmap='RdYlGn', cbar=True, 
                    xticklabels=range(8), yticklabels=range(self.num_classes),
                    ax=ax, annot=True, fmt='d')
        ax.set_title('Binary Pattern for Each Defect Class (8-bit)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Bit Position', fontsize=12)
        ax.set_ylabel('Class Index', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def plot_sample_gallery(self, samples_per_row=7):
        """
        Display one representative sample from each class in a grid.
        
        Args:
            samples_per_row (int): Number of samples to display per row
        """
        num_rows = (self.num_classes + samples_per_row - 1) // samples_per_row
        fig, axes = plt.subplots(num_rows, samples_per_row, 
                                figsize=(samples_per_row*3, num_rows*3))
        axes = axes.flatten()
        
        for class_idx in range(self.num_classes):
            # Find first sample with this class label
            sample_indices = np.where(self.y == class_idx)[0]
            
            if len(sample_indices) > 0:
                first_sample_idx = sample_indices[0]
                wafer_map = self.X[first_sample_idx]
                pattern_str = self.class_to_pattern[class_idx]
                
                # Display the wafer map
                axes[class_idx].imshow(wafer_map, cmap='viridis')
                axes[class_idx].set_title(f'Class {class_idx}\n({pattern_str})', 
                                         fontsize=9, fontweight='bold')
                axes[class_idx].axis('off')
        
        # Hide extra subplots
        for idx in range(self.num_classes, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Sample Wafer Map from Each of 38 Defect Classes', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.show()
    
    def get_sample_by_class(self, class_idx, sample_num=0):
        """
        Get a specific sample from a given class.
        
        Args:
            class_idx (int): Class index (0-37)
            sample_num (int): Which sample from that class (0 = first)
        
        Returns:
            tuple: (wafer_map, pattern_str)
        """
        sample_indices = np.where(self.y == class_idx)[0]
        if sample_num >= len(sample_indices):
            raise IndexError(f"Class {class_idx} has only {len(sample_indices)} samples")
        
        idx = sample_indices[sample_num]
        wafer_map = self.X[idx]
        pattern_str = self.class_to_pattern[class_idx]
        return wafer_map, pattern_str
    
    def print_summary(self):
        """Print a comprehensive summary of the dataset."""
        print("\n" + "="*80)
        print("WAFER DEFECT DATASET SUMMARY")
        print("="*80)
        print(f"Total samples: {len(self.X):,}")
        print(f"Image dimensions: {self.X.shape[1]} × {self.X.shape[2]} pixels")
        print(f"Number of classes: {self.num_classes}")
        print(f"Pixel value range: {self.X.min():.4f} - {self.X.max():.4f}")
        print(f"Mean pixel value: {self.X.mean():.4f}")
        print(f"Std pixel value: {self.X.std():.4f}")
        
        unique, counts = np.unique(self.y, return_counts=True)
        print(f"\nClass distribution:")
        print(f"  Min samples per class: {counts.min()}")
        print(f"  Max samples per class: {counts.max()}")
        print(f"  Mean samples per class: {counts.mean():.1f}")
        print("="*80)


# Example usage and demonstrations (runs when script is executed directly)
if __name__ == "__main__":
    # Initialize the data loader
    print("Initializing WaferDataLoader...\n")
    loader = WaferDataLoader()
    
    # Print summary
    loader.print_summary()
    
    # Print class distribution
    loader.print_class_distribution()
    
    # Plot class distribution
    loader.plot_class_distribution()
    
    # Print pattern table
    loader.print_pattern_table()
    
    # Plot pattern heatmap
    loader.plot_pattern_heatmap()
    
    # Plot sample gallery
    loader.plot_sample_gallery()
    
    print("\n✓ Data loading and visualization complete!")


