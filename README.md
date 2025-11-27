# Wafer Defect Classification (Final project for ECH684)
Deep learning to identify the defect type in semiconductor wafers
![38 defect types in semi conductor wafer chips](output.png)
## Step-by-Step Setup Guide


## 📋 Overview

### Step 1: Load Conda Module

```bash
module load conda
conda create -n wafer_defect python=3.11 -y
conda activate wafer_defect
conda install tqdm numpy pandas matplotlib scipy scikit-learn jupyter ipykernel -y
conda install -c conda-forge opencv -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m ipykernel install --user --name=wafer_defect --display-name="Python (Wafer Defect)"
```
---



