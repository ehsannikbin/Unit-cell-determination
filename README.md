# Electron Diffraction Unit Cell Determination Pipeline

![Python Version](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This repository contains a computational pipeline designed to determine crystal unit cells from **randomly oriented electron diffraction patterns**.

The workflow bridges the gap between raw microscope data and crystallographic parameters. It consists of two main stages:
1.  **Facet Selection:** Interactive visualization and extraction of reciprocal lattice vectors.
2.  **Unit Cell Determination:** (In development) Algorithms to solve the unit cell parameters based on the extracted vectors.

## Repository Structure

The code is organized to support a multi-step workflow:

```text
.
├── scripts/
│   ├── 01_facet_selector.py      # GUI for processing patterns and extracting vectors
│   ├── 02_unit_cell_finder.py    # (Planned) Solves unit cell from .csv data
│   └── utils.py                  # Shared physical constants (h, m0, etc.)
├── input_data/                   # Place your .lst or .h5 files here
├── output_data/                  # Destination for .csv results
├── environment.yml               # Conda environment configuration
└── README.md                     # Project documentation
```

## 1. Installation

To ensure all dependencies (GUI, math, plotting) work correctly, please use the provided Conda environment.

**1. Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/REPO_NAME.git
cd REPO_NAME
```

**2. Create the environment:**
```bash
conda env create -f environment.yml
```

**3. Activate the environment:**
```bash
conda activate diffraction_env
```

## 2. Usage: Step 1 (Facet Selector)

This script calculates the autocorrelation of diffraction patterns to help users identify and select valid crystallographic facets. It uses **multiprocessing** to handle large datasets efficiently.

**Run the script:**
```bash
python scripts/01_facet_selector.py
```

### GUI Features & Controls

* **File Loading & Performance:**
    * **Max Load:** Set a limit (e.g., 500) to randomly subsample large datasets (e.g., if you have 10,000 patterns but only need a statistical sample).
    * **Processors:** Adjust the number of CPU cores used for parallel autocorrelation calculations.
    * **Progress Bar:** Visual feedback during the loading and processing phase.

* **Visualization:**
    * **Toggle AC:** Switch between the raw diffraction pattern and the Autocorrelation view.
    * **Zoom:** Use the **Mouse Wheel** to zoom in/out. The canvas automatically resizes to fill the window.
    * **Show Peaks/Facets:** Toggle overlays on and off.

* **Navigation:**
    * **Buttons:** Use the **Previous/Next** buttons.
    * **Keyboard:** Use **Left/Right Arrow Keys** on your keyboard (Note: these keys are disabled while typing in text boxes to prevent accidental navigation).
    * **Jump:** Type a specific pattern number to jump directly to it.

* **Filtering:**
    * **Include this facet:** Check/Uncheck this box to determine if the current pattern's vectors should be saved to the output file.
    * **Parameters:** Adjust **AC Threshold** and **Min Separation** to fine-tune peak detection.

## 3. Input & Output Data

### Input Format (.h5)
The software reads HDF5 files. Each file is expected to contain the following datasets:
* `/data`: The raw diffraction images.
* `/peaks`: Peak positions (x, y) pre-calculated by peak-finding software.
* `/center`: The direct beam center coordinates (x, y).

### Output Format (.csv)
The results are saved as a **CSV file** (default: `facet_vectors.csv`) containing the reciprocal lattice vectors ($s_1, s_2$) and the angle between them ($\theta$).

**Example Output:**
| s0_Ainv | s1_Ainv | angle_deg |
| :--- | :--- | :--- |
| 0.04512 | 0.05201 | 59.8 |
| 0.06100 | 0.06100 | 89.9 |
| ... | ... | ... |

*(Note: Patterns marked as excluded in the GUI are skipped in this file).*

## Dependencies
* **Python 3.12**
* **PySide6** (Qt GUI framework)
* **Matplotlib** (Plotting and visualization)
* **NumPy & SciPy** (Math and image processing)
* **h5py** (Data handling)

## License
This project is licensed under the MIT License.
