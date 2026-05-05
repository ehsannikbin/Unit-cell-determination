# Electron Diffraction Unit Cell Determination Pipeline

![Python Version](https://img.shields.io/badge/python-3.14-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This repository contains a computational pipeline designed to determine crystal unit cells from **randomly oriented electron diffraction patterns**.

The workflow bridges the gap between raw microscope data and crystallographic parameters. It consists of two main stages:
1.  **Facet Selection:** Interactive visualization and extraction of reciprocal lattice vectors.
2.  **Unit Cell Determination:** A hybrid optimization algorithm (Global + Local) to solve unit cell parameters based on the extracted vectors.


## 1. Installation

To ensure all dependencies (GUI, math, plotting, JIT compilation) work correctly, please use the provided Conda environment.

**1. Clone the repository:**
```bash
git clone https://github.com/ehsannikbin/Unit-cell-determination.git
cd Unit-cell-determination
```

**2. Create the environment:**
```bash
conda create -n unit_cell python=3.14 h5py numpy matplotlib scipy pyside6 numba -c conda-forge
```

**3. Activate the environment:**
```bash
conda activate unit_cell
```

## 2. Usage: Step 1 (Facet Selector)

This script calculates the autocorrelation of diffraction patterns to help users identify and select valid crystallographic facets. It uses **multiprocessing** to handle large datasets efficiently.

**Run the script:**
```bash
python scripts/01_facet_selector.py
```

### GUI Features & Controls

* **File Loading & Performance:**
    * **Max Load:** Randomly subsample large datasets (e.g., limit to 500 patterns for statistical sampling).
    * **Processors:** Adjust the number of CPU cores used for parallel autocorrelation calculations.
    * **Progress Bar:** Visual feedback during the loading phase.

* **Visualization:**
    * **Toggle AC:** Switch between the raw diffraction pattern and the Autocorrelation view.
    * **Zoom:** Use the **Mouse Wheel** to zoom in/out.
    * **Show Peaks/Facets:** Toggle overlays on and off.

* **Navigation:**
    * **Keyboard:** Use **Left/Right Arrow Keys** to navigate.
    * **Jump:** Type a specific pattern number to jump directly to it.

* **Filtering:**
    * **Include this facet:** Check/Uncheck this box to determine if the current pattern's vectors should be saved. You only need to include ~10-30 facets from the dataset.
    * **Parameters:** Adjust **AC Threshold** and **Min Separation** to fine-tune peak detection.

---

## 3. Usage: Step 2 (Unit Cell Finder)

This script takes the output from Step 1 (`facet_vectors.csv`) and determines the unit cell parameters ($a, b, c, \alpha, \beta, \gamma$). It uses **Numba-accelerated** Differential Evolution (Global Search) followed by L-BFGS-B (Local Refinement) to find the best fit.

**Run the script:**
```bash
python scripts/02_unit_cell_finder.py
```

### GUI Features & Controls

* **Input & Output:**
    * **Files:** Browse to select your input `.csv` file.
    * **Custom Naming:** Define custom filenames for the output run log and unit cell result file.

* **Crystal Settings:**
    * **System:** Select from 7 crystal systems (Triclinic, Monoclinic, Orthorhombic, etc.). The parameter range inputs dynamically update based on your selection.
    * **Centering:** Choose the centering type (P, I, F, A, B, C, R).
    * **Ranges:** Define Min/Max constraints for real-space parameters ($\text{\AA}$ and Degrees) to guide the optimizer.

* **Algorithm Controls:**
    * **Max hkl (M):** Define the search radius for integer indices (e.g., $\pm 6$).
    * **Optimization:** Adjust **Population Size** and **Max Iterations** for the Global Search.
    * **Tolerances:** Set the Relative Length Tolerance and Absolute Cosine Tolerance for matching vector pairs.
    * **Processors:** Control parallelization. Set to `-1` to use all cores, or `1` for serial execution (often faster for smaller datasets due to reduced overhead).

* **Advanced Features:**
    * **Outlier Rejection:** Optional 2-stage refinement. The algorithm first finds a consensus cell, then drops the worst-fitting patterns (e.g., worst 20%) and re-refines the cell to improve accuracy.

## 4. Input & Output Data

### Input Format (Step 1)
The software reads HDF5 files containing:
* `/data`: Raw diffraction images.
* `/peaks`: Pre-calculated peak positions.
* `/center`: Direct beam center coordinates.

### Intermediate Data (Step 1 Output -> Step 2 Input)
The results from Step 1 are saved as a **CSV file** containing reciprocal lattice vectors ($s_1, s_2$) and the angle ($\theta$).

| s0_Ainv | s1_Ainv | angle_deg |
| :--- | :--- | :--- |
| 0.04512 | 0.05201 | 59.8 |
| 0.06100 | 0.06100 | 89.9 |

### Final Output (Step 2 Output)
The final unit cell is saved as a text file (e.g., `unit_cell.txt`):

```text
# Crystal system = monoclinic
a = 13.731 A
b = 9.203 A
c = 8.497 A
alpha = 90.000 deg
beta = 100.060 deg
gamma = 90.000 deg
```

## Dependencies
* **Python 3.14**
* **PySide6** (GUI)
* **Numba** (JIT Compilation for high-performance math)
* **NumPy & SciPy** (Optimization and array processing)
* **Matplotlib** (Visualization)
* **h5py** (HDF5 file handling)

## License
This project is licensed under the MIT License.
