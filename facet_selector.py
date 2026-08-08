import sys
import os
import h5py
import numpy as np
import random
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import warnings

# --- Suppress Matplotlib aspect ratio warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.*")
# -------------------------------------------------

from scipy import ndimage
from scipy.ndimage import gaussian_filter
import multiprocessing

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Qt imports
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

# ----------------- USER PARAMETERS -----------------
show_peaks = True
show_facets = True
autocorr_peak_min_sep = 4
autocorr_threshold_rel = 0.1
ignore_radius = 3
energy_keV = 90.0
camera_length_m = 0.403
pixel_size_m = 55e-6
include_patterns_by_default = False
DEFAULT_MAX_PATTERNS = 500
min_ang = 14.0
max_ang = 96.0
# ---------------------------------------------------

h = 6.62607015e-34
m0 = 9.10938356e-31
e_charge = 1.602176634e-19
c = 299792458.0

def compute_wavelength(e_keV):
    V = e_keV * 1e3
    return h / np.sqrt(2*m0*e_charge*V*(1 + e_charge*V/(2*m0*c**2)))

wavelength_m = compute_wavelength(energy_keV)

# ---------- Math / Physics Functions ---------- 
def autocorr_from_peaks(peaks, shape):
    ac = np.zeros(shape, dtype=float)
    cx, cy = shape[1]//2, shape[0]//2
    n = len(peaks)
    for i in range(n):
        for j in range(i+1, n):
            dx = peaks[j,0] - peaks[i,0]
            dy = peaks[j,1] - peaks[i,1]
            px, py = int(cx + dx), int(cy + dy)
            if 0 <= px < shape[1] and 0 <= py < shape[0]:
                ac[py, px] += 1
            px, py = int(cx - dx), int(cy - dy)
            if 0 <= px < shape[1] and 0 <= py < shape[0]:
                ac[py, px] += 1
    return ac

def find_autocorr_peaks(ac, threshold_rel=0.1, min_separation=3):
    ac_s = gaussian_filter(ac, sigma=0.5)
    neighborhood_size = 3
    local_max = (ac_s == ndimage.maximum_filter(ac_s, size=neighborhood_size))
    thresh = np.max(ac_s) * threshold_rel
    candidates = np.where((ac_s > thresh) & local_max)
    peaks = list(zip(candidates[1], candidates[0]))  # x,y
    if not peaks:
        return []
    peaks_sorted = sorted(peaks, key=lambda p: ac_s[p[1], p[0]], reverse=True)
    selected = []
    for p in peaks_sorted:
        ok = True
        for q in selected:
            if np.hypot(p[0]-q[0], p[1]-q[1]) < min_separation:
                ok = False
                break
        if ok:
            selected.append(p)
    return selected

def choose_principal_facet(ac_peaks, ac_shape, center_xy):
    if len(ac_peaks) < 2:
        return None
    ac_cx = (ac_shape[1] - 1) / 2.0
    ac_cy = (ac_shape[0] - 1) / 2.0
    
    vecs = [np.array([p[0] - ac_cx, p[1] - ac_cy]) for p in ac_peaks]
    vec_norms = [np.linalg.norm(v) for v in vecs]
    
    # Filter out center peak (zero-length)
    vecs, ac_peaks = zip(*[(v,p) for v,p,n in zip(vecs, ac_peaks, vec_norms) if n > 1e-6])
    vecs, ac_peaks = list(vecs), list(ac_peaks)
    if len(vecs) < 2:
        return None

    # --- FIX 1: Use stable sort to prevent identical lengths from randomly swapping ---
    lengths_initial = [np.linalg.norm(v) for v in vecs]
    sort_idx = np.argsort(lengths_initial, kind='stable')
    vecs = [vecs[i] for i in sort_idx]
    ac_peaks = [ac_peaks[i] for i in sort_idx]
    # -------------------------------------------------------------------------

    unique_vecs, unique_peaks = [], []
    for v, p in zip(vecs, ac_peaks):
        keep = True
        for u in unique_vecs:
            # Check for collinearity
            cosang = np.clip(np.dot(v,u)/(np.linalg.norm(v)*np.linalg.norm(u)), -1.0, 1.0)
            ang = np.degrees(np.arccos(cosang))
            
            # --- FIX 2: Only filter collinear (< 5 deg). 
            # Do NOT filter anti-collinear (180 deg) so we keep Friedel opposites.
            if ang < 5:
                keep = False
                break
        if keep:
            unique_vecs.append(v)
            unique_peaks.append(p)
            
    if len(unique_vecs) < 2:
        return None
        
    for i in range(len(unique_vecs)):
        v1 = unique_vecs[i]
        for j in range(i+1, len(unique_vecs)):
            v2 = unique_vecs[j]
            cosang = np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1.0, 1.0)
            ang = np.degrees(np.arccos(cosang))
            if min_ang < ang < max_ang:
                return unique_peaks[i], unique_peaks[j]
                
    # Fallback if no pair meets the angle criteria
    # Find the first pair that is NOT a 180-degree Friedel pair
    for i in range(1, len(unique_vecs)):
        cosang = np.clip(np.dot(unique_vecs[0], unique_vecs[i])/(np.linalg.norm(unique_vecs[0])*np.linalg.norm(unique_vecs[i])), -1.0, 1.0)
        ang = np.degrees(np.arccos(cosang))
        if abs(ang - 180) > 5:
            return unique_peaks[0], unique_peaks[i]
            
    return unique_peaks[0], unique_peaks[1]

def compute_lattice_conformity(ac_peaks, v1, v2, ac_cx, ac_cy, tol_pixels=3.0, scaling_factor=0.02):
    """
    Projects all AC peaks onto the integer grid formed by v1 and v2.
    Returns the fraction of peaks that land within tol_pixels of a perfect grid point.
    """
    if not ac_peaks or v1 is None or v2 is None:
        return 0.0
    
    # Create transformation matrix M from basis vectors v1, v2
    M = np.column_stack((v1, v2))
    if abs(np.linalg.det(M)) < 1e-6:
        return 0.0 # Vectors are collinear
        
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return 0.0

    conforming_count = 0
    for p in ac_peaks:
        vec = np.array([p[0] - ac_cx, p[1] - ac_cy])
        
        # Find fractional coordinates in the basis [v1, v2]
        nm = M_inv.dot(vec)
        
        # Find the nearest perfect integer grid point
        n_int, m_int = np.round(nm[0]), np.round(nm[1])
        ideal_vec = n_int * v1 + m_int * v2
        
        # Check physical pixel distance between real peak and ideal grid point
        dist = np.linalg.norm(vec - ideal_vec)

        dynamic_tol = tol_pixels + (scaling_factor * np.linalg.norm(ideal_vec))

        if dist <= dynamic_tol:
            conforming_count += 1
            
    return conforming_count / len(ac_peaks)

def ac_peak_to_detector_peak(ac_peak, ac_shape, detector_shape, center_det):
    ac_cx = (ac_shape[1]-1)/2.0
    ac_cy = (ac_shape[0]-1)/2.0
    dx = ac_peak[0] - ac_cx
    dy = ac_peak[1] - ac_cy
    px = center_det[0] + dx
    py = center_det[1] + dy
    return (np.clip(px, 0, detector_shape[1]-1),
            np.clip(py, 0, detector_shape[0]-1))

def detector_pixel_to_s_Ainv(px, py, cx, cy, pixel_size_m, camera_length_m, wavelength_m):
    dx_m = (px - cx)*pixel_size_m
    dy_m = (py - cy)*pixel_size_m
    r = np.hypot(dx_m, dy_m)
    theta = np.arctan2(r, camera_length_m)/2
    s = (2/wavelength_m)*np.sin(theta)
    return s/1e10

# ---------------- Worker Function for Multiprocessing ----------------
def worker_process_file(args):
    fname, target_frames, cfg = args
    results = []
    
    p_thresh = cfg['autocorr_threshold_rel']
    p_min_sep = cfg['autocorr_peak_min_sep']
    p_ignore = cfg['ignore_radius']
    p_pixel = cfg['pixel_size_m']
    p_cam = cfg['camera_length_m']
    p_wave = cfg['wavelength_m']
    p_tol = cfg.get('tol_pixels', 3.0)
    p_scale = cfg.get('scaling_factor', 0.02)
    
    try:
        with h5py.File(fname, 'r') as fh:
            data = fh['data']
            nPeaks_ds = fh['peaks']['nPeaks']
            peakX_ds = fh['peaks']['peakXPosRaw']
            peakY_ds = fh['peaks']['peakYPosRaw']
            cx_ds = fh['center']['center_x']
            cy_ds = fh['center']['center_y']
            
            for i in target_frames:
                try:
                    img = data[i][()]
                    n = int(nPeaks_ds[i])
                    pxs = np.array(peakX_ds[i][:n], float)
                    pys = np.array(peakY_ds[i][:n], float)
                    peaks = np.column_stack((pxs, pys)) if pxs.size else np.zeros((0,2))
                    cx, cy = float(cx_ds[i]), float(cy_ds[i])
                    
                    res_item = {
                        'img': img,
                        'peaks': peaks,
                        'center': (cx, cy),
                        'facet_info': {'p0':None,'p1':None,'s0':None,'s1':None,'angle':None},
                        'facet_tuple': (None, None, None),
                        'ac_peaks': [],
                        'conformity_score': 0.0
                    }

                    if len(peaks) < cfg.get('min_spots', 20):
                        results.append(res_item)
                        continue

                    ac = autocorr_from_peaks(peaks, img.shape)
                    ac_peaks = find_autocorr_peaks(ac, p_thresh, p_min_sep)
                    ac_cx, ac_cy = (ac.shape[1]-1)/2.0, (ac.shape[0]-1)/2.0
                    
                    # Store filtered ac peaks
                    ac_peaks_filtered = [(x,y) for (x,y) in ac_peaks if np.hypot(x-ac_cx,y-ac_cy)>=p_ignore]
                    res_item['ac_peaks'] = ac_peaks_filtered

                    chosen = choose_principal_facet(ac_peaks_filtered, ac.shape, (cx,cy)) if len(ac_peaks_filtered)>=2 else None

                    if chosen:
                        p0_ac, p1_ac = chosen
                        p0_det = ac_peak_to_detector_peak(p0_ac, ac.shape, img.shape, (cx,cy))
                        p1_det = ac_peak_to_detector_peak(p1_ac, ac.shape, img.shape, (cx,cy))
                        s0 = detector_pixel_to_s_Ainv(*p0_det, cx, cy, p_pixel, p_cam, p_wave)
                        s1 = detector_pixel_to_s_Ainv(*p1_det, cx, cy, p_pixel, p_cam, p_wave)
                        
                        v0 = np.array([(p0_det[0]-cx)*p_pixel, (p0_det[1]-cy)*p_pixel])
                        v1 = np.array([(p1_det[0]-cx)*p_pixel, (p1_det[1]-cy)*p_pixel])
                        cosang = np.dot(v0,v1)/(np.linalg.norm(v0)*np.linalg.norm(v1))
                        angle = np.degrees(np.arccos(np.clip(cosang,-1,1)))
                        
                        # Compute initial conformity score
                        v1_ac = np.array([p0_ac[0] - ac_cx, p0_ac[1] - ac_cy])
                        v2_ac = np.array([p1_ac[0] - ac_cx, p1_ac[1] - ac_cy])
                        conf = compute_lattice_conformity(ac_peaks_filtered, v1_ac, v2_ac, ac_cx, ac_cy, tol_pixels=p_tol, scaling_factor=p_scale)
                        
                        res_item['facet_info'] = {'p0':p0_det,'p1':p1_det,'s0':s0,'s1':s1,'angle':angle}
                        res_item['facet_tuple'] = (s0, s1, angle)
                        res_item['conformity_score'] = conf
                    
                    results.append(res_item)
                except Exception as frame_err:
                    print(f"Error processing frame {i} in {fname}: {frame_err}")
                    
    except Exception as e:
        print(f"Worker failed on file {fname}: {e}")
        
    return results

# ---------------- helper to load files ----------------
def load_files_from_folder(folder):
    candidates = []
    if not os.path.exists(folder): return []
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith('.h5') or fn.lower().endswith('.hdf5'):
            candidates.append(os.path.join(folder, fn))
    return candidates

def load_from_files_list_or_folder(path_or_folder):
    if os.path.isdir(path_or_folder):
        folder = path_or_folder
        files = load_files_from_folder(folder)
        return files, folder
    if not os.path.exists(path_or_folder):
        return [], os.path.dirname(os.path.abspath(sys.argv[0]))
    folder = os.path.dirname(os.path.abspath(path_or_folder))
    files = []
    with open(path_or_folder, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if os.path.isabs(line):
                if os.path.exists(line):
                    files.append(line)
            else:
                candidate = os.path.join(folder, line)
                if os.path.exists(candidate):
                    files.append(candidate)
    if not files:
        files = load_files_from_folder(folder)
    return files, folder

def process_file_paths(file_paths, max_patterns, num_processors, min_spots_val, tol_pixels_val, scaling_factor_val, progress_callback=None):
    patterns_all, peaks_all, centers_all, facet_info_all, facet_tuples = [], [], [], [], []
    ac_peaks_all, conformity_scores_all = [], []
    
    all_indices_map = [] 
    if progress_callback:
        progress_callback(0, 100, "Scanning files (Single thread)...")

    for i, fname in enumerate(file_paths):
        try:
            with h5py.File(fname, 'r') as fh:
                if 'data' in fh:
                    n_frames = fh['data'].shape[0]
                    for j in range(n_frames):
                        all_indices_map.append((i, j))
        except Exception as e:
            print(f"Warning: failed to scan {fname}: {e}")
            
    total_found = len(all_indices_map)
    if total_found == 0:
        return [], [], [], [], [], [], []
    
    if total_found > max_patterns:
        print(f"Dataset has {total_found} patterns. Randomly sampling {max_patterns} patterns.")
        selected_map = sorted(random.sample(all_indices_map, max_patterns))
    else:
        selected_map = all_indices_map
        
    file_to_frames = {}
    for f_idx, frame_idx in selected_map:
        if f_idx not in file_to_frames:
            file_to_frames[f_idx] = []
        file_to_frames[f_idx].append(frame_idx)
        
    total_to_process = len(selected_map)
    processed_count = 0
    
    config = {
        'autocorr_threshold_rel': autocorr_threshold_rel,
        'autocorr_peak_min_sep': autocorr_peak_min_sep,
        'ignore_radius': ignore_radius,
        'pixel_size_m': pixel_size_m,
        'camera_length_m': camera_length_m,
        'wavelength_m': wavelength_m, 
        'min_spots': min_spots_val, 
        'tol_pixels': tol_pixels_val,
        'scaling_factor': scaling_factor_val
    }
    
    tasks = []
    for f_idx, frames in file_to_frames.items():
        fname = file_paths[f_idx]
        tasks.append((fname, frames, config))
        
    if progress_callback:
        progress_callback(0, total_to_process, "Starting parallel processing...")

    with multiprocessing.Pool(processes=num_processors) as pool:
        for batch_results in pool.imap_unordered(worker_process_file, tasks):
            for res in batch_results:
                patterns_all.append(res['img'])
                peaks_all.append(res['peaks'])
                centers_all.append(res['center'])
                facet_info_all.append(res['facet_info'])
                facet_tuples.append(res['facet_tuple'])
                ac_peaks_all.append(res['ac_peaks'])
                conformity_scores_all.append(res['conformity_score'])
                
                processed_count += 1
                if progress_callback:
                      progress_callback(processed_count, total_to_process, f"Processed {processed_count}/{total_to_process}")

    return patterns_all, peaks_all, centers_all, facet_info_all, facet_tuples, ac_peaks_all, conformity_scores_all

# ---------------- main default setup ----------------
script_folder = os.path.dirname(os.path.abspath(sys.argv[0]))
default_files_lst = os.path.join(script_folder, "files.lst")
initial_file_paths, initial_folder = [], script_folder
if os.path.exists(default_files_lst):
     _, initial_folder = load_from_files_list_or_folder(default_files_lst)
else:
    cwd_files = os.path.join(os.getcwd(), "files.lst")
    if os.path.exists(cwd_files):
        _, initial_folder = load_from_files_list_or_folder(cwd_files)

# Start empty
patterns_all, peaks_all, centers_all, facet_info_all, facet_tuples, ac_peaks_all, conformity_scores_all = [], [], [], [], [], [], []

# ---------------- Qt Viewer ----------------
class QtViewer(QtWidgets.QMainWindow):
    def __init__(self, images, centers, peaks, facets, facet_tuples, ac_peaks, conf_scores, default_folder):
        super().__init__()
        self.images = images
        self.centers = centers
        self.peaks = peaks
        self.facets = facets
        self.facet_tuples = facet_tuples
        self.ac_peaks_all = ac_peaks
        self.conformity_scores = conf_scores
        self.grouped_df = None

        self.included = np.full(len(facet_tuples), include_patterns_by_default, dtype=bool)

        self.idx = 0
        self.show_ac = False
        self.show_ac_peaks = True
        self.user_changed_thresh = False

        self.current_folder = default_folder
        self.current_file_paths = [] 
        self.output_basename = "facet_vectors"

        self.params = {
            'show_peaks': show_peaks,
            'show_facets': show_facets,
            'autocorr_peak_min_sep': autocorr_peak_min_sep,
            'autocorr_threshold_rel': autocorr_threshold_rel,
            'ignore_radius': ignore_radius,
            'energy_keV': energy_keV,
            'camera_length_m': camera_length_m,
            'pixel_size_m': pixel_size_m,
        }
        self.wavelength_m = compute_wavelength(self.params['energy_keV'])

        self._build_ui()
        self._update_plot()
        QtWidgets.QApplication.instance().installEventFilter(self)

    def _build_ui(self):
        self.setWindowTitle("Facet Visualizer (Qt)")
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # Left: matplotlib canvas
        self.fig = Figure(figsize=(6,6))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_aspect('equal', adjustable='datalim')
        self.canvas.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self.canvas, stretch=3)

        # Right: controls
        ctrl = QtWidgets.QWidget()
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl)
        layout.addWidget(ctrl, stretch=1)

        # File load area
        file_group = QtWidgets.QGroupBox("Files")
        fg_layout = QtWidgets.QVBoxLayout(file_group)
        # self.files_label = QtWidgets.QLabel(self._short_path(self.current_folder))
        # fg_layout.addWidget(self.files_label)
        
        # Row 1: Max patterns + Processors
        row1 = QtWidgets.QHBoxLayout()
        
        # Max Load
        self.lbl_max = QtWidgets.QLabel("Max Load:")
        self.max_pat_spin = QtWidgets.QSpinBox()
        self.max_pat_spin.setRange(1, 1000000)
        self.max_pat_spin.setValue(DEFAULT_MAX_PATTERNS)
        self.max_pat_spin.setToolTip("Maximum number of patterns to randomly subsample")
        
        # Processors
        self.lbl_proc = QtWidgets.QLabel("Processors:")
        self.proc_spin = QtWidgets.QSpinBox()
        self.proc_spin.setRange(1, 128)
        try:
            self.proc_spin.setValue(os.cpu_count() or 1)
        except:
            self.proc_spin.setValue(4)
        self.proc_spin.setToolTip("Number of parallel processes to use")

        row1.addWidget(self.lbl_max)
        row1.addWidget(self.max_pat_spin)
        row1.addWidget(self.lbl_proc)
        row1.addWidget(self.proc_spin)
        fg_layout.addLayout(row1)

        # Row 2: Load Button
        row2 = QtWidgets.QHBoxLayout()
        self.btn_load_list = QtWidgets.QPushButton("Load files list")
        self.btn_load_list.setFocusPolicy(Qt.NoFocus)
        row2.addWidget(self.btn_load_list)
        fg_layout.addLayout(row2)

        ctrl_layout.addWidget(file_group)

        # Index navigation & Jump combined
        nav_layout = QtWidgets.QHBoxLayout()
        self.prev_btn = QtWidgets.QPushButton("Previous")
        self.next_btn = QtWidgets.QPushButton("Next")
        self.prev_btn.setFocusPolicy(Qt.NoFocus)
        self.next_btn.setFocusPolicy(Qt.NoFocus)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)

        nav_layout.addWidget(QtWidgets.QLabel("Pattern"))
        self.pattern_spin = QtWidgets.QSpinBox()
        self.pattern_spin.setRange(1, max(1, len(self.images)))
        self.pattern_spin.setValue(self.idx + 1)
        # Prevents rapid jumping while typing multi-digit numbers (waits for Enter/Click)
        self.pattern_spin.setKeyboardTracking(False) 
        nav_layout.addWidget(self.pattern_spin)
        
        self.total_patterns_label = QtWidgets.QLabel(f"/ {len(self.images)}")
        nav_layout.addWidget(self.total_patterns_label)
        ctrl_layout.addLayout(nav_layout)

        # Toggles
        
        cb_layout = QtWidgets.QHBoxLayout()
        self.show_peaks_cb = QtWidgets.QCheckBox("Show peaks")
        self.show_peaks_cb.setChecked(self.params['show_peaks'])
        self.show_peaks_cb.setFocusPolicy(Qt.NoFocus)
        cb_layout.addWidget(self.show_peaks_cb)
        
        self.show_facets_cb = QtWidgets.QCheckBox("Show facets")
        self.show_facets_cb.setChecked(self.params['show_facets'])
        self.show_facets_cb.setFocusPolicy(Qt.NoFocus)
        cb_layout.addWidget(self.show_facets_cb)

        self.show_ac_circles_cb = QtWidgets.QCheckBox("Show AC peaks")
        self.show_ac_circles_cb.setChecked(True)
        self.show_ac_circles_cb.setFocusPolicy(Qt.NoFocus)
        cb_layout.addWidget(self.show_ac_circles_cb)
        
        ctrl_layout.addLayout(cb_layout)

        # Include facet
        include_group = QtWidgets.QGroupBox("Inclusion")
        inc_layout = QtWidgets.QHBoxLayout(include_group)
        self.include_cb = QtWidgets.QCheckBox("Include this facet")
        if len(self.included) > 0:
            self.include_cb.setChecked(bool(self.included[self.idx]))
        else:
            self.include_cb.setChecked(bool(include_patterns_by_default))
        self.include_cb.setFocusPolicy(Qt.NoFocus)
        inc_layout.addWidget(self.include_cb)
        
        self.lbl_selected_count = QtWidgets.QLabel("Selected: 0")
        inc_layout.addWidget(self.lbl_selected_count)
        inc_layout.addStretch()
        ctrl_layout.addWidget(include_group)


        # --- COMPACTED AC OPTIONS ---
        thresh_group = QtWidgets.QGroupBox("Autocorrelation Options")
        thresh_layout = QtWidgets.QGridLayout(thresh_group)
        
        self.min_spots_spin = QtWidgets.QSpinBox()
        self.min_spots_spin.setRange(0, 500)
        self.min_spots_spin.setValue(20)
        thresh_layout.addWidget(QtWidgets.QLabel("Min spots"), 0, 0)
        thresh_layout.addWidget(self.min_spots_spin, 0, 1)
        
        self.ac_thresh_spin = QtWidgets.QDoubleSpinBox()
        self.ac_thresh_spin.setRange(0.01, 1.0)
        self.ac_thresh_spin.setSingleStep(0.01)
        self.ac_thresh_spin.setValue(self.params['autocorr_threshold_rel'])
        thresh_layout.addWidget(QtWidgets.QLabel("AC Thresh"), 0, 2)
        thresh_layout.addWidget(self.ac_thresh_spin, 0, 3)

        self.ac_minsep_spin = QtWidgets.QSpinBox()
        self.ac_minsep_spin.setRange(1, 50)
        self.ac_minsep_spin.setValue(int(self.params['autocorr_peak_min_sep']))
        thresh_layout.addWidget(QtWidgets.QLabel("Min sep"), 1, 0)
        thresh_layout.addWidget(self.ac_minsep_spin, 1, 1)

        self.ignore_radius_spin = QtWidgets.QSpinBox()
        self.ignore_radius_spin.setRange(0, 50)
        self.ignore_radius_spin.setValue(int(self.params['ignore_radius']))
        thresh_layout.addWidget(QtWidgets.QLabel("Ignore rad"), 1, 2)
        thresh_layout.addWidget(self.ignore_radius_spin, 1, 3)
        self.toggle_ac_btn = QtWidgets.QPushButton("Toggle AC")
        self.toggle_ac_btn.setFocusPolicy(Qt.NoFocus)
        thresh_layout.addWidget(self.toggle_ac_btn, 2, 0, 1, 4)
        
        ctrl_layout.addWidget(thresh_group)

        # --- COMPACTED INSTRUMENT PARAMS ---
        instr_group = QtWidgets.QGroupBox("Instrument Params")
        ig_layout = QtWidgets.QGridLayout(instr_group)
        
        self.energy_spin = QtWidgets.QDoubleSpinBox()
        self.energy_spin.setRange(1.0, 1000.0)
        self.energy_spin.setSingleStep(1.0)
        self.energy_spin.setValue(self.params['energy_keV'])
        ig_layout.addWidget(QtWidgets.QLabel("Energy (keV)"), 0, 0)
        ig_layout.addWidget(self.energy_spin, 0, 1)
        
        self.camlen_spin = QtWidgets.QDoubleSpinBox()
        self.camlen_spin.setRange(1e-6, 10.0)
        self.camlen_spin.setDecimals(6)
        self.camlen_spin.setSingleStep(0.001)
        self.camlen_spin.setValue(self.params['camera_length_m'])
        ig_layout.addWidget(QtWidgets.QLabel("Cam len (m)"), 0, 2)
        ig_layout.addWidget(self.camlen_spin, 0, 3)
        
        self.pixsize_spin = QtWidgets.QDoubleSpinBox()
        self.pixsize_spin.setRange(0.001, 10000.0)
        self.pixsize_spin.setDecimals(3)
        self.pixsize_spin.setValue(self.params['pixel_size_m'] * 1e6)
        ig_layout.addWidget(QtWidgets.QLabel("Pix size (µm)"), 1, 0)
        ig_layout.addWidget(self.pixsize_spin, 1, 1)
        ctrl_layout.addWidget(instr_group)

        # --- COMPACTED LATTICE FILTER ---
        filter_group = QtWidgets.QGroupBox("Quality & Lattice Filter")
        filter_layout = QtWidgets.QGridLayout(filter_group)
        
        self.tol_pixels_spin = QtWidgets.QDoubleSpinBox()
        self.tol_pixels_spin.setRange(0.1, 20.0)
        self.tol_pixels_spin.setSingleStep(0.5)
        self.tol_pixels_spin.setValue(3.0)
        filter_layout.addWidget(QtWidgets.QLabel("Grid Tol (px)"), 0, 0)
        filter_layout.addWidget(self.tol_pixels_spin, 0, 1)

        self.scaling_factor_spin = QtWidgets.QDoubleSpinBox()
        self.scaling_factor_spin.setRange(0.0, 0.20)
        self.scaling_factor_spin.setSingleStep(0.005)
        self.scaling_factor_spin.setDecimals(3)
        self.scaling_factor_spin.setValue(0.02)
        filter_layout.addWidget(QtWidgets.QLabel("Dist Scaling"), 0, 2)
        filter_layout.addWidget(self.scaling_factor_spin, 0, 3)

        self.conformity_thresh_spin = QtWidgets.QDoubleSpinBox()
        self.conformity_thresh_spin.setRange(0.0, 1.0)
        self.conformity_thresh_spin.setSingleStep(0.05)
        self.conformity_thresh_spin.setValue(0.90)
        filter_layout.addWidget(QtWidgets.QLabel("Min Conform"), 1, 0)
        filter_layout.addWidget(self.conformity_thresh_spin, 1, 1)

        self.btn_auto_filter = QtWidgets.QPushButton("Apply Filter to All")
        self.btn_auto_filter.setFocusPolicy(Qt.NoFocus)
        filter_layout.addWidget(self.btn_auto_filter, 2, 0, 1, 4) # Row 2, spans 4 columns
        ctrl_layout.addWidget(filter_group)

        # --- NEW: CLUSTERING GROUP ---
        cluster_group = QtWidgets.QGroupBox("Clustering & Analysis")
        cluster_layout = QtWidgets.QGridLayout(cluster_group)
        
        self.tol_s_spin = QtWidgets.QDoubleSpinBox()
        self.tol_s_spin.setRange(0.0001, 1.0)
        self.tol_s_spin.setDecimals(4)
        self.tol_s_spin.setSingleStep(0.001)
        self.tol_s_spin.setValue(0.003)
        cluster_layout.addWidget(QtWidgets.QLabel("s Tol (Å⁻¹)"), 0, 0)
        cluster_layout.addWidget(self.tol_s_spin, 0, 1)

        self.tol_angle_spin = QtWidgets.QDoubleSpinBox()
        self.tol_angle_spin.setRange(0.1, 90.0)
        self.tol_angle_spin.setSingleStep(0.5)
        self.tol_angle_spin.setValue(2.5)
        cluster_layout.addWidget(QtWidgets.QLabel("Angle Tol (°)"), 0, 2)
        cluster_layout.addWidget(self.tol_angle_spin, 0, 3)

        self.min_samples_spin = QtWidgets.QSpinBox()
        self.min_samples_spin.setRange(1, 1000)
        self.min_samples_spin.setValue(2)
        cluster_layout.addWidget(QtWidgets.QLabel("Min Samples"), 1, 0)
        cluster_layout.addWidget(self.min_samples_spin, 1, 1)

        # --- MOVED: Grouped count label moved up to row 1 (next to Min Samples) ---
        self.lbl_grouped_count = QtWidgets.QLabel("Grouped: 0")
        cluster_layout.addWidget(self.lbl_grouped_count, 1, 2, 1, 2)

        self.btn_cluster = QtWidgets.QPushButton("Group Facets")
        self.btn_cluster.setFocusPolicy(Qt.NoFocus)
        cluster_layout.addWidget(self.btn_cluster, 2, 0, 1, 4)
        ctrl_layout.addWidget(cluster_group)

        # Output
        out_group = QtWidgets.QGroupBox("Output")
        out_layout = QtWidgets.QFormLayout(out_group)
        self.output_name_edit = QtWidgets.QLineEdit("facet_vectors")
        out_layout.addRow("Output basename", self.output_name_edit)
        self.outfolder_label = QtWidgets.QLabel(self._short_path(self.current_folder))
        out_layout.addRow("Save folder", self.outfolder_label)
        out_group.setLayout(out_layout)
        ctrl_layout.addWidget(out_group)

        # Save buttons
        save_layout = QtWidgets.QHBoxLayout()
        self.save_btn = QtWidgets.QPushButton("Save Facets")
        self.save_btn.setFocusPolicy(Qt.NoFocus)
        save_layout.addWidget(self.save_btn)
        
        self.save_grouped_btn = QtWidgets.QPushButton("Save Grouped Facets")
        self.save_grouped_btn.setFocusPolicy(Qt.NoFocus)
        self.save_grouped_btn.setEnabled(False) # Starts inactive
        save_layout.addWidget(self.save_grouped_btn)
        
        ctrl_layout.addLayout(save_layout)
        
        self.save_img_btn = QtWidgets.QPushButton("Save current images (Diff + AC)")
        self.save_img_btn.setFocusPolicy(Qt.NoFocus)
        ctrl_layout.addWidget(self.save_img_btn)

        ctrl_layout.addStretch()
        self.status_label = QtWidgets.QLabel("")
        ctrl_layout.addWidget(self.status_label)

        # Connections
        self.prev_btn.clicked.connect(self.on_prev)
        self.next_btn.clicked.connect(self.on_next)
        self.toggle_ac_btn.clicked.connect(self.on_toggle_ac)
        self.show_peaks_cb.stateChanged.connect(self.on_show_peaks_changed)
        self.show_facets_cb.stateChanged.connect(self.on_show_facets_changed)
        self.include_cb.stateChanged.connect(self.on_include_changed)
        
        self.btn_auto_filter.clicked.connect(self.on_auto_filter)

        self.btn_cluster.clicked.connect(self.on_group_facets)
        self.save_grouped_btn.clicked.connect(self.on_save_grouped)
        
        self.ac_thresh_spin.valueChanged.connect(self.on_ac_thresh_changed)
        self.ac_minsep_spin.valueChanged.connect(self.on_params_changed)
        self.ignore_radius_spin.valueChanged.connect(self.on_params_changed)
        self.show_ac_circles_cb.stateChanged.connect(self.on_update_plot_req)
        self.camlen_spin.valueChanged.connect(self.on_instrument_params_changed)
        self.pixsize_spin.valueChanged.connect(self.on_instrument_params_changed)
        self.energy_spin.valueChanged.connect(self.on_energy_changed)
        
        self.save_btn.clicked.connect(self.on_save)
        self.save_img_btn.clicked.connect(self.on_save_current_images)
        self.btn_load_list.clicked.connect(self.load_files_list_dialog)
        self.pattern_spin.valueChanged.connect(self.on_pattern_jump)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.canvas.draw()
        
    def eventFilter(self, obj, event):
        # Intercept key presses globally
        if event.type() == QtCore.QEvent.Type.KeyPress:
            if event.key() in (Qt.Key_Left, Qt.Key_Right):
                focused_widget = QtWidgets.QApplication.focusWidget()
                
                # Allow Left/Right cursor movement ONLY inside the Output Name text box
                if focused_widget is self.output_name_edit:
                    return False # Do not intercept, let it behave normally
                
                # Otherwise, lock globally to pattern navigation
                if event.key() == Qt.Key_Left:
                    self.on_prev()
                elif event.key() == Qt.Key_Right:
                    self.on_next()
                    
                return True # Consume the event so spinboxes don't steal it
                
        # For all other events, process normally
        return super().eventFilter(obj, event)

    def _short_path(self, p):
        if not p:
            return ""
        return p if len(p) < 60 else "..." + p[-57:]
    
    def update_selection_count(self):
        if len(self.included) > 0:
            count = np.sum(self.included)
            self.lbl_selected_count.setText(f"Selected: {count}")
        else:
            self.lbl_selected_count.setText("Selected: 0")

    def _on_scroll(self, event):
        ax = self.ax
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return
        base_scale = 1.3
        scale_factor = 1/base_scale if event.button == 'up' else base_scale
        
        xlim = np.array(ax.get_xlim())
        ylim = np.array(ax.get_ylim())
        new_width = (xlim[1] - xlim[0]) * scale_factor
        new_height = (ylim[0] - ylim[1]) * scale_factor
        
        relx = (xdata - xlim[0]) / (xlim[1] - xlim[0])
        rely = (ydata - ylim[1]) / (ylim[0] - ylim[1])
        
        new_xlim = [xdata - new_width * relx, xdata + new_width * (1 - relx)]
        new_ylim = [ydata + new_height * (1 - rely), ydata - new_height * rely]
        
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        self.canvas.draw_idle()

    # ---------------- file loading handlers ----------------
    def load_files_list_dialog(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open files list", self.current_folder,
                                                      "List files (*.lst *.txt);;All files (*)")
        if not fn:
            return
        file_paths, folder = load_from_files_list_or_folder(fn)
        if not file_paths:
            QtWidgets.QMessageBox.warning(self, "No files", f"No valid files found in {fn}")
            return
        
        self.current_folder = folder
        self.current_file_paths = file_paths
        
        max_patterns_val = self.max_pat_spin.value()
        num_proc_val = self.proc_spin.value()
        min_spots_val = self.min_spots_spin.value()
        tol_val = self.tol_pixels_spin.value()
        scale_val = self.scaling_factor_spin.value()

        progress = QtWidgets.QProgressDialog("Loading and processing patterns...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0) 
        progress.setValue(0)
        
        def update_progress(val, total, message):
            progress.setLabelText(message)
            if total > 0:
                progress.setMaximum(total)
                progress.setValue(val)
            QtWidgets.QApplication.processEvents()

        self._set_loaded_data(file_paths, folder, max_patterns_val, num_proc_val, min_spots_val, tol_val, scale_val, progress_callback=update_progress)
        progress.setValue(progress.maximum())
        progress.close()

    def _set_loaded_data(self, file_paths, folder, max_patterns, num_processors, min_spots_val, tol_val, scale_val, progress_callback=None):
        images, peaks, centers, facets, tuples, ac_peaks, conf_scores = process_file_paths(file_paths, max_patterns, num_processors, min_spots_val, tol_val, scale_val, progress_callback=progress_callback)
        self.images = images
        self.peaks = peaks
        self.centers = centers
        self.facets = facets
        self.facet_tuples = tuples
        self.ac_peaks_all = ac_peaks
        self.conformity_scores = conf_scores
        
        self.included = np.full(len(self.facet_tuples), include_patterns_by_default, dtype=bool)
        self.idx = 0
        self.current_folder = folder
        # self.files_label.setText(self._short_path(folder))
        self.outfolder_label.setText(self._short_path(folder))
        
        self.include_cb.blockSignals(True)
        self.include_cb.setChecked(bool(self.included[self.idx]) if len(self.included) > 0 else bool(include_patterns_by_default))
        self.include_cb.blockSignals(False)
        
        if len(self.images) > 0:
            self._recompute_facet(0)
            
        has_imgs = len(self.images) > 0
        self.pattern_spin.blockSignals(True)
        self.pattern_spin.setRange(1, max(1, len(self.images)))
        self.pattern_spin.setValue(self.idx + 1)
        self.pattern_spin.setEnabled(has_imgs)
        self.pattern_spin.blockSignals(False)
        self.total_patterns_label.setText(f"/ {len(self.images)}")
        self._update_plot()
        
        self.setFocus()
        self.update_selection_count()

    # ---------------- navigation / UI handlers ----------------
    def on_prev(self):
        if len(self.images) == 0:
            return
        self.idx = (self.idx - 1) % len(self.images)
        self.include_cb.blockSignals(True)
        self.include_cb.setChecked(bool(self.included[self.idx]))
        self.include_cb.blockSignals(False)

        if not self.user_changed_thresh:
            self.ac_thresh_spin.blockSignals(True)
            self.ac_thresh_spin.setValue(autocorr_threshold_rel)
            self.params['autocorr_threshold_rel'] = autocorr_threshold_rel
            self.ac_thresh_spin.blockSignals(False)
            self._recompute_facet(index=self.idx)
        else:
            self._recompute_facet(index=self.idx)

        self._update_plot()

    def on_next(self):
        if len(self.images) == 0:
            return
        self.idx = (self.idx + 1) % len(self.images)
        self.include_cb.blockSignals(True)
        self.include_cb.setChecked(bool(self.included[self.idx]))
        self.include_cb.blockSignals(False)

        if not self.user_changed_thresh:
            self.ac_thresh_spin.blockSignals(True)
            self.ac_thresh_spin.setValue(autocorr_threshold_rel)
            self.params['autocorr_threshold_rel'] = autocorr_threshold_rel
            self.ac_thresh_spin.blockSignals(False)
            self._recompute_facet(index=self.idx)
        else:
            self._recompute_facet(index=self.idx)

        self._update_plot()

    def on_toggle_ac(self):
        self.show_ac = not self.show_ac
        self._update_plot()

    def on_show_peaks_changed(self, state):
        self.params['show_peaks'] = bool(state == Qt.Checked)
        self._update_plot()

    def on_show_facets_changed(self, state):
        self.params['show_facets'] = bool(state == Qt.Checked)
        if self.params['show_facets'] and len(self.images) > 0:
            self._recompute_facet(index=self.idx)
        self._update_plot()
        
    def on_update_plot_req(self, state):
        self._update_plot()

    def on_include_changed(self, state):
        if len(self.included) == 0:
            return
        checked = self.include_cb.isChecked()
        self.included[self.idx] = bool(checked)
        self._update_plot()
        self.update_selection_count()
        
    def on_auto_filter(self):
        if len(self.images) == 0: 
            return
            
        min_score = self.conformity_thresh_spin.value()
        tol = self.tol_pixels_spin.value()
        scale = self.scaling_factor_spin.value()
        
        kept = 0
        for i in range(len(self.images)):
            # If no vectors were found, or spots were too few, skip
            if self.facets[i]['p0'] is None or not self.ac_peaks_all[i]:
                self.included[i] = False
                self.conformity_scores[i] = 0.0
                continue
                
            ac_cx, ac_cy = (self.images[i].shape[1]-1)/2.0, (self.images[i].shape[0]-1)/2.0
            
            p0_det = self.facets[i]['p0']
            p1_det = self.facets[i]['p1']
            cx, cy = self.centers[i]
            
            # The vectors v1, v2 in AC coordinates are identical to detector coords relative to center
            v1_ac = np.array([p0_det[0] - cx, p0_det[1] - cy])
            v2_ac = np.array([p1_det[0] - cx, p1_det[1] - cy])
            
            score = compute_lattice_conformity(self.ac_peaks_all[i], v1_ac, v2_ac, ac_cx, ac_cy, tol_pixels=tol, scaling_factor=scale)
            self.conformity_scores[i] = score
            
            if score >= min_score:
                self.included[i] = True
                kept += 1
            else:
                self.included[i] = False
                
        self._update_plot()
        msg = f"Lattice filter kept {kept} patterns out of {len(self.images)}."
        self.status_label.setText(msg)
        QtWidgets.QMessageBox.information(self, "Auto-Filter Applied", msg)
        self.update_selection_count()

    def on_energy_changed(self, val):
        self.params['energy_keV'] = float(val)
        self.wavelength_m = compute_wavelength(self.params['energy_keV'])
        if len(self.images) > 0:
            self._recalculate_all_instrument_params()
            self._update_plot()

    def on_params_changed(self, *args):
        self.params['autocorr_peak_min_sep'] = int(self.ac_minsep_spin.value())
        self.params['ignore_radius'] = int(self.ignore_radius_spin.value())
        if len(self.images) > 0:
            self._recompute_facet(index=self.idx)
            self._update_plot()

    def on_instrument_params_changed(self, *args):
        self.params['camera_length_m'] = float(self.camlen_spin.value())
        self.params['pixel_size_m'] = float(self.pixsize_spin.value()) * 1e-6
        if len(self.images) > 0:
            self._recalculate_all_instrument_params()
            self._update_plot()

    def on_ac_thresh_changed(self, val):
        self.user_changed_thresh = True
        self.params['autocorr_threshold_rel'] = float(val)
        if len(self.images) > 0:
            self._recompute_facet(index=self.idx)
            self._update_plot()

    def on_pattern_jump(self, val):
        if len(self.images) == 0:
            return
            
        new_idx = val - 1 # UI is 1-based, internal data is 0-based
        
        if new_idx < 0 or new_idx >= len(self.images) or new_idx == self.idx:
            return
            
        self.idx = new_idx
        self.include_cb.blockSignals(True)
        self.include_cb.setChecked(bool(self.included[self.idx]))
        self.include_cb.blockSignals(False)

        if not self.user_changed_thresh:
            self.ac_thresh_spin.blockSignals(True)
            self.ac_thresh_spin.setValue(autocorr_threshold_rel)
            self.params['autocorr_threshold_rel'] = autocorr_threshold_rel
            self.ac_thresh_spin.blockSignals(False)
            self._recompute_facet(index=self.idx)
        else:
            self._recompute_facet(index=self.idx)

        self._update_plot()

    def _recompute_facet(self, index):
        if len(self.images) == 0:
            return

        if len(self.peaks[index]) < self.min_spots_spin.value():
            self.facets[index] = {'p0':None,'p1':None,'s0':None,'s1':None,'angle':None}
            self.facet_tuples[index] = (None, None, None)
            self.ac_peaks_all[index] = []
            self.conformity_scores[index] = 0.0
            return

        cx, cy = self.centers[index]
        img = self.images[index]
        peaks = self.peaks[index]
        ac = autocorr_from_peaks(peaks, img.shape)
        ac_peaks = find_autocorr_peaks(ac, threshold_rel=self.params['autocorr_threshold_rel'],
                                       min_separation=self.params['autocorr_peak_min_sep'])
        ac_cx, ac_cy = (ac.shape[1]-1)/2.0, (ac.shape[0]-1)/2.0
        ac_peaks_filtered = [(x,y) for (x,y) in ac_peaks if np.hypot(x-ac_cx,y-ac_cy)>=self.params['ignore_radius']]
        self.ac_peaks_all[index] = ac_peaks_filtered
        
        chosen = choose_principal_facet(ac_peaks_filtered, ac.shape, (cx,cy)) if len(ac_peaks_filtered)>=2 else None

        if not chosen:
            self.facets[index] = {'p0':None,'p1':None,'s0':None,'s1':None,'angle':None}
            self.facet_tuples[index] = (None, None, None)
            self.conformity_scores[index] = 0.0
            return

        p0_ac, p1_ac = chosen
        p0_det = ac_peak_to_detector_peak(p0_ac, ac.shape, img.shape, (cx,cy))
        p1_det = ac_peak_to_detector_peak(p1_ac, ac.shape, img.shape, (cx,cy))
        s0 = detector_pixel_to_s_Ainv(*p0_det, cx, cy,
                                      self.params['pixel_size_m'],
                                      self.params['camera_length_m'],
                                      self.wavelength_m)
        s1 = detector_pixel_to_s_Ainv(*p1_det, cx, cy,
                                      self.params['pixel_size_m'],
                                      self.params['camera_length_m'],
                                      self.wavelength_m)
        v0 = np.array([(p0_det[0]-cx)*self.params['pixel_size_m'], (p0_det[1]-cy)*self.params['pixel_size_m']])
        v1 = np.array([(p1_det[0]-cx)*self.params['pixel_size_m'], (p1_det[1]-cy)*self.params['pixel_size_m']])
        cosang = np.dot(v0,v1)/(np.linalg.norm(v0)*np.linalg.norm(v1))
        angle = np.degrees(np.arccos(np.clip(cosang,-1,1)))
        
        v1_ac = np.array([p0_ac[0] - ac_cx, p0_ac[1] - ac_cy])
        v2_ac = np.array([p1_ac[0] - ac_cx, p1_ac[1] - ac_cy])
        conf = compute_lattice_conformity(ac_peaks_filtered, v1_ac, v2_ac, ac_cx, ac_cy, tol_pixels=self.tol_pixels_spin.value(), scaling_factor=self.scaling_factor_spin.value())
        
        self.facets[index] = {'p0':p0_det,'p1':p1_det,'s0':s0,'s1':s1,'angle':angle}
        self.facet_tuples[index] = (s0, s1, angle)
        self.conformity_scores[index] = conf

    def _recalculate_all_instrument_params(self):
        wl = self.wavelength_m
        cam = self.params['camera_length_m']
        pix = self.params['pixel_size_m']
        
        for i in range(len(self.images)):
            f = self.facets[i]
            if f['p0'] is not None:
                cx, cy = self.centers[i]
                p0_det = f['p0']
                p1_det = f['p1']
                
                s0 = detector_pixel_to_s_Ainv(*p0_det, cx, cy, pix, cam, wl)
                s1 = detector_pixel_to_s_Ainv(*p1_det, cx, cy, pix, cam, wl)
                
                v0 = np.array([(p0_det[0]-cx)*pix, (p0_det[1]-cy)*pix])
                v1 = np.array([(p1_det[0]-cx)*pix, (p1_det[1]-cy)*pix])
                cosang = np.dot(v0,v1)/(np.linalg.norm(v0)*np.linalg.norm(v1))
                angle = np.degrees(np.arccos(np.clip(cosang,-1,1)))
                
                self.facets[i]['s0'] = s0
                self.facets[i]['s1'] = s1
                self.facets[i]['angle'] = angle
                self.facet_tuples[i] = (s0, s1, angle)

    def _update_plot(self):
        self.ax.clear()
        if len(self.images) == 0:
            self.ax.set_title("No images loaded")
            self.total_patterns_label.setText("/ 0")
            self.canvas.draw_idle()
            return

        if self.show_ac:
            ac = autocorr_from_peaks(self.peaks[self.idx], self.images[self.idx].shape)
            self.ax.imshow(np.log1p(ac), cmap='gray')
            
            if self.show_ac_circles_cb.isChecked():
                ac_peaks = find_autocorr_peaks(ac, threshold_rel=self.params['autocorr_threshold_rel'],
                                               min_separation=self.params['autocorr_peak_min_sep'])
                for px, py in ac_peaks:
                    self.ax.scatter(px, py, s=20, edgecolors='cyan', facecolors='none', linewidths=1.2)
                
            ac_cx, ac_cy = self.images[self.idx].shape[1]/2, self.images[self.idx].shape[0]/2
            self.ax.scatter(ac_cx, ac_cy, s=80, marker='x', color='lime')
            self.ax.set_title(f"Autocorr - Pattern {self.idx}")
        else:
            img = np.log1p(self.images[self.idx])
            self.ax.imshow(img, cmap='gray')
            cx, cy = self.centers[self.idx]
            self.ax.scatter(cx, cy, c='lime', s=40, marker='x')
            if self.show_peaks_cb.isChecked():
                for px, py in self.peaks[self.idx]:
                    self.ax.add_patch(Circle((px, py), 3, color='red', fill=False, lw=0.8))
            f = self.facets[self.idx]
            if self.show_facets_cb.isChecked() and f['p0'] is not None:
                p0, p1 = f['p0'], f['p1']
                self.ax.plot([cx, p0[0]], [cy, p0[1]], 'c-', lw=2)
                self.ax.plot([cx, p1[0]], [cy, p1[1]], 'c-', lw=2)
                s0, s1, ang = self.facet_tuples[self.idx]
                if s0 is not None:
                    score = self.conformity_scores[self.idx]
                    self.ax.text(0.02, 0.05,
                                 f"s₁={s0:.3f} Å⁻¹   s₂={s1:.3f} Å⁻¹   θ={ang:.1f}°\nConf: {score*100:.1f}%",
                                 color='yellow', fontsize=10, transform=self.ax.transAxes,
                                 bbox=dict(facecolor='black', alpha=0.4, pad=3))
            self.ax.set_title(f"Pattern {self.idx} (Included: {self.included[self.idx]})")

        self.ax.set_xlim(0, self.images[self.idx].shape[1])
        self.ax.set_ylim(self.images[self.idx].shape[0], 0)
        self.canvas.draw_idle()

        self.include_cb.blockSignals(True)
        self.include_cb.setChecked(bool(self.included[self.idx]))
        self.include_cb.blockSignals(False)
        self.pattern_spin.blockSignals(True)
        self.pattern_spin.setValue(self.idx + 1)
        self.pattern_spin.blockSignals(False)

    def on_save(self):
        if len(self.facet_tuples) == 0:
            QtWidgets.QMessageBox.information(self, "Nothing to save", "No facets to save.")
            return

        folder = self.current_folder if self.current_folder else os.path.dirname(os.path.abspath(sys.argv[0]))
        basename = self.output_name_edit.text().strip()
        if not basename:
            basename = "facet_vectors"

        included_indices = [i for i, inc in enumerate(self.included) if inc]
        included_tuples = [self.facet_tuples[i] for i in included_indices]

        # .csv (Index removed)
        csv_path = os.path.join(folder, f"{basename}.csv")
        header = "s0_Ainv,s1_Ainv,angle_deg\n"
        with open(csv_path, "w") as fcsv:
            fcsv.write(header)
            for i, tup in zip(included_indices, included_tuples):
                s0, s1, ang = tup
                if s0 is None:
                    fcsv.write(",,\n")
                else:
                    fcsv.write(f"{s0:.6e},{s1:.6e},{ang:.6f}\n")

        msg = f"Saved {len(included_indices)} facets to:\n{csv_path}"
        self.status_label.setText(msg)
        print(msg)

    def on_group_facets(self):
        valid_data = []
        for i, inc in enumerate(self.included):
            if inc:
                s0, s1, ang = self.facet_tuples[i]
                if s0 is not None:
                    valid_data.append({'s0_Ainv': s0, 's1_Ainv': s1, 'angle_deg': ang})

        if len(valid_data) < 2:
            QtWidgets.QMessageBox.warning(self, "Not enough data", "Need at least 2 valid/included facets to cluster.")
            return

        df = pd.DataFrame(valid_data)
        
        s_cols = np.sort(df[['s0_Ainv', 's1_Ainv']].values, axis=1)
        df['s_min'] = s_cols[:, 0]
        df['s_max'] = s_cols[:, 1]

        tol_s = self.tol_s_spin.value()
        tol_angle = self.tol_angle_spin.value()
        min_samples = self.min_samples_spin.value()

        features = np.column_stack([
            df['s_min'] / tol_s,
            df['s_max'] / tol_s,
            df['angle_deg'] / tol_angle
        ])

        clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5, linkage='complete')
        df['cluster'] = clusterer.fit_predict(features)

        grouped_df = df.groupby('cluster').agg({
            's_min': 'mean',
            's_max': 'mean',
            'angle_deg': 'mean',
            's0_Ainv': 'count' 
        }).reset_index(drop=True)

        grouped_df = grouped_df[grouped_df['s0_Ainv'] >= min_samples]

        final_df = pd.DataFrame({
            'support_count': grouped_df['s0_Ainv'], 
            's0_Ainv': grouped_df['s_min'],
            's1_Ainv': grouped_df['s_max'],
            'angle_deg': grouped_df['angle_deg']
        }).sort_values(by='support_count', ascending=False).reset_index(drop=True)

        # Store in memory instead of saving to file
        self.grouped_df = final_df
        
        # Update GUI elements
        num_groups = len(self.grouped_df)
        self.lbl_grouped_count.setText(f"Grouped: {num_groups}")
        self.save_grouped_btn.setEnabled(True) # Activate the save button

        msg = f"Clustered {len(df)} facets into {num_groups} unique groups.\nYou can now save the grouped facets."
        self.status_label.setText(f"Grouped into {num_groups} facets.")
        QtWidgets.QMessageBox.information(self, "Clustering Complete", msg)

    def on_save_grouped(self):
        if self.grouped_df is None or self.grouped_df.empty:
            return
            
        folder = self.current_folder if self.current_folder else os.path.dirname(os.path.abspath(sys.argv[0]))
        basename = self.output_name_edit.text().strip()
        if not basename:
            basename = "facet_vectors"
            
        csv_path = os.path.join(folder, f"{basename}_grouped.csv")
        self.grouped_df.to_csv(csv_path, index=False)

        msg = f"Saved {len(self.grouped_df)} grouped facets to:\n{csv_path}"
        self.status_label.setText(msg)
        print(msg)
        QtWidgets.QMessageBox.information(self, "Grouped Facets Saved", msg)

    def on_save_current_images(self):
        if len(self.images) == 0:
            QtWidgets.QMessageBox.information(self, "No Images", "No images loaded to save.")
            return

        folder = self.current_folder if self.current_folder else os.path.dirname(os.path.abspath(sys.argv[0]))
        idx = self.idx
        img = self.images[idx]
        peaks = self.peaks[idx]
        cx, cy = self.centers[idx]
        
        # --- 1. Save CLEAN Diffraction Pattern (No graphics, exact pixel size) ---
        diff_path_clean = os.path.join(folder, f"pattern_{idx:04d}_diff_clean.png")
        plt.imsave(diff_path_clean, np.log1p(img), cmap='gray')

        # --- 2. Save CLEAN Autocorrelation (No graphics, exact pixel size) ---
        ac = autocorr_from_peaks(peaks, img.shape)
        ac_path_clean = os.path.join(folder, f"pattern_{idx:04d}_ac_clean.png")
        plt.imsave(ac_path_clean, np.log1p(ac), cmap='gray')

        # --- 3. Save ANNOTATED Diffraction Pattern (High Res, Graphics on) ---
        fig_diff = Figure(figsize=(10, 10), dpi=300)
        ax_diff = fig_diff.add_subplot(111)
        ax_diff.imshow(np.log1p(img), cmap='gray')
        ax_diff.scatter(cx, cy, c='lime', s=40, marker='x')

        if self.show_peaks_cb.isChecked():
            for px, py in peaks:
                ax_diff.add_patch(Circle((px, py), 3, color='red', fill=False, lw=0.8))
        
        f = self.facets[idx]
        info_text = ""
        if self.show_facets_cb.isChecked() and f['p0'] is not None:
            p0, p1 = f['p0'], f['p1']
            ax_diff.plot([cx, p0[0]], [cy, p0[1]], 'c-', lw=2)
            ax_diff.plot([cx, p1[0]], [cy, p1[1]], 'c-', lw=2)
            s0, s1, ang = self.facet_tuples[idx]
            if s0 is not None:
                info_text = f"s1={s0:.4f} Ainv, s2={s1:.4f} Ainv, ang={ang:.2f} deg, Conf={self.conformity_scores[idx]*100:.1f}%"
        
        if info_text:
            ax_diff.set_xlabel(info_text, fontsize=14, color='black', labelpad=10)

        ax_diff.set_title(f"Pattern {idx}", fontsize=14)
        ax_diff.set_xticks([])
        ax_diff.set_yticks([])

        diff_path_annotated = os.path.join(folder, f"pattern_{idx:04d}_diff_annotated.png")
        fig_diff.savefig(diff_path_annotated, bbox_inches='tight', dpi=300)
        
        # --- 4. Save ANNOTATED Autocorrelation ---
        fig_ac = Figure(figsize=(10, 10), dpi=300)
        ax_ac = fig_ac.add_subplot(111)
        ax_ac.imshow(np.log1p(ac), cmap='gray')
        
        ac_cx, ac_cy = img.shape[1]/2, img.shape[0]/2
        ax_ac.scatter(ac_cx, ac_cy, s=80, marker='x', color='lime')

        if self.show_ac_circles_cb.isChecked():
            ac_peaks = find_autocorr_peaks(ac, threshold_rel=self.params['autocorr_threshold_rel'],
                                           min_separation=self.params['autocorr_peak_min_sep'])
            for px, py in ac_peaks:
                ax_ac.scatter(px, py, s=20, edgecolors='cyan', facecolors='none', linewidths=1.2)
        
        ax_ac.set_title(f"Autocorr {idx}", fontsize=14)
        ax_ac.set_xticks([])
        ax_ac.set_yticks([])
        
        ac_path_annotated = os.path.join(folder, f"pattern_{idx:04d}_ac_annotated.png")
        fig_ac.savefig(ac_path_annotated, bbox_inches='tight', dpi=300)
        
        msg = f"Saved 4 images for pattern {idx} to:\n{folder}"
        self.status_label.setText(f"Saved images for pattern {idx}")
        print(msg)
        QtWidgets.QMessageBox.information(self, "Images Saved", msg)

    def closeEvent(self, event):
        event.accept()

if __name__ == "__main__":
    # Support for multiprocessing on Windows
    multiprocessing.freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    default_folder = initial_folder if initial_folder else script_folder
    viewer = QtViewer(patterns_all, centers_all, peaks_all, facet_info_all, facet_tuples, ac_peaks_all, conformity_scores_all, default_folder)
    viewer.resize(1300, 900)
    viewer.show()
    sys.exit(app.exec())
    