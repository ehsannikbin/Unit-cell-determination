import sys
import os
import time
import csv
import numpy as np
import multiprocessing
from math import radians

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                               QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, 
                               QFileDialog, QGroupBox, QCheckBox, QFormLayout, 
                               QSplitter, QMessageBox, QSizePolicy)
from PySide6.QtCore import Qt, QThread, Signal, QObject

from scipy.optimize import differential_evolution, minimize
from numba import njit

# =============================================================================
#  PART 1: GLOBAL ALGORITHMS (Must be top-level for Multiprocessing)
# =============================================================================

@njit(fastmath=True, cache=True)
def fast_objective_loop(patterns, U, H, G, tol_len_rel, tol_cos_abs):
    """
    Optimized loop matching your original logic.
    Includes the 'Fallback' (best 8) and Cosine Tolerance.
    """
    n_cand = U.shape[0]
    
    # Precompute G @ U.T
    GU_rows = np.zeros((n_cand, 3))
    for i in range(n_cand):
        for j in range(3):
            val = 0.0
            for k in range(3):
                val += U[i, k] * G[j, k]
            GU_rows[i, j] = val

    # Precompute lengths
    pred_lens_all = np.zeros(n_cand)
    for i in range(n_cand):
        dot = 0.0
        for k in range(3):
            dot += U[i, k] * GU_rows[i, k]
        pred_lens_all[i] = np.sqrt(max(0.0, dot) + 1e-16)

    # Precompute H weights
    H_sq = np.zeros(n_cand)
    for i in range(n_cand):
        s = 0.0
        for k in range(3):
            s += H[i, k]**2
        H_sq[i] = s

    total_residual = 0.0
    eps = 1e-12
    
    n_patterns = patterns.shape[0]
    
    for p_idx in range(n_patterns):
        l1 = patterns[p_idx, 0]
        l2 = patterns[p_idx, 1]
        cos_obs = patterns[p_idx, 2]
        
        idx1 = []
        idx2 = []
        
        # Pass 1: Filter
        for i in range(n_cand):
            re1 = abs(pred_lens_all[i] - l1) / (l1 + eps)
            re2 = abs(pred_lens_all[i] - l2) / (l2 + eps)
            
            if re1 <= tol_len_rel: idx1.append(i)
            if re2 <= tol_len_rel: idx2.append(i)
            
        # --- FALLBACK MECHANISM ---
        if len(idx1) == 0:
            errs = np.abs(pred_lens_all - l1)
            sorted_indices = np.argsort(errs)
            for k in range(8): idx1.append(sorted_indices[k])
                
        if len(idx2) == 0:
            errs = np.abs(pred_lens_all - l2)
            sorted_indices = np.argsort(errs)
            for k in range(8): idx2.append(sorted_indices[k])

        len_idx1 = len(idx1)
        len_idx2 = len(idx2)

        # Heuristic Cap
        if len_idx1 * len_idx2 > 2500:
             if len_idx1 > 50: len_idx1 = 50
             if len_idx2 > 50: len_idx2 = 50

        # Prepare Observed 2x2 matrix data
        sin_obs = np.sqrt(max(0.0, 1.0 - cos_obs**2))
        p11 = l1
        p12 = l2 * cos_obs
        p22 = l2 * sin_obs
        p_norms_sq = l1**2 + l2**2
        
        min_r_pattern = 1e20 # Large number

        for ii in range(len_idx1):
            i = idx1[ii]
            for jj in range(len_idx2):
                j = idx2[jj]
                
                cx = U[i,1]*U[j,2] - U[i,2]*U[j,1]
                cy = U[i,2]*U[j,0] - U[i,0]*U[j,2]
                cz = U[i,0]*U[j,1] - U[i,1]*U[j,0]
                cross_sq = cx*cx + cy*cy + cz*cz
                
                if cross_sq < 1e-14:
                    continue

                dot_val = 0.0
                for k in range(3):
                    dot_val += U[i,k] * GU_rows[j,k]
                
                denom = pred_lens_all[i] * pred_lens_all[j] + eps
                cp = dot_val / denom
                
                if cp > 1.0: cp = 1.0
                if cp < -1.0: cp = -1.0
                
                # --- Cos Tolerance Check ---
                if abs(cp - cos_obs) > tol_cos_abs:
                    continue
                
                sp = np.sqrt(1.0 - cp**2)
                ql1 = pred_lens_all[i]
                ql2 = pred_lens_all[j]

                t2 = ql2 * cp
                t3 = ql2 * sp
                
                A11 = ql1 * p11 + t2 * p12
                A12 = t2 * p22
                A21 = t3 * p12
                A22 = t3 * p22
                
                trace_AAT = A11**2 + A12**2 + A21**2 + A22**2
                det_A = A11*A22 - A12*A21
                sumS = np.sqrt(trace_AAT + 2.0 * abs(det_A))
                
                q_norms_sq = ql1**2 + ql2**2
                
                r_raw = p_norms_sq + q_norms_sq - 2.0 * sumS
                if r_raw < 0.0: r_raw = 0.0
                
                weight = H_sq[i] + H_sq[j]
                weighted_res = r_raw * weight
                
                if weighted_res < min_r_pattern:
                    min_r_pattern = weighted_res
        
        total_residual += min_r_pattern

    return total_residual


# --- Converters and Wrapper (Pickleable) ---

def free_to_metric_vector(free, system):
    if system == 'triclinic':
        a,b,c,alpha_deg,beta_deg,gamma_deg = free
        alpha, beta, gamma = radians(alpha_deg), radians(beta_deg), radians(gamma_deg)
    elif system == 'monoclinic':
        a,b,c,beta_deg = free
        alpha, gamma = np.pi/2, np.pi/2
        beta = radians(beta_deg)
    elif system == 'orthorhombic':
        a,b,c = free
        alpha = beta = gamma = np.pi/2
    elif system == 'tetragonal':
        a,c = free; b = a; alpha = beta = gamma = np.pi/2
    elif system == 'hexagonal':
        a,c = free; b = a; gamma = 2*np.pi/3; alpha = beta = np.pi/2
    elif system == 'rhombohedral':
        a,alpha_deg = free; b = a; c = a; alpha = beta = gamma = radians(alpha_deg)
    elif system == 'cubic':
        a = free[0]; b = a; c = a; alpha = beta = gamma = np.pi/2
    else:
        raise ValueError("Unknown system")

    g11 = a*a; g22 = b*b; g33 = c*c
    g12 = a*b*np.cos(gamma)
    g13 = a*c*np.cos(beta)
    g23 = b*c*np.cos(alpha)
    return np.array([g11, g22, g33, g12, g13, g23], dtype=float)

def vec6_to_G(vec6):
    g11,g22,g33,g12,g13,g23 = vec6
    return np.array([[g11, g12, g13], [g12, g22, g23], [g13, g23, g33]], dtype=float)

# IMPORTANT: This wrapper must accept ALL data as arguments.
# This allows 'differential_evolution' to pickle it and send it to workers.
def objective_wrapper(free, patterns, U, candidates, system, tol_len_rel, tol_cos_abs):
    vec6 = free_to_metric_vector(free, system)
    G = vec6_to_G(vec6)

    # PD Check
    try:
        eigs = np.linalg.eigvalsh(G)
        if np.any(eigs <= 1e-12):
            return 1e12 + np.sum(np.abs(eigs[eigs <= 0]))*1e8
    except:
        return 1e12

    return fast_objective_loop(patterns, U, candidates, G, tol_len_rel, tol_cos_abs)

# =============================================================================
#  PART 2: WORKER THREAD
# =============================================================================

class OptimizationWorker(QObject):
    log_signal = Signal(str)
    finished_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.start_time = 0

    def log(self, msg):
        self.log_signal.emit(msg)

    def run(self):
        try:
            s = self.settings
            self.start_time = time.time()
            self.log(f"--- Starting Optimization ({s['system']}) ---")
            
            # 1. Load Data
            pattern_data = self.read_csv(s['file_path'])
            if not pattern_data:
                self.error_signal.emit("No valid data found in CSV.")
                return

            patterns = np.array([(l1, l2, np.cos(np.deg2rad(theta))) 
                                 for (l1, l2, theta) in pattern_data], dtype=float)
            
            # 2. Build Candidates
            candidates = self.build_integer_candidates(s['M'], s['centering'])
            U = candidates.astype(float)
            self.log(f"Generated {len(candidates)} candidates.")

            # 3. Bounds
            bounds = self.get_bounds(s['system'], s['ranges'])

            # 4. Arguments
            # (patterns, U, candidates, system, tol_len, tol_cos)
            args = (patterns, U, candidates, s['system'], s['tol_len_rel'], s['tol_cos_abs'])

            # 5. Global Search (DE)
            self.log("Starting Differential Evolution (global search)...")
            self.log("Compiling JIT function (warmup)...")
            dummy = np.mean(bounds, axis=1)
            objective_wrapper(dummy, *args)
            self.log("Compilation done.")

            # Define Callback with Closure to access self
            def de_callback(xk, convergence):
                # We can't log every single step or GUI floods, but Scipy calls this once per generation
                # We need generation count. Scipy doesn't pass it, so we use a mutable counter
                de_callback.iter += 1
                val = objective_wrapper(xk, *args)
                if val < de_callback.best: de_callback.best = val
                
                # Mimic original log format:
                # [DE gen X] time=Ys obj=Z best=Z conv=C
                if de_callback.iter % 10 == 0:
                    t_el = time.time() - self.start_time
                    msg = f"[DE gen {de_callback.iter}] time={t_el:.1f}s obj={val:.6g} best={de_callback.best:.6g} conv={convergence:.4g}"
                    self.log(msg)
            
            de_callback.iter = 0
            de_callback.best = np.inf

            # WORKERS: Pass the user setting (-1 = All)
            n_workers = s['workers']
            
            res_de = differential_evolution(
                objective_wrapper, 
                bounds,
                args=args, 
                strategy=s['de_strategy'],
                maxiter=s['de_maxiter'],
                popsize=s['de_popsize'],
                tol=s['de_tol'],
                mutation=(0.5, 1.9) if s['de_strategy']=='rand1bin' else (0.2, 1.5),
                recombination=0.7,
                callback=de_callback,
                workers=n_workers,
                polish=False
            )
            self.log(f"DE finished. time={time.time()-self.start_time:.1f}s best_fun={res_de.fun:.6g}")

            # 7. Outlier Rejection
            current_x = res_de.x
            
            if s['use_outliers']:
                self.log(f"\n--- Outlier Detection ---")
                # Calculate scores (One pass)
                scores = self.calculate_scores(patterns, U, candidates, current_x, s)
                threshold = np.percentile(scores, s['keep_pct'])
                good_mask = scores <= threshold
                n_keep = np.sum(good_mask)
                
                self.log(f"Total Patterns: {len(patterns)}")
                self.log(f"Score Threshold: {threshold:.6g}")
                self.log(f"Retaining: {n_keep}")
                self.log(f"Dropping: {len(patterns) - n_keep}")
                
                if n_keep < 5:
                    self.log("WARNING: Too few patterns left. Skipping refinement.")
                else:
                    patterns = patterns[good_mask]
                    # Update args
                    args = (patterns, U, candidates, s['system'], s['tol_len_rel'], s['tol_cos_abs'])
            
            # 8. Local Refinement
            self.log("Starting local refinement (L-BFGS-B)...")
            t0 = time.time()
            res_local = minimize(objective_wrapper, current_x, args=args, method='L-BFGS-B', bounds=bounds,
                                 options={'maxiter':1000, 'ftol':1e-12})
            self.log(f"Local refine finished in {time.time()-t0:.1f}s. fun={res_local.fun:.6g}")

            # 9. Format Results
            final_free = res_local.x
            
            # Reciprocal
            vec6 = free_to_metric_vector(final_free, s['system'])
            G_final = vec6_to_G(vec6)
            astar, bstar, cstar, alstar, bestar, gastar = self.lattice_from_metric(G_final)
            
            # Real
            G_real = np.linalg.inv(G_final)
            a, b, c, al, be, ga = self.lattice_from_metric(G_real)

            # Build strings for logging (Exactly like original)
            recip_str = (f"\n=== Final Reciprocal Lattice Parameters ===\n"
                         f"a* = {astar:.6f} 1/Å\n"
                         f"b* = {bstar:.6f} 1/Å\n"
                         f"c* = {cstar:.6f} 1/Å\n"
                         f"alpha* = {alstar:.6f} deg\n"
                         f"beta* = {bestar:.6f} deg\n"
                         f"gamma* = {gastar:.6f} deg")
            
            real_str = (f"\n=== Final Real-Space Lattice Parameters ===\n"
                        f"a = {a:.3f} Å\n"
                        f"b = {b:.3f} Å\n"
                        f"c = {c:.3f} Å\n"
                        f"alpha = {al:.3f} deg\n"
                        f"beta  = {be:.3f} deg\n"
                        f"gamma = {ga:.3f} deg")
            
            self.log(recip_str)
            self.log(real_str)
            self.log(f"\nTotal runtime: {time.time()-self.start_time:.1f}s")

           # Custom formatted string
            formatted_output = (f"# Crystal system = {s['system']}\n"
                                f"a= {a:.3f} Å\n"
                                f"b= {b:.3f} Å\n"
                                f"c= {c:.3f} Å\n"
                                f"alpha = {al:.3f} deg\n"
                                f"beta = {be:.3f} deg\n"
                                f"gamma = {ga:.3f} deg")

            # Result Dict
            results = {
                'params_str': formatted_output,
                'system': s['system']
            }
            self.finished_signal.emit(results)

        except Exception as e:
            self.error_signal.emit(str(e))
            import traceback
            traceback.print_exc()

    # Helpers
    def read_csv(self, fpath):
        data = []
        try:
            with open(fpath, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3:
                        try:
                            data.append((float(row[-3]), float(row[-2]), float(row[-1])))
                        except: pass
        except Exception as e:
            self.error_signal.emit(f"CSV Read Error: {e}")
        return data

    def build_integer_candidates(self, M, centering):
        cand = []
        def allowed(h, k, l, cent):
            cent = cent.upper()
            if cent == 'P': return True
            if cent == 'I': return ((h + k + l) % 2) == 0
            if cent == 'F': return (h%2 == k%2) and (k%2 == l%2)
            if cent == 'A': return ((k + l) % 2) == 0
            if cent == 'B': return ((h + l) % 2) == 0
            if cent == 'C': return ((h + k) % 2) == 0
            if cent == 'R': return ((-h + k + l)%3 == 0) or ((h - k + l)%3 == 0)
            return True
        for h in range(-M, M+1):
            for k in range(-M, M+1):
                for l in range(-M, M+1):
                    if h==0 and k==0 and l==0: continue
                    if allowed(h, k, l, centering):
                        cand.append((h,k,l))
        return np.array(cand, dtype=int)

    def get_bounds(self, system, ranges):
        def r2rec(rng): return (1.0/rng[1], 1.0/rng[0])
        r_a, r_b, r_c = ranges[0], ranges[1], ranges[2]
        r_al, r_be, r_ga = ranges[3], ranges[4], ranges[5]
        if system == 'triclinic': return [r2rec(r_a), r2rec(r_b), r2rec(r_c), r_al, r_be, r_ga]
        elif system == 'monoclinic':
            b_min, b_max = sorted(r_be)
            return [r2rec(r_a), r2rec(r_b), r2rec(r_c), (180-b_max, 180-b_min)]
        elif system == 'orthorhombic': return [r2rec(r_a), r2rec(r_b), r2rec(r_c)]
        elif system == 'tetragonal': return [r2rec(r_a), r2rec(r_c)]
        elif system == 'hexagonal': return [r2rec(r_a), r2rec(r_c)]
        elif system == 'rhombohedral': return [r2rec(r_a), r_al]
        elif system == 'cubic': return [r2rec(r_a)]
        return []

    def lattice_from_metric(self, G):
        a = np.sqrt(G[0,0])
        b = np.sqrt(G[1,1])
        c = np.sqrt(G[2,2])
        cos_al = np.clip(G[1,2]/(b*c), -1, 1)
        cos_be = np.clip(G[0,2]/(a*c), -1, 1)
        cos_ga = np.clip(G[0,1]/(a*b), -1, 1)
        return a, b, c, np.degrees(np.arccos(cos_al)), np.degrees(np.arccos(cos_be)), np.degrees(np.arccos(cos_ga))

    def calculate_scores(self, patterns, U, candidates, free_params, s):
        # Helper to run one pass and return scores for outlier rejection
        # Must re-create G and run a specialized scoring loop
        vec6 = free_to_metric_vector(free_params, s['system'])
        G = vec6_to_G(vec6)
        return _fast_score_calc(patterns, U, candidates, G, s['tol_len_rel'], s['tol_cos_abs'])

@njit(fastmath=True)
def _fast_score_calc(patterns, U, H, G, tol_len_rel, tol_cos_abs):
    # Almost identical to objective_loop but saves individual scores
    n_cand = U.shape[0]
    GU_rows = np.zeros((n_cand, 3))
    for i in range(n_cand):
        for j in range(3):
            val=0.0
            for k in range(3): val += U[i,k]*G[j,k]
            GU_rows[i,j] = val
    pred_lens = np.zeros(n_cand)
    for i in range(n_cand):
        val=0.0
        for k in range(3): val+=U[i,k]*GU_rows[i,k]
        pred_lens[i] = np.sqrt(val)
    H_sq = np.zeros(n_cand)
    for i in range(n_cand):
        val=0.0
        for k in range(3): val+=H[i,k]**2
        H_sq[i]=val
    
    scores = np.zeros(len(patterns))
    eps=1e-12
    
    for p in range(len(patterns)):
        l1, l2, cos_obs = patterns[p]
        idx1 = []
        idx2 = []
        for i in range(n_cand):
            if abs(pred_lens[i]-l1)/l1 <= tol_len_rel: idx1.append(i)
            if abs(pred_lens[i]-l2)/l2 <= tol_len_rel: idx2.append(i)
        
        # Fallback
        if len(idx1)==0:
             errs = np.abs(pred_lens[i] - l1)
             # Basic sort manually or just penalize to save speed in outlier check
             # For outlier check, if it requires fallback, it's likely a bad point anyway
             scores[p] = 1e6; continue
        if len(idx2)==0:
             scores[p] = 1e6; continue

        sin_obs = np.sqrt(max(0.0, 1.0-cos_obs**2))
        p_norms = l1**2 + l2**2
        min_r = 1e9
        
        lim1 = len(idx1) if len(idx1)<50 else 50
        lim2 = len(idx2) if len(idx2)<50 else 50
        
        for ii in range(lim1):
            i = idx1[ii]
            for jj in range(lim2):
                j = idx2[jj]
                cx = U[i,1]*U[j,2] - U[i,2]*U[j,1]
                cy = U[i,2]*U[j,0] - U[i,0]*U[j,2]
                cz = U[i,0]*U[j,1] - U[i,1]*U[j,0]
                if (cx*cx+cy*cy+cz*cz) < 1e-12: continue
                
                dot = 0.0
                for k in range(3): dot+=U[i,k]*GU_rows[j,k]
                cp = dot/(pred_lens[i]*pred_lens[j]+eps)
                if cp>1: cp=1
                if cp<-1: cp=-1
                
                if abs(cp-cos_obs) > tol_cos_abs: continue

                sp = np.sqrt(1.0-cp**2)
                ql1, ql2 = pred_lens[i], pred_lens[j]
                
                t2=ql2*cp; t3=ql2*sp
                A11 = ql1*l1 + t2*l2*cos_obs
                A12 = t2*l2*sin_obs
                A21 = t3*l2*cos_obs
                A22 = t3*l2*sin_obs
                
                tr = A11**2 + A12**2 + A21**2 + A22**2
                det = A11*A22 - A12*A21
                sumS = np.sqrt(tr + 2*abs(det))
                r = p_norms + ql1**2 + ql2**2 - 2*sumS
                if r<0: r=0
                w_r = r * (H_sq[i]+H_sq[j])
                if w_r < min_r: min_r = w_r
        scores[p] = min_r
    return scores

# =============================================================================
#  PART 3: GUI MAIN WINDOW
# =============================================================================

class CrystalApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diffraction Cell Finder")
        self.resize(1150, 850)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # --- Left Panel ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(420)
        
        # 1. File Selection
        file_group = QGroupBox("I/O Settings")
        file_layout = QFormLayout()
        
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select facet_vectors.csv...")
        browse_btn = QPushButton("...")
        browse_btn.setFixedWidth(40)
        browse_btn.clicked.connect(self.browse_file)
        
        h_file = QHBoxLayout()
        h_file.addWidget(self.path_edit)
        h_file.addWidget(browse_btn)
        
        self.log_name = QLineEdit("run_log.txt")
        self.cell_name = QLineEdit("unit_cell.txt")
        
        file_layout.addRow("Input CSV:", h_file)
        file_layout.addRow("Log Name:", self.log_name)
        file_layout.addRow("Cell Name:", self.cell_name)
        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)
        
        # 2. Crystal System
        crys_group = QGroupBox("Crystal Settings")
        crys_layout = QFormLayout()
        
        self.sys_combo = QComboBox()
        self.sys_combo.addItems(['triclinic', 'monoclinic', 'orthorhombic', 
                                 'tetragonal', 'hexagonal', 'rhombohedral', 'cubic'])
        self.sys_combo.currentTextChanged.connect(self.update_range_inputs)
        
        self.cent_combo = QComboBox()
        self.cent_combo.addItems(['P', 'I', 'F', 'A', 'B', 'C', 'R'])
        
        crys_layout.addRow("System:", self.sys_combo)
        crys_layout.addRow("Centering:", self.cent_combo)
        crys_group.setLayout(crys_layout)
        left_layout.addWidget(crys_group)
        
        # 3. Ranges
        range_group = QGroupBox("Parameter Ranges (Å and Deg)")
        self.range_layout = QFormLayout()
        self.inputs = {}
        labels = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
        defaults = [(4, 30), (4, 30), (4, 30), (90, 120), (90, 120), (90, 120)]
        for lbl, (dmin, dmax) in zip(labels, defaults):
            w = QWidget()
            l = QHBoxLayout(w)
            l.setContentsMargins(0,0,0,0)
            sb1 = QDoubleSpinBox(); sb1.setRange(0.1, 999); sb1.setValue(dmin)
            sb2 = QDoubleSpinBox(); sb2.setRange(0.1, 999); sb2.setValue(dmax)
            sb1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            sb2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            l.addWidget(sb1); l.addWidget(QLabel("-")); l.addWidget(sb2)
            #l.addStretch()
            self.inputs[lbl] = (w, sb1, sb2)
            self.range_layout.addRow(f"{lbl}:", w)
        range_group.setLayout(self.range_layout)
        left_layout.addWidget(range_group)
        
        # 4. Algo Settings
        algo_group = QGroupBox("Algorithm Settings")
        algo_layout = QFormLayout()
        
        self.sb_M = QSpinBox(); self.sb_M.setValue(6)
        self.sb_pop = QSpinBox(); self.sb_pop.setRange(10, 500); self.sb_pop.setValue(100)
        self.sb_iter = QSpinBox(); self.sb_iter.setRange(10, 5000); self.sb_iter.setValue(1000)
        self.sb_workers = QSpinBox(); self.sb_workers.setRange(-1, 64); self.sb_workers.setValue(-1)
        self.sb_workers.setToolTip("-1 = All Cores")
        self.sb_tol_rel = QDoubleSpinBox(); self.sb_tol_rel.setSingleStep(0.01); self.sb_tol_rel.setValue(0.2)
        self.sb_tol_cos = QDoubleSpinBox(); self.sb_tol_cos.setSingleStep(0.01); self.sb_tol_cos.setValue(0.15)
        
        self.combo_strat = QComboBox()
        self.combo_strat.addItems(['rand1bin', 'best1bin'])
        
        self.chk_outlier = QCheckBox("Outlier Rejection")
        self.sb_keep = QSpinBox(); self.sb_keep.setRange(10, 100); self.sb_keep.setValue(80)
        
        algo_layout.addRow("Max hkl (M):", self.sb_M)
        algo_layout.addRow("Pop Size:", self.sb_pop)
        algo_layout.addRow("Max Iter:", self.sb_iter)
        algo_layout.addRow("Processors:", self.sb_workers)
        algo_layout.addRow("Len Tol (Rel):", self.sb_tol_rel)
        algo_layout.addRow("Cos Tol (Abs):", self.sb_tol_cos)
        algo_layout.addRow("Strategy:", self.combo_strat)
        algo_layout.addRow(self.chk_outlier, self.sb_keep)
        algo_layout.setItem(7, QFormLayout.LabelRole, algo_layout.itemAt(7, QFormLayout.FieldRole))
        
        algo_group.setLayout(algo_layout)
        left_layout.addWidget(algo_group)
        
        # Run Button
        self.run_btn = QPushButton("RUN OPTIMIZATION")
        self.run_btn.setFixedHeight(45)
        self.run_btn.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #2b5b84; color: white;")
        self.run_btn.clicked.connect(self.start_optimization)
        left_layout.addWidget(self.run_btn)
        
        # --- Right Panel: Log ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        lbl_log = QLabel("Run Log:")
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setStyleSheet("font-family: Consolas, monospace; font-size: 10pt; background: #f0f0f0;")
        right_layout.addWidget(lbl_log)
        right_layout.addWidget(self.text_log)
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        layout.addWidget(splitter)
        
        self.update_range_inputs(self.sys_combo.currentText())

    def browse_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if f: self.path_edit.setText(f)

    def update_range_inputs(self, system):
        for k, v in self.inputs.items():
            v[0].setVisible(False)
            self.range_layout.labelForField(v[0]).setVisible(False)
        vis = []
        if system == 'triclinic': vis = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
        elif system == 'monoclinic': vis = ['a', 'b', 'c', 'beta']
        elif system == 'orthorhombic': vis = ['a', 'b', 'c']
        elif system == 'tetragonal': vis = ['a', 'c']
        elif system == 'hexagonal': vis = ['a', 'c']
        elif system == 'rhombohedral': vis = ['a', 'alpha']
        elif system == 'cubic': vis = ['a']
        for k in vis:
            self.inputs[k][0].setVisible(True)
            self.range_layout.labelForField(self.inputs[k][0]).setVisible(True)

    def start_optimization(self):
        csv_path = self.path_edit.text()
        if not os.path.exists(csv_path):
            QMessageBox.critical(self, "Error", "File not found!")
            return

        self.run_btn.setEnabled(False)
        self.text_log.clear()
        
        ranges = []
        for k in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']:
            w, sb1, sb2 = self.inputs[k]
            ranges.append((sb1.value(), sb2.value()))

        settings = {
            'file_path': csv_path,
            'log_name': self.log_name.text(),
            'cell_name': self.cell_name.text(),
            'system': self.sys_combo.currentText(),
            'centering': self.cent_combo.currentText(),
            'ranges': ranges,
            'M': self.sb_M.value(),
            'tol_len_rel': self.sb_tol_rel.value(),
            'tol_cos_abs': self.sb_tol_cos.value(),
            'de_popsize': self.sb_pop.value(),
            'de_maxiter': self.sb_iter.value(),
            'de_strategy': self.combo_strat.currentText(),
            'de_tol': 1e-6,
            'use_outliers': self.chk_outlier.isChecked(),
            'keep_pct': self.sb_keep.value(),
            'workers': self.sb_workers.value()
        }

        self.thread = QThread()
        self.worker = OptimizationWorker(settings)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log_signal.connect(self.text_log.append)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.finished_signal.connect(self.thread.quit)
        self.worker.finished_signal.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_error(self, msg):
        self.text_log.append(f"\nERROR: {msg}")
        self.run_btn.setEnabled(True)

    def on_finished(self, res):
        self.run_btn.setEnabled(True)
        try:
            base_dir = os.path.dirname(self.path_edit.text())
            
            # Save Log
            log_path = os.path.join(base_dir, self.log_name.text())
            with open(log_path, "w", encoding='utf-8') as f:
                f.write(self.text_log.toPlainText())
            
            # Save Clean Cell File
            cell_path = os.path.join(base_dir, self.cell_name.text())
            with open(cell_path, "w", encoding='utf-8') as f:
                #f.write(f"{res['system']}\n")
                f.write(res['params_str'])
            
            self.text_log.append(f"\nSaved files to:\n{log_path}\n{cell_path}")
            
        except Exception as e:
            self.text_log.append(f"Error saving files: {e}")

if __name__ == "__main__":
    multiprocessing.freeze_support() # Essential for Windows executable/multiprocessing
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = CrystalApp()
    window.show()
    sys.exit(app.exec())
