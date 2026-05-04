import sys
import os
import time
import csv
import numpy as np
import multiprocessing
from math import radians

# --- IMPORTS FOR CELL REDUCTION ---
try:
    from pymatgen.core import Lattice, Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    print("Warning: pymatgen not installed. Cell reduction feature will be disabled.")

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                               QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, 
                               QFileDialog, QGroupBox, QCheckBox, QFormLayout, 
                               QSplitter, QMessageBox, QSizePolicy)
from PySide6.QtCore import Qt, QThread, Signal, QObject

import nlopt
from scipy.optimize import differential_evolution, minimize
from numba import njit

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Maximum number of hkl planes to evaluate per observed vector length.
MAX_PLANES_DEFAULT = 8

# False = Use closest 2 neighbors if tolerance fails
# True  = Apply high penalty (Hard Fallback) - Stricter, discards bad fits immediately
USE_HARD_FALLBACK = False

# --- NLOPT DIRECT CONFIGURATION ---
DIRECT_XTOL_REL_LEN_DEFAULT = 1e-2  # 1.0% relative precision for 1/a, 1/b, 1/c
DIRECT_XTOL_ABS_ANG_DEFAULT = 0.5  # 0.5 degrees absolute precision for angles

# =============================================================================
#  PART 1: GLOBAL ALGORITHMS
# =============================================================================

@njit(fastmath=True, cache=True)
def fast_objective_loop(patterns, U, H, G, max_planes, tol_len_rel, tol_ang_abs_rad, use_hard_fallback):
    n_cand = U.shape[0]
    
    GU_rows = np.zeros((n_cand, 3))
    for i in range(n_cand):
        for j in range(3):
            val = 0.0
            for k in range(3): val += U[i, k] * G[j, k]
            GU_rows[i, j] = val

    pred_lens_all = np.zeros(n_cand)
    for i in range(n_cand):
        dot = 0.0
        for k in range(3): dot += U[i, k] * GU_rows[i, k]
        pred_lens_all[i] = np.sqrt(max(0.0, dot) + 1e-16)

    H_sq = np.zeros(n_cand)
    for i in range(n_cand):
        H_sq[i] = H[i,0]**2 + H[i,1]**2 + H[i,2]**2

    sorted_indices = np.argsort(pred_lens_all)
    sorted_lens = pred_lens_all[sorted_indices]

    total_residual = 0.0
    eps = 1e-12
    n_patterns = patterns.shape[0]
    
    # Ensure array is at least size 2 for the fallback mechanism
    safe_len = max(2, max_planes)
    idx1 = np.zeros(safe_len, dtype=np.int64)
    idx2 = np.zeros(safe_len, dtype=np.int64)
    
    for p_idx in range(n_patterns):
        l1 = patterns[p_idx, 0]
        l2 = patterns[p_idx, 1]
        cos_obs = patterns[p_idx, 2]
        theta_obs_rad = np.arccos(cos_obs)
        
        # --- 1. Filter l1 with Two-Pointer Expansion ---
        insert_idx1 = np.searchsorted(sorted_lens, l1)
        count1 = 0
        left, right = insert_idx1 - 1, insert_idx1
        
        while count1 < max_planes:
            err_left = 1e10
            err_right = 1e10
            if left >= 0: err_left = abs(sorted_lens[left] - l1) / (l1 + eps)
            if right < n_cand: err_right = abs(sorted_lens[right] - l1) / (l1 + eps)
            
            if err_left > tol_len_rel and err_right > tol_len_rel: break
                
            if err_left <= err_right:
                idx1[count1] = sorted_indices[left]
                left -= 1
            else:
                idx1[count1] = sorted_indices[right]
                right += 1
            count1 += 1

        # --- 2. Filter l2 with Two-Pointer Expansion ---
        insert_idx2 = np.searchsorted(sorted_lens, l2)
        count2 = 0
        left, right = insert_idx2 - 1, insert_idx2
        
        while count2 < max_planes:
            err_left = 1e10
            err_right = 1e10
            if left >= 0: err_left = abs(sorted_lens[left] - l2) / (l2 + eps)
            if right < n_cand: err_right = abs(sorted_lens[right] - l2) / (l2 + eps)
            
            if err_left > tol_len_rel and err_right > tol_len_rel: break
                
            if err_left <= err_right:
                idx2[count2] = sorted_indices[left]
                left -= 1
            else:
                idx2[count2] = sorted_indices[right]
                right += 1
            count2 += 1

        # --- 3. FALLBACK MECHANISM ---
        if count1 == 0 or count2 == 0:
            if use_hard_fallback:
                total_residual += 1e9
                continue
            else:
                if count1 == 0:
                    p1, p2 = insert_idx1 - 1, insert_idx1
                    if p1 < 0: p1, p2 = 0, min(1, n_cand - 1)
                    elif p2 >= n_cand: p2, p1 = n_cand - 1, max(0, n_cand - 2)
                    idx1[0], idx1[1] = sorted_indices[p1], sorted_indices[p2]
                    count1 = 2
                if count2 == 0:
                    p1, p2 = insert_idx2 - 1, insert_idx2
                    if p1 < 0: p1, p2 = 0, min(1, n_cand - 1)
                    elif p2 >= n_cand: p2, p1 = n_cand - 1, max(0, n_cand - 2)
                    idx2[0], idx2[1] = sorted_indices[p1], sorted_indices[p2]
                    count2 = 2

        sin_obs = np.sqrt(max(0.0, 1.0 - cos_obs**2))
        p11, p12, p22 = l1, l2 * cos_obs, l2 * sin_obs
        p_norms_sq = l1**2 + l2**2
        min_r_pattern = 1e20 

        best_invalid_diff = 1e10 
        best_invalid_i = -1
        best_invalid_j = -1

        # --- EVALUATE NEIGHBORS ---
        for ii in range(count1):
            i = idx1[ii]
            for jj in range(count2):
                j = idx2[jj]
                
                cx = U[i,1]*U[j,2] - U[i,2]*U[j,1]
                cy = U[i,2]*U[j,0] - U[i,0]*U[j,2]
                cz = U[i,0]*U[j,1] - U[i,1]*U[j,0]
                if (cx*cx + cy*cy + cz*cz) < 1e-14: continue

                dot_val = 0.0
                for k in range(3): dot_val += U[i,k] * GU_rows[j,k]
                
                cp = dot_val / (pred_lens_all[i] * pred_lens_all[j] + eps)
                if cp > 1.0: cp = 1.0
                elif cp < -1.0: cp = -1.0
                
                angle_diff = abs(np.arccos(cp) - theta_obs_rad)
                if angle_diff > tol_ang_abs_rad: 
                    if angle_diff < best_invalid_diff:
                        best_invalid_diff = angle_diff
                        best_invalid_i = i
                        best_invalid_j = j
                    continue
                
                sp = np.sqrt(1.0 - cp**2)
                ql1, ql2 = pred_lens_all[i], pred_lens_all[j]
                
                t2, t3 = ql2 * cp, ql2 * sp
                A11, A12 = ql1 * p11 + t2 * p12, t2 * p22
                A21, A22 = t3 * p12, t3 * p22
                
                tr = A11**2 + A12**2 + A21**2 + A22**2
                det_A = A11*A22 - A12*A21
                sumS = np.sqrt(tr + 2.0 * abs(det_A))
                
                r_raw = p_norms_sq + ql1**2 + ql2**2 - 2.0 * sumS
                if r_raw < 0.0: r_raw = 0.0
                
                weighted_res = r_raw * (H_sq[i] + H_sq[j])
                if weighted_res < min_r_pattern: min_r_pattern = weighted_res
        
        # --- The Seamless Geometric Fallback ---
        if min_r_pattern == 1e20 and best_invalid_i != -1:
            # Nothing passed the angle check. Evaluate the absolute closest failing pair!
            i = best_invalid_i
            j = best_invalid_j
            
            dot_val = 0.0
            for k in range(3): dot_val += U[i,k] * GU_rows[j,k]
            cp = dot_val / (pred_lens_all[i] * pred_lens_all[j] + eps)
            if cp > 1.0: cp = 1.0
            elif cp < -1.0: cp = -1.0
            
            sp = np.sqrt(1.0 - cp**2)
            ql1, ql2 = pred_lens_all[i], pred_lens_all[j]
            t2, t3 = ql2 * cp, ql2 * sp
            
            A11, A12 = ql1 * p11 + t2 * p12, t2 * p22
            A21, A22 = t3 * p12, t3 * p22
            
            tr = A11**2 + A12**2 + A21**2 + A22**2
            det_A = A11*A22 - A12*A21
            sumS = np.sqrt(tr + 2.0 * abs(det_A))
            
            r_raw = p_norms_sq + ql1**2 + ql2**2 - 2.0 * sumS
            if r_raw < 0.0: r_raw = 0.0
            min_r_pattern = r_raw * (H_sq[i] + H_sq[j])

        # If it's still 1e20 here, it means all pairs were perfectly collinear (cx, cy, cz < 1e-14)
        # That is a true impossibility, so a 1e9 cliff is actually appropriate here.
        if min_r_pattern == 1e20:
            min_r_pattern = 1e9
                
        total_residual += min_r_pattern

    return total_residual

@njit(fastmath=True, cache=True)
def get_best_assignments(patterns, U, H, G, max_planes, tol_len_rel, tol_ang_abs_rad, use_hard_fallback):
    n_cand = U.shape[0]
    GU_rows = np.zeros((n_cand, 3))
    for i in range(n_cand):
        for j in range(3):
            val = 0.0
            for k in range(3): val += U[i, k] * G[j, k]
            GU_rows[i, j] = val

    pred_lens_all = np.zeros(n_cand)
    for i in range(n_cand):
        dot = 0.0
        for k in range(3): dot += U[i, k] * GU_rows[i, k]
        pred_lens_all[i] = np.sqrt(max(0.0, dot) + 1e-16)

    H_sq = np.zeros(n_cand)
    for i in range(n_cand):
        H_sq[i] = H[i,0]**2 + H[i,1]**2 + H[i,2]**2

    sorted_indices = np.argsort(pred_lens_all)
    sorted_lens = pred_lens_all[sorted_indices]

    n_patterns = patterns.shape[0]
    assignments = np.zeros((n_patterns, 2), dtype=np.int64)
    safe_len = max(2, max_planes)
    idx1 = np.zeros(safe_len, dtype=np.int64)
    idx2 = np.zeros(safe_len, dtype=np.int64)
    eps = 1e-12
    
    for p_idx in range(n_patterns):
        l1 = patterns[p_idx, 0]
        l2 = patterns[p_idx, 1]
        cos_obs = patterns[p_idx, 2]
        theta_obs_rad = np.arccos(cos_obs)
        
        insert_idx1 = np.searchsorted(sorted_lens, l1)
        count1 = 0
        left, right = insert_idx1 - 1, insert_idx1
        while count1 < max_planes:
            err_left = 1e10
            err_right = 1e10
            if left >= 0: err_left = abs(sorted_lens[left] - l1) / (l1 + eps)
            if right < n_cand: err_right = abs(sorted_lens[right] - l1) / (l1 + eps)
            if err_left > tol_len_rel and err_right > tol_len_rel: break
            if err_left <= err_right:
                idx1[count1] = sorted_indices[left]; left -= 1
            else:
                idx1[count1] = sorted_indices[right]; right += 1
            count1 += 1

        insert_idx2 = np.searchsorted(sorted_lens, l2)
        count2 = 0
        left, right = insert_idx2 - 1, insert_idx2
        while count2 < max_planes:
            err_left = 1e10
            err_right = 1e10
            if left >= 0: err_left = abs(sorted_lens[left] - l2) / (l2 + eps)
            if right < n_cand: err_right = abs(sorted_lens[right] - l2) / (l2 + eps)
            if err_left > tol_len_rel and err_right > tol_len_rel: break
            if err_left <= err_right:
                idx2[count2] = sorted_indices[left]; left -= 1
            else:
                idx2[count2] = sorted_indices[right]; right += 1
            count2 += 1

        if count1 == 0 or count2 == 0:
            if use_hard_fallback:
                assignments[p_idx, 0] = -1; assignments[p_idx, 1] = -1; continue
            else:
                if count1 == 0:
                    p1, p2 = insert_idx1 - 1, insert_idx1
                    if p1 < 0: p1, p2 = 0, min(1, n_cand - 1)
                    elif p2 >= n_cand: p2, p1 = n_cand - 1, max(0, n_cand - 2)
                    idx1[0], idx1[1] = sorted_indices[p1], sorted_indices[p2]; count1 = 2
                if count2 == 0:
                    p1, p2 = insert_idx2 - 1, insert_idx2
                    if p1 < 0: p1, p2 = 0, min(1, n_cand - 1)
                    elif p2 >= n_cand: p2, p1 = n_cand - 1, max(0, n_cand - 2)
                    idx2[0], idx2[1] = sorted_indices[p1], sorted_indices[p2]; count2 = 2

        sin_obs = np.sqrt(max(0.0, 1.0 - cos_obs**2))
        p11, p12, p22 = l1, l2 * cos_obs, l2 * sin_obs
        p_norms_sq = l1**2 + l2**2
        
        min_r_pattern = 1e20 
        best_i = -1; best_j = -1

        best_invalid_diff = 1e10 
        best_invalid_i = -1
        best_invalid_j = -1

        for ii in range(count1):
            i = idx1[ii]
            for jj in range(count2):
                j = idx2[jj]
                cx = U[i,1]*U[j,2] - U[i,2]*U[j,1]
                cy = U[i,2]*U[j,0] - U[i,0]*U[j,2]
                cz = U[i,0]*U[j,1] - U[i,1]*U[j,0]
                if (cx*cx + cy*cy + cz*cz) < 1e-14: continue

                dot_val = 0.0
                for k in range(3): dot_val += U[i,k] * GU_rows[j,k]
                cp = dot_val / (pred_lens_all[i] * pred_lens_all[j] + eps)
                if cp > 1.0: cp = 1.0
                elif cp < -1.0: cp = -1.0
                
                angle_diff = abs(np.arccos(cp) - theta_obs_rad)
                if angle_diff > tol_ang_abs_rad: 
                    if angle_diff < best_invalid_diff:
                        best_invalid_diff = angle_diff
                        best_invalid_i = i
                        best_invalid_j = j
                    continue
                
                sp = np.sqrt(1.0 - cp**2)
                ql1, ql2 = pred_lens_all[i], pred_lens_all[j]
                t2, t3 = ql2 * cp, ql2 * sp
                A11, A12 = ql1 * p11 + t2 * p12, t2 * p22
                A21, A22 = t3 * p12, t3 * p22
                
                tr = A11**2 + A12**2 + A21**2 + A22**2
                det = A11*A22 - A12*A21
                sumS = np.sqrt(tr + 2.0 * abs(det))
                
                r_raw = p_norms_sq + ql1**2 + ql2**2 - 2.0 * sumS
                if r_raw < 0.0: r_raw = 0.0
                
                weighted_res = r_raw * (H_sq[i] + H_sq[j])
                if weighted_res < min_r_pattern:
                    min_r_pattern = weighted_res
                    best_i, best_j = i, j

        if min_r_pattern == 1e20 and best_invalid_i != -1:
            i = best_invalid_i
            j = best_invalid_j
            
            dot_val = 0.0
            for k in range(3): dot_val += U[i,k] * GU_rows[j,k]
            cp = dot_val / (pred_lens_all[i] * pred_lens_all[j] + eps)
            if cp > 1.0: cp = 1.0
            elif cp < -1.0: cp = -1.0
            
            sp = np.sqrt(1.0 - cp**2)
            ql1, ql2 = pred_lens_all[i], pred_lens_all[j]
            t2, t3 = ql2 * cp, ql2 * sp
            
            A11, A12 = ql1 * p11 + t2 * p12, t2 * p22
            A21, A22 = t3 * p12, t3 * p22
            
            tr = A11**2 + A12**2 + A21**2 + A22**2
            det = A11*A22 - A12*A21
            sumS = np.sqrt(tr + 2.0 * abs(det))
            
            r_raw = p_norms_sq + ql1**2 + ql2**2 - 2.0 * sumS
            if r_raw < 0.0: r_raw = 0.0
            
            best_i = i
            best_j = j
                    
        assignments[p_idx, 0] = best_i
        assignments[p_idx, 1] = best_j
        
    return assignments

@njit(fastmath=True, cache=True)
def smooth_local_objective(patterns, U, H, G, assignments):
    """A purely continuous mathematical function for L-BFGS-B. No loops, no searching."""
    n_cand = U.shape[0]
    GU_rows = np.zeros((n_cand, 3))
    for i in range(n_cand):
        for j in range(3):
            val = 0.0
            for k in range(3): val += U[i, k] * G[j, k]
            GU_rows[i, j] = val

    pred_lens_all = np.zeros(n_cand)
    for i in range(n_cand):
        dot = 0.0
        for k in range(3): dot += U[i, k] * GU_rows[i, k]
        pred_lens_all[i] = np.sqrt(max(0.0, dot) + 1e-16)

    H_sq = np.zeros(n_cand)
    for i in range(n_cand):
        H_sq[i] = H[i,0]**2 + H[i,1]**2 + H[i,2]**2

    total_residual = 0.0
    eps = 1e-12
    
    for p_idx in range(patterns.shape[0]):
        i = assignments[p_idx, 0]
        j = assignments[p_idx, 1]
        
        if i == -1 or j == -1:
            total_residual += 1e9
            continue

        l1 = patterns[p_idx, 0]
        l2 = patterns[p_idx, 1]
        cos_obs = patterns[p_idx, 2]
        
        sin_obs = np.sqrt(max(0.0, 1.0 - cos_obs**2))
        p11 = l1; p12 = l2 * cos_obs; p22 = l2 * sin_obs
        p_norms_sq = l1**2 + l2**2

        dot_val = 0.0
        for k in range(3): dot_val += U[i,k] * GU_rows[j,k]
        cp = dot_val / (pred_lens_all[i] * pred_lens_all[j] + eps)
        if cp > 1.0: cp = 1.0
        elif cp < -1.0: cp = -1.0
        
        sp = np.sqrt(1.0 - cp**2)
        ql1 = pred_lens_all[i]; ql2 = pred_lens_all[j]
        t2 = ql2 * cp; t3 = ql2 * sp
        
        A11 = ql1 * p11 + t2 * p12; A12 = t2 * p22
        A21 = t3 * p12; A22 = t3 * p22
        
        tr = A11**2 + A12**2 + A21**2 + A22**2
        det = A11*A22 - A12*A21
        sumS = np.sqrt(tr + 2.0 * abs(det))
        
        r_raw = p_norms_sq + ql1**2 + ql2**2 - 2.0 * sumS
        if r_raw < 0.0: r_raw = 0.0
        
        total_residual += (r_raw * (H_sq[i] + H_sq[j]))

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
def objective_wrapper(free, patterns, U, candidates, system, max_planes, tol_len_rel, tol_ang_abs_rad, use_hard_fallback):
    vec6 = free_to_metric_vector(free, system)
    G = vec6_to_G(vec6)

    # PD Check
    try:
        eigs = np.linalg.eigvalsh(G)
        if np.any(eigs <= 1e-12):
            return 1e12 + np.sum(np.abs(eigs[eigs <= 0]))*1e8
    except:
        return 1e12

    # Pass it to the loop
    return fast_objective_loop(patterns, U, candidates, G, max_planes, tol_len_rel, tol_ang_abs_rad, use_hard_fallback)

def local_objective_wrapper(free, patterns, U, candidates, system, assignments):
    vec6 = free_to_metric_vector(free, system)
    G = vec6_to_G(vec6)
    
    try:
        eigs = np.linalg.eigvalsh(G)
        if np.any(eigs <= 1e-12):
            return 1e12 + np.sum(np.abs(eigs[eigs <= 0]))*1e8
    except:
        return 1e12

    return smooth_local_objective(patterns, U, candidates, G, assignments)

# --- CELL REDUCTION FUNCTION ---
def get_lattice_point_group(a, b, c, alpha, beta, gamma, 
                            scan_range=(0.01, 1.0), step=0.05):
    """
    Determines the Point Group and Bravais Lattice of an experimental unit cell.
    Optimized for reliability when atomic positions are unknown.
    """
    if not PYMATGEN_AVAILABLE:
        return {"status": "Failure", "reason": "Pymatgen not installed"}
    
    # Create Dummy Structure
    try:
        lat = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        dummy_struct = Structure(lat, ["H"], [[0, 0, 0]])
    except Exception as e:
        return {"error": str(e)}

    best_sg_number = 0
    best_result = None
    best_tol = 0.0

    # 1. Scan Tolerances to find highest geometric symmetry
    # This ensures we don't miss Centered lattices due to large cell errors
    for tol in np.arange(scan_range[0], scan_range[1], step):
        sga = SpacegroupAnalyzer(dummy_struct, symprec=tol)
        try:
            sg_num = sga.get_space_group_number()
            if sg_num >= best_sg_number:
                best_sg_number = sg_num
                best_tol = tol
                best_result = sga
        except:
            continue

    if not best_result:
        return {"status": "Failure", "reason": "No valid lattice found"}

    # 2. Extract Reliable Data
    dataset = best_result.get_symmetry_dataset()
    conv_struct = best_result.get_conventional_standard_structure()
    
    # Point Group (International/Hermann-Mauguin notation)
    point_group = dataset['pointgroup']
    
    # Bravais Lattice Symbol (e.g., cF, mC, oI)
    crystal_sys = best_result.get_crystal_system()
    lat_type = dataset['international'][0] # P, I, F, C, A, R
    
    # Map system names to Pearson symbol letters
    sys_map = {
        "triclinic": "a", "monoclinic": "m", "orthorhombic": "o", 
        "tetragonal": "t", "trigonal": "h", "hexagonal": "h", "cubic": "c"
    }
    pearson_symbol = f"{sys_map.get(crystal_sys, '?')}{lat_type}"

    return {
        "status": "Success",
        "tolerance_used": f"{best_tol:.2f} Å",
        "crystal_system": crystal_sys.title(),
        "point_group": point_group,
        "bravais_lattice": pearson_symbol, # e.g., mC, cF
        "lattice_centering": f"{lat_type}-Centered",
        
        # Conventional Parameters (The standardized box)
        "std_a": np.round(conv_struct.lattice.a, 4),
        "std_b": np.round(conv_struct.lattice.b, 4),
        "std_c": np.round(conv_struct.lattice.c, 4),
        "std_alpha": np.round(conv_struct.lattice.alpha, 3),
        "std_beta":  np.round(conv_struct.lattice.beta, 3),
        "std_gamma": np.round(conv_struct.lattice.gamma, 3),
    }

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
            
            # --- NEW FEATURE: RANDOM SUBSET SELECTION ---
            total_loaded = len(patterns)
            req_subset = s.get('subset_count', -1)
            
            if req_subset != -1 and req_subset < total_loaded and req_subset > 0:
                self.log(f"Randomly selecting {req_subset} facets out of {total_loaded}...")
                indices = np.random.choice(total_loaded, req_subset, replace=False)
                patterns = patterns[indices]
                self.log(f"Subset selection complete. Working with {len(patterns)} facets.")
            else:
                self.log(f"Using all {total_loaded} facets.")
            
            # 2. Build Candidates
            candidates = self.build_integer_candidates(s['M'], s['centering'])
            U = candidates.astype(float)
            self.log(f"Generated {len(candidates)} candidates.")

            # 3. Bounds
            bounds = self.get_bounds(s['system'], s['ranges'])

            # 4. Arguments
            args = (patterns, U, candidates, s['system'], s['max_planes'], s['tol_len_rel'], s['tol_ang_abs_rad'], USE_HARD_FALLBACK)

            # 5. Global Search
            self.log(f"Starting {s['algorithm']} (global search)...")
            self.log("Compiling JIT function (warmup)...")
            dummy = np.mean(bounds, axis=1)
            objective_wrapper(dummy, *args)
            self.log("Compilation done.")

            if s['algorithm'] == 'Differential Evolution':
                def de_callback(xk, convergence):
                    de_callback.iter += 1
                    val = objective_wrapper(xk, *args)
                    if val < de_callback.best: de_callback.best = val
                    
                    if de_callback.iter % 10 == 0:
                        t_el = time.time() - self.start_time
                        msg = f"[DE gen {de_callback.iter}] time={t_el:.1f}s obj={val:.6g} best={de_callback.best:.6g} conv={convergence:.4g}"
                        self.log(msg)
                
                de_callback.iter = 0
                de_callback.best = np.inf
    
                n_workers = s['workers']
                
                res_global = differential_evolution(
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

            else:
                self.log("Starting NLopt DIRECT (Global Search)...")
                
                # 1. Prepare Bounds for NLopt
                lb = [b[0] for b in bounds]
                ub = [b[1] for b in bounds]
                
                # 2. Build mixed tolerance array based on Crystal System
                xtol_abs_array = []
                sys_type = s['system']
                ang_tol = s['direct_xtol_ang']
                
                # Lengths get 0.0 (disabled absolute), forcing them to use the global relative tolerance.
                # Angles get the specific absolute tolerance.
                if sys_type == 'triclinic': 
                    xtol_abs_array = [0.0]*3 + [ang_tol]*3
                elif sys_type == 'monoclinic': 
                    xtol_abs_array = [0.0]*3 + [ang_tol]
                elif sys_type == 'orthorhombic': 
                    xtol_abs_array = [0.0]*3
                elif sys_type in ['tetragonal', 'hexagonal']: 
                    xtol_abs_array = [0.0]*2
                elif sys_type == 'rhombohedral': 
                    xtol_abs_array = [0.0, ang_tol]
                elif sys_type == 'cubic': 
                    xtol_abs_array = [0.0]

                # 3. Setup NLopt Object
                algo = nlopt.GN_DIRECT_L_RAND if s['direct_loc_bias'] else nlopt.GN_DIRECT
                opt = nlopt.opt(algo, len(bounds))
                opt.set_lower_bounds(lb)
                opt.set_upper_bounds(ub)
                opt.set_maxeval(s['direct_maxfun'])
                
                # Apply the mixed constraints
                opt.set_xtol_rel(s['direct_xtol_len'])  # Global relative fallback
                opt.set_xtol_abs(xtol_abs_array)       # Specific absolute overrides

                # Tracking variables for the logger
                eval_count = [0]
                best_fun = [np.inf]
                best_x = [None]
                
                # 4. Objective Wrapper
                def nlopt_objective(x, grad):
                    val = objective_wrapper(
                        x, patterns, U, candidates, s['system'], 
                        s['max_planes'], s['tol_len_rel'], s['tol_ang_abs_rad'], USE_HARD_FALLBACK
                    )
                    eval_count[0] += 1
                    
                    if val < best_fun[0]:
                        best_fun[0] = val
                        best_x[0] = np.copy(x)
                        
                    if eval_count[0] % 10000 == 0:
                        t_el = time.time() - self.start_time
                        self.log(f"[NLopt eval {eval_count[0]}/{s['direct_maxfun']}] time={t_el:.1f}s | best_obj={best_fun[0]:.6g}")
                        
                    return val

                opt.set_min_objective(nlopt_objective)
                
                # 5. Execute Optimization
                x0 = np.mean(bounds, axis=1) # NLopt requires a starting array to launch
                
                try:
                    res_x = opt.optimize(x0)
                    res_fun = opt.last_optimum_value()
                except nlopt.RoundoffLimited:
                    self.log("NLopt halted early: Reached floating-point precision limits (RoundoffLimited).")
                    res_x = best_x[0]
                    res_fun = best_fun[0]
                except Exception as e:
                    self.log(f"NLopt halted with message: {e}")
                    res_x = best_x[0]
                    res_fun = best_fun[0]
                    
                # Safety fallback in case it exited on evaluation 0
                if res_x is None:
                    res_x = best_x[0]
                    res_fun = best_fun[0]
                
                # Create a dummy result object so the downstream L-BFGS-B logic still works
                class DummyResult: pass
                res_global = DummyResult()
                res_global.x = res_x
                res_global.fun = res_fun

            self.log(f"Global Search finished. time={time.time()-self.start_time:.1f}s best_fun={res_global.fun:.6g}")

            # 7. Outlier Rejection
            current_x = res_global.x
            
            if s['use_outliers']:
                self.log(f"\n--- Outlier Detection ---")
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
                    args = (patterns, U, candidates, s['system'])
            
            # 8. Local Refinement (Index Locking)
            self.log("Starting local refinement (L-BFGS-B)...")
            t0 = time.time()
            
            # Extract the best fixed integer assignments from the global DE result
            vec6_best = free_to_metric_vector(current_x, s['system'])
            G_best = vec6_to_G(vec6_best)
            best_assignments = get_best_assignments(patterns, U, candidates, G_best, s['max_planes'], s['tol_len_rel'], s['tol_ang_abs_rad'], USE_HARD_FALLBACK)
            
            # Pass the fixed assignments into the smooth local objective
            local_args = (patterns, U, candidates, s['system'], best_assignments)
            
            res_local = minimize(local_objective_wrapper, current_x, args=local_args, 
                                 method='L-BFGS-B', bounds=bounds,
                                 options={'maxiter':1000, 'ftol':1e-12})
                                 
            self.log(f"Local refine finished in {time.time()-t0:.1f}s. fun={res_local.fun:.6g} (Was: {res_global.fun:.6g})")

            # 9. Format Results
            final_free = res_local.x
            
            # Reciprocal
            vec6 = free_to_metric_vector(final_free, s['system'])
            G_final = vec6_to_G(vec6)
            astar, bstar, cstar, alstar, bestar, gastar = self.lattice_from_metric(G_final)
            
            # Real
            G_real = np.linalg.inv(G_final)
            a, b, c, al, be, ga = self.lattice_from_metric(G_real)

            # Build strings for logging
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
                                f"a = {a:.3f} Å\n"
                                f"b = {b:.3f} Å\n"
                                f"c = {c:.3f} Å\n"
                                f"alpha = {al:.3f} deg\n"
                                f"beta = {be:.3f} deg\n"
                                f"gamma = {ga:.3f} deg")

            # Result Dict
            results = {
                'params_str': formatted_output,
                'system': s['system'],
                # NEW: Pass Raw parameters for Reduction Step
                'raw_params': (a, b, c, al, be, ga) 
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
        vec6 = free_to_metric_vector(free_params, s['system'])
        G = vec6_to_G(vec6)
        return _fast_score_calc(patterns, U, candidates, G, s['max_planes'], s['tol_len_rel'], s['tol_ang_abs_rad'], USE_HARD_FALLBACK)

@njit(fastmath=True)
def _fast_score_calc(patterns, U, H, G, max_planes, tol_len_rel, tol_ang_abs_rad, use_hard_fallback):
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
        
    sorted_indices = np.argsort(pred_lens)
    sorted_lens = pred_lens[sorted_indices]
    
    scores = np.zeros(len(patterns))
    eps=1e-12
    safe_len = max(2, max_planes)
    idx1 = np.zeros(safe_len, dtype=np.int64)
    idx2 = np.zeros(safe_len, dtype=np.int64)
    
    for p in range(len(patterns)):
        l1, l2, cos_obs = patterns[p]
        theta_obs_rad = np.arccos(cos_obs)
        insert_idx1 = np.searchsorted(sorted_lens, l1)
        count1 = 0
        left, right = insert_idx1 - 1, insert_idx1
        while count1 < max_planes:
            err_left = 1e10; err_right = 1e10
            if left >= 0: err_left = abs(sorted_lens[left] - l1) / (l1 + eps)
            if right < n_cand: err_right = abs(sorted_lens[right] - l1) / (l1 + eps)
            if err_left > tol_len_rel and err_right > tol_len_rel: break
            if err_left <= err_right: idx1[count1] = sorted_indices[left]; left -= 1
            else: idx1[count1] = sorted_indices[right]; right += 1
            count1 += 1

        insert_idx2 = np.searchsorted(sorted_lens, l2)
        count2 = 0
        left, right = insert_idx2 - 1, insert_idx2
        while count2 < max_planes:
            err_left = 1e10; err_right = 1e10
            if left >= 0: err_left = abs(sorted_lens[left] - l2) / (l2 + eps)
            if right < n_cand: err_right = abs(sorted_lens[right] - l2) / (l2 + eps)
            if err_left > tol_len_rel and err_right > tol_len_rel: break
            if err_left <= err_right: idx2[count2] = sorted_indices[left]; left -= 1
            else: idx2[count2] = sorted_indices[right]; right += 1
            count2 += 1

        if count1 == 0 or count2 == 0:
            if use_hard_fallback:
                scores[p] = 1e9; continue
            else:
                if count1 == 0:
                    p1, p2 = insert_idx1 - 1, insert_idx1
                    if p1 < 0: p1, p2 = 0, min(1, n_cand - 1)
                    elif p2 >= n_cand: p2, p1 = n_cand - 1, max(0, n_cand - 2)
                    idx1[0], idx1[1] = sorted_indices[p1], sorted_indices[p2]; count1 = 2
                if count2 == 0:
                    p1, p2 = insert_idx2 - 1, insert_idx2
                    if p1 < 0: p1, p2 = 0, min(1, n_cand - 1)
                    elif p2 >= n_cand: p2, p1 = n_cand - 1, max(0, n_cand - 2)
                    idx2[0], idx2[1] = sorted_indices[p1], sorted_indices[p2]; count2 = 2

        sin_obs = np.sqrt(max(0.0, 1.0-cos_obs**2))
        p_norms = l1**2 + l2**2
        min_r = 1e20
        
        best_invalid_diff = 1e10 
        best_invalid_i = -1
        best_invalid_j = -1
        
        for ii in range(count1):
            i = idx1[ii]
            for jj in range(count2):
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
                
                angle_diff = abs(np.arccos(cp) - theta_obs_rad)
                if angle_diff > tol_ang_abs_rad: 
                    if angle_diff < best_invalid_diff:
                        best_invalid_diff = angle_diff
                        best_invalid_i = i
                        best_invalid_j = j
                    continue
                
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

        if min_r == 1e20 and best_invalid_i != -1:
            i = best_invalid_i
            j = best_invalid_j
            
            dot = 0.0
            for k in range(3): dot += U[i,k] * GU_rows[j,k]
            cp = dot / (pred_lens[i] * pred_lens[j] + eps)
            if cp > 1.0: cp = 1.0
            elif cp < -1.0: cp = -1.0
            
            sp = np.sqrt(1.0 - cp**2)
            ql1, ql2 = pred_lens[i], pred_lens[j]
            t2 = ql2 * cp; t3 = ql2 * sp
            
            A11 = ql1*l1 + t2*l2*cos_obs
            A12 = t2*l2*sin_obs
            A21 = t3*l2*cos_obs
            A22 = t3*l2*sin_obs
            
            tr = A11**2 + A12**2 + A21**2 + A22**2
            det = A11*A22 - A12*A21
            sumS = np.sqrt(tr + 2*abs(det))
            r = p_norms + ql1**2 + ql2**2 - 2*sumS
            if r < 0: r = 0
            min_r = r * (H_sq[i] + H_sq[j])

        if min_r == 1e20:
            min_r = 1e9

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
        
        # State variable for max facets in loaded file
        self.loaded_total_facets = 0 
        self.last_res = None # Stores last optimization result
        
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
        
        # --- NEW FACET SELECTION WIDGETS ---
        self.lbl_facets_total = QLabel("Total Facets in File: N/A")
        
        self.sb_facets_use = QSpinBox()
        self.sb_facets_use.setRange(-1, 9999999) # Large range
        self.sb_facets_use.setValue(-1)
        self.sb_facets_use.setSpecialValueText("All") # -1 displays as "All"
        self.sb_facets_use.setToolTip("Select number of facets to use randomly. -1 uses all available.")
        # Connect to validation function
        self.sb_facets_use.editingFinished.connect(self.validate_facet_count)
        self.sb_facets_use.valueChanged.connect(self.validate_facet_count)

        file_layout.addRow("Input CSV:", h_file)
        file_layout.addRow(self.lbl_facets_total)       # New Row
        file_layout.addRow("Facets to use:", self.sb_facets_use) # New Row
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
        main_algo_layout = QVBoxLayout()
        
        from PySide6.QtWidgets import QGridLayout
        grid = QGridLayout()
        
        # --- Row 0: M & Tolerances ---
        self.sb_M = QSpinBox(); self.sb_M.setValue(6)
        self.sb_tol_rel = QDoubleSpinBox(); self.sb_tol_rel.setSingleStep(0.01); self.sb_tol_rel.setValue(0.15)
        
        grid.addWidget(QLabel("Max hkl (M):"), 0, 0)
        grid.addWidget(self.sb_M, 0, 1)
        grid.addWidget(QLabel("Len Tol (Rel):"), 0, 2)
        grid.addWidget(self.sb_tol_rel, 0, 3)
        
        # --- Row 1: K & Tolerances ---
        self.sb_max_planes = QSpinBox(); self.sb_max_planes.setRange(2, 500); self.sb_max_planes.setValue(MAX_PLANES_DEFAULT)
        self.sb_tol_ang = QDoubleSpinBox(); self.sb_tol_ang.setRange(0.1, 45.0); self.sb_tol_ang.setSingleStep(0.5); self.sb_tol_ang.setValue(5.0)
        
        grid.addWidget(QLabel("Max Planes:"), 1, 0)
        grid.addWidget(self.sb_max_planes, 1, 1)
        grid.addWidget(QLabel("Angle Tol (°):"), 1, 2)
        grid.addWidget(self.sb_tol_ang, 1, 3)
        
        # --- Row 2: Global Search Selection ---
        self.combo_algo = QComboBox()
        self.combo_algo.addItems(['Differential Evolution', 'Direct'])
        self.sb_iter = QSpinBox(); self.sb_iter.setRange(10, 999999); self.sb_iter.setSingleStep(100); self.sb_iter.setValue(2000)
        self.lbl_iter = QLabel("Max Iter:")
        grid.addWidget(QLabel("Global Search:"), 2, 0)
        grid.addWidget(self.combo_algo, 2, 1)
        grid.addWidget(self.lbl_iter, 2, 2)
        grid.addWidget(self.sb_iter, 2, 3)
        
        # --- Row 3 & 4: DE Specific Widgets ---
        self.lbl_pop = QLabel("Pop Size:")
        self.sb_pop = QSpinBox(); self.sb_pop.setRange(10, 500); self.sb_pop.setValue(100)
        self.lbl_strat = QLabel("Strategy:")
        self.combo_strat = QComboBox(); self.combo_strat.addItems(['best1bin', 'rand1bin'])
        self.lbl_workers = QLabel("Processors:")
        self.sb_workers = QSpinBox(); self.sb_workers.setRange(-1, 64); self.sb_workers.setValue(-1)
        
        grid.addWidget(self.lbl_pop, 3, 0)
        grid.addWidget(self.sb_pop, 3, 1)
        grid.addWidget(self.lbl_strat, 3, 2)
        grid.addWidget(self.combo_strat, 3, 3)
        grid.addWidget(self.lbl_workers, 4, 0)
        grid.addWidget(self.sb_workers, 4, 1)
        
        # --- Row 5 & 6: DIRECT Specific Widgets ---
        self.lbl_maxfun = QLabel("Max Evals:")
        self.sb_maxfun = QSpinBox(); self.sb_maxfun.setRange(1000, 10000000); self.sb_maxfun.setSingleStep(100000); self.sb_maxfun.setValue(1000000)
        
        self.lbl_xtol_len = QLabel("Len Prec (Rel):")
        self.sb_xtol_len = QDoubleSpinBox(); self.sb_xtol_len.setRange(1e-8, 0.5); self.sb_xtol_len.setDecimals(5); self.sb_xtol_len.setSingleStep(0.001); self.sb_xtol_len.setValue(DIRECT_XTOL_REL_LEN_DEFAULT)
        self.sb_xtol_len.setToolTip("Relative tolerance for reciprocal lengths (e.g., 0.0001 = 0.01% precision)")

        self.lbl_xtol_ang = QLabel("Ang Prec (Abs°):")
        self.sb_xtol_ang = QDoubleSpinBox(); self.sb_xtol_ang.setRange(1e-5, 10.0); self.sb_xtol_ang.setDecimals(4); self.sb_xtol_ang.setSingleStep(0.05); self.sb_xtol_ang.setValue(DIRECT_XTOL_ABS_ANG_DEFAULT)
        self.sb_xtol_ang.setToolTip("Absolute tolerance for angles in degrees")

        self.chk_local_bias = QCheckBox("Locally Biased (RAND)")
        self.chk_local_bias.setChecked(True)
        self.chk_local_bias.setToolTip("Uses NLOPT_GN_DIRECT_L_RAND. Uncheck for standard NLOPT_GN_DIRECT.")
        
        grid.addWidget(self.lbl_maxfun, 5, 0)
        grid.addWidget(self.sb_maxfun, 5, 1)
        grid.addWidget(self.lbl_xtol_len, 5, 2)
        grid.addWidget(self.sb_xtol_len, 5, 3)
        
        grid.addWidget(self.lbl_xtol_ang, 6, 0)
        grid.addWidget(self.sb_xtol_ang, 6, 1)
        grid.addWidget(self.chk_local_bias, 6, 2, 1, 2)
        
        main_algo_layout.addLayout(grid)
        
        # --- Bottom Section: Outlier Rejection ---
        outlier_layout = QHBoxLayout()
        self.chk_outlier = QCheckBox("Outlier Rejection")
        self.sb_keep = QSpinBox(); self.sb_keep.setRange(10, 100); self.sb_keep.setValue(80)
        
        outlier_layout.addWidget(self.chk_outlier)
        outlier_layout.addWidget(self.sb_keep)
        outlier_layout.addStretch() # Pushes the widgets to the left
        
        main_algo_layout.addLayout(outlier_layout)
        algo_group.setLayout(main_algo_layout)
        left_layout.addWidget(algo_group)
        
        # Hook up the dynamic visibility toggle
        self.combo_algo.currentTextChanged.connect(self.update_algo_inputs)
        
        # Call it once manually to set the initial GUI state
        self.update_algo_inputs(self.combo_algo.currentText())
        
        # Run Button
        self.run_btn = QPushButton("RUN OPTIMIZATION")
        self.run_btn.setFixedHeight(45)
        self.run_btn.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #2b5b84; color: white;")
        self.run_btn.clicked.connect(self.start_optimization)
        left_layout.addWidget(self.run_btn)
        
        # --- NEW: CELL REDUCTION BUTTON ---
        self.reduce_btn = QPushButton("FIND CONVENTIONAL CELL")
        self.reduce_btn.setFixedHeight(40)
        self.reduce_btn.setStyleSheet("font-weight: bold; font-size: 12px; background-color: #555555; color: white;")
        self.reduce_btn.setEnabled(False) # Disabled until opt is done
        self.reduce_btn.clicked.connect(self.run_cell_reduction)
        left_layout.addWidget(self.reduce_btn)
        
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

    def update_algo_inputs(self, algo_name):
        is_de = (algo_name == 'Differential Evolution')
        
        # 1. Toggle DE Widgets (Show if DE is selected)
        for w in [self.lbl_iter, self.sb_iter, self.lbl_pop, self.sb_pop, 
                  self.lbl_strat, self.combo_strat, self.lbl_workers, self.sb_workers]:
            w.setVisible(is_de)
            
        # 2. Toggle DIRECT Widgets (Show if DIRECT is selected)
        for w in [self.lbl_maxfun, self.sb_maxfun, self.lbl_xtol_len, self.sb_xtol_len, 
                  self.lbl_xtol_ang, self.sb_xtol_ang, self.chk_local_bias]:
            w.setVisible(not is_de)


    def browse_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if f: 
            self.path_edit.setText(f)
            # --- NEW: Count lines immediately to update GUI ---
            try:
                count = 0
                with open(f, 'r', encoding='utf-8') as csvf:
                    reader = csv.reader(csvf)
                    for row in reader:
                        if len(row) >= 3:
                            try:
                                # Ensure it's valid data
                                float(row[-3]); float(row[-2]); float(row[-1])
                                count += 1
                            except: pass
                
                self.loaded_total_facets = count
                self.lbl_facets_total.setText(f"Total Facets in File: {count}")
                self.sb_facets_use.setValue(-1) # Default to All when loading new file
            except Exception as e:
                self.lbl_facets_total.setText("Total Facets in File: Error")
                self.loaded_total_facets = 0

    def validate_facet_count(self):
        """
        Ensures the user input for facet count does not exceed total loaded facets.
        """
        val = self.sb_facets_use.value()
        # If val is -1, it means "All", which is valid.
        if val == -1:
            return
            
        if self.loaded_total_facets > 0:
            if val > self.loaded_total_facets:
                self.sb_facets_use.setValue(self.loaded_total_facets)
        elif self.loaded_total_facets == 0 and val > 0:
             # If no file loaded yet, maybe just let it be or reset?
             # But requirement says "When user loads file... if user inputs bigger..."
             # So we enforce mostly when file is known.
             pass

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
        self.reduce_btn.setEnabled(False) # Disable reduction during run
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
            'max_planes': self.sb_max_planes.value(),
            'tol_len_rel': self.sb_tol_rel.value(),
            'tol_ang_abs_rad': np.radians(self.sb_tol_ang.value()),
            'de_popsize': self.sb_pop.value(),
            'de_maxiter': self.sb_iter.value(),
            'de_strategy': self.combo_strat.currentText(),
            'de_tol': 1e-5,
            'use_outliers': self.chk_outlier.isChecked(),
            'keep_pct': self.sb_keep.value(),
            'workers': self.sb_workers.value(),

            'subset_count': self.sb_facets_use.value(),
            'algorithm': self.combo_algo.currentText(), 

            'direct_maxfun': self.sb_maxfun.value(),
            'direct_xtol_len': self.sb_xtol_len.value(),
            'direct_xtol_ang': self.sb_xtol_ang.value(),
            'direct_loc_bias': self.chk_local_bias.isChecked()
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
        self.last_res = res # Store for reduction
        self.reduce_btn.setEnabled(True) # Enable reduction button
        
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

    def run_cell_reduction(self):
        if not self.last_res or 'raw_params' not in self.last_res:
            self.text_log.append("\nError: No valid cell parameters available. Run optimization first.")
            return

        if not PYMATGEN_AVAILABLE:
             self.text_log.append("\nError: Pymatgen is not installed. Cannot reduce cell.")
             return

        a, b, c, al, be, ga = self.last_res['raw_params']
        self.text_log.append("\n--- Running Cell Reduction & Symmetry Analysis ---")
        
        # Run reduction
        result = get_lattice_point_group(a, b, c, al, be, ga)
        
        if result.get("status") == "Failure":
            self.text_log.append(f"Reduction Failed: {result.get('reason', 'Unknown')}")
            if "error" in result:
                 self.text_log.append(f"Details: {result['error']}")
            return

        # Format Output
        out_str = (f"\n\n========================================\n"
                   f"   CONVENTIONAL CELL & SYMMETRY REPORT  \n"
                   f"========================================\n"
                   f"Crystal System   : {result['crystal_system']}\n"
                   f"Point Group      : {result['point_group']}\n"
                   f"Bravais Lattice  : {result['bravais_lattice']}\n"
                   f"Lattice Centering: {result['lattice_centering']}\n"
                   f"Tolerance Used   : {result['tolerance_used']}\n\n"
                   f"Conventional Parameters:\n"
                   f"a     = {result['std_a']:.4f} Å\n"
                   f"b     = {result['std_b']:.4f} Å\n"
                   f"c     = {result['std_c']:.4f} Å\n"
                   f"alpha = {result['std_alpha']:.3f}°\n"
                   f"beta  = {result['std_beta']:.3f}°\n"
                   f"gamma = {result['std_gamma']:.3f}°\n")
        
        self.text_log.append(out_str)
        
        # Append to file
        try:
            base_dir = os.path.dirname(self.path_edit.text())
            cell_path = os.path.join(base_dir, self.cell_name.text())
            with open(cell_path, "a", encoding='utf-8') as f:
                f.write(out_str)
            self.text_log.append(f"Appended results to {cell_path}")
            
            # Also append to log file on disk
            log_path = os.path.join(base_dir, self.log_name.text())
            with open(log_path, "a", encoding='utf-8') as f:
                 f.write(out_str)
            
        except Exception as e:
            self.text_log.append(f"Error appending to files: {e}")


if __name__ == "__main__":
    multiprocessing.freeze_support() # Essential for Windows executable/multiprocessing
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = CrystalApp()
    window.show()
    sys.exit(app.exec())
    