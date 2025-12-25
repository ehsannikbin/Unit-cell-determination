# Unit-cell-determination
This toolkit determines unit cell parameters from randomly oriented electron diffraction patterns.

Workflow:

1- Facet Selection (facet_selector.py):
    Loads raw .h5 diffraction patterns form a file list.
    User interactively selects valid facets.
    Output: A .csv file containing reciprocal vectors ($s_1, s_2$) and angles.
    
2- Unit Cell Determination (unit_cell_finder.py):
    Input: The .csv file from step 1.
    Algorithms calculate the best-fit unit cell.
