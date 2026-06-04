#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def select_sources_spatially_debug(catalog, max_sources, image_shape, selection_mode="uniform", random_seed=42):
    """Debug version of source selection with visualization."""
    
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Get valid sources with positions
    valid_sources = []
    positions = []
    for row in catalog:
        if "XWIN_IMAGE" in row.colnames and "YWIN_IMAGE" in row.colnames:
            x, y = row["XWIN_IMAGE"], row["YWIN_IMAGE"]
            if np.isfinite(x) and np.isfinite(y):
                valid_sources.append(row)
                positions.append((x, y))
    
    print(f"Total valid sources: {len(valid_sources)}")
    print(f"Image shape: {image_shape}")
    print(f"Max sources requested: {max_sources}")
    
    if len(valid_sources) <= max_sources:
        return valid_sources, positions
    
    if selection_mode == "uniform":
        # Uniform spatial grid sampling
        img_h, img_w = image_shape
        n_grid = int(np.ceil(np.sqrt(max_sources * 2)))  # Grid size for uniform sampling
        
        print(f"Grid size: {n_grid}x{n_grid}")
        
        selected_sources = []
        selected_positions = []
        grid_cells = {}  # Dictionary to store sources per grid cell
        
        # Assign sources to grid cells
        for i, source in enumerate(valid_sources):
            x, y = source["XWIN_IMAGE"], source["YWIN_IMAGE"]
            
            # Debug: print coordinate ranges
            if i < 5:
                print(f"Source {i}: x={x:.1f}, y={y:.1f}")
            
            grid_x = min(int(x / img_w * n_grid), n_grid - 1)
            grid_y = min(int(y / img_h * n_grid), n_grid - 1)
            cell_key = (grid_x, grid_y)
            
            if cell_key not in grid_cells:
                grid_cells[cell_key] = []
            grid_cells[cell_key].append((source, positions[i]))
        
        print(f"Number of populated grid cells: {len(grid_cells)}")
        print(f"Grid cells with sources: {sorted(grid_cells.keys())}")
        
        # Select sources from each grid cell
        sources_per_cell = max(1, max_sources // len(grid_cells))
        remaining_slots = max_sources - (sources_per_cell * len(grid_cells))
        
        print(f"Sources per cell: {sources_per_cell}, remaining slots: {remaining_slots}")
        
        for cell_key in sorted(grid_cells.keys()):
            cell_sources = grid_cells[cell_key]
            # Randomly select sources from this cell
            n_select = min(sources_per_cell + (1 if remaining_slots > 0 else 0), len(cell_sources))
            if remaining_slots > 0:
                remaining_slots -= 1
            
            if len(cell_sources) > 0:
                np.random.shuffle(cell_sources)
                for source, pos in cell_sources[:n_select]:
                    selected_sources.append(source)
                    selected_positions.append(pos)
        
        # If we still need more sources, add randomly from remaining
        if len(selected_sources) < max_sources:
            remaining_needed = max_sources - len(selected_sources)
            used_indices = set()
            # Find indices of already selected sources
            for selected_source in selected_sources:
                for i, source in enumerate(valid_sources):
                    if source == selected_source:
                        used_indices.add(i)
                        break
            
            available_indices = [i for i in range(len(valid_sources)) if i not in used_indices]
            if available_indices:
                np.random.shuffle(available_indices)
                for i in available_indices[:remaining_needed]:
                    selected_sources.append(valid_sources[i])
                    selected_positions.append(positions[i])
        
        print(f"Finally selected {len(selected_sources)} sources")
        return selected_sources[:max_sources], selected_positions[:max_sources]
    
    elif selection_mode == "random":
        # Random selection across entire image
        indices = np.random.choice(len(valid_sources), max_sources, replace=False)
        selected_sources = [valid_sources[i] for i in indices]
        selected_positions = [positions[i] for i in indices]
        return selected_sources, selected_positions
    
    elif selection_mode == "first":
        # Original behavior: take first N sources
        return valid_sources[:max_sources], positions[:max_sources]

def visualize_source_selection(all_positions, selected_positions, image_shape, selection_mode):
    """Visualize source selection for debugging."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    img_h, img_w = image_shape
    
    # Plot all sources
    if all_positions:
        all_x, all_y = zip(*all_positions)
        ax1.scatter(all_x, all_y, c='blue', alpha=0.3, s=10, label='All sources')
        ax1.set_xlim(0, img_w)
        ax1.set_ylim(0, img_h)
        ax1.set_title(f'All Sources ({len(all_positions)})')
        ax1.set_xlabel('X [pixels]')
        ax1.set_ylabel('Y [pixels]')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot selected sources
    if selected_positions:
        sel_x, sel_y = zip(*selected_positions)
        ax2.scatter(sel_x, sel_y, c='red', alpha=0.8, s=20, label=f'Selected ({len(selected_positions)})')
        ax2.set_xlim(0, img_w)
        ax2.set_ylim(0, img_h)
        ax2.set_title(f'Selected Sources - {selection_mode.title()} Mode')
        ax2.set_xlabel('X [pixels]')
        ax2.set_ylabel('Y [pixels]')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add grid for uniform mode
        if selection_mode == "uniform":
            n_grid = int(np.ceil(np.sqrt(len(selected_positions) * 2)))
            for i in range(n_grid + 1):
                x_line = i * img_w / n_grid
                y_line = i * img_h / n_grid
                ax2.axvline(x_line, color='gray', alpha=0.2, linestyle='--')
                ax2.axhline(y_line, color='gray', alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'debug_source_selection_{selection_mode}.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved debug plot: debug_source_selection_{selection_mode}.pdf")

# Test with simulated data
if __name__ == "__main__":
    # Create a mock catalog with sources clustered in different regions
    from astropy.table import Table
    
    # Simulate sources with various distributions
    x_positions = []
    y_positions = []
    
    # Add sources in different regions
    np.random.seed(42)
    
    # Cluster 1: lower-left (dense)
    for i in range(200):
        x_positions.append(np.random.normal(200, 50))
        y_positions.append(np.random.normal(200, 50))
    
    # Cluster 2: upper-right (sparse)
    for i in range(50):
        x_positions.append(np.random.normal(800, 100))
        y_positions.append(np.random.normal(800, 100))
    
    # Cluster 3: scattered across middle
    for i in range(100):
        x_positions.append(np.random.uniform(300, 700))
        y_positions.append(np.random.uniform(300, 700))
    
    # Filter to image bounds and create catalog
    valid_x = []
    valid_y = []
    for x, y in zip(x_positions, y_positions):
        if 0 <= x < 1000 and 0 <= y < 1000:
            valid_x.append(x)
            valid_y.append(y)
    
    catalog = Table()
    catalog["XWIN_IMAGE"] = valid_x
    catalog["YWIN_IMAGE"] = valid_y
    
    print(f"Simulated catalog with {len(catalog)} sources")
    
    # Test different selection modes
    image_shape = (1000, 1000)
    max_sources = 100
    
    for mode in ["first", "random", "uniform"]:
        print(f"\n=== Testing {mode.upper()} mode ===")
        selected, positions = select_sources_spatially_debug(
            catalog, max_sources, image_shape, mode, random_seed=42
        )
        
        # Get all positions for visualization
        all_positions = [(x, y) for x, y in zip(catalog["XWIN_IMAGE"], catalog["YWIN_IMAGE"])]
        
        visualize_source_selection(all_positions, positions, image_shape, mode)
