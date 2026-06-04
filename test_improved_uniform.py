#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

def select_sources_improved_uniform(catalog, max_sources, image_shape, random_seed=42):
    """Improved uniform selection that ensures coverage across entire image."""
    
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
    
    # True uniform spatial grid sampling - ensure coverage across entire image
    img_h, img_w = image_shape
    
    # Create a grid that covers the entire image
    n_grid = int(np.ceil(np.sqrt(max_sources)))
    grid_spacing_x = img_w / n_grid
    grid_spacing_y = img_h / n_grid
    
    print(f"Grid size: {n_grid}x{n_grid}")
    print(f"Grid spacing: {grid_spacing_x:.1f} x {grid_spacing_y:.1f} pixels")
    
    selected_sources = []
    selected_positions = []
    grid_cells = {}  # Dictionary to store sources per grid cell
    
    # Assign sources to grid cells
    for i, source in enumerate(valid_sources):
        x, y = source["XWIN_IMAGE"], source["YWIN_IMAGE"]
        
        # Debug: print coordinate ranges
        if i < 5:
            print(f"Source {i}: x={x:.1f}, y={y:.1f}")
        
        grid_x = min(int(x / grid_spacing_x), n_grid - 1)
        grid_y = min(int(y / grid_spacing_y), n_grid - 1)
        cell_key = (grid_x, grid_y)
        
        if cell_key not in grid_cells:
            grid_cells[cell_key] = []
        grid_cells[cell_key].append((source, positions[i]))
    
    print(f"Number of populated grid cells: {len(grid_cells)}")
    print(f"Grid cells with sources: {sorted(grid_cells.keys())}")
    
    # Create all possible grid cells (even empty ones)
    all_grid_cells = [(gx, gy) for gx in range(n_grid) for gy in range(n_grid)]
    
    # First, try to select one source from each grid cell that has sources
    selected_count = 0
    for cell_key in all_grid_cells:
        if selected_count >= max_sources:
            break
        
        if cell_key in grid_cells and len(grid_cells[cell_key]) > 0:
            # Randomly select one source from this cell
            np.random.shuffle(grid_cells[cell_key])
            source, pos = grid_cells[cell_key][0]
            selected_sources.append(source)
            selected_positions.append(pos)
            selected_count += 1
    
    print(f"Selected {selected_count} sources from different grid cells")
    
    # If we still need more sources, fill remaining slots from populated cells
    if selected_count < max_sources:
        remaining_needed = max_sources - selected_count
        
        # Create a list of all remaining sources
        remaining_sources = []
        for cell_key in all_grid_cells:
            if cell_key in grid_cells:
                cell_sources = grid_cells[cell_key]
                if len(cell_sources) > 1:  # Skip the one we already took
                    remaining_sources.extend(cell_sources[1:])
        
        # Randomly select from remaining sources
        if remaining_sources:
            np.random.shuffle(remaining_sources)
            additional_needed = min(remaining_needed, len(remaining_sources))
            for source, pos in remaining_sources[:additional_needed]:
                selected_sources.append(source)
                selected_positions.append(pos)
                selected_count += 1
    
    print(f"After filling: {selected_count} sources selected")
    
    # If still not enough (sparse catalog), add from any available sources
    if selected_count < max_sources:
        remaining_needed = max_sources - selected_count
        used_sources = set(selected_sources)
        available_sources = [(s, p) for s, p in zip(valid_sources, positions) if s not in used_sources]
        
        if available_sources:
            np.random.shuffle(available_sources)
            additional_needed = min(remaining_needed, len(available_sources))
            for source, pos in available_sources[:additional_needed]:
                selected_sources.append(source)
                selected_positions.append(pos)
                selected_count += 1
    
    print(f"Finally selected {selected_count} sources")
    return selected_sources[:max_sources], selected_positions[:max_sources]

def visualize_selection(all_positions, selected_positions, image_shape, mode):
    """Visualize source selection."""
    
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
        ax2.set_title(f'Selected Sources - {mode.title()} Mode')
        ax2.set_xlabel('X [pixels]')
        ax2.set_ylabel('Y [pixels]')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add grid for uniform mode
        if mode == "uniform":
            n_grid = int(np.ceil(np.sqrt(len(selected_positions))))
            for i in range(n_grid + 1):
                x_line = i * img_w / n_grid
                y_line = i * img_h / n_grid
                ax2.axvline(x_line, color='gray', alpha=0.2, linestyle='--')
                ax2.axhline(y_line, color='gray', alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'test_improved_uniform_{mode}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: test_improved_uniform_{mode}.png")

# Test with simulated one-sided data
if __name__ == "__main__":
    # Create a catalog that's heavily one-sided (like the real data might be)
    catalog = Table()
    x_positions = []
    y_positions = []
    
    np.random.seed(42)
    
    # Most sources in lower-left quadrant (simulating one-sided real data)
    for i in range(300):
        x_positions.append(np.random.normal(200, 80))  # Centered at x=200
        y_positions.append(np.random.normal(200, 80))  # Centered at y=200
    
    # Very few sources in upper-right quadrant
    for i in range(20):
        x_positions.append(np.random.normal(800, 100))
        y_positions.append(np.random.normal(800, 100))
    
    # Filter to image bounds
    valid_x = []
    valid_y = []
    for x, y in zip(x_positions, y_positions):
        if 0 <= x < 1000 and 0 <= y < 1000:
            valid_x.append(x)
            valid_y.append(y)
    
    catalog["XWIN_IMAGE"] = valid_x
    catalog["YWIN_IMAGE"] = valid_y
    
    print(f"Created one-sided catalog with {len(catalog)} sources")
    
    # Test the improved uniform selection
    image_shape = (1000, 1000)
    max_sources = 100
    
    selected, positions = select_sources_improved_uniform(
        catalog, max_sources, image_shape, random_seed=42
    )
    
    # Get all positions for visualization
    all_positions = [(x, y) for x, y in zip(catalog["XWIN_IMAGE"], catalog["YWIN_IMAGE"])]
    
    visualize_selection(all_positions, positions, image_shape, "uniform")
