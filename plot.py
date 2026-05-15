#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting utilities for subtraction checks, source diagnostics, and light curves.
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import plotting utilities with fallback
try:
    from plotting_utils import get_divergent_color, get_marker_size
except ImportError:
    logger.warning("plotting_utils module not found, some plotting features may be limited")
    get_divergent_color = None
    get_marker_size = None

try:
    from lightcurve import (
        _normalize_photometry_columns,
        BAND_COLORS,
        canonical_band_label_map_from_filter_series,
        canonical_bands_from_filter_series,
        filter_value_matches_band,
        photometry_filter_series,
    )
except ImportError:
    _normalize_photometry_columns = None
    BAND_COLORS = None
    canonical_band_label_map_from_filter_series = None
    canonical_bands_from_filter_series = None
    filter_value_matches_band = None
    photometry_filter_series = None


class Plot:

    # =============================================================================
    #
    # =============================================================================
    def __init__(self, input_yaml):
        """
        Initialize the Plot class with input YAML configuration.

        Parameters:
        input_yaml (dict): Configuration dictionary with file paths and other settings.
        """
        self.input_yaml = input_yaml

    # =============================================================================
    #
    # =============================================================================

    def subtraction_check(
        self,
        image,
        ref,
        diff,
        inset_size=21,
        expected_location=[],
        fitted_location=[],
        mask=None,
        aperture_size=7,
        aligned_sources=None,
        matching_sources=None,
        masked_sources=None,
        weight_map_sci=None,
        weight_map_ref=None,
        wcs_sci=None,
        wcs_ref=None,
        target_ra=None,
        target_dec=None,
        masked_source_centers=None,
    ):
        """
        Perform and visualize a subtraction check of astronomical images using zscale with percentile cleaning.

        Parameters:
        image (ndarray): Main image data.
        ref (ndarray): Reference image data.
        diff (ndarray): Difference image data.
        expected_location (list): Location [x, y] for the inset (default is empty list).
        inset_size (int): Size of the inset around the marked location (default 21).
        mask (ndarray or None): Optional mask to overlay on the images.
        aligned_sources (list): List of aligned sources.
        matching_sources (DataFrame): DataFrame of consistent sources to mark with crosses.
        weight_map_sci (ndarray or None): Science image weight map for display.
        weight_map_ref (ndarray or None): Reference image weight map for display.
        wcs_sci (astropy.wcs.WCS or None): WCS object for science image.
        wcs_ref (astropy.wcs.WCS or None): WCS object for reference image.
        target_ra (float or None): Target RA in degrees for marking on weight maps.
        target_dec (float or None): Target Dec in degrees for marking on weight maps.
        masked_source_centers (list or None): List of (x, y) tuples for masked point source centers to mark with red 'x'.
        """
        import matplotlib.pyplot as plt
        from functions import set_size
        from matplotlib.gridspec import GridSpec
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        from matplotlib.patches import ConnectionPatch, Rectangle
        from matplotlib import colors
        import numpy as np
        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines
        import matplotlib.patches as patches
        from astropy.visualization import ZScaleInterval

        try:
            # Setup
            dir_path = os.path.dirname(os.path.realpath(__file__))
            plt.style.use(os.path.join(dir_path, "autophot.mplstyle"))

            base = os.path.splitext(os.path.basename(self.input_yaml["fpath"]))[0]
            write_dir = os.path.dirname(self.input_yaml["fpath"])
            save_path = os.path.join(
                write_dir, f"Subtraction_Check_{base}.png"
            )

            # Create zscale interval object
            zscale = ZScaleInterval()

            images = {"Image": image, "Reference": ref, "Difference": diff}
            
            # Store dimensions for each image
            image_dims = {}
            
            # Debug: Log image shapes and data quality
            for key, img_data in images.items():
                finite_frac = np.sum(np.isfinite(img_data)) / img_data.size * 100
                img_h, img_w = img_data.shape
                image_dims[key] = (img_w, img_h)  # (width, height)
                logger.info(
                    f"subtraction_check: {key} shape={img_data.shape}, "
                    f"finite={finite_frac:.1f}%, "
                    f"range=[{np.nanmin(img_data):.2e}, {np.nanmax(img_data):.2e}]"
                )

            # Compute vmin, vmax per image using zscale with percentile cleaning
            vmins = {}
            vmaxs = {}
            for key, img_data in images.items():
                # First apply percentile cleaning to remove extreme values
                valid_data = img_data[np.isfinite(img_data)]
                if len(valid_data) == 0:
                    logger.warning(f"subtraction_check: {key} has no valid data, using fallback vmin/vmax")
                    vmins[key] = np.nanmin(img_data)
                    vmaxs[key] = np.nanmax(img_data)
                    continue
                try:
                    lower, upper = np.percentile(valid_data, [0.5, 99.5])
                    cleaned_data = np.clip(img_data, lower, upper)
                    # Then apply zscale to get optimal display range
                    vmin, vmax = zscale.get_limits(cleaned_data)
                    vmins[key] = vmin
                    vmaxs[key] = vmax
                except Exception as e:
                    logger.warning(f"subtraction_check: zscale failed for {key}, using min/max: {e}")
                    vmins[key] = np.nanmin(img_data)
                    vmaxs[key] = np.nanmax(img_data)

            # Set up figure with 1x3 layout for three images (Image, Reference, Difference)
            fig = plt.figure(figsize=(18, 5), constrained_layout=False)
            gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1], wspace=0.3)
            axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
            plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.1)

            img_height, img_width = image.shape
            margin = 0.05
            inset_axes_list = []

            def get_inset_side(inset_anchor, size, axis_len):
                if inset_anchor > axis_len * (1 - margin):
                    return "low", inset_anchor - size, 0.0
                elif inset_anchor < axis_len * margin:
                    return "high", inset_anchor + size, 1.0
                else:
                    side = "high" if inset_anchor < axis_len / 2 else "low"
                    main = (
                        inset_anchor + size if side == "high" else inset_anchor - size
                    )
                    inset = 1.0 if side == "high" else 0.0
                    return side, main, inset

            # Plot images with zscale scaling (Image, Reference, Difference)
            image_titles = ["Image", "Reference", "Difference"]
            # Use first image's shape for all panels to ensure identical display
            if "Image" not in images:
                logger.error("subtraction_check: 'Image' key not found in images dictionary")
                return 0
            ref_width = images["Image"].shape[1]
            ref_height = images["Image"].shape[0]
            # Log image shapes for debugging
            for title in image_titles:
                logger.info(f"subtraction_check: {title} shape={images[title].shape}")
            for i, (ax, title) in enumerate(zip(axes, image_titles)):
                img_data = images[title]
                cmap = plt.get_cmap("viridis").copy()
                cmap.set_bad(color="white")
                ax.imshow(
                    img_data,
                    origin="lower",
                    aspect="equal",
                    cmap=cmap,
                    vmin=vmins[title],
                    vmax=vmaxs[title],
                )
                # Set identical axis limits for all panels based on first image
                ax.set_xlim(0, ref_width)
                ax.set_ylim(0, ref_height)
                # Panel titles for readability; no legends are drawn in these axes.
                ax.set_title(title, fontsize=10, pad=5)
                ax.set_xlabel("X [Pixel]", fontsize=9)
                ax.set_ylabel("Y [Pixel]", fontsize=9)

            square_size = int(inset_size * 2)  # Width/height of the square in pixels
            if matching_sources is not None and len(matching_sources) > 0:
                x_col = y_col = None
                for xc, yc in (
                    ("x_pix", "y_pix"),
                    ("X_IMAGE_REF_SCI_MEAN", "Y_IMAGE_REF_SCI_MEAN"),
                    ("X_IMAGE_SCI", "Y_IMAGE_SCI"),
                    ("x_center", "y_center"),
                ):
                    if xc in matching_sources.columns and yc in matching_sources.columns:
                        x_col, y_col = xc, yc
                        break
                if x_col is None or y_col is None:
                    logger.warning(
                        "subtraction_check: matching_sources missing expected x/y columns; available=%s",
                        list(matching_sources.columns),
                    )
                else:
                    half_size = square_size / 2
                    valid_markers_total = 0
                    skipped_markers_total = 0
                    
                    # Debug: Log coordinate and image info
                    x_vals = matching_sources[x_col].values
                    y_vals = matching_sources[y_col].values
                    logger.info(
                        f"subtraction_check: Using columns ({x_col}, {y_col}) for {len(matching_sources)} sources"
                    )
                    logger.info(
                        f"subtraction_check: Coordinate ranges: "
                        f"X=[{np.nanmin(x_vals):.1f}, {np.nanmax(x_vals):.1f}], "
                        f"Y=[{np.nanmin(y_vals):.1f}, {np.nanmax(y_vals):.1f}]"
                    )
                    
                    # Map axes to panel names for correct dimension lookup
                    panel_names = ["Image", "Reference", "Difference"]
                    
                    for panel_idx, ax in enumerate(axes):  # Both panels
                        panel_name = panel_names[panel_idx]
                        panel_width, panel_height = image_dims[panel_name]
                        valid_markers = 0
                        skipped_markers = 0
                        non_finite = 0
                        
                        for idx, (x_pix, y_pix) in enumerate(zip(
                            matching_sources[x_col], matching_sources[y_col]
                        )):
                            # Skip invalid coordinates
                            if not (np.isfinite(x_pix) and np.isfinite(y_pix)):
                                non_finite += 1
                                skipped_markers += 1
                                continue

                            # Coordinates are already 0-based (converted on ingestion).
                            x_plot = float(x_pix)
                            y_plot = float(y_pix)

                            # Plot all sources regardless of bounds to show full matching
                            rect = patches.Rectangle(
                                (x_plot - half_size, y_plot - half_size),
                                square_size,
                                square_size,
                                linewidth=0.5,
                                edgecolor="b",
                                facecolor="none",
                                alpha=0.5,
                            )
                            ax.add_patch(rect)
                            valid_markers += 1
                        
                        valid_markers_total += valid_markers
                        skipped_markers_total += skipped_markers
                        
                        if skipped_markers > 0:
                            logger.info(
                                f"subtraction_check: {panel_name} panel - plotted {valid_markers} markers, "
                                f"skipped {skipped_markers} ({non_finite} non-finite). "
                                f"Panel dims: {panel_width}x{panel_height}"
                            )
                    
                    if skipped_markers_total > 0:
                        logger.info(
                            f"subtraction_check: Total - plotted {valid_markers_total} markers, "
                            f"skipped {skipped_markers_total} across all panels"
                        )

            # Plot variable sources as red "x" and annotate with otype
            if masked_sources is not None and len(masked_sources) > 0:
                cross_len = square_size / 4  # Length of each arm of the cross
                skipped_masked = 0
                # Check if required columns exist
                required_cols = ["x_pix", "y_pix", "OTYPE_opt", "MAIN_ID"]
                missing_cols = [col for col in required_cols if col not in masked_sources.columns]
                if missing_cols:
                    logger.warning(f"subtraction_check: masked_sources missing columns {missing_cols}")
                else:
                    for x, y, otype, name in zip(
                        masked_sources["x_pix"],
                        masked_sources["y_pix"],
                        masked_sources["OTYPE_opt"],
                        masked_sources["MAIN_ID"],
                    ):
                        # Skip invalid coordinates
                        if not (np.isfinite(x) and np.isfinite(y)):
                            skipped_masked += 1
                            continue

                        # Convert to 0-indexed if needed
                        x_plot = float(x) - 1 if x > 0 else float(x)
                        y_plot = float(y) - 1 if y > 0 else float(y)

                        # Check if coordinates are within image bounds
                        if not (0 <= x_plot < img_width and 0 <= y_plot < img_height):
                            skipped_masked += 1
                            continue

                        x = x_plot
                        y = y_plot

                        if "SN*" in otype:
                            otype = name
                            circle = mpatches.Circle(
                                (x, y),
                                cross_len * 2,
                                edgecolor="#FF0000",
                                facecolor="none",
                                zorder=4,
                                lw=0.5,
                            )
                            if len(axes) > 0:
                                axes[0].add_patch(circle)
                        # else:
                        #     axes[0].plot(
                        #         [x - cross_len, x + cross_len],
                        #         [y - cross_len, y + cross_len],
                        #         color="#FF0000",
                        #         lw=0.5,
                        #         zorder=2,
                        #     )
                        #     axes[0].plot(
                        #         [x - cross_len, x + cross_len],
                        #         [y + cross_len, y - cross_len],
                        #         color="#FF0000",
                        #         lw=0.5,
                        #         zorder=2,
                        #     )
                        if len(axes) > 0:
                            axes[0].annotate(
                                otype,
                                xy=(x, y),
                                xytext=(x, y + cross_len / 2),
                                ha="center",
                                va="bottom",
                                fontsize=3,
                                color="#FF0000",
                                zorder=3,
                            )

            # Plot variable sources (masked from flux calibration) as red "x" markers
            if masked_sources is not None and len(masked_sources) > 0:
                cross_len = square_size / 4  # Length of each arm of the cross
                masked_count = 0
                if "x_pix" in masked_sources.columns and "y_pix" in masked_sources.columns:
                    for x, y in zip(masked_sources["x_pix"], masked_sources["y_pix"]):
                        # Skip invalid coordinates
                        if not (np.isfinite(x) and np.isfinite(y)):
                            continue

                        # Convert to 0-indexed if needed
                        x_plot = float(x) - 1 if x > 0 else float(x)
                        y_plot = float(y) - 1 if y > 0 else float(y)

                        # Check if coordinates are within image bounds
                        if not (0 <= x_plot < img_width and 0 <= y_plot < img_height):
                            continue

                        x = x_plot
                        y = y_plot

                        # Plot red "x" on all three panels
                        for ax in axes:
                            ax.plot(
                                [x - cross_len, x + cross_len],
                                [y - cross_len, y + cross_len],
                                color="#FF0000",
                                lw=0.5,
                                zorder=2,
                            )
                            ax.plot(
                                [x - cross_len, x + cross_len],
                                [y + cross_len, y - cross_len],
                                color="#FF0000",
                                lw=0.5,
                                zorder=2,
                            )
                        masked_count += 1
                logger.info(f"Plotted {masked_count} variable sources (masked from flux calibration) as red 'x' markers")

            # Add insets and other features
            for i, (title, img_data) in enumerate(images.items()):
                ax = axes[i]

                if expected_location and len(expected_location) == 2:
                    try:
                        x, y = map(int, expected_location)
                        x = max(inset_size, min(x, img_width - inset_size))
                        y = max(inset_size, min(y, img_height - inset_size))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"subtraction_check: invalid expected_location format: {e}")
                        continue

                    x_side, con_x_main, con_x_inset = get_inset_side(
                        x, inset_size, img_width
                    )
                    y_side, con_y_main, con_y_inset = get_inset_side(
                        y, inset_size, img_height
                    )
                    inset_loc = f'{"upper" if y_side == "high" else "lower"} {"right" if x_side == "high" else "left"}'

                    # Inset
                    ax_inset = inset_axes(ax, width="30%", height="30%", loc=inset_loc)
                    cmap = plt.get_cmap("viridis").copy()
                    cmap.set_bad(color="white")
                    ax_inset.imshow(
                        img_data,
                        origin="lower",
                        aspect="auto",
                        cmap=cmap,
                        vmin=vmins[title],
                        vmax=vmaxs[title],
                    )
                    ax_inset.set_xlim(x - inset_size, x + inset_size)
                    ax_inset.set_ylim(y - inset_size, y + inset_size)
                    ax_inset.set_xticks([])
                    ax_inset.set_yticks([])
                    for spine in ax_inset.spines.values():
                        spine.set_color("#000000")
                        spine.set_linewidth(0.5)
                    inset_axes_list.append(ax_inset)

                    # Rectangle on main image
                    rect = Rectangle(
                        (x - inset_size, y - inset_size),
                        2 * inset_size,
                        2 * inset_size,
                        linewidth=0.5,
                        edgecolor="#FF0000",
                        facecolor="none",
                    )
                    ax.add_patch(rect)

                    # Connection lines
                    if inset_loc == "upper left":
                        corners = [
                            ((x - inset_size, y - inset_size), (0, 0)),
                            ((x + inset_size, y + inset_size), (1, 1)),
                        ]
                    elif inset_loc == "upper right":
                        corners = [
                            ((x + inset_size, y - inset_size), (1, 0)),
                            ((x - inset_size, y + inset_size), (0, 1)),
                        ]
                    elif inset_loc == "lower left":
                        corners = [
                            ((x + inset_size, y - inset_size), (1, 0)),
                            ((x - inset_size, y + inset_size), (0, 1)),
                        ]
                    elif inset_loc == "lower right":
                        corners = [
                            ((x - inset_size, y - inset_size), (0, 0)),
                            ((x + inset_size, y + inset_size), (1, 1)),
                        ]

                    for main_pt, inset_pt in corners:
                        fig.add_artist(
                            ConnectionPatch(
                                xyA=main_pt,
                                coordsA=ax.transData,
                                xyB=inset_pt,
                                coordsB=ax_inset.transAxes,
                                axesA=ax,
                                axesB=ax_inset,
                                color="#FF0000",
                                linewidth=0.5,
                            )
                        )

            # Optional mask overlay (skip difference image)
            if mask is not None:
                red_overlay = colors.ListedColormap(["none", "#FF0000"])
                for i, ax in enumerate(fig.axes[:-1]):
                    if ax not in inset_axes_list and i != 2:  # Skip difference image (index 2)
                        ax.imshow(mask, cmap=red_overlay, alpha=0.5, origin="lower")

            # Add markers for expected and fitted locations
            if fitted_location and len(fitted_location) == 2:
                radius = aperture_size  # Circle radius in pixels (diameter = inset_size // 4)

                if len(inset_axes_list) >= 3:
                    for ax in inset_axes_list[2:]:
                        # Red hollow circle at fitted location
                        circle = mpatches.Circle(
                            fitted_location,
                            edgecolor="#FF0000",
                            facecolor="none",
                            linewidth=0.5,
                            transform=ax.transData,
                        )
                        ax.add_patch(circle)

                        # Green cross at expected location (2 lines)
                        if expected_location and len(expected_location) == 2:
                            x, y = expected_location
                    cross_len = aperture_size / 2  # half-length of each arm

                    # hline = mlines.Line2D(
                    #     [x - cross_len, x + cross_len],
                    #     [y, y],
                    #     color="#0000FF",
                    #     linewidth=0.5,
                    #     transform=ax.transData,
                    # )
                    # vline = mlines.Line2D(
                    #     [x, x],
                    #     [y - cross_len, y + cross_len],
                    #     color="#0000FF",
                    #     linewidth=0.5,
                    #     transform=ax.transData,
                    # )
                    # ax.add_line(hline)
                    # ax.add_line(vline)


            # Save figure (PNG only)
            fig.savefig(
                save_path, dpi=150, bbox_inches="tight", facecolor="white"
            )
            plt.close(fig)
            return 1

        except Exception as exc:
            import sys

            exc_type, _, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            line = exc_tb.tb_lineno if exc_tb is not None else -1
            logger.error(
                "subtraction_check failed: %s in %s:%d",
                exc_type.__name__,
                fname,
                line,
                exc_info=True,
            )
            return 0

    # =============================================================================
    #
    # =============================================================================

    def crowding_target(
        self,
        image,
        center,
        segmentation,
        neighbor_mask,
        box_half_size=50,
        aperture_radius=None,
        title_extra="",
    ):
        """
        Save a diagnostic plot centered on the target showing segmentation and neighbor mask.

        Parameters
        ----------
        image : 2D ndarray
            Full image.
        center : tuple (x, y)
            Target center in full-image pixels.
        segmentation : 2D int ndarray
            Segmentation labels in cutout coordinates (0=background).
        neighbor_mask : 2D bool ndarray
            Neighbor mask in cutout coordinates (True=neighbor pixels).
        box_half_size : int
            Half-size of cutout in pixels.
        aperture_radius : float or None
            If provided, draw a circular aperture at the target position (cutout coords).
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from astropy.visualization import ZScaleInterval
        from matplotlib import colors
        import matplotlib.patches as mpatches
        from functions import set_size

        cx, cy = center
        if cx is None or cy is None:
            return
        cx = float(cx)
        cy = float(cy)

        ny, nx = image.shape[:2]
        x0 = int(max(0, np.floor(cx - box_half_size)))
        x1 = int(min(nx, np.ceil(cx + box_half_size + 1)))
        y0 = int(max(0, np.floor(cy - box_half_size)))
        y1 = int(min(ny, np.ceil(cy + box_half_size + 1)))

        cut = np.asarray(image[y0:y1, x0:x1], dtype=float)
        if cut.size == 0:
            return

        seg = np.asarray(segmentation)
        nmask = np.asarray(neighbor_mask, dtype=bool)
        if seg.shape != cut.shape or nmask.shape != cut.shape:
            logger.warning(
                "crowding_target: shape mismatch cut=%s seg=%s mask=%s",
                cut.shape,
                seg.shape,
                nmask.shape,
            )
            return

        # Plot setup
        dir_path = os.path.dirname(os.path.realpath(__file__))
        plt.style.use(os.path.join(dir_path, "autophot.mplstyle"))

        base = os.path.splitext(os.path.basename(self.input_yaml["fpath"]))[0]
        write_dir = os.path.dirname(self.input_yaml["fpath"])
        save_path = os.path.join(write_dir, f"Crowding_Target_{base}.png")

        zscale = ZScaleInterval()
        finite = cut[np.isfinite(cut)]
        if finite.size:
            lower, upper = np.percentile(finite, [0.5, 99.5])
            vmin, vmax = zscale.get_limits(np.clip(cut, lower, upper))
        else:
            vmin, vmax = np.nanmin(cut), np.nanmax(cut)

        fig, axes = plt.subplots(
            1, 3, figsize=set_size(540, 1), constrained_layout=True
        )

        titles = ["Target cutout", "Segmentation", "Neighbor mask"]
        for ax, t in zip(axes, titles):
            ax.set_title(t, fontsize=7, pad=2)
            ax.set_xlabel("X [Pixel]")
            ax.set_ylabel("Y [Pixel]")

        tx = cx - x0
        ty = cy - y0

        # Panel 1
        cmap_vir = plt.get_cmap("viridis").copy()
        cmap_vir.set_bad(color="white")
        axes[0].imshow(
            cut, origin="lower", cmap=cmap_vir, vmin=vmin, vmax=vmax
        )
        axes[0].axvline(tx, color="#0000FF", lw=0.6, alpha=0.9)
        axes[0].axhline(ty, color="#0000FF", lw=0.6, alpha=0.9)
        if (
            aperture_radius is not None
            and np.isfinite(aperture_radius)
            and aperture_radius > 0
        ):
            axes[0].add_patch(
                mpatches.Circle(
                    (tx, ty),
                    float(aperture_radius),
                        edgecolor="#0000FF",
                    facecolor="none",
                    lw=0.8,
                )
            )

        # Panel 2
        axes[1].imshow(
            cut, origin="lower", cmap=cmap_vir, vmin=vmin, vmax=vmax
        )
        levels = np.unique(seg)
        levels = levels[levels > 0]
        if levels.size:
            axes[1].contour(
                seg,
                levels=levels,
                    colors="#00AA00",
                linewidths=0.4,
                alpha=0.9,
            )
        axes[1].axvline(tx, color="#0000FF", lw=0.6, alpha=0.9)
        axes[1].axhline(ty, color="#0000FF", lw=0.6, alpha=0.9)

        # Panel 3
        axes[2].imshow(
            cut, origin="lower", cmap=cmap_vir, vmin=vmin, vmax=vmax
        )
        overlay = colors.ListedColormap(["none", "#FF0000"])
        axes[2].imshow(nmask.astype(int), origin="lower", cmap=overlay, alpha=0.35)
        axes[2].axvline(tx, color="#0000FF", lw=0.6, alpha=0.9)
        axes[2].axhline(ty, color="#0000FF", lw=0.6, alpha=0.9)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    def source_check(
        self,
        image,
        targetSources=None,
        psfSources=None,
        catalogSources=None,
        FWHMSources=None,
        subtracted=False,
        variable_sources=None,
        mask=None,
    ):
        """
        Create a plot to check sources in astronomical images and overlay various source markers.
        FWHM sources are plotted as colored circles with a gradient corresponding to their FWHM values.
        PSF sources are plotted as crosses.
        Reference (catalog) sources are now plotted as squares.

        Parameters:
        image (ndarray): Image data to be plotted.
        targetSources (dict or None): Dictionary with 'x_pix' and 'y_pix' for target sources (optional).
        psfSources (dict or None): Dictionary with 'x_pix' and 'y_pix' for PSF sources (optional).
        catalogSources (dict or None): Dictionary with 'x_pix' and 'y_pix' for catalog sources (optional).
        FWHMSources (dict or None): Dictionary with 'x_pix', 'y_pix', and 'fwhm' for FWHM sources (optional).
        subtracted (bool): Whether the image is subtracted (used for filename) (default is False).
        mask (ndarray or None): Optional mask to overlay on the image (default is None).
        """

        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from functions import set_size
            from astropy.visualization import (
                ImageNormalize,
                LinearStretch,
                ZScaleInterval,
            )
            from matplotlib.patches import Circle, Rectangle
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            from matplotlib.lines import Line2D
            from scipy.spatial import cKDTree

            # Set up matplotlib style
            dir_path = os.path.dirname(os.path.realpath(__file__))
            plt.style.use(os.path.join(dir_path, "autophot.mplstyle"))

            # Extract file path and base name
            fpath = self.input_yaml["fpath"]
            base = os.path.basename(fpath)
            write_dir = os.path.dirname(fpath)
            base = os.path.splitext(base)[0]

            # Get photometry radius and scale from input YAML
            phot_cfg = self.input_yaml.get("photometry") or {}
            ap_size_fwhm = phot_cfg.get("aperture_size", 1.7)
            radius = float(ap_size_fwhm) * float(self.input_yaml["fwhm"])
            scale = self.input_yaml["scale"]

            # Create the figure with WCS axes if available
            plt.ioff()  # Turn interactive mode off
            fig = plt.figure(figsize=set_size(540, 1))
            
            # Try to use WCSAxes for RA/Dec display
            wcs = None
            skip_tight_layout = False
            try:
                from astropy.wcs import WCS
                from astropy.io import fits

                def _pick_celestial_wcs(_fpath: str):
                    """
                    Return a 2D celestial WCS suitable for plotting, or None.

                    Some FITS store the science image/WCS in an extension rather than the
                    primary HDU. Also, WCSAxes works best with an explicitly 2D celestial
                    WCS (`w.celestial`) even when the full WCS has extra axes.
                    """
                    try:
                        with fits.open(_fpath, memmap=False) as hdul:
                            for hdu in hdul:
                                hdr = getattr(hdu, "header", None)
                                if hdr is None:
                                    continue
                                # Must look like an image header.
                                if int(hdr.get("NAXIS", 0)) < 2:
                                    continue
                                w = WCS(hdr, fix=True, relax=True)
                                if not getattr(w, "has_celestial", False):
                                    continue
                                wc = w.celestial
                                # Guard against malformed celestial WCS that produces empty coords
                                if getattr(wc, "pixel_n_dim", 0) >= 2:
                                    return wc
                    except Exception:
                        return None
                    return None

                wcs = _pick_celestial_wcs(fpath)
                if wcs is not None:
                    ax1 = fig.add_subplot(111, projection=wcs)
                    # Guard: some malformed headers can yield a WCSAxes instance whose
                    # coordinate helpers are empty; attempting ax1.coords[0] then raises
                    # "index 0 is out of bounds...". In that case fall back to pixels.
                    try:
                        n_coords = len(getattr(ax1, "coords", []))
                    except Exception:
                        n_coords = 0
                    if n_coords < 2:
                        raise ValueError(
                            f"WCSAxes has insufficient coords (n={n_coords})"
                        )
                    # Configure coordinate axes with minimal overlap
                    # Main coordinates: RA/Dec on bottom/left
                    ax1.coords[0].set_ticklabel_position('b')
                    ax1.coords[0].set_axislabel_position('b')
                    ax1.coords[0].set_axislabel("RA", fontsize=6, minpad=0.3)
                    ax1.coords[0].set_major_formatter('hh:mm')
                    
                    ax1.coords[1].set_ticklabel_position('l')
                    ax1.coords[1].set_axislabel_position('l')
                    ax1.coords[1].set_axislabel("Dec", fontsize=6, minpad=0.3)
                    ax1.coords[1].set_major_formatter('dd:mm')
                    
                    # Hide default frame labels
                    ax1.set_xlabel("")
                    ax1.set_ylabel("")
                    
                    # Add pixel coordinates on top/right without labels
                    ax1.coords[0].set_ticks_position('bt')
                    ax1.coords[1].set_ticks_position('lr')
                    ax1.coords[0].set_ticklabel_position('b')
                    ax1.coords[1].set_ticklabel_position('l')
                    
                    # Disable coordinate grid
                    ax1.coords.grid(False)
                    
                    # Skip tight_layout for WCS axes (it can fail)
                    skip_tight_layout = True
                    
                    logger.info("Source check plot: using RA/Dec WCS axes")
                else:
                    logger.warning("Source check plot: WCS has no celestial component, using pixel axes")
                    raise ValueError("WCS has no celestial component")
            except Exception as e:
                # Fallback to regular axes with pixel coordinates only
                logger.warning(f"Source check plot: WCS axes failed ({e}), using pixel coordinates only")
                # Clear the figure to remove any partially-created WCS axes
                fig.clf()
                ax1 = fig.add_subplot(111)
                ax1.set_xlabel("X (pixels)", fontsize=6)
                ax1.set_ylabel("Y (pixels)", fontsize=6)
                skip_tight_layout = False

            # Normalize and plot the image
            norm = ImageNormalize(
                image, interval=ZScaleInterval(), stretch=LinearStretch()
            )
            # Set NaN values to display as white
            cmap = plt.get_cmap("viridis")
            cmap.set_bad(color='white')
            im = ax1.imshow(
                image,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                interpolation=None,
                norm=norm,
            )

            # Harmonize pixel-origin conventions for overlay markers.
            # Some WCS-derived catalogs can be 1-based while detected sources
            # are 0-based; estimate and correct a global +/-1 px shift for display.
            marker_dx = 0.0
            marker_dy = 0.0
            if (
                catalogSources is not None
                and FWHMSources is not None
                and len(catalogSources) > 5
                and len(FWHMSources) > 5
                and {"x_pix", "y_pix"}.issubset(catalogSources.columns)
                and {"x_pix", "y_pix"}.issubset(FWHMSources.columns)
            ):
                try:
                    cat_xy0 = np.vstack(
                        [catalogSources["x_pix"].values, catalogSources["y_pix"].values]
                    ).T.astype(float)
                    det_xy0 = np.vstack(
                        [FWHMSources["x_pix"].values, FWHMSources["y_pix"].values]
                    ).T.astype(float)
                    finite_cat = np.isfinite(cat_xy0).all(axis=1)
                    finite_det = np.isfinite(det_xy0).all(axis=1)
                    cat_xy0 = cat_xy0[finite_cat]
                    det_xy0 = det_xy0[finite_det]
                    if len(cat_xy0) > 5 and len(det_xy0) > 5:
                        tree_det0 = cKDTree(det_xy0)
                        d0, idx0 = tree_det0.query(cat_xy0, k=1)
                        keep0 = d0 < 3.0
                        if np.count_nonzero(keep0) >= 10:
                            delta0 = det_xy0[idx0[keep0]] - cat_xy0[keep0]
                            med_dx = float(np.nanmedian(delta0[:, 0]))
                            med_dy = float(np.nanmedian(delta0[:, 1]))
                            cand_dx = float(np.round(med_dx))
                            cand_dy = float(np.round(med_dy))
                            if (
                                abs(cand_dx) <= 1.0
                                and abs(cand_dy) <= 1.0
                                and (abs(cand_dx) + abs(cand_dy)) > 0.0
                                and abs(med_dx - cand_dx) < 0.35
                                and abs(med_dy - cand_dy) < 0.35
                            ):
                                marker_dx, marker_dy = cand_dx, cand_dy
                                logger.info(
                                    "SourceCheck marker alignment: applying catalog overlay shift (dx=%+.0f, dy=%+.0f) px.",
                                    marker_dx,
                                    marker_dy,
                                )
                except Exception:
                    pass

            # Plot the target source as a circle
            edge_color = get_divergent_color('target') if get_divergent_color else 'blue'
            circle = Circle(
                (self.input_yaml["target_x_pix"], self.input_yaml["target_y_pix"]),
                radius,
                edgecolor=edge_color,
                facecolor="none",
                zorder=4,
                lw=1.0,
            )
            ax1.add_patch(circle)
            # Get target name with TNS prefix if available
            target_name = self.input_yaml["target_name"]
            name_prefix = self.input_yaml.get("name_prefix", "")
            objname = self.input_yaml.get("objname", target_name)
            # Add prefix if it exists and is not already in the name
            if name_prefix and name_prefix.strip() and not target_name.startswith(name_prefix):
                display_name = f"{name_prefix}{objname}"
            else:
                display_name = target_name

            ax1.text(
                self.input_yaml["target_x_pix"],
                self.input_yaml["target_y_pix"] + radius + 2,
                display_name,
                color="#FFD700",
                fontsize=4,
                ha="center",
                va="bottom",
                fontweight="bold",
            )

            # Plot PSF sources as hexagons with 4*FWHM width
            if psfSources is not None and len(psfSources) > 0:
                from matplotlib.patches import RegularPolygon
                fwhm = float(self.input_yaml.get("fwhm", 5.0))
                hex_radius = 2.0 * fwhm  # Radius of hexagon = 2*FWHM (width = 4*FWHM)
                for x, y in zip(psfSources["x_pix"], psfSources["y_pix"]):
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue
                    hexagon = RegularPolygon(
                        (x, y),
                        numVertices=6,
                        radius=hex_radius,
                        orientation=np.pi/6,  # Point up
                        edgecolor=get_divergent_color('psf'),
                        facecolor="none",
                        linewidth=0.5,
                        label="PSF sources" if x == psfSources["x_pix"].iloc[0] else None,
                        zorder=1,
                    )
                    ax1.add_patch(hexagon)

            # Plot reference (catalog) sources as squares
            if catalogSources is not None:
                fwhm = float(self.input_yaml.get("fwhm", 5.0))
                square_size = 4.0 * fwhm  # Size of the square = 4 * FWHM
                for x, y in zip(catalogSources["x_pix"], catalogSources["y_pix"]):
                    x = float(x) + marker_dx
                    y = float(y) + marker_dy
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue
                    lower_left = (x - square_size / 2, y - square_size / 2)
                    square = Rectangle(
                        lower_left,
                        square_size,
                        square_size,
                        edgecolor=get_divergent_color('reference'),
                        facecolor="none",
                        label="Reference Sources",
                        zorder=1,
                        lw=0.5,
                    )
                    ax1.add_patch(square)

            # Plot FWHM sources as colored circles with gradient
            # print(FWHMSources.columns)
            if FWHMSources is not None and "fwhm" in FWHMSources:

                if len(FWHMSources) > 0:
                    # Get FWHM values and normalize for colormap
                    fwhm_values = np.array(FWHMSources["fwhm"])
                    norm_fwhm = Normalize(
                        vmin=np.nanmin(fwhm_values), vmax=np.nanmax(fwhm_values)
                    )
                    # Perceptually-uniform sequential map for FWHM scaling.
                    cmap = plt.get_cmap("gray")

                    # Create a ScalarMappable for the colorbar
                    sm = ScalarMappable(norm=norm_fwhm, cmap=cmap)
                    sm.set_array([])

                    # Plot each FWHM source with color corresponding to its FWHM value
                    for x, y, fwhm in zip(
                        FWHMSources["x_pix"], FWHMSources["y_pix"], fwhm_values
                    ):
                        if np.isnan(fwhm):
                            continue
                        color = cmap(norm_fwhm(fwhm))
                        circle = Circle(
                            (x, y),
                            fwhm,
                            edgecolor=color,
                            facecolor="none",
                            label="FWHM Sources",
                            zorder=2,
                            lw=0.8,
                            ls="-",
                        )
                        ax1.add_patch(circle)

                    # Add colorbar
                    cbar = fig.colorbar(sm, ax=ax1, pad=0.02, aspect=40)
                    cbar.set_label("FWHM (pixels)", fontsize=7)
                    cbar.ax.tick_params(labelsize=6)

            # from matplotlib.patches import Rectangle

            # Plot variable sources as yellow "x" and annotate with otype
            if variable_sources is not None:

                if len(variable_sources) > 0:

                    cross_len = scale / 4  # Length of each arm of the cross

                    for x, y, otype, name in zip(
                        variable_sources["x_pix"],
                        variable_sources["y_pix"],
                        variable_sources["OTYPE_opt"],
                        variable_sources["MAIN_ID"],
                    ):
                        x = float(x) + marker_dx
                        y = float(y) + marker_dy
                        if not (np.isfinite(x) and np.isfinite(y)):
                            continue
                        # Skip sources that fall on NaN regions
                        ix, iy = int(round(x)), int(round(y))
                        if 0 <= ix < image.shape[1] and 0 <= iy < image.shape[0]:
                            if np.isnan(image[iy, ix]):
                                continue

                        # Draw "x" as two lines rotated 45 degrees (divergent color for distinct visibility)
                        ax1.plot(
                            [x - cross_len, x + cross_len],
                            [y - cross_len, y + cross_len],
                            color=get_divergent_color('cross'),
                            lw=1.0,
                            zorder=3,
                        )
                        ax1.plot(
                            [x - cross_len, x + cross_len],
                            [y + cross_len, y - cross_len],
                            color=get_divergent_color('cross'),
                            lw=1.0,
                            zorder=3,
                        )

                        if "SN*" in otype:
                            otype = name
                        # Annotate otype centered below the cross
                        ax1.annotate(
                            otype,
                            xy=(x, y),
                            xytext=(
                                x,
                                y + cross_len * 2,
                            ),  # offset down in display (pixels)
                            ha="center",
                            va="bottom",
                            fontsize=4,
                            color=get_divergent_color('cross'),
                            zorder=4,
                        )

            # Optional: distortion/residual vectors (catalog -> detected/FWHM sources).
            # This helps visualize non-uniform astrometric residuals across the field.
            distortion_rms_text = None
            distortion_grid_artist = None
            try:
                align_cfg = self.input_yaml.get("alignment", {})
                show_vec = bool(
                    align_cfg.get("plot_source_check_distortion_vectors", True)
                )
                max_sep = float(align_cfg.get("plot_source_check_max_sep_pix", 6.0))
                max_vec = int(align_cfg.get("plot_source_check_max_vectors", 300))
                min_vec = int(align_cfg.get("plot_source_check_min_vectors", 10))
                show_grid_map = bool(
                    align_cfg.get("plot_source_check_distortion_grid_map", False)
                )
                if (
                    show_vec
                    and catalogSources is not None
                    and FWHMSources is not None
                    and len(catalogSources) >= min_vec
                    and len(FWHMSources) >= min_vec
                    and {"x_pix", "y_pix"}.issubset(catalogSources.columns)
                    and {"x_pix", "y_pix"}.issubset(FWHMSources.columns)
                ):
                    cat_xy = np.vstack(
                        [catalogSources["x_pix"].values, catalogSources["y_pix"].values]
                    ).T
                    det_xy = np.vstack(
                        [FWHMSources["x_pix"].values, FWHMSources["y_pix"].values]
                    ).T
                    tree = cKDTree(det_xy)
                    dists, idx = tree.query(cat_xy, k=1, distance_upper_bound=max_sep)
                    keep = np.isfinite(dists) & (idx < len(det_xy))
                    if np.count_nonzero(keep) >= min_vec:
                        cat_m = cat_xy[keep]
                        det_m = det_xy[idx[keep]]
                        u = det_m[:, 0] - cat_m[:, 0]
                        v = det_m[:, 1] - cat_m[:, 1]
                        # Note: distortion contour overlays have been removed. We only
                        # show per-source residual vectors (optionally) and the RMS text.
                        if len(cat_m) > max_vec:
                            sel = np.linspace(0, len(cat_m) - 1, max_vec, dtype=int)
                            cat_m = cat_m[sel]
                            u = u[sel]
                            v = v[sel]
                        ax1.quiver(
                            cat_m[:, 0],
                            cat_m[:, 1],
                            u,
                            v,
                            angles="xy",
                            scale_units="xy",
                            scale=1.0,
                            color=get_divergent_color('positive'),
                            alpha=0.65,
                            width=0.0020,
                            zorder=5,
                        )
                        distortion_rms = float(np.sqrt(np.mean(u * u + v * v)))
                        distortion_rms_text = f"Distortion residual RMS: {distortion_rms:.2f} px"
                        
            except Exception as e:
                logger.debug("Distortion vector overlay skipped: %s", e)

            # Overlay the mask if provided
            if mask is not None:
                from matplotlib import colors

                # White overlay for masked regions
                mask_cmap = colors.ListedColormap(["none", "white"])
                ax1.imshow(mask, cmap=mask_cmap, alpha=1.0, origin="lower")

            # Optional colorbar for distortion grid-map magnitude.
            if distortion_grid_artist is not None:
                try:
                    align_cfg = self.input_yaml.get("alignment", {})
                    show_grid_cbar = bool(
                        align_cfg.get(
                            "plot_source_check_distortion_grid_colorbar",
                            True,
                        )
                    )
                    if show_grid_cbar:
                        cbar = fig.colorbar(
                            distortion_grid_artist,
                            ax=ax1,
                            pad=0.01,
                            aspect=35,
                        )
                        cbar.set_label(
                            "Distortion magnitude [px]",
                            fontsize=6,
                        )
                        cbar.ax.tick_params(labelsize=5)
                except Exception as e:
                    logger.debug("Distortion grid colorbar skipped: %s", e)

            # Set axis labels and limits
            ax1.set_xlabel("X [Pixel]")
            ax1.set_ylabel("Y [Pixel]")
            ax1.set_xlim(0, image.shape[1])
            ax1.set_ylim(0, image.shape[0])
    
            # Create and add legend
            handles, labels = ax1.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            leg = ax1.legend(
                by_label.values(),
                by_label.keys(),
                loc="lower center",
                bbox_to_anchor=(0.5, 1.0),
                frameon=False,
                handlelength=1.5,
                handletextpad=0.5,
                ncol = 3
            )

            # Finalize figure layout
            if not skip_tight_layout:
                fig.tight_layout()
            ax1.set_aspect("equal", adjustable="box")

            # Save figure
            if not subtracted:
                save_loc = os.path.join(
                    write_dir, "SourceCheck_" + base + ".png"
                )
            else:
                save_loc = os.path.join(
                    write_dir, "Subtracted_SourceCheck_" + base + ".png"
                )

            fig.savefig(
                save_loc, dpi=150, bbox_extra_artists=[leg], facecolor="white"
            )
            plt.close()

        except Exception as exc:
            import sys

            exc_type, _, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            line = exc_tb.tb_lineno if exc_tb is not None else -1
            logger.error(
                "SourceCheck failed: %s in %s:%d",
                exc_type.__name__,
                fname,
                line,
                exc_info=True,
            )

    # =============================================================================
    #
    # =============================================================================
    @staticmethod
    def plot_lightcurve(
        output_file,
        snr_limit=3,
        beta_limit=0.5,
        fwhm=3,
        method="PSF",
        reference_epoch=0,
        redshift=0,
        show_limits=False,
        show_details=True,
        default_size=(540, 1),
        ls="",
        show: bool = False,
    ):
        """
        Plot lightcurve with detections and optional upper limits.
        Detection = SNR >= snr_limit and (beta > beta_limit if 'beta' present).
        Non-detections are plotted as upper limits at limiting_inst_mag (instrumental limiting magnitude) when show_limits=True.

        Parameters
        ----------
        ls : str
            Line style for connecting detection points (default "" for no line).
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from functions import set_size

        # Default color array (legacy). If `lightcurve.BAND_COLORS` is available
        # it will override known filter colors using `databases/filters.yml`.
        cols = {
            "u": "dodgerblue",
            "g": "g",
            "r": "r",
            "i": "goldenrod",
            "z": "k",
            "y": "0.5",
            "w": "firebrick",
            "Y": "0.5",
            "U": "slateblue",
            "B": "b",
            "V": "yellowgreen",
            "R": "crimson",
            "I": "chocolate",
            "G": "salmon",
            "E": "salmon",
            "J": "darkred",
            "H": "orangered",
            "K": "saddlebrown",
            "S": "mediumorchid",
            "D": "purple",
            "A": "midnightblue",
            "F": "hotpink",
            "N": "magenta",
            "o": "darkorange",
            "c": "cyan",
            "W": "forestgreen",
            "Q": "peru",
        }
        palette = cols
        if isinstance(BAND_COLORS, dict) and BAND_COLORS:
            palette = {**cols, **BAND_COLORS}

        # Maintains order from blue to red effective wavelength
        bandlist = "FSDNAuUBgcVwrRoGEiIzyYJHKWQ"

        markers = ["x", "o", "^", "s", "D"]
        marker_iterator = iter(markers)

        data = pd.read_csv(output_file)
        if _normalize_photometry_columns is not None:
            data = _normalize_photometry_columns(data)
        if data.columns.duplicated().any():
            data = data.loc[:, ~data.columns.duplicated()].copy()

        filter_series = (
            photometry_filter_series(data) if photometry_filter_series else None
        )
        uniform_mag = (
            f"mag_{method_low}" in data.columns
            and f"mag_{method_low}_err" in data.columns
        )
        plotted_uniform_without_filter = False
        use_filter_bands = (
            uniform_mag
            and filter_series is not None
            and filter_series.notna().any()
            and canonical_bands_from_filter_series is not None
        )
        label_map = {}
        if (
            use_filter_bands
            and canonical_band_label_map_from_filter_series is not None
            and filter_series is not None
        ):
            try:
                label_map = canonical_band_label_map_from_filter_series(filter_series)
            except Exception:
                label_map = {}
        loop_bands = (
            canonical_bands_from_filter_series(filter_series)
            if use_filter_bands
            else list(bandlist)
        )
        if use_filter_bands and not loop_bands:
            loop_bands = list(bandlist)

        if redshift != 0:
            from functions import get_distance_modulus

            dm = get_distance_modulus(redshift)
        else:
            dm = 0

        fig = plt.figure(figsize=set_size(*default_size))
        ax1 = fig.add_subplot(111)
        ax1.invert_yaxis()

        # Color selection:
        # - If band is known, use the pipeline palette in `cols`.
        # - Otherwise, fall back to Matplotlib's default color cycle so unknown
        #   filters are still distinguishable.
        _generic_cycle = (
            plt.rcParams.get("axes.prop_cycle", None).by_key().get("color", [])
            if plt.rcParams.get("axes.prop_cycle", None) is not None
            else []
        )
        if not _generic_cycle:
            _generic_cycle = ["k"]
        _unknown_color_map = {}

        def _color_for_band(band_label: str) -> str:
            if band_label in palette:
                return palette[band_label]
            bl = str(band_label).strip().lower()
            if bl in palette:
                return palette[bl]
            if bl not in _unknown_color_map:
                _unknown_color_map[bl] = _generic_cycle[
                    len(_unknown_color_map) % len(_generic_cycle)
                ]
            return _unknown_color_map[bl]

        num_detect = 0
        num_nondetect = 0
        method_low = str(method).strip().lower()
        method_u = str(method).strip().upper()
        for b in loop_bands:
            # Prefer new uniform columns, then fall back to legacy per-band columns.
            band = b + "_" + method
            if uniform_mag:
                mag_col = f"mag_{method_low}"
                err_col = f"mag_{method_low}_err"
            elif band in data.columns and (band + "_err") in data.columns:
                mag_col = band
                err_col = band + "_err"
            else:
                continue

            # Long-form CSV: assign rows to this band (gp→g, Sloan_g→g, etc.).
            if use_filter_bands:
                mask = filter_series.map(
                    lambda rv: filter_value_matches_band(b, rv)
                    if filter_value_matches_band
                    else str(rv).strip().lower() == str(b).strip().lower()
                )
                mask = mask.fillna(False)
                data_band = data.loc[mask].copy()
            elif filter_series is not None and filter_series.notna().any():
                mask = filter_series.map(
                    lambda rv: filter_value_matches_band(b, rv)
                    if filter_value_matches_band
                    else str(rv).strip().lower() == str(b).strip().lower()
                )
                mask = mask.fillna(False)
                data_band = data.loc[mask].copy()
            elif uniform_mag:
                if plotted_uniform_without_filter:
                    continue
                data_band = data.copy()
                plotted_uniform_without_filter = True
            else:
                data_band = data.copy()

            mag_num = pd.to_numeric(data_band[mag_col], errors="coerce")
            data_band = data_band[np.isfinite(mag_num)].copy()
            if data_band.empty:
                continue

            col_lc = {str(c).lower(): c for c in data_band.columns}

            # Resolve SNR: match lightcurve-style priority; headers are usually lowercase.
            if method_u == "PSF" and "snr_psf" in col_lc:
                snr = np.asarray(data_band[col_lc["snr_psf"]], dtype=float)
            elif (
                method_u == "PSF"
                and "flux_psf" in col_lc
                and "flux_psf_err" in col_lc
            ):
                err_psf = np.asarray(data_band[col_lc["flux_psf_err"]], dtype=float)
                snr = np.divide(
                    np.asarray(data_band[col_lc["flux_psf"]], dtype=float),
                    err_psf,
                    out=np.full(len(data_band), np.nan),
                    where=(err_psf > 0) & np.isfinite(err_psf),
                )
            elif method_u == "AP" and "snr_ap" in col_lc:
                snr = np.asarray(data_band[col_lc["snr_ap"]], dtype=float)
            elif "snr" in col_lc:
                snr = np.asarray(data_band[col_lc["snr"]], dtype=float)
            elif "snr_psf" in col_lc:
                snr = np.asarray(data_band[col_lc["snr_psf"]], dtype=float)
            elif "snr_ap" in col_lc:
                snr = np.asarray(data_band[col_lc["snr_ap"]], dtype=float)
            else:
                err_vals = np.asarray(
                    pd.to_numeric(data_band[err_col], errors="coerce"), dtype=float
                )
                mag_vals = np.asarray(
                    pd.to_numeric(data_band[mag_col], errors="coerce"), dtype=float
                )
                snr = np.divide(
                    mag_vals,
                    err_vals,
                    out=np.zeros(len(data_band)),
                    where=err_vals > 0,
                )

            # Detection: SNR >= snr_limit and (optionally) beta > beta_limit
            beta_ok = (
                (data_band["beta"] > beta_limit)
                if "beta" in data_band.columns
                else np.ones(len(data_band), dtype=bool)
            )
            detects_idx = np.isfinite(snr) & (snr >= snr_limit) & beta_ok
            nondetects_idx = ~detects_idx

            detects = data_band[detects_idx]
            nondetects = data_band[nondetects_idx]

            num_detect += len(detects)
            num_nondetect += len(nondetects)

            marker = next(marker_iterator)

            if not detects.empty:
                x_det = pd.to_numeric(detects["mjd"], errors="coerce") - reference_epoch
                leg_label = (
                    label_map.get(str(b).strip().lower(), str(b))
                    if use_filter_bands
                    else str(b)
                )
                ax1.errorbar(
                    x_det,
                    detects[mag_col],
                    yerr=detects[err_col],
                    c=_color_for_band(b),
                    ls="",
                    capsize=2,
                    marker=marker,
                    label=leg_label,
                )
                # Optional: draw line connecting detection points
                if ls:
                    sorted_detects = detects.sort_values("mjd")
                    x_line = (
                        pd.to_numeric(sorted_detects["mjd"], errors="coerce")
                        - reference_epoch
                    )
                    ax1.plot(
                        x_line,
                        sorted_detects[mag_col],
                        color=_color_for_band(b),
                        linestyle=ls,
                        linewidth=0.8,
                        alpha=0.6,
                        zorder=0,
                    )
            if show_limits and not nondetects.empty:
                # Upper limits: plot at limiting magnitude (fainter than this = non-detection)
                if "limiting_inst_mag" in nondetects.columns and np.any(
                    np.isfinite(nondetects["limiting_inst_mag"])
                ):
                    y_lim = nondetects["limiting_inst_mag"]
                elif "lmag" in nondetects.columns and np.any(np.isfinite(nondetects["lmag"])):
                    # Backwards compatibility when plotting older output CSVs.
                    y_lim = nondetects["lmag"]
                else:
                    y_lim = nondetects[mag_col]
                x_nd = (
                    pd.to_numeric(nondetects["mjd"], errors="coerce") - reference_epoch
                )
                ax1.errorbar(
                    x_nd,
                    y_lim,
                    c=_color_for_band(b),
                    ls="",
                    marker="v",
                    markersize=5,
                    markerfacecolor="none",
                    markeredgewidth=0.5,
                    alpha=0.85,
                    zorder=0,
                )

        ax1.set_ylim(ax1.get_ylim())
        if redshift != 0.0:

            ax11 = ax1.twinx()
            ax11.set_xlim(ax1.get_xlim())
            # autophot_input['target_ra'], autophot_input['target_dec'] = 122.920076, -54.651908

            ax11.set_ylim(ax1.get_ylim() - dm)
            ax11.set_ylabel("Abs. Magnitude")

        ax1.set_ylabel("App. Magnitude")

        if reference_epoch != 0.0:
            ax1.set_xlabel(rf"Days since {reference_epoch}")
        else:
            ax1.set_xlabel("Modified Julian Date")

        if show_details:
            text = rf"# detect {num_detect}" + "\n" + rf"# nondetect {num_nondetect}"
            plt.text(
                0.02,
                0.95,
                text,
                transform=ax1.transAxes,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(facecolor="white", alpha=0.8),
            )

        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        ncols = len(by_label.values()) % 3
        if ncols == 0:
            ncols = len(by_label.values())
        ax1.legend(
            by_label.values(),
            by_label.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            frameon=False,
            handlelength=1.5,
            handletextpad=0.5,
        )

        # Only show interactively when requested. In batch mode (Agg backend),
        # calling show() emits warnings and is not useful.
        if bool(show):
            plt.show()

        return

    def plot_wcs_vs_psf_offset(self, sources, imageWCS=None):
        """
        Diagnostic plot: dx vs dy between PSF fitted position and WCS catalog position.

        dx = x_fit - x_pix (PSF fitted minus WCS catalog)
        dy = y_fit - y_pix (PSF fitted minus WCS catalog)

        Plots with error bars on both axes to identify systematic offsets or outliers.

        Parameters:
        sources (pd.DataFrame): DataFrame with columns x_pix, y_pix, x_fit, y_fit,
                                x_fit_err, y_fit_err.
        imageWCS (astropy.wcs.WCS, optional): WCS object for logging.
        """
        import matplotlib.pyplot as plt
        from functions import set_size
        import numpy as np

        try:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            plt.style.use(os.path.join(dir_path, "autophot.mplstyle"))

            base = os.path.splitext(os.path.basename(self.input_yaml["fpath"]))[0]
            write_dir = os.path.dirname(self.input_yaml["fpath"])
            save_path = os.path.join(write_dir, f"WCS_vs_PSF_Offset_{base}.png")

            # Filter to sources with valid PSF fits
            valid = (
                sources["x_pix"].notna()
                & sources["y_pix"].notna()
                & sources["x_fit"].notna()
                & sources["y_fit"].notna()
            )
            df = sources[valid].copy()

            if len(df) == 0:
                logger.warning("No valid PSF fits for WCS vs PSF offset plot")
                return

            # Compute offsets
            df["dx"] = df["x_fit"] - df["x_pix"]
            df["dy"] = df["y_fit"] - df["y_pix"]

            # Get errors (use catalog position error if available, otherwise PSF fit error)
            # For dx error: sqrt(x_fit_err^2 + x_pix_err^2). If x_pix_err not available, use x_fit_err only.
            dx_err = df.get("x_fit_err", np.nan).copy()
            dy_err = df.get("y_fit_err", np.nan).copy()

            # Filter to sources with finite errors if available
            finite_err = dx_err.notna() & dy_err.notna()
            has_errors = finite_err.any()
            
            if has_errors:
                df_plot = df[finite_err].copy()
                # Exclude sources with very large position errors (> FWHM)
                # These indicate problematic fits and should not dominate the plot
                fwhm = float(self.input_yaml.get("fwhm", 3.0))
                reasonable_err = (
                    (df_plot["x_fit_err"] <= fwhm) & (df_plot["y_fit_err"] <= fwhm)
                )
                n_before = len(df_plot)
                df_plot = df_plot[reasonable_err].copy()
                n_excluded = n_before - len(df_plot)
                if n_excluded > 0:
                    logger.info(
                        f"WCS vs PSF offset plot: excluded {n_excluded}/{n_before} sources "
                        f"with position errors > FWHM ({fwhm:.1f} px)"
                    )
            else:
                # No error columns available, plot without error bars
                df_plot = df.copy()
                logger.info("WCS vs PSF offset plot: no error columns available, plotting without error bars")

            # Create plot
            width_pt = 5.5 * 72.27
            aspect = 1.0
            fig, ax = plt.subplots(figsize=set_size(width_pt, aspect=aspect))

            # Scatter with error bars if available, otherwise simple scatter
            if has_errors:
                ax.errorbar(
                    df_plot["dx"],
                    df_plot["dy"],
                    xerr=df_plot["x_fit_err"],
                    yerr=df_plot["y_fit_err"],
                    fmt="o",
                    markersize=3,
                    capsize = 1,
                    markerfacecolor="dodgerblue",
                    markeredgecolor="none",
                    ecolor="gray",
                    elinewidth=0.5,
                    alpha=0.7,
                    zorder=2,
                )
            else:
                ax.scatter(
                    df_plot["dx"],
                    df_plot["dy"],
                    s=9,
                    marker="o",
                    facecolor="dodgerblue",
                    edgecolor="none",
                    alpha=0.7,
                    zorder=2,
                )

            # Zero lines
            # ax.axhline(0, color="red", lw=1.0, ls="--", alpha=0.5, zorder=1)
            # ax.axvline(0, color="red", lw=1.0, ls="--", alpha=0.5, zorder=1)

            # Median offset lines
            med_dx = np.nanmedian(df_plot["dx"])
            med_dy = np.nanmedian(df_plot["dy"])
            # ax.axhline(med_dy, color="orange", lw=1.2, ls="-", alpha=0.6, zorder=1, label=f"Median dy = {med_dy:.3f}")
            # ax.axvline(med_dx, color="orange", lw=1.2, ls="-", alpha=0.6, zorder=1, label=f"Median dx = {med_dx:.3f}")

            # RMS
            rms_dx = np.sqrt(np.nanmean(df_plot["dx"]**2))
            rms_dy = np.sqrt(np.nanmean(df_plot["dy"]**2))

            # Symmetric square axes with (0,0) at centre
            if has_errors:
                _lim = max(
                    np.nanmax(np.abs(df_plot["dx"] + df_plot["x_fit_err"])),
                    np.nanmax(np.abs(df_plot["dy"] + df_plot["y_fit_err"])),
                    1.0,
                ) * 1.1
            else:
                _lim = max(
                    np.nanmax(np.abs(df_plot["dx"])),
                    np.nanmax(np.abs(df_plot["dy"])),
                    1.0,
                ) * 1.1
            ax.set_xlim(-_lim, _lim)
            ax.set_ylim(-_lim, _lim)
            ax.axhline(0, color="red", lw=0.8, ls="--", alpha=0.5, zorder=1)
            ax.axvline(0, color="red", lw=0.8, ls="--", alpha=0.5, zorder=1)

            ax.set_xlabel(r"$\Delta x = x_{\mathrm{PSF}} - x_{\mathrm{WCS}}$ [px]")
            ax.set_ylabel(r"$\Delta y = y_{\mathrm{PSF}} - y_{\mathrm{WCS}}$ [px]")

            # Add upper and right axes for arcsecond offsets
            # Get pixel scale from input_yaml or WCS
            pixel_scale = None
            if "pixel_scale" in self.input_yaml:
                pixel_scale = float(self.input_yaml["pixel_scale"])
            elif imageWCS is not None:
                # Try to get pixel scale from WCS
                try:
                    from astropy.wcs import utils as wcs_utils
                    pixel_scale = wcs_utils.proj_plane_pixel_scales(imageWCS)[0] * 3600  # Convert to arcsec
                except:
                    pass

            if pixel_scale is not None and pixel_scale > 0:
                # Create twin axes for arcsecond display
                ax_top = ax.twiny()
                ax_right = ax.twinx()
                # set_aspect is incompatible with shared/twin axes; the symmetric
                # ±_lim xlim/ylim already enforces a square data region.

                # Set the limits for twin axes to match the main axes
                ax_top.set_xlim(ax.get_xlim())
                ax_right.set_ylim(ax.get_ylim())

                # Convert pixel limits to arcseconds
                x_lim_arcsec = np.array(ax.get_xlim()) * pixel_scale
                y_lim_arcsec = np.array(ax.get_ylim()) * pixel_scale

                # Set tick locations and labels for arcseconds
                ax_top.set_xticks(ax.get_xticks())
                ax_top.set_xticklabels([f"{x*pixel_scale:.2f}" for x in ax.get_xticks()])
                ax_top.set_xlabel(r"$\Delta$RA [arcsec]", fontsize="small")

                ax_right.set_yticks(ax.get_yticks())
                ax_right.set_yticklabels([f"{y*pixel_scale:.2f}" for y in ax.get_yticks()])
                ax_right.set_ylabel(r"$\Delta$Dec [arcsec]", fontsize="small")

                # Hide the tick labels on the opposite sides of twin axes
                ax_top.tick_params(axis="x", which="both", labeltop=True, labelbottom=False)
                ax_right.tick_params(axis="y", which="both", labelright=True, labelleft=False)
            else:
                # No twin axes — safe to enforce equal aspect
                ax.set_aspect("equal", adjustable="box")
            # ax.set_title(f"WCS vs PSF Position Offset (N={len(df_plot)})")
            # ax.legend(loc="upper right", fontsize="small", framealpha=0.9)
            ax.grid(True, ls="-", alpha=0.25, zorder=0)

            # Add text with statistics
            stats_text = (
                f"Median: ({med_dx:.3f}, {med_dy:.3f}) px\n"
                f"RMS: ({rms_dx:.3f}, {rms_dy:.3f}) px"
            )
            ax.text(
                0.05,
                0.95,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
                fontsize="small",
            )

            # Add source count in bottom left
            ax.text(
                0.05,
                0.05,
                f"N = {len(df_plot)}",
                transform=ax.transAxes,
                verticalalignment="bottom",
                horizontalalignment="left",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
                fontsize="small",
            )

            fig.tight_layout()
            fig.savefig(save_path, dpi=150, facecolor="white")
            plt.close(fig)

            logger.info(f"Saved WCS vs PSF offset plot: {save_path}")
            logger.info(
                f"WCS vs PSF offset statistics: Median=({med_dx:.3f}, {med_dy:.3f}) px, RMS=({rms_dx:.3f}, {rms_dy:.3f}) px"
            )

        except Exception as e:
            logger.warning(f"WCS vs PSF offset plot failed: {e}")
