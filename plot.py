#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting utilities for subtraction checks, source diagnostics, and light curves.
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from plotting_utils import (
        apply_autophot_mplstyle,
        get_divergent_color,
        get_marker_size,
        get_plot_color,
        get_plot_ext,
        safe_tight_layout,
        PLOT_COLORS,
    )
except ImportError:
    logger.warning("plotting_utils module not found, some plotting features may be limited")
    apply_autophot_mplstyle = lambda: None
    get_divergent_color = None
    get_marker_size = None
    get_plot_color = None
    get_plot_ext = lambda _iy=None: ".png"
    safe_tight_layout = lambda fig=None, **kw: fig.tight_layout(**kw) if fig is not None else None
    PLOT_COLORS = {}

try:
    from lightcurve import (
        _normalize_photometry_columns,
        _time_axis_transform,
        BAND_COLORS,
        canonical_band_label_map_from_filter_series,
        canonical_bands_from_filter_series,
        filter_value_matches_band,
        photometry_filter_series,
    )
except ImportError:
    _normalize_photometry_columns = None
    _time_axis_transform = None
    BAND_COLORS = None
    canonical_band_label_map_from_filter_series = None
    canonical_bands_from_filter_series = None
    filter_value_matches_band = None
    photometry_filter_series = None


def _object_label(otype, name, max_len: int = 20):
    """Label text for a SIMBAD object marker.

    Prefers the object name (``MAIN_ID``, e.g. "NGC 1068" or
    "2MASX J...") and falls back to the object type code
    (``OTYPE_opt``, e.g. "G", "QSO") when no name is available.
    Returns ``None`` when neither is usable -- including a bare "SN*"
    type code with no real name, which carries no information.
    Labels longer than ``max_len`` are truncated as ``first8...last8``.
    """
    label = None
    if isinstance(name, str) and name.strip() and name.strip().lower() != "nan":
        label = name.strip()
    elif (
        isinstance(otype, str)
        and otype.strip()
        and otype.strip().lower() != "nan"
    ):
        if "SN*" in otype:
            return None
        label = otype.strip()
    if label is not None and len(label) > max_len:
        label = f"{label[:8]}...{label[-8:]}"
    return label


class Plot:
    """Diagnostic plotting utilities for AutoPHOT.

    Provides methods for subtraction checks, source overlays, crowding
    diagnostics, light-curve plotting, and WCS-vs-PSF offset visualisation.
    """

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
        kernel_half_width=None,
        diff_decorrelated=None,
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
        diff_decorrelated (ndarray or None): Decorrelated difference image for additional panel (SFFT v1.5.0+). 
                                              If provided, main diff is non-decorrelated (for photometry), 
                                              diff_decorrelated is decorrelated (for detection).
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
            apply_autophot_mplstyle()

            base = os.path.splitext(os.path.basename(self.input_yaml["fpath"]))[0]
            write_dir = os.path.dirname(self.input_yaml["fpath"])
            save_path = os.path.join(
                write_dir, f"Subtraction_Check_{base}{get_plot_ext(self.input_yaml)}"
            )

            zscale = ZScaleInterval()

            images = {"Image": image, "Reference": ref, "Difference": diff}

            if diff_decorrelated is not None:
                images["Decorrelated (Detection)"] = diff_decorrelated

            image_dims = {}

            for key, img_data in images.items():
                finite_frac = np.sum(np.isfinite(img_data)) / img_data.size * 100
                img_h, img_w = img_data.shape
                image_dims[key] = (img_w, img_h)  # (width, height)
                logger.debug(
                    f"subtraction_check: {key} shape={img_data.shape}, "
                    f"finite={finite_frac:.1f}%, "
                    f"range=[{np.nanmin(img_data):.2e}, {np.nanmax(img_data):.2e}]"
                )

            vmins = {}
            vmaxs = {}
            for key, img_data in images.items():
                # Clip extremes so zscale limits are not driven by a few bad pixels.
                valid_data = img_data[np.isfinite(img_data)]
                if len(valid_data) == 0:
                    logger.warning("subtraction_check: %s has no valid data, using fallback vmin/vmax", key)
                    vmins[key] = np.nanmin(img_data)
                    vmaxs[key] = np.nanmax(img_data)
                    continue
                try:
                    lower, upper = np.percentile(valid_data, [0.5, 99.5])
                    cleaned_data = np.clip(img_data, lower, upper)
                    vmin, vmax = zscale.get_limits(cleaned_data)
                    vmins[key] = vmin
                    vmaxs[key] = vmax
                except Exception as e:
                    logger.warning("subtraction_check: zscale failed for %s, using min/max: %s", key, e)
                    vmins[key] = np.nanmin(img_data)
                    vmaxs[key] = np.nanmax(img_data)

            n_images = len(images)
            # Panels use aspect="equal" on the Image dimensions, so size the
            # figure so each gridspec cell matches the image aspect ratio;
            # otherwise equal-aspect shrinks the axes inside wide cells and
            # leaves gaps between panels no wspace setting can remove.
            img_h0, img_w0 = image.shape
            aspect = img_w0 / img_h0
            ax_h = 4.2  # inches of axes height per panel
            left, right, bottom, top = 0.05, 0.98, 0.11, 0.88
            wspace = 0.04
            fig_w = n_images * ax_h * aspect * (1 + wspace) / (right - left)
            fig_h = ax_h / (top - bottom)
            fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
            gs = GridSpec(1, n_images, figure=fig, wspace=wspace)
            axes = [fig.add_subplot(gs[0, i]) for i in range(n_images)]
            plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

            img_height, img_width = image.shape
            margin = 0.05
            inset_axes_list = []
            panel_to_inset = {}

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

            image_titles = list(images.keys())
            # All panels share the first image's shape so display axes are identical.
            if "Image" not in images:
                logger.error("subtraction_check: 'Image' key not found in images dictionary")
                return 0
            ref_width = images["Image"].shape[1]
            ref_height = images["Image"].shape[0]
            for title in image_titles:
                logger.debug("subtraction_check: %s shape=%s", title, images[title].shape)
            for i, (ax, title) in enumerate(zip(axes, image_titles)):
                img_data = images[title]
                # Grayscale keeps colored markers readable.
                cmap = plt.get_cmap(PLOT_COLORS.get('image_cmap', 'gray')).copy()
                cmap.set_bad(color=PLOT_COLORS.get('nan_color', 'magenta'))
                ax.imshow(
                    img_data,
                    origin="lower",
                    aspect="equal",
                    cmap=cmap,
                    vmin=vmins[title],
                    vmax=vmaxs[title],
                )
                ax.set_xlim(0, ref_width)
                ax.set_ylim(0, ref_height)
                ax.set_title(title, fontsize=10, pad=5)
                ax.set_xlabel("X [Pixel]", fontsize=9)
                if i == 0:
                    ax.set_ylabel("Y [Pixel]", fontsize=9)
                else:
                    # Panels share the same y limits; repeat labels add clutter.
                    ax.set_ylabel("")
                    ax.tick_params(axis="y", labelleft=False)

            # Marker box shows the full kernel when known, else the inset extent.
            if kernel_half_width is not None and kernel_half_width > 0:
                square_size = int(kernel_half_width * 2)
            else:
                square_size = int(inset_size * 2)
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

                    x_vals = matching_sources[x_col].values
                    y_vals = matching_sources[y_col].values
                    logger.debug(
                        f"subtraction_check: Using columns ({x_col}, {y_col}) for {len(matching_sources)} sources"
                    )
                    logger.debug(
                        f"subtraction_check: Coordinate ranges: "
                        f"X=[{np.nanmin(x_vals):.1f}, {np.nanmax(x_vals):.1f}], "
                        f"Y=[{np.nanmin(y_vals):.1f}, {np.nanmax(y_vals):.1f}]"
                    )
                    
                    # Panel dims differ per image; look up by name.
                    panel_names = list(images.keys())

                    for panel_idx, ax in enumerate(axes):
                        panel_name = panel_names[panel_idx]
                        panel_width, panel_height = image_dims[panel_name]
                        valid_markers = 0
                        skipped_markers = 0
                        non_finite = 0
                        
                        for idx, (x_pix, y_pix) in enumerate(zip(
                            matching_sources[x_col], matching_sources[y_col]
                        )):
                            if not (np.isfinite(x_pix) and np.isfinite(y_pix)):
                                non_finite += 1
                                skipped_markers += 1
                                continue

                            # Coordinates are already 0-based (converted on ingestion).
                            x_plot = float(x_pix)
                            y_plot = float(y_pix)

                            # Draw all matched sources, even out-of-bounds, so the
                            # full matching result stays visible.
                            rect = patches.Rectangle(
                                (x_plot - half_size, y_plot - half_size),
                                square_size,
                                square_size,
                                linewidth=0.9,
                                edgecolor=PLOT_COLORS.get('matched_box', '#00CFFF'),
                                facecolor="none",
                                alpha=0.9,
                            )
                            ax.add_patch(rect)
                            valid_markers += 1
                        
                        valid_markers_total += valid_markers
                        skipped_markers_total += skipped_markers
                        
                        if skipped_markers > 0:
                            logger.debug(
                                f"subtraction_check: {panel_name} panel - plotted {valid_markers} markers, "
                                f"skipped {skipped_markers} ({non_finite} non-finite). "
                                f"Panel dims: {panel_width}x{panel_height}"
                            )
                    
                    if skipped_markers_total > 0:
                        logger.debug(
                            f"subtraction_check: Total - plotted {valid_markers_total} markers, "
                            f"skipped {skipped_markers_total} across all panels"
                        )

            if masked_sources is not None and len(masked_sources) > 0:
                cross_len = square_size / 4
                skipped_masked = 0
                required_cols = ["x_pix", "y_pix", "OTYPE_opt", "MAIN_ID"]
                missing_cols = [col for col in required_cols if col not in masked_sources.columns]
                if missing_cols:
                    logger.warning("subtraction_check: masked_sources missing columns %s", missing_cols)
                else:
                    for x, y, otype, name in zip(
                        masked_sources["x_pix"],
                        masked_sources["y_pix"],
                        masked_sources["OTYPE_opt"],
                        masked_sources["MAIN_ID"],
                    ):
                        if not (np.isfinite(x) and np.isfinite(y)):
                            skipped_masked += 1
                            continue

                        # masked_sources x_pix arrive 1-based; subtract 1 when positive.
                        x_plot = float(x) - 1 if x > 0 else float(x)
                        y_plot = float(y) - 1 if y > 0 else float(y)

                        if not (0 <= x_plot < img_width and 0 <= y_plot < img_height):
                            skipped_masked += 1
                            continue

                        x = x_plot
                        y = y_plot

                        if isinstance(otype, str) and "SN*" in otype:
                            circle = mpatches.Circle(
                                (x, y),
                                cross_len * 2,
                                edgecolor=PLOT_COLORS.get('target', '#FF0000'),
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
                        _label = _object_label(otype, name)
                        if _label and len(axes) > 0:
                            axes[0].annotate(
                                _label,
                                xy=(x, y),
                                xytext=(x, y + cross_len / 2),
                                ha="center",
                                va="bottom",
                                fontsize=3,
                                color=PLOT_COLORS.get('target', '#FF0000'),
                                zorder=3,
                            )

            # masked_sources are sources excluded from flux calibration.
            if masked_sources is not None and len(masked_sources) > 0:
                cross_len = square_size / 4
                masked_count = 0
                if "x_pix" in masked_sources.columns and "y_pix" in masked_sources.columns:
                    for x, y in zip(masked_sources["x_pix"], masked_sources["y_pix"]):
                        if not (np.isfinite(x) and np.isfinite(y)):
                            continue

                        # masked_sources x_pix arrive 1-based; subtract 1 when positive.
                        x_plot = float(x) - 1 if x > 0 else float(x)
                        y_plot = float(y) - 1 if y > 0 else float(y)

                        if not (0 <= x_plot < img_width and 0 <= y_plot < img_height):
                            continue

                        x = x_plot
                        y = y_plot

                        for ax in axes:
                            ax.plot(
                                [x - cross_len, x + cross_len],
                                [y - cross_len, y + cross_len],
                                color=PLOT_COLORS.get('target', '#FF0000'),
                                lw=0.5,
                                zorder=2,
                            )
                            ax.plot(
                                [x - cross_len, x + cross_len],
                                [y + cross_len, y - cross_len],
                                color=PLOT_COLORS.get('target', '#FF0000'),
                                lw=0.5,
                                zorder=2,
                            )
                        masked_count += 1
                logger.debug("Plotted %s variable sources (masked from flux calibration) as red 'x' markers", masked_count)

            for i, (title, img_data) in enumerate(images.items()):
                ax = axes[i]

                if expected_location and len(expected_location) == 2:
                    try:
                        x, y = map(int, expected_location)
                        x = max(inset_size, min(x, img_width - inset_size))
                        y = max(inset_size, min(y, img_height - inset_size))
                    except (ValueError, TypeError) as e:
                        logger.warning("subtraction_check: invalid expected_location format: %s", e)
                        continue

                    x_side, con_x_main, con_x_inset = get_inset_side(
                        x, inset_size, img_width
                    )
                    y_side, con_y_main, con_y_inset = get_inset_side(
                        y, inset_size, img_height
                    )
                    inset_loc = f'{"upper" if y_side == "high" else "lower"} {"right" if x_side == "high" else "left"}'

                    ax_inset = inset_axes(ax, width="30%", height="30%", loc=inset_loc)
                    cmap = plt.get_cmap(PLOT_COLORS.get('image_cmap', 'gray')).copy()
                    cmap.set_bad(color=PLOT_COLORS.get('nan_color', 'magenta'))
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
                    # Red frame ties the inset to the red zoom rectangle
                    # on the main panel.
                    for spine in ax_inset.spines.values():
                        spine.set_color(PLOT_COLORS.get('target', '#FF0000'))
                        spine.set_linewidth(1.0)
                    inset_axes_list.append(ax_inset)
                    panel_to_inset[i] = ax_inset

                    rect = Rectangle(
                        (x - inset_size, y - inset_size),
                        2 * inset_size,
                        2 * inset_size,
                        linewidth=0.5,
                        edgecolor=PLOT_COLORS.get('target', '#FF0000'),
                        facecolor="none",
                    )
                    ax.add_patch(rect)

                    # Connect matching rectangle/inset corners for each inset location.
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
                                color=PLOT_COLORS.get('target', '#FF0000'),
                                linewidth=0.5,
                            )
                        )

            # Optional mask overlay (skip difference and decorrelated main
            # panels; their insets still get it so the zoomed view flags the
            # masked region around the target).
            if mask is not None:
                red_overlay = colors.ListedColormap(["none", PLOT_COLORS.get('mask_overlay', '#FF0000')])
                for i, ax in enumerate(fig.axes[:-1]):
                    if ax not in inset_axes_list:
                        ax_inset = panel_to_inset.get(i)
                        if ax_inset is not None:
                            ax_inset.imshow(mask, cmap=red_overlay, alpha=0.5, origin="lower")
                        if i == 2 or (n_images == 4 and i == 3):
                            continue
                        ax.imshow(mask, cmap=red_overlay, alpha=0.5, origin="lower")

            if fitted_location and len(fitted_location) == 2:
                radius = aperture_size

                if len(inset_axes_list) >= 3:
                    for ax in inset_axes_list[2:]:
                        circle = mpatches.Circle(
                            fitted_location,
                            edgecolor=PLOT_COLORS.get('target', '#FF0000'),
                            facecolor="none",
                            linewidth=0.5,
                            transform=ax.transData,
                        )
                        ax.add_patch(circle)

                        # Green cross at expected location is disabled (see
                        # commented block below); cross_len kept for it.
                        if expected_location and len(expected_location) == 2:
                            x, y = expected_location
                            cross_len = aperture_size / 2

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


            # Legend for the overlay markers; only list what was actually drawn.
            legend_handles = []
            if matching_sources is not None and len(matching_sources) > 0:
                legend_handles.append(
                    mpatches.Rectangle(
                        (0, 0), 1, 1, facecolor="none",
                        edgecolor=PLOT_COLORS.get('matched_box', '#00CFFF'),
                        linewidth=0.9, label="Matched sources",
                    )
                )
            if mask is not None:
                legend_handles.append(
                    mpatches.Patch(
                        facecolor=PLOT_COLORS.get('mask_overlay', 'magenta'),
                        alpha=0.5, label="Masked pixels",
                    )
                )
            if masked_sources is not None and len(masked_sources) > 0:
                legend_handles.append(
                    mlines.Line2D(
                        [], [], marker="x", linestyle="None", markersize=6,
                        color=PLOT_COLORS.get('target', '#FF0000'),
                        label="Masked variable",
                    )
                )
            if legend_handles:
                fig.legend(
                    handles=legend_handles,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.0),
                    ncol=len(legend_handles),
                    fontsize=8,
                    frameon=False,
                    handlelength=1.4,
                    columnspacing=1.5,
                )

            fig.savefig(
                save_path, dpi=150, bbox_inches="tight", facecolor=PLOT_COLORS.get('figure_facecolor', 'white')
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
        import matplotlib.lines as mlines

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

        apply_autophot_mplstyle()

        base = os.path.splitext(os.path.basename(self.input_yaml["fpath"]))[0]
        write_dir = os.path.dirname(self.input_yaml["fpath"])
        save_path = os.path.join(
            write_dir, f"Crowding_Target_{base}{get_plot_ext(self.input_yaml)}"
        )

        zscale = ZScaleInterval()
        finite = cut[np.isfinite(cut)]
        if finite.size:
            lower, upper = np.percentile(finite, [0.5, 99.5])
            vmin, vmax = zscale.get_limits(np.clip(cut, lower, upper))
        else:
            vmin, vmax = np.nanmin(cut), np.nanmax(cut)

        # Equal-aspect panels shrink inside mismatched figure cells, leaving
        # gaps no wspace can remove. All three panels show the same cutout,
        # so size the figure from cut.shape so each cell fits the image.
        img_h, img_w = cut.shape
        aspect = img_w / img_h
        ax_h = 3.0
        left, right, bottom, top = 0.055, 0.99, 0.10, 0.86
        wspace = 0.05
        fig_w = 3 * ax_h * aspect * (1 + wspace) / (right - left)
        fig_h = ax_h / (top - bottom)

        fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))
        fig.subplots_adjust(
            left=left, right=right, top=top, bottom=bottom, wspace=wspace
        )

        titles = ["Target cutout", "Segmentation", "Neighbor mask"]
        for i, (ax, t) in enumerate(zip(axes, titles)):
            ax.set_title(t, fontsize=8, pad=2)
            ax.set_xlabel("X [Pixel]")
            if i == 0:
                ax.set_ylabel("Y [Pixel]")
            else:
                # Panels share the same y extent; repeat labels add clutter.
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)

        tx = cx - x0
        ty = cy - y0

        # Track which overlays were drawn so the figure legend lists only
        # markers that are actually visible.
        aperture_drawn = False
        seg_drawn = False

        cmap_vir = plt.get_cmap(PLOT_COLORS.get('image_cmap_alt', 'viridis')).copy()
        cmap_vir.set_bad(color=PLOT_COLORS.get('nan_color', 'magenta'))
        axes[0].imshow(
            cut, origin="lower", cmap=cmap_vir, vmin=vmin, vmax=vmax
        )
        axes[0].axvline(tx, color=PLOT_COLORS.get('reference', '#0072B2'), lw=0.6, alpha=0.9)
        axes[0].axhline(ty, color=PLOT_COLORS.get('reference', '#0072B2'), lw=0.6, alpha=0.9)
        if (
            aperture_radius is not None
            and np.isfinite(aperture_radius)
            and aperture_radius > 0
        ):
            axes[0].add_patch(
                mpatches.Circle(
                    (tx, ty),
                    float(aperture_radius),
                        edgecolor=PLOT_COLORS.get('reference', '#0072B2'),
                    facecolor="none",
                    lw=0.8,
                )
            )
            aperture_drawn = True

        axes[1].imshow(
            cut, origin="lower", cmap=cmap_vir, vmin=vmin, vmax=vmax
        )
        levels = np.unique(seg)
        levels = levels[levels > 0]
        if levels.size:
            axes[1].contour(
                seg,
                levels=levels,
                colors=PLOT_COLORS.get('segmentation', '#00AA00'),
                linewidths=0.4,
                alpha=0.9,
            )
            seg_drawn = True
        axes[1].axvline(tx, color=PLOT_COLORS.get('reference', '#0072B2'), lw=0.6, alpha=0.9)
        axes[1].axhline(ty, color=PLOT_COLORS.get('reference', '#0072B2'), lw=0.6, alpha=0.9)

        axes[2].imshow(
            cut, origin="lower", cmap=cmap_vir, vmin=vmin, vmax=vmax
        )
        overlay = colors.ListedColormap(["none", PLOT_COLORS.get('mask_overlay', '#FF0000')])
        axes[2].imshow(nmask.astype(int), origin="lower", cmap=overlay, alpha=0.35)
        axes[2].axvline(tx, color=PLOT_COLORS.get('reference', '#0072B2'), lw=0.6, alpha=0.9)
        axes[2].axhline(ty, color=PLOT_COLORS.get('reference', '#0072B2'), lw=0.6, alpha=0.9)

        handles = [
            mlines.Line2D(
                [0], [0],
                color=PLOT_COLORS.get('reference', '#0072B2'),
                lw=0.8,
                label="Target",
            )
        ]
        if aperture_drawn:
            handles.append(
                mlines.Line2D(
                    [0], [0],
                    marker="o",
                    color=PLOT_COLORS.get('reference', '#0072B2'),
                    markerfacecolor="none",
                    markersize=6,
                    linestyle="None",
                    label="Aperture",
                )
            )
        if seg_drawn:
            handles.append(
                mlines.Line2D(
                    [0], [0],
                    color=PLOT_COLORS.get('segmentation', '#00AA00'),
                    lw=0.8,
                    label="Segmentation",
                )
            )
        if np.any(nmask):
            handles.append(
                mpatches.Patch(
                    facecolor=PLOT_COLORS.get('mask_overlay', '#FF0000'),
                    edgecolor="none",
                    alpha=0.35,
                    label="Neighbor mask",
                )
            )
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(handles),
            fontsize=8,
            frameon=False,
        )

        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PLOT_COLORS.get('figure_facecolor', 'white'))
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

            apply_autophot_mplstyle()

            fpath = self.input_yaml["fpath"]
            base = os.path.basename(fpath)
            write_dir = os.path.dirname(fpath)
            base = os.path.splitext(base)[0]

            phot_cfg = self.input_yaml.get("photometry") or {}
            ap_size_fwhm = phot_cfg.get("aperture_size", 1.7)
            radius = float(ap_size_fwhm) * float(self.input_yaml["fwhm"])
            scale = self.input_yaml["scale"]

            plt.ioff()
            fig = plt.figure(figsize=set_size(540, 1))

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
                    ax1.coords[0].set_ticklabel_position('b')
                    ax1.coords[0].set_axislabel_position('b')
                    ax1.coords[0].set_axislabel("RA", fontsize=6, minpad=0.3)
                    ax1.coords[0].set_major_formatter('hh:mm')
                    
                    ax1.coords[1].set_ticklabel_position('l')
                    ax1.coords[1].set_axislabel_position('l')
                    ax1.coords[1].set_axislabel("Dec", fontsize=6, minpad=0.3)
                    ax1.coords[1].set_major_formatter('dd:mm')
                    
                    ax1.set_xlabel("")
                    ax1.set_ylabel("")

                    ax1.coords[0].set_ticks_position('bt')
                    ax1.coords[1].set_ticks_position('lr')
                    ax1.coords[0].set_ticklabel_position('b')
                    ax1.coords[1].set_ticklabel_position('l')

                    ax1.coords.grid(False)

                    # tight_layout can fail on WCSAxes.
                    skip_tight_layout = True
                    
                    logger.info("Source check plot: using RA/Dec WCS axes")
                else:
                    logger.debug("Source check plot: WCS has no celestial component, using pixel axes")
                    raise ValueError("WCS has no celestial component")
            except Exception as e:
                logger.debug("Source check plot: WCS axes failed (%s), using pixel coordinates only", e)
                # Drop any partially-created WCS axes before re-adding a plain one.
                fig.clf()
                ax1 = fig.add_subplot(111)
                ax1.set_xlabel("X (pixels)", fontsize=6)
                ax1.set_ylabel("Y (pixels)", fontsize=6)
                skip_tight_layout = False

            norm = ImageNormalize(
                image, interval=ZScaleInterval(), stretch=LinearStretch()
            )
            cmap = plt.get_cmap(PLOT_COLORS.get('image_cmap', 'gray'))
            cmap.set_bad(color=PLOT_COLORS.get('nan_color', 'magenta'))
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

            edge_color = PLOT_COLORS.get('target', '#FF0000')
            circle = Circle(
                (self.input_yaml["target_x_pix"], self.input_yaml["target_y_pix"]),
                radius,
                edgecolor=edge_color,
                facecolor="none",
                zorder=4,
                lw=1.0,
            )
            ax1.add_patch(circle)
            # Target name text removed to avoid overlapping with other annotations

            if psfSources is not None and len(psfSources) > 0:
                from matplotlib.patches import RegularPolygon
                fwhm = float(self.input_yaml.get("fwhm", 5.0))
                hex_radius = 2.0 * fwhm  # hexagon width across flats = 4*FWHM
                for x, y in zip(psfSources["x_pix"], psfSources["y_pix"]):
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue
                    hexagon = RegularPolygon(
                        (x, y),
                        numVertices=6,
                        radius=hex_radius,
                        orientation=np.pi/6,  # Point up
                        edgecolor=PLOT_COLORS.get('psf', '#00AA00'),
                        facecolor="none",
                        linewidth=0.5,
                        label="PSF sources" if x == psfSources["x_pix"].iloc[0] else None,
                        zorder=1,
                    )
                    ax1.add_patch(hexagon)

            if catalogSources is not None:
                fwhm = float(self.input_yaml.get("fwhm", 5.0))
                square_size = 4.0 * fwhm
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
                        edgecolor=PLOT_COLORS.get('reference', '#0072B2'),
                        facecolor="none",
                        label="Reference Sources",
                        zorder=1,
                        lw=0.5,
                    )
                    ax1.add_patch(square)

            if FWHMSources is not None and "fwhm" in FWHMSources:

                if len(FWHMSources) > 0:
                    fwhm_values = np.array(FWHMSources["fwhm"])
                    norm_fwhm = Normalize(
                        vmin=np.nanmin(fwhm_values), vmax=np.nanmax(fwhm_values)
                    )
                    # Orange colormap for FWHM scaling (distinct from red target and blue catalog).
                    cmap = plt.get_cmap(PLOT_COLORS.get('fwhm_sources', 'Oranges'))

                    sm = ScalarMappable(norm=norm_fwhm, cmap=cmap)
                    sm.set_array([])

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

                    cbar = fig.colorbar(sm, ax=ax1, pad=0.02, aspect=40)
                    cbar.set_label("FWHM (pixels)", fontsize=7)
                    cbar.ax.tick_params(labelsize=6)

            if variable_sources is not None:

                if len(variable_sources) > 0:

                    cross_len = scale / 4
                    _gold = PLOT_COLORS.get('variable', '#FFD700')

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

                        ax1.plot(
                            [x - cross_len, x + cross_len],
                            [y - cross_len, y + cross_len],
                            color=_gold,
                            lw=1.0,
                            zorder=3,
                        )
                        ax1.plot(
                            [x - cross_len, x + cross_len],
                            [y + cross_len, y - cross_len],
                            color=_gold,
                            lw=1.0,
                            zorder=3,
                        )

                        # Prefer the source name, fall back to otype. NaN or
                        # non-string labels get no annotation.
                        _label = _object_label(otype, name)

                        if _label:
                            ax1.annotate(
                                _label,
                                xy=(x, y),
                                xytext=(
                                    x,
                                    y + cross_len * 2,
                                ),
                                ha="center",
                                va="bottom",
                                fontsize=4,
                                color=_gold,
                                zorder=4,
                            )

            # Residual vectors (catalog -> detected) show non-uniform
            # astrometric residuals across the field.
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
                        # NOTE: distortion contour overlays were removed; only
                        # per-source residual vectors and the RMS text remain.
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
                            color=PLOT_COLORS.get('positive', '#FF8C00') if PLOT_COLORS else (get_divergent_color('positive') if get_divergent_color else '#FF8C00'),
                            alpha=0.65,
                            width=0.0020,
                            zorder=5,
                        )
                        distortion_rms = float(np.sqrt(np.mean(u * u + v * v)))
                        distortion_rms_text = f"Distortion residual RMS: {distortion_rms:.2f} px"
                        
            except Exception as e:
                logger.debug("Distortion vector overlay skipped: %s", e)

            if mask is not None:
                from matplotlib import colors

                mask_cmap = colors.ListedColormap(["none", PLOT_COLORS.get('mask_overlay_alt', 'white')])
                ax1.imshow(mask, cmap=mask_cmap, alpha=1.0, origin="lower")

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

            ax1.set_xlabel("X [Pixel]")
            ax1.set_ylabel("Y [Pixel]")
            ax1.set_xlim(0, image.shape[1])
            ax1.set_ylim(0, image.shape[0])
    
            handles, labels = ax1.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            leg = ax1.legend(
                by_label.values(),
                by_label.keys(),
                loc="lower center",
                bbox_to_anchor=(0.5, 1.0),
                frameon=False,
                fontsize=8,
                handlelength=1.5,
                handletextpad=0.5,
                ncol=3,
            )

            # Leave headroom at top for the legend.
            if not skip_tight_layout:
                safe_tight_layout(fig, rect=[0, 0, 1, 0.92])
            ax1.set_aspect("equal", adjustable="box")

            _ext = get_plot_ext(self.input_yaml)
            if not subtracted:
                save_loc = os.path.join(
                    write_dir, f"Source_Check_{base}{_ext}"
                )
            else:
                save_loc = os.path.join(
                    write_dir, f"Source_Check_Subtracted_{base}{_ext}"
                )

            fig.savefig(
                save_loc, dpi=150, bbox_inches="tight", bbox_extra_artists=[leg],
                facecolor=PLOT_COLORS.get('figure_facecolor', 'white')
            )
            plt.close(fig)

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
        adaptive_snr_selection=False,
        input_yaml=None,
        max_plot_err=0.5,
        chi2_marginal_threshold=5.0,
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
        # Switch to an interactive backend before any pyplot figure creation
        # so that plt.show() actually displays the window when show=True.
        # plt.switch_backend() works after pyplot is already imported, unlike
        # matplotlib.use() which silently fails once pyplot is loaded.
        if show:
            import matplotlib
            import matplotlib.pyplot as _plt_check

            current_backend = str(_plt_check.get_backend()).lower()
            if "agg" in current_backend:
                for backend in ("QtAgg", "TkAgg"):
                    try:
                        matplotlib.use(backend, force=True)
                        break
                    except Exception:
                        continue

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from functions import set_size

        # Use the shared per-band palette from lightcurve.py (BAND_COLORS).
        # If BAND_COLORS is unavailable, fall back to a minimal built-in dict.
        if isinstance(BAND_COLORS, dict) and BAND_COLORS:
            palette = dict(BAND_COLORS)
        else:
            palette = {
                "u": "dodgerblue", "g": "g", "r": "r", "i": "goldenrod",
                "z": "k", "y": "0.5", "w": "firebrick", "Y": "0.5",
                "U": "slateblue", "B": "b", "V": "yellowgreen", "R": "crimson",
                "I": "chocolate", "G": "salmon", "E": "salmon",
                "J": "darkred", "H": "orangered", "K": "saddlebrown",
                "S": "mediumorchid", "D": "purple", "A": "midnightblue",
                "F": "#8E4585", "N": "#CC79A7", "o": "darkorange",
                "c": "#17A2B8", "W": "forestgreen", "Q": "peru",
            }

        # Maintains order from blue to red effective wavelength
        bandlist = "FSDNAuUBgcVwrRoGEiIzyYJHKWQ"

        markers = ["x", "o", "^", "s", "D"]
        marker_iterator = iter(markers)

        data = pd.read_csv(output_file)
        if _normalize_photometry_columns is not None:
            data = _normalize_photometry_columns(data)
        
        adaptive_limit_col = None
        if adaptive_snr_selection and input_yaml:
            lim_cfg = input_yaml.get("limiting_magnitude") or {}
            if lim_cfg.get("adaptive_snr_selection", False):
                all_snr_values = []
                for col in data.columns:
                    if 'snr' in col.lower():
                        try:
                            snr_vals = pd.to_numeric(data[col], errors='coerce')
                            all_snr_values.extend(snr_vals.dropna().tolist())
                        except Exception:
                            continue
                
                if all_snr_values:
                    median_snr = np.median(all_snr_values)
                    lim_cfg = input_yaml.get("limiting_magnitude") or {}
                    snr_thresholds = lim_cfg.get("snr_thresholds", [3.0, 5.0])

                    # Median S/N < 3 -> show the 5sigma limit, else 3sigma.
                    if median_snr < 3.0 and len(snr_thresholds) >= 2:
                        higher_threshold = sorted(snr_thresholds)[1]
                        adaptive_limit_col = f'Limit_{higher_threshold:.1f}S2N'.replace('.', 'p')
                        if adaptive_limit_col not in data.columns:
                            adaptive_limit_col = "Limit_5p0S2N"
                        logger.info("Adaptive S/N selection: median S/N=%.2f < 3, using %ssigma limiting magnitude", median_snr, higher_threshold)
                    else:
                        lower_threshold = sorted(snr_thresholds)[0]
                        adaptive_limit_col = f'Limit_{lower_threshold:.1f}S2N'.replace('.', 'p')
                        if adaptive_limit_col not in data.columns:
                            adaptive_limit_col = "Limit_3p0S2N"
                        logger.info("Adaptive S/N selection: median S/N=%.2f >= 3, using %ssigma limiting magnitude", median_snr, lower_threshold)
        if data.columns.duplicated().any():
            data = data.loc[:, ~data.columns.duplicated()].copy()

        # Intra-night data (< 1 day span): switch to minutes/hours since the
        # first observation instead of raw MJD (shared with lightcurve.py).
        if _time_axis_transform is not None and "mjd" in data.columns:
            x_transform, lc_xlabel, subday_unit = _time_axis_transform(
                data["mjd"].values, reference_epoch
            )
        else:
            x_transform = lambda m: pd.to_numeric(m, errors="coerce") - reference_epoch
            lc_xlabel = None
            subday_unit = None

        filter_series = (
            photometry_filter_series(data) if photometry_filter_series else None
        )
        method_low = str(method).strip().lower()
        method_u = str(method).strip().upper()
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

        apply_autophot_mplstyle()

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
        plotted_marginal_chi2 = False
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

            # Long-form CSV: assign rows to this band (gp->g, Sloan_g->g, etc.).
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
                if max_plot_err is not None and max_plot_err > 0:
                    err_vals = pd.to_numeric(detects[err_col], errors="coerce")
                    good_err = err_vals.notna() & (err_vals <= max_plot_err)
                    n_removed = int((~good_err).sum())
                    if n_removed > 0:
                        removed_mjds = detects.loc[~good_err, "mjd"].tolist()
                        logger.warning(
                            "plot_lightcurve: %d detection(s) in band %s excluded from plot "
                            "due to magnitude error > %.2f mag. MJDs: %s",
                            n_removed, str(b), max_plot_err,
                            ", ".join(f"{m:.5f}" for m in removed_mjds),
                        )
                        detects = detects[good_err].copy()

            if not detects.empty:
                x_det = x_transform(detects["mjd"])
                leg_label = (
                    label_map.get(str(b).strip().lower(), str(b))
                    if use_filter_bands
                    else str(b)
                )
                # Split detections by chi2 quality: marginal detections
                # (reduced_chi2 > threshold) get white face + faded edge.
                _band_c = _color_for_band(b)
                chi2_marginal_enabled = (
                    chi2_marginal_threshold is not None
                    and float(chi2_marginal_threshold) > 0
                    and "reduced_chi2" in detects.columns
                )
                if chi2_marginal_enabled:
                    _chi2_vals = pd.to_numeric(
                        detects["reduced_chi2"], errors="coerce"
                    )
                    marginal_mask = _chi2_vals > float(chi2_marginal_threshold)
                else:
                    marginal_mask = pd.Series(False, index=detects.index)

                good_detects = detects[~marginal_mask]
                marginal_detects = detects[marginal_mask]

                if not good_detects.empty:
                    ax1.errorbar(
                        x_det[~marginal_mask],
                        good_detects[mag_col],
                        yerr=good_detects[err_col],
                        c=_band_c,
                        ls="",
                        capsize=1.5,
                        elinewidth=0.5,
                        marker=marker,
                        label=leg_label,
                    )
                if not marginal_detects.empty:
                    plotted_marginal_chi2 = True
                    ax1.errorbar(
                        x_det[marginal_mask],
                        marginal_detects[mag_col],
                        yerr=marginal_detects[err_col],
                        c=_band_c,
                        ls="",
                        capsize=1.5,
                        elinewidth=0.5,
                        marker=marker,
                        markerfacecolor="white",
                        markeredgewidth=0.8,
                        alpha=0.5,
                        label=leg_label if good_detects.empty else "",
                    )
                if ls:
                    sorted_detects = detects.sort_values("mjd")
                    x_line = x_transform(sorted_detects["mjd"])
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
                # Upper limits sit at the limiting magnitude (fainter = non-detection).
                # Column priority: adaptive S/N selection, then per-threshold
                # limiting_mag_<n>s2n, then the standard columns.
                if adaptive_limit_col and adaptive_limit_col in nondetects.columns:
                    y_lim = nondetects[adaptive_limit_col]
                elif f"limiting_mag_{snr_limit:.0f}s2n" in nondetects.columns and np.any(
                    np.isfinite(nondetects[f"limiting_mag_{snr_limit:.0f}s2n"])
                ):
                    y_lim = nondetects[f"limiting_mag_{snr_limit:.0f}s2n"]
                elif "limiting_inst_mag" in nondetects.columns and np.any(
                    np.isfinite(nondetects["limiting_inst_mag"])
                ):
                    y_lim = nondetects["limiting_inst_mag"]
                elif "lmag" in nondetects.columns and np.any(np.isfinite(nondetects["lmag"])):
                    # Backwards compatibility when plotting older output CSVs.
                    y_lim = nondetects["lmag"]
                else:
                    y_lim = nondetects[mag_col]
                x_nd = x_transform(nondetects["mjd"])
                ax1.errorbar(
                    x_nd,
                    y_lim,
                    c=_color_for_band(b),
                    ls="",
                    marker="v",
                    markersize=5,
                    capsize=5 / 4,
                    elinewidth=0.5,
                    markerfacecolor="none",
                    markeredgewidth=0.5,
                    alpha=0.85,
                    zorder=0,
                )

        ax1.set_ylim(ax1.get_ylim())
        if redshift != 0.0:

            ax11 = ax1.twinx()
            ax11.set_xlim(ax1.get_xlim())

            ax11.set_ylim(ax1.get_ylim() - dm)
            ax11.set_ylabel("Absolute Magnitude [mag]")

        ax1.set_ylabel("Apparent Magnitude [mag]")

        if lc_xlabel is not None:
            ax1.set_xlabel(lc_xlabel)
        elif reference_epoch != 0.0:
            ax1.set_xlabel(rf"Days since {reference_epoch}")
        else:
            ax1.set_xlabel("Modified Julian Date")

        if subday_unit == "min":
            from matplotlib.ticker import MaxNLocator as _MaxNLocator

            ax1.xaxis.set_major_locator(_MaxNLocator(integer=True, nbins=8))

        if show_details:
            text = rf"# detect {num_detect}" + "\n" + rf"# nondetect {num_nondetect}"
            plt.text(
                0.02,
                0.95,
                text,
                transform=ax1.transAxes,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(facecolor=PLOT_COLORS.get('stats_bbox', 'white'), alpha=0.8),
            )

        handles, labels = ax1.get_legend_handles_labels()
        if plotted_marginal_chi2:
            from matplotlib.lines import Line2D as _L2
            _marg_label = f"Marginal (high chi^2, >{chi2_marginal_threshold:g})"
            if _marg_label not in labels:
                _marg_handle = _L2(
                    [0], [0],
                    color="black",
                    marker="o",
                    markersize=5,
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                    alpha=0.5,
                    ls="",
                    label=_marg_label,
                )
                handles.append(_marg_handle)
                labels.append(_marg_label)
        by_label = dict(zip(labels, handles))

        _n_entries = len(by_label.values())
        ncols = 3 if _n_entries >= 8 else (2 if _n_entries >= 5 else 1)
        ax1.legend(
            by_label.values(),
            by_label.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            frameon=False,
            fontsize=8,
            handlelength=1.5,
            handletextpad=0.5,
            ncol=ncols,
        )
        safe_tight_layout(fig, rect=[0, 0, 1, 0.94])

        if bool(show):
            plt.show()
        else:
            plt.close(fig)

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
            apply_autophot_mplstyle()

            base = os.path.splitext(os.path.basename(self.input_yaml["fpath"]))[0]
            write_dir = os.path.dirname(self.input_yaml["fpath"])
            save_path = os.path.join(
                write_dir, f"WCS_vs_PSF_Offset_{base}{get_plot_ext(self.input_yaml)}"
            )

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

            df["dx"] = df["x_fit"] - df["x_pix"]
            df["dy"] = df["y_fit"] - df["y_pix"]

            # Only PSF fit errors are used; catalog position errors are not
            # propagated into the error bars.
            dx_err = df.get("x_fit_err", np.nan).copy()
            dy_err = df.get("y_fit_err", np.nan).copy()

            finite_err = dx_err.notna() & dy_err.notna()
            has_errors = finite_err.any()
            
            if has_errors:
                df_plot = df[finite_err].copy()
                # Position errors > FWHM flag bad fits; they would dominate the plot.
                fwhm = float(self.input_yaml.get("fwhm", 3.0))
                reasonable_err = (
                    (df_plot["x_fit_err"] <= fwhm) & (df_plot["y_fit_err"] <= fwhm)
                )
                n_before = len(df_plot)
                df_plot = df_plot[reasonable_err].copy()
                n_excluded = n_before - len(df_plot)
                if n_excluded > 0:
                    logger.debug(
                        "WCS vs PSF offset plot: excluded %d/%d sources "
                        "with position errors > FWHM (%.1f px)",
                        n_excluded, n_before, fwhm,
                    )
            else:
                df_plot = df.copy()
                logger.info("WCS vs PSF offset plot: no error columns available, plotting without error bars")

            width_pt = 5.5 * 72.27
            aspect = 1.0
            fig, ax = plt.subplots(figsize=set_size(width_pt, aspect=aspect))

            # Color points by distance from target.
            _target_x = self.input_yaml.get("target_x_pix")
            _target_y = self.input_yaml.get("target_y_pix")
            if _target_x is not None and _target_y is not None and np.isfinite(_target_x) and np.isfinite(_target_y):
                _dist_from_target = np.sqrt(
                    (df_plot["x_pix"].values - float(_target_x)) ** 2
                    + (df_plot["y_pix"].values - float(_target_y)) ** 2
                )
            else:
                _dist_from_target = None

            # Error bars drawn first so they sit behind the points.
            if has_errors:
                ax.errorbar(
                    df_plot["dx"],
                    df_plot["dy"],
                    xerr=df_plot["x_fit_err"],
                    yerr=df_plot["y_fit_err"],
                    fmt="none",
                    ecolor=PLOT_COLORS.get('error_bar', '#999999'),
                    elinewidth=0.5,
                    capsize=1.5,
                    alpha=0.5,
                    zorder=1,
                )

            _sc_obj = None
            if _dist_from_target is not None:
                _cmap = plt.get_cmap(PLOT_COLORS.get('scatter_cmap', 'viridis'))
                _sc_obj = ax.scatter(
                    df_plot["dx"],
                    df_plot["dy"],
                    s=12,
                    c=_dist_from_target,
                    cmap=_cmap,
                    marker="o",
                    edgecolor="none",
                    alpha=0.8,
                    zorder=3,
                )
            else:
                ax.scatter(
                    df_plot["dx"],
                    df_plot["dy"],
                    s=12,
                    marker="o",
                    facecolor=PLOT_COLORS.get('scatter_primary', '#0072B2'),
                    edgecolor="none",
                    alpha=0.7,
                    zorder=3,
                )

            med_dx = np.nanmedian(df_plot["dx"])
            med_dy = np.nanmedian(df_plot["dy"])

            rms_dx = np.sqrt(np.nanmean(df_plot["dx"]**2))
            rms_dy = np.sqrt(np.nanmean(df_plot["dy"]**2))

            # Get pixel scale before setting limits so we can enforce a
            # minimum range of max(1 px, 1 arcsec) whichever is larger.
            pixel_scale = None
            if "pixel_scale" in self.input_yaml:
                pixel_scale = float(self.input_yaml["pixel_scale"])
            elif imageWCS is not None:
                try:
                    from astropy.wcs import utils as wcs_utils
                    pixel_scale = wcs_utils.proj_plane_pixel_scales(imageWCS)[0] * 3600
                except Exception:
                    pass

            # Minimum half-range: 1 pixel or 1 arcsec (in pixels), whichever is larger
            _min_lim = 1.0
            if pixel_scale is not None and pixel_scale > 0:
                _min_lim = max(_min_lim, 1.0 / pixel_scale)

            # Symmetric square axes with (0,0) at centre
            if has_errors:
                _lim = max(
                    np.nanmax(np.abs(df_plot["dx"] + df_plot["x_fit_err"])),
                    np.nanmax(np.abs(df_plot["dy"] + df_plot["y_fit_err"])),
                    _min_lim,
                ) * 1.1
            else:
                _lim = max(
                    np.nanmax(np.abs(df_plot["dx"])),
                    np.nanmax(np.abs(df_plot["dy"])),
                    _min_lim,
                ) * 1.1
            ax.set_xlim(-_lim, _lim)
            ax.set_ylim(-_lim, _lim)
            ax.axhline(0, color=PLOT_COLORS.get('zero_line', '#FF0000'), lw=0.8, ls="--", alpha=0.5, zorder=1)
            ax.axvline(0, color=PLOT_COLORS.get('zero_line', '#FF0000'), lw=0.8, ls="--", alpha=0.5, zorder=1)

            ax.set_xlabel(r"$\Delta x = x_{\mathrm{PSF}} - x_{\mathrm{WCS}}$ [px]")
            ax.set_ylabel(r"$\Delta y = y_{\mathrm{PSF}} - y_{\mathrm{WCS}}$ [px]")

            if pixel_scale is not None and pixel_scale > 0:
                # Twin axes show the same offsets in arcsec.
                ax_top = ax.twiny()
                ax_right = ax.twinx()
                # set_aspect is incompatible with shared/twin axes; the symmetric
                # +/-_lim xlim/ylim already enforces a square data region.

                ax_top.set_xlim(ax.get_xlim())
                ax_right.set_ylim(ax.get_ylim())

                x_lim_arcsec = np.array(ax.get_xlim()) * pixel_scale
                y_lim_arcsec = np.array(ax.get_ylim()) * pixel_scale

                ax_top.set_xticks(ax.get_xticks())
                ax_top.set_xticklabels([f"{x*pixel_scale:.2f}" for x in ax.get_xticks()])
                ax_top.set_xlabel(r"$\Delta$RA [arcsec]", fontsize="small")

                ax_right.set_yticks(ax.get_yticks())
                ax_right.set_yticklabels([f"{y*pixel_scale:.2f}" for y in ax.get_yticks()])
                ax_right.set_ylabel(r"$\Delta$Dec [arcsec]", fontsize="small")

                ax_top.tick_params(axis="x", which="both", labeltop=True, labelbottom=False)
                ax_right.tick_params(axis="y", which="both", labelright=True, labelleft=False)
            else:
                # No twin axes - safe to enforce equal aspect.
                ax.set_aspect("equal", adjustable="box")

            # --- Colorbar (manually positioned to avoid twin axis overlap) ---
            if _sc_obj is not None:
                _cax = fig.add_axes([0.88, 0.12, 0.025, 0.80])
                cbar = fig.colorbar(_sc_obj, cax=_cax)
                cbar.set_label("Distance from target [px]", fontsize="small")
                cbar.ax.tick_params(labelsize="x-small")

            # ax.set_title(f"WCS vs PSF Position Offset (N={len(df_plot)})")
            # ax.legend(loc="upper right", fontsize="small", framealpha=0.9)
            ax.grid(True, ls="-", alpha=0.25, zorder=0)

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
                bbox=dict(facecolor=PLOT_COLORS.get('stats_bbox', 'white'), alpha=0.75, edgecolor="none"),
                fontsize="small",
            )

            ax.text(
                0.05,
                0.05,
                f"N = {len(df_plot)}",
                transform=ax.transAxes,
                verticalalignment="bottom",
                horizontalalignment="left",
                bbox=dict(facecolor=PLOT_COLORS.get('stats_bbox', 'white'), alpha=0.75, edgecolor="none"),
                fontsize="small",
            )

            # Right margin: only reserve space when a colorbar or the
            # arcsec twin-axis labels actually need it.
            _has_twin = pixel_scale is not None and pixel_scale > 0
            _right = 0.78 if _sc_obj is not None else (0.86 if _has_twin else 0.95)
            fig.subplots_adjust(left=0.12, right=_right, top=0.92, bottom=0.12)

            fig.savefig(save_path, dpi=150, facecolor=PLOT_COLORS.get('figure_facecolor', 'white'))
            plt.close(fig)

            logger.debug("Saved WCS vs PSF offset plot: %s", save_path)
            logger.info(
                f"WCS vs PSF offset:\tmedian=({med_dx:.3f}, {med_dy:.3f}) px, "
                f"RMS=({rms_dx:.3f}, {rms_dy:.3f}) px"
            )

        except Exception as e:
            logger.warning("WCS vs PSF offset plot failed: %s", e)

    def plot_alignment_offset(
        self,
        sci_fpath: str,
        template_fpath: str,
        match_radius_arcsec: float = 2.0,
    ):
        """Diagnostic plot: dx vs dy between matched sources in aligned images.

        Runs SExtractor on both the aligned science and template images,
        cross-matches detected sources by RA/Dec, and compares their pixel
        positions.  In a perfectly aligned image pair every matched source
        should have dx ~ 0, dy ~ 0.

        Parameters
        ----------
        sci_fpath : str
            Path to the aligned science FITS image.
        template_fpath : str
            Path to the aligned template FITS image.
        match_radius_arcsec : float
            Maximum sky separation for cross-matching (default 2 arcsec).
        """
        import matplotlib.pyplot as plt
        from functions import set_size
        import numpy as np
        from astropy.io import fits
        from astropy.wcs import WCS
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        try:
            apply_autophot_mplstyle()

            base = os.path.splitext(os.path.basename(self.input_yaml["fpath"]))[0]
            write_dir = os.path.dirname(self.input_yaml["fpath"])
            save_path = os.path.join(
                write_dir, f"Alignment_Offset_{base}{get_plot_ext(self.input_yaml)}"
            )

            # --- Detect sources in both aligned images using SExtractor -----
            # SExtractor is used consistently across the entire pipeline for
            # source detection, avoiding centroiding systematics between
            # different detectors.
            from templates import _detect_sextractor_sources

            fwhm_pix = float(self.input_yaml.get("fwhm", 3.0))

            sci_xy, _, _, sci_errx, sci_erry = _detect_sextractor_sources(
                sci_fpath, input_yaml=self.input_yaml, fwhm_pix=fwhm_pix,
                thresh=5.0, fwhm_min=1.5, return_errors=True,
            )
            ref_xy, _, _, ref_errx, ref_erry = _detect_sextractor_sources(
                template_fpath, input_yaml=self.input_yaml, fwhm_pix=fwhm_pix,
                thresh=5.0, fwhm_min=1.5, return_errors=True,
            )

            if sci_xy is None or ref_xy is None:
                logger.warning(
                    "Alignment offset plot: SExtractor returned no sources "
                    "for one or both images."
                )
                return

            if len(sci_xy) < 3 or len(ref_xy) < 3:
                logger.warning(
                    "Alignment offset plot: too few sources detected "
                    "(sci=%d, ref=%d); need >= 3 each.",
                    len(sci_xy), len(ref_xy),
                )
                return

            sci_header = fits.getheader(sci_fpath)
            ref_header = fits.getheader(template_fpath)
            sci_wcs = WCS(sci_header, naxis=2)
            ref_wcs = WCS(ref_header, naxis=2)

            # SExtractor centroids are 0-based; errors come from its error
            # ellipse decomposition.
            sci_x, sci_y = sci_xy[:, 0], sci_xy[:, 1]
            ref_x, ref_y = ref_xy[:, 0], ref_xy[:, 1]
            sci_errx = np.asarray(sci_errx, float)
            sci_erry = np.asarray(sci_erry, float)
            ref_errx = np.asarray(ref_errx, float)
            ref_erry = np.asarray(ref_erry, float)

            from scipy.spatial import cKDTree

            sci_xy = np.column_stack((sci_x, sci_y))
            ref_xy = np.column_stack((ref_x, ref_y))
            tree_ref = cKDTree(ref_xy)
            tree_sci = cKDTree(sci_xy)
            distance, idx_ref = tree_ref.query(sci_xy, k=1)
            _, idx_sci = tree_sci.query(ref_xy, k=1)
            idx = np.arange(len(sci_xy), dtype=int)
            fwhm = float(self.input_yaml.get("fwhm", 3.0))
            max_pixel_separation = max(2.0 * fwhm, 1.0)
            matched = (idx_sci[idx_ref] == idx) & np.isfinite(distance) & (
                distance <= max_pixel_separation
            )

            dx_all = sci_x[matched] - ref_x[idx_ref[matched]]
            dy_all = sci_y[matched] - ref_y[idx_ref[matched]]
            dx_err_all = np.sqrt(
                sci_errx[matched] ** 2 + ref_errx[idx_ref[matched]] ** 2
            )
            dy_err_all = np.sqrt(
                sci_erry[matched] ** 2 + ref_erry[idx_ref[matched]] ** 2
            )
            sci_x_all = sci_x[matched]
            sci_y_all = sci_y[matched]

            n_matched = len(dx_all)
            if n_matched < 2:
                logger.warning(
                    "Alignment offset plot: only %d mutual pixel matches within %.1f px; skipping plot.",
                    n_matched, max_pixel_separation,
                )
                return

            logger.debug(
                "Alignment offset plot: %d mutual pixel matches within %.1f px",
                n_matched, max_pixel_separation,
            )

            # Position errors > FWHM flag blends/edge detections; they would
            # dominate the plot.
            fwhm = float(self.input_yaml.get("fwhm", 3.0))
            reasonable_err = (dx_err_all <= fwhm) & (dy_err_all <= fwhm)
            n_before = len(dx_all)
            dx_all = dx_all[reasonable_err]
            dy_all = dy_all[reasonable_err]
            dx_err_all = dx_err_all[reasonable_err]
            dy_err_all = dy_err_all[reasonable_err]
            sci_x_all = sci_x_all[reasonable_err]
            sci_y_all = sci_y_all[reasonable_err]
            n_excluded = n_before - len(dx_all)
            if n_excluded > 0:
                logger.debug(
                    f"Alignment offset plot: excluded {n_excluded}/{n_before} sources "
                    f"with position errors > FWHM ({fwhm:.1f} px)"
                )

            if len(dx_all) < 2:
                logger.warning(
                    "Alignment offset plot: no sources remain after error filtering."
                )
                return

            from astropy.stats import sigma_clip as _sc

            n_matched = len(dx_all)
            if n_matched >= 8:
                dx_clipped = _sc(dx_all, sigma=2.5, maxiters=3)
                dy_clipped = _sc(dy_all, sigma=2.5, maxiters=3)
                both_ok = ~dx_clipped.mask & ~dy_clipped.mask
                if np.sum(both_ok) >= 3:
                    dx_plot = dx_all[both_ok]
                    dy_plot = dy_all[both_ok]
                    dx_err_plot = dx_err_all[both_ok]
                    dy_err_plot = dy_err_all[both_ok]
                    sci_x_plot = sci_x_all[both_ok]
                    sci_y_plot = sci_y_all[both_ok]
                else:
                    dx_plot = dx_all
                    dy_plot = dy_all
                    dx_err_plot = dx_err_all
                    dy_err_plot = dy_err_all
                    sci_x_plot = sci_x_all
                    sci_y_plot = sci_y_all
            else:
                dx_plot = dx_all
                dy_plot = dy_all
                dx_err_plot = dx_err_all
                dy_err_plot = dy_err_all
                sci_x_plot = sci_x_all
                sci_y_plot = sci_y_all

            med_dx = float(np.nanmedian(dx_plot))
            med_dy = float(np.nanmedian(dy_plot))
            rms_dx = float(np.sqrt(np.nanmean(dx_plot**2)))
            rms_dy = float(np.sqrt(np.nanmean(dy_plot**2)))

            # Same layout as WCS_vs_PSF_Offset.
            width_pt = 5.5 * 72.27
            aspect = 1.0
            fig, ax = plt.subplots(figsize=set_size(width_pt, aspect=aspect))

            has_errors = np.all(np.isfinite(dx_err_plot)) and np.all(np.isfinite(dy_err_plot))

            # Color points by distance from target.
            _target_x = self.input_yaml.get("target_x_pix")
            _target_y = self.input_yaml.get("target_y_pix")
            if _target_x is not None and _target_y is not None and np.isfinite(_target_x) and np.isfinite(_target_y):
                _dist_from_target = np.sqrt(
                    (sci_x_plot - float(_target_x)) ** 2
                    + (sci_y_plot - float(_target_y)) ** 2
                )
            else:
                _dist_from_target = None

            # Error bars drawn first so they sit behind the points.
            if has_errors:
                ax.errorbar(
                    dx_plot,
                    dy_plot,
                    xerr=dx_err_plot,
                    yerr=dy_err_plot,
                    fmt="none",
                    ecolor=PLOT_COLORS.get('error_bar', '#999999'),
                    elinewidth=0.5,
                    capsize=1.5,
                    alpha=0.5,
                    zorder=1,
                )

            _sc_obj = None
            if _dist_from_target is not None:
                _cmap = plt.get_cmap(PLOT_COLORS.get('scatter_cmap', 'viridis'))
                _sc_obj = ax.scatter(
                    dx_plot,
                    dy_plot,
                    s=12,
                    c=_dist_from_target,
                    cmap=_cmap,
                    marker="o",
                    edgecolor="none",
                    alpha=0.8,
                    zorder=3,
                )
            else:
                ax.scatter(
                    dx_plot,
                    dy_plot,
                    s=12,
                    marker="o",
                    facecolor=PLOT_COLORS.get('scatter_primary', '#0072B2'),
                    edgecolor="none",
                    alpha=0.7,
                    zorder=3,
                )

            # Get pixel scale before setting limits so we can enforce a
            # minimum range of max(1 px, 1 arcsec) whichever is larger.
            pixel_scale = None
            if "pixel_scale" in self.input_yaml:
                pixel_scale = float(self.input_yaml["pixel_scale"])
            else:
                try:
                    from astropy.wcs import utils as wcs_utils
                    pixel_scale = (
                        wcs_utils.proj_plane_pixel_scales(sci_wcs)[0] * 3600
                    )
                except Exception:
                    pass

            # Minimum half-range: 1 pixel or 1 arcsec (in pixels), whichever is larger
            _min_lim = 1.0
            if pixel_scale is not None and pixel_scale > 0:
                _min_lim = max(_min_lim, 1.0 / pixel_scale)

            # Symmetric square axes with (0,0) at centre
            if has_errors:
                _lim = max(
                    np.nanmax(np.abs(dx_plot + dx_err_plot)),
                    np.nanmax(np.abs(dy_plot + dy_err_plot)),
                    _min_lim,
                ) * 1.1
            else:
                _lim = max(
                    np.nanmax(np.abs(dx_plot)),
                    np.nanmax(np.abs(dy_plot)),
                    _min_lim,
                ) * 1.1
            ax.set_xlim(-_lim, _lim)
            ax.set_ylim(-_lim, _lim)
            ax.axhline(0, color=PLOT_COLORS.get('zero_line', '#FF0000'), lw=0.8, ls="--", alpha=0.5, zorder=1)
            ax.axvline(0, color=PLOT_COLORS.get('zero_line', '#FF0000'), lw=0.8, ls="--", alpha=0.5, zorder=1)

            ax.set_xlabel(
                r"$\Delta x = x_{\mathrm{sci}} - x_{\mathrm{ref}}$ [px]"
            )
            ax.set_ylabel(
                r"$\Delta y = y_{\mathrm{sci}} - y_{\mathrm{ref}}$ [px]"
            )

            if pixel_scale is not None and pixel_scale > 0:
                ax_top = ax.twiny()
                ax_right = ax.twinx()

                ax_top.set_xlim(ax.get_xlim())
                ax_right.set_ylim(ax.get_ylim())

                ax_top.set_xticks(ax.get_xticks())
                ax_top.set_xticklabels(
                    [f"{x*pixel_scale:.2f}" for x in ax.get_xticks()]
                )
                ax_top.set_xlabel(r"$\Delta$RA [arcsec]", fontsize="small")

                ax_right.set_yticks(ax.get_yticks())
                ax_right.set_yticklabels(
                    [f"{y*pixel_scale:.2f}" for y in ax.get_yticks()]
                )
                ax_right.set_ylabel(r"$\Delta$Dec [arcsec]", fontsize="small")

                ax_top.tick_params(
                    axis="x", which="both", labeltop=True, labelbottom=False
                )
                ax_right.tick_params(
                    axis="y", which="both", labelright=True, labelleft=False
                )
            else:
                ax.set_aspect("equal", adjustable="box")

            # --- Colorbar (manually positioned to avoid twin axis overlap) ---
            if _sc_obj is not None:
                _cax = fig.add_axes([0.88, 0.12, 0.025, 0.80])
                cbar = fig.colorbar(_sc_obj, cax=_cax)
                cbar.set_label("Distance from target [px]", fontsize="small")
                cbar.ax.tick_params(labelsize="x-small")

            ax.grid(True, ls="-", alpha=0.25, zorder=0)

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
                bbox=dict(facecolor=PLOT_COLORS.get('stats_bbox', 'white'), alpha=0.75, edgecolor="none"),
                fontsize="small",
            )

            ax.text(
                0.05,
                0.05,
                f"N = {len(dx_plot)}",
                transform=ax.transAxes,
                verticalalignment="bottom",
                horizontalalignment="left",
                bbox=dict(facecolor=PLOT_COLORS.get('stats_bbox', 'white'), alpha=0.75, edgecolor="none"),
                fontsize="small",
            )

            # Right margin: only reserve space when a colorbar or the
            # arcsec twin-axis labels actually need it.
            _has_twin = pixel_scale is not None and pixel_scale > 0
            _right = 0.78 if _sc_obj is not None else (0.86 if _has_twin else 0.95)
            fig.subplots_adjust(left=0.12, right=_right, top=0.92, bottom=0.12)

            fig.savefig(save_path, dpi=150, facecolor=PLOT_COLORS.get('figure_facecolor', 'white'))
            plt.close(fig)

            logger.debug("Saved alignment offset plot: %s", save_path)
            logger.info(
                f"Alignment offset:\tmedian=({med_dx:.3f}, {med_dy:.3f}) px, "
                f"RMS=({rms_dx:.3f}, {rms_dy:.3f}) px, N={n_matched}"
            )

        except Exception as e:
            logger.warning("Alignment offset plot failed: %s", e)

    def plot_match_sources(
        self,
        sci_image,
        tpl_image,
        sci_matched_xy,
        tpl_matched_xy,
        sci_all_xy=None,
        tpl_all_xy=None,
        method_label="spalipy",
        sci_fwhm=None,
        tpl_fwhm=None,
    ):
        """Side-by-side plot of matched sources on science and template images.

        Mirrors the SCAMP/SWarp ``plot_matched_sources_side_by_side`` but
        works with in-memory arrays and astropy Tables instead of SExtractor
        LDAC catalogs.

        Parameters
        ----------
        sci_image : ndarray
            Science image data (2D).
        tpl_image : ndarray
            Template image data (2D, before alignment).
        sci_matched_xy : array-like, shape (N, 2)
            Matched source (x, y) positions in the science image.
        tpl_matched_xy : array-like, shape (N, 2)
            Corresponding source (x, y) positions in the template image
            (same ordering as ``sci_matched_xy``).
        sci_all_xy : array-like or None
            All detected sources in the science image (for context).
        tpl_all_xy : array-like or None
            All detected sources in the template image (for context).
        method_label : str
            Label for the plot title (e.g. "spalipy" or "WCS-seeded").
        sci_fwhm : float or None
            FWHM in pixels for the science image.  Circle radius is 3*FWHM.
            Falls back to ``self.input_yaml["fwhm"]`` if None.
        tpl_fwhm : float or None
            FWHM in pixels for the template image.  Circle radius is 3*FWHM.
            Falls back to ``self.input_yaml["fwhm"]`` if None.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from astropy.visualization import (
                ImageNormalize,
                LinearStretch,
                ZScaleInterval,
            )
            from matplotlib.patches import Circle

            apply_autophot_mplstyle()
            plt.ioff()

            base = os.path.splitext(
                os.path.basename(self.input_yaml["fpath"])
            )[0]
            write_dir = os.path.dirname(self.input_yaml["fpath"])
            save_path = os.path.join(
                write_dir, f"Matched_Sources_{base}{get_plot_ext(self.input_yaml)}"
            )

            sci_matched_xy = np.asarray(sci_matched_xy, float)
            tpl_matched_xy = np.asarray(tpl_matched_xy, float)
            n_matched = len(sci_matched_xy)

            # Equal-aspect panels shrink inside mismatched figure cells,
            # leaving gaps no wspace can remove. Science and template shapes
            # can differ, so give each cell a width ratio matching its image
            # aspect and size the figure from the mean aspect.
            _sci_h, _sci_w = sci_image.shape[:2]
            _tpl_h, _tpl_w = tpl_image.shape[:2]
            _aspects = [_sci_w / _sci_h, _tpl_w / _tpl_h]
            ax_h = 4.0
            left, right, bottom, top = 0.06, 0.98, 0.10, 0.88
            # When the image heights differ both panels keep their y tick
            # labels and axis label, so the gap must fit them (~0.7 in).
            _gap_in = 0.7 if _sci_h != _tpl_h else 0.25
            wspace = _gap_in / (ax_h * sum(_aspects) / 2)
            fig_w = (
                (ax_h * sum(_aspects) + _gap_in) / (right - left)
            )
            fig_h = ax_h / (top - bottom)
            fig, (ax1, ax2) = plt.subplots(
                1, 2,
                figsize=(fig_w, fig_h),
                gridspec_kw={"width_ratios": _aspects, "wspace": wspace},
            )
            fig.subplots_adjust(
                left=left, right=right, top=top, bottom=bottom, wspace=wspace
            )

            cmap = plt.get_cmap(PLOT_COLORS.get('image_cmap', 'gray')).copy()
            cmap.set_bad(color=PLOT_COLORS.get('nan_color', 'magenta'))

            for i, (ax, img, title) in enumerate([
                (ax1, sci_image, "Science"),
                (ax2, tpl_image, "Template"),
            ]):
                img_f = np.asarray(img, dtype=np.float32)
                z = ZScaleInterval()
                vmin, vmax = z.get_limits(img_f)
                norm = ImageNormalize(
                    img_f, interval=ZScaleInterval(), stretch=LinearStretch()
                )
                ax.imshow(
                    img_f, cmap=cmap, norm=norm,
                    origin="lower", aspect="equal",
                )
                ax.set_title(title)
                ax.set_xlabel("X [Pixel]")
                # Only suppress the duplicate y labels when both panels
                # share the same y extent.
                if i == 0 or _sci_h != _tpl_h:
                    ax.set_ylabel("Y [Pixel]")
                else:
                    ax.set_ylabel("")
                    ax.tick_params(axis="y", labelleft=False)

            _fwhm_default = float(self.input_yaml.get("fwhm", 3.0))
            _r_sci = 3.0 * (float(sci_fwhm) if sci_fwhm is not None else _fwhm_default)
            _r_tpl = 3.0 * (float(tpl_fwhm) if tpl_fwhm is not None else _fwhm_default)

            def _find_unmatched(all_xy, matched_xy):
                """Return indices of all_xy entries not near any matched source."""
                if all_xy is None or len(all_xy) == 0:
                    return np.array([], dtype=int)
                all_xy = np.asarray(all_xy, float)
                if matched_xy is None or len(matched_xy) == 0:
                    return np.arange(len(all_xy))
                matched_xy = np.asarray(matched_xy, float)
                from scipy.spatial import cKDTree
                tree = cKDTree(matched_xy)
                d, _ = tree.query(all_xy, k=1)
                # >1 px from any matched source = unmatched.
                return np.where(d > 1.0)[0]

            # Track which overlays were drawn so the figure legend lists only
            # markers that are actually visible.
            _drew_unmatched = False
            _drew_target = False
            if sci_all_xy is not None:
                sci_all = np.asarray(sci_all_xy, float)
                _sci_unmatched = _find_unmatched(sci_all, sci_matched_xy)
                if len(_sci_unmatched) > 0:
                    ax1.scatter(
                        sci_all[_sci_unmatched, 0],
                        sci_all[_sci_unmatched, 1],
                        marker="x", s=12, c=PLOT_COLORS.get('unmatched', '#FF0000'), alpha=0.6,
                        linewidths=0.5, zorder=3,
                    )
                    _drew_unmatched = True
            if tpl_all_xy is not None:
                tpl_all = np.asarray(tpl_all_xy, float)
                _tpl_unmatched = _find_unmatched(tpl_all, tpl_matched_xy)
                if len(_tpl_unmatched) > 0:
                    ax2.scatter(
                        tpl_all[_tpl_unmatched, 0],
                        tpl_all[_tpl_unmatched, 1],
                        marker="x", s=12, c=PLOT_COLORS.get('unmatched', '#FF0000'), alpha=0.6,
                        linewidths=0.5, zorder=3,
                    )
                    _drew_unmatched = True

            for (sx, sy) in sci_matched_xy:
                ax1.add_patch(Circle(
                    (sx, sy), _r_sci,
                    edgecolor=PLOT_COLORS.get('matched', '#0072B2'), facecolor="none",
                    linewidth=0.5, zorder=5,
                ))
            for (tx, ty) in tpl_matched_xy:
                ax2.add_patch(Circle(
                    (tx, ty), _r_tpl,
                    edgecolor=PLOT_COLORS.get('matched', '#0072B2'), facecolor="none",
                    linewidth=0.5, zorder=5,
                ))

            _max_label = min(n_matched, 50)
            _step = max(1, n_matched // _max_label)
            for i in range(0, n_matched, _step):
                sx, sy = sci_matched_xy[i]
                tx, ty = tpl_matched_xy[i]
                ax1.text(sx, sy + _r_sci + 1, str(i),
                         color=PLOT_COLORS.get('matched', '#0072B2'), fontsize=4, ha="center", va="bottom")
                ax2.text(tx, ty + _r_tpl + 1, str(i),
                         color=PLOT_COLORS.get('matched', '#0072B2'), fontsize=4, ha="center", va="bottom")

            from matplotlib.patches import Rectangle as _Rect
            from matplotlib.lines import Line2D as _L2
            _target_x = self.input_yaml.get("target_x_pix")
            _target_y = self.input_yaml.get("target_y_pix")
            _target_name = self.input_yaml.get("target_name", "")
            _name_prefix = self.input_yaml.get("name_prefix", "")
            _objname = self.input_yaml.get("objname", _target_name)
            if _name_prefix and _name_prefix.strip() and not str(_target_name).startswith(_name_prefix):
                _display_name = f"{_name_prefix}{_objname}"
            else:
                _display_name = str(_target_name) if _target_name else ""
            _box_size = 4.0 * _fwhm_default
            _box_half = _box_size / 2.0
            _target_xy_tpl = None
            if _target_x is not None and _target_y is not None and np.isfinite(_target_x) and np.isfinite(_target_y):
                ax1.add_patch(_Rect(
                    (_target_x - _box_half, _target_y - _box_half),
                    _box_size, _box_size,
                    edgecolor=PLOT_COLORS.get('variable', '#FFD700'), facecolor="none",
                    linewidth=1.5, linestyle="-", zorder=10,
                ))
                _drew_target = True
                if _display_name:
                    ax1.text(
                        _target_x, _target_y + _box_half + 2,
                        _display_name,
                        color=PLOT_COLORS.get('variable_label', '#B8860B'), fontsize=5, fontweight="bold",
                        ha="center", va="bottom", zorder=11,
                    )
                # Template image needs WCS conversion from the science frame.
                try:
                    from astropy.wcs import WCS
                    from astropy.io import fits as _fits
                    _fpath = self.input_yaml["fpath"]
                    with _fits.open(_fpath, memmap=False) as _hdul:
                        _sci_wcs = WCS(_hdul[0].header, fix=True, relax=True)
                    _ra, _dec = _sci_wcs.all_pix2world(_target_x, _target_y, 0)
                    _tpl_fpath = self.input_yaml.get("template_path") or self.input_yaml.get("templateFpath")
                    if _tpl_fpath and os.path.exists(_tpl_fpath):
                        with _fits.open(_tpl_fpath, memmap=False) as _hdul2:
                            _tpl_wcs = WCS(_hdul2[0].header, fix=True, relax=True)
                        _tx, _ty = _tpl_wcs.all_world2pix(_ra, _dec, 0)
                        if np.isfinite(_tx) and np.isfinite(_ty):
                            _target_xy_tpl = (_tx, _ty)
                            ax2.add_patch(_Rect(
                                (_tx - _box_half, _ty - _box_half),
                                _box_size, _box_size,
                                edgecolor=PLOT_COLORS.get('variable', '#FFD700'), facecolor="none",
                                linewidth=1.5, linestyle="-", zorder=10,
                            ))
                            if _display_name:
                                ax2.text(
                                    _tx, _ty + _box_half + 2,
                                    _display_name,
                                    color=PLOT_COLORS.get('variable_label', '#B8860B'), fontsize=5, fontweight="bold",
                                    ha="center", va="bottom", zorder=11,
                                )
                except Exception:
                    pass

            _legend_handles = []
            # Matched count already appears in the stats box, so no legend
            # entry is needed for the matched circles.
            if _drew_unmatched:
                _legend_handles.append(_L2([0], [0], marker="x", color=PLOT_COLORS.get('unmatched', '#FF0000'),
                                           markersize=5, linestyle="None",
                                           label="Unmatched"))
            if _drew_target:
                _legend_handles.append(_L2([0], [0], marker="s", color=PLOT_COLORS.get('variable', '#FFD700'),
                                           markerfacecolor="none", markersize=5,
                                           linestyle="None", label="Transient"))
            if _legend_handles:
                # Figure-level legend keeps marker overlays from covering it.
                fig.legend(
                    handles=_legend_handles,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.0),
                    ncol=len(_legend_handles),
                    fontsize=8,
                    frameon=False,
                )

            _n_sci_unmatched = len(_find_unmatched(
                np.asarray(sci_all_xy, float) if sci_all_xy is not None else np.empty((0, 2)),
                sci_matched_xy,
            ))
            _n_tpl_unmatched = len(_find_unmatched(
                np.asarray(tpl_all_xy, float) if tpl_all_xy is not None else np.empty((0, 2)),
                tpl_matched_xy,
            ))
            _stats = (
                f"Matching sources: {n_matched}\n"
                f"Ignored in Science: {_n_sci_unmatched}\n"
                f"Ignored in Reference: {_n_tpl_unmatched}\n"
                f"Method: {method_label}"
            )
            ax1.text(
                0.02, 0.98, _stats,
                transform=ax1.transAxes, fontsize=6,
                verticalalignment="top", horizontalalignment="left",
                bbox=dict(facecolor=PLOT_COLORS.get('stats_bbox', 'white'), edgecolor=PLOT_COLORS.get('legend_edgecolor', '#999999'), alpha=0.9, boxstyle="round,pad=0.3"),
            )

            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=PLOT_COLORS.get('figure_facecolor', 'white'))
            plt.close(fig)
            logger.info(
                "Match sources plot saved: %s (%d matched sources)",
                os.path.basename(save_path), n_matched,
            )
        except Exception as e:
            logger.warning("Match sources plot failed: %s", e)
