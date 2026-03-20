#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting utilities for subtraction checks, source diagnostics, and light curves.
"""

import logging
import os

logger = logging.getLogger(__name__)


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

            # Compute vmin, vmax per image using zscale with percentile cleaning
            vmins = {}
            vmaxs = {}
            for key, img_data in images.items():
                # First apply percentile cleaning to remove extreme values
                valid_data = img_data[np.isfinite(img_data)]
                lower, upper = np.percentile(valid_data, [0.5, 99.5])
                cleaned_data = np.clip(img_data, lower, upper)

                # Then apply zscale to get optimal display range
                vmin, vmax = zscale.get_limits(cleaned_data)
                vmins[key] = vmin
                vmaxs[key] = vmax

            # Set up figure
            fig = plt.figure(figsize=set_size(540, 1), constrained_layout=True)
            gs = GridSpec(1, 3, figure=fig, wspace=0.1)

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

            # Create axes and plot images with zscale scaling
            axes = []
            for i, (title, img_data) in enumerate(images.items()):
                ax = fig.add_subplot(gs[i])
                ax.imshow(
                    img_data,
                    origin="lower",
                    aspect="auto",
                    cmap="bone",
                    vmin=vmins[title],
                    vmax=vmaxs[title],
                )
                ax.set_title(title)
                ax.set_xlabel("X [Pixel]")
                ax.set_ylabel("Y [Pixel]")
                axes.append(ax)

            square_size = int(inset_size * 2)  # Width/height of the square in pixels
            if matching_sources is not None and len(matching_sources) > 0:
                half_size = square_size / 2
                for ax in axes:  # Only first two panels (Image and Reference)
                    for x_pix, y_pix in zip(
                        matching_sources["x_pix"], matching_sources["y_pix"]
                    ):
                        x_pix -= 1
                        y_pix -= 1
                        rect = patches.Rectangle(
                            (x_pix - half_size, y_pix - half_size),
                            square_size,
                            square_size,
                            linewidth=0.5,
                            edgecolor="b",
                            facecolor="none",
                            alpha=0.5,
                        )
                        ax.add_patch(rect)

            # Plot variable sources as red "x" and annotate with otype
            if masked_sources is not None and len(masked_sources) > 0:
                cross_len = square_size / 4  # Length of each arm of the cross
                for x, y, otype, name in zip(
                    masked_sources["x_pix"],
                    masked_sources["y_pix"],
                    masked_sources["OTYPE_opt"],
                    masked_sources["MAIN_ID"],
                ):
                    x -= 1
                    y -= 1
                    if "SN*" in otype:
                        otype = name
                        circle = mpatches.Circle(
                            (x, y),
                            cross_len * 2,
                            edgecolor="red",
                            facecolor="none",
                            zorder=4,
                            lw=0.5,
                        )
                        axes[0].add_patch(
                            circle
                        )  # Assuming add to first ax, adjust if needed
                    else:
                        axes[0].plot(
                            [x - cross_len, x + cross_len],
                            [y - cross_len, y + cross_len],
                            color="red",
                            lw=0.5,
                            zorder=2,
                        )
                        axes[0].plot(
                            [x - cross_len, x + cross_len],
                            [y + cross_len, y - cross_len],
                            color="red",
                            lw=0.5,
                            zorder=2,
                        )
                    axes[0].annotate(
                        otype,
                        xy=(x, y),
                        xytext=(x, y + cross_len / 2),
                        ha="center",
                        va="bottom",
                        fontsize=3,
                        color="red",
                        zorder=3,
                    )

            # Add insets and other features
            for i, (title, img_data) in enumerate(images.items()):
                ax = axes[i]

                if expected_location:
                    x, y = map(int, expected_location)
                    x = max(inset_size, min(x, img_width - inset_size))
                    y = max(inset_size, min(y, img_height - inset_size))

                    x_side, con_x_main, con_x_inset = get_inset_side(
                        x, inset_size, img_width
                    )
                    y_side, con_y_main, con_y_inset = get_inset_side(
                        y, inset_size, img_height
                    )
                    inset_loc = f'{"upper" if y_side == "high" else "lower"} {"right" if x_side == "high" else "left"}'

                    # Inset
                    ax_inset = inset_axes(ax, width="30%", height="30%", loc=inset_loc)
                    ax_inset.imshow(
                        img_data,
                        origin="lower",
                        aspect="auto",
                        cmap="bone",
                        vmin=vmins[title],
                        vmax=vmaxs[title],
                    )
                    ax_inset.set_xlim(x - inset_size, x + inset_size)
                    ax_inset.set_ylim(y - inset_size, y + inset_size)
                    ax_inset.set_xticks([])
                    ax_inset.set_yticks([])
                    for spine in ax_inset.spines.values():
                        spine.set_color("red")
                        spine.set_linewidth(0.5)
                    inset_axes_list.append(ax_inset)

                    # Rectangle on main image
                    rect = Rectangle(
                        (x - inset_size, y - inset_size),
                        2 * inset_size,
                        2 * inset_size,
                        linewidth=0.5,
                        edgecolor="red",
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
                                color="red",
                                linewidth=0.5,
                            )
                        )

            # Optional mask overlay
            if mask is not None:
                red_overlay = colors.ListedColormap(["none", "red"])
                for ax in fig.axes:
                    if ax not in inset_axes_list:
                        ax.imshow(mask, cmap=red_overlay, alpha=0.5, origin="lower")

            # Add markers for expected and fitted locations
            if fitted_location and len(fitted_location) == 2:
                radius = aperture_size  # Circle radius in pixels (diameter = inset_size // 4)

                for ax in inset_axes_list[2:]:
                    # Red hollow circle at fitted location
                    circle = mpatches.Circle(
                        fitted_location,
                        radius=radius,
                        edgecolor="red",
                        facecolor="none",
                        linewidth=0.5,
                        transform=ax.transData,
                    )
                    ax.add_patch(circle)

                    # Green cross at expected location (2 lines)
                    x, y = expected_location
                    cross_len = aperture_size / 2  # half-length of each arm

                    hline = mlines.Line2D(
                        [x - cross_len, x + cross_len],
                        [y, y],
                        color="green",
                        linewidth=0.5,
                        transform=ax.transData,
                    )
                    vline = mlines.Line2D(
                        [x, x],
                        [y - cross_len, y + cross_len],
                        color="green",
                        linewidth=0.5,
                        transform=ax.transData,
                    )
                    ax.add_line(hline)
                    ax.add_line(vline)

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
            ax.set_title(t)
            ax.set_xlabel("X [Pixel]")
            ax.set_ylabel("Y [Pixel]")

        tx = cx - x0
        ty = cy - y0

        # Panel 1
        axes[0].imshow(cut, origin="lower", cmap="bone", vmin=vmin, vmax=vmax)
        axes[0].axvline(tx, color="cyan", lw=0.6, alpha=0.9)
        axes[0].axhline(ty, color="cyan", lw=0.6, alpha=0.9)
        if (
            aperture_radius is not None
            and np.isfinite(aperture_radius)
            and aperture_radius > 0
        ):
            axes[0].add_patch(
                mpatches.Circle(
                    (tx, ty),
                    float(aperture_radius),
                    edgecolor="cyan",
                    facecolor="none",
                    lw=0.8,
                )
            )

        # Panel 2
        axes[1].imshow(cut, origin="lower", cmap="bone", vmin=vmin, vmax=vmax)
        levels = np.unique(seg)
        levels = levels[levels > 0]
        if levels.size:
            axes[1].contour(
                seg, levels=levels, colors="lime", linewidths=0.4, alpha=0.9
            )
        axes[1].axvline(tx, color="cyan", lw=0.6, alpha=0.9)
        axes[1].axhline(ty, color="cyan", lw=0.6, alpha=0.9)

        # Panel 3
        axes[2].imshow(cut, origin="lower", cmap="bone", vmin=vmin, vmax=vmax)
        overlay = colors.ListedColormap(["none", "red"])
        axes[2].imshow(nmask.astype(int), origin="lower", cmap=overlay, alpha=0.35)
        axes[2].axvline(tx, color="cyan", lw=0.6, alpha=0.9)
        axes[2].axhline(ty, color="cyan", lw=0.6, alpha=0.9)

        fig.suptitle(f"Crowding diagnostic {title_extra}".strip())
        fig.savefig(save_path, dpi=300)
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

            # Set up matplotlib style
            dir_path = os.path.dirname(os.path.realpath(__file__))
            plt.style.use(os.path.join(dir_path, "autophot.mplstyle"))

            # Extract file path and base name
            fpath = self.input_yaml["fpath"]
            base = os.path.basename(fpath)
            write_dir = os.path.dirname(fpath)
            base = os.path.splitext(base)[0]

            # Get photometry radius and scale from input YAML
            radius = self.input_yaml["photometry"]["ap_size"] * self.input_yaml["fwhm"]
            scale = self.input_yaml["scale"]

            # Create the figure
            plt.ioff()  # Turn interactive mode off
            fig = plt.figure(figsize=set_size(330, 1))
            ax1 = fig.add_subplot(111)

            # Normalize and plot the image
            norm = ImageNormalize(
                image, interval=ZScaleInterval(), stretch=LinearStretch()
            )
            im = ax1.imshow(
                image,
                origin="lower",
                aspect="auto",
                cmap="grey_r",
                interpolation=None,
                norm=norm,
            )

            # Title summarising what is shown
            title_main = "SourceCheck: target, PSF, FWHM, catalog"
            if subtracted:
                title_main += " (subtracted image)"
            ax1.set_title(title_main, fontsize=6)

            # Plot the target source as a circle
            circle = Circle(
                (self.input_yaml["target_x_pix"], self.input_yaml["target_y_pix"]),
                radius,
                edgecolor="#D55E00",
                facecolor="none",
                zorder=4,
                lw=0.5,
            )
            ax1.add_patch(circle)
            ax1.text(
                self.input_yaml["target_x_pix"],
                self.input_yaml["target_y_pix"] + radius + 2,
                self.input_yaml["target_name"],
                color="#D55E00",
                fontsize=3,
                ha="center",
                va="bottom",
            )

            # Plot PSF sources as a clean, consistent marker layer
            if psfSources is not None and len(psfSources) > 0:
                ax1.scatter(
                    psfSources["x_pix"],
                    psfSources["y_pix"],
                    s=max(8, 0.6 * scale),
                    marker="+",
                    linewidths=0.5,
                    color="#009E73",
                    label="PSF sources",
                    zorder=1,
                )

            # Plot reference (catalog) sources as squares
            if catalogSources is not None and not self.input_yaml["HST_mode"]:
                for x, y in zip(catalogSources["x_pix"], catalogSources["y_pix"]):
                    square_size = scale  # Size of the square
                    lower_left = (x - square_size / 2, y - square_size / 2)
                    square = Rectangle(
                        lower_left,
                        square_size,
                        square_size,
                        edgecolor="#0072B2",
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
                    cmap = plt.get_cmap("plasma")  # Choose a colormap

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
                            lw=0.5,
                            ls="-",
                        )
                        ax1.add_patch(circle)

                    # Add colorbar
                    cbar = fig.colorbar(sm, ax=ax1, pad=0.0, aspect=40)
                    cbar.set_label("FWHM (pixels)", fontsize=6)
                    cbar.ax.tick_params(labelsize=5)

            # from matplotlib.patches import Rectangle

            # Plot variable sources as red "x" and annotate with otype
            if variable_sources is not None and not self.input_yaml["HST_mode"]:

                if len(variable_sources) > 0:

                    cross_len = scale / 4  # Length of each arm of the cross

                    for x, y, otype, name in zip(
                        variable_sources["x_pix"],
                        variable_sources["y_pix"],
                        variable_sources["OTYPE_opt"],
                        variable_sources["MAIN_ID"],
                    ):

                        # Draw "x" as two lines rotated 45 degrees (magenta for strong contrast)
                        ax1.plot(
                            [x - cross_len, x + cross_len],
                            [y - cross_len, y + cross_len],
                            color="#CC79A7",
                            lw=0.5,
                            zorder=2,
                        )
                        ax1.plot(
                            [x - cross_len, x + cross_len],
                            [y + cross_len, y - cross_len],
                            color="#CC79A7",
                            lw=0.5,
                            zorder=2,
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
                            fontsize=3,
                            color="#CC79A7",
                            zorder=3,
                        )

            # Overlay the mask if provided
            if mask is not None:
                from matplotlib import colors

                # Cyan overlay for masks to avoid confusion with other markers
                mask_cmap = colors.ListedColormap(["none", "#00BFC4"])
                ax1.imshow(mask, cmap=mask_cmap, alpha=0.5, origin="lower")

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
                ncol=4,
                frameon=False,
                columnspacing=1.5,
                fontsize=3,
            )

            # Finalize figure layout
            fig.tight_layout()
            ax1.set_aspect("equal", adjustable="box")

            # Save figure
            if not subtracted:
                save_loc = os.path.join(
                    write_dir, "Sourcecheck_" + base + ".png"
                )
            else:
                save_loc = os.path.join(
                    write_dir, "Subtracted_Sourcecheck_" + base + ".png"
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
    ):
        """
        Plot lightcurve with detections and optional upper limits.
        Detection = SNR >= snr_limit and (beta > beta_limit if 'beta' present).
        Non-detections are plotted as upper limits at lmag (limiting magnitude) when show_limits=True.
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from functions import set_size

        # Color array taken from Superbol by Matt Nicholl
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

        # Maintains order from blue to red effective wavelength
        bandlist = "FSDNAuUBgcVwrRoGEiIzyYJHKWQ"

        markers = ["x", "o", "^", "s", "D"]
        marker_iterator = iter(markers)

        data = pd.read_csv(output_file)

        if redshift != 0:
            from functions import get_distance_modulus

            dm = get_distance_modulus(redshift)
        else:
            dm = 0

        fig = plt.figure(figsize=set_size(*default_size))
        ax1 = fig.add_subplot(111)
        ax1.invert_yaxis()

        num_detect = 0
        num_nondetect = 0
        for b in bandlist:
            band = b + "_" + method
            if band not in data.columns:
                continue
            err_col = band + "_err"
            if err_col not in data.columns:
                continue

            data_band = data[~np.isnan(data[band])].copy()
            if data_band.empty:
                continue

            # Resolve SNR: same priority as lightcurve.plot_lightcurve (PSF/AP then generic)
            if method == "PSF" and "SNR_PSF" in data_band.columns:
                snr = np.asarray(data_band["SNR_PSF"], dtype=float)
            elif (
                method == "PSF"
                and "flux_PSF" in data_band.columns
                and "flux_PSF_err" in data_band.columns
            ):
                err_psf = np.asarray(data_band["flux_PSF_err"], dtype=float)
                snr = np.divide(
                    data_band["flux_PSF"],
                    err_psf,
                    out=np.full(len(data_band), np.nan),
                    where=(err_psf > 0) & np.isfinite(err_psf),
                )
            elif method == "AP" and "SNR_AP" in data_band.columns:
                snr = np.asarray(data_band["SNR_AP"], dtype=float)
            elif "snr" in data_band.columns:
                snr = np.asarray(data_band["snr"], dtype=float)
            elif "SNR" in data_band.columns:
                snr = np.asarray(data_band["SNR"], dtype=float)
            else:
                err_vals = np.asarray(data_band[err_col], dtype=float)
                snr = np.divide(
                    data_band[band],
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
                ax1.errorbar(
                    detects.mjd - reference_epoch,
                    detects[band],
                    yerr=detects[err_col],
                    c=cols.get(b, "k"),
                    ls="",
                    capsize=2,
                    marker=marker,
                    label=b,
                )
            if show_limits and not nondetects.empty:
                # Upper limits: plot at limiting magnitude (fainter than this = non-detection)
                y_lim = (
                    nondetects["lmag"]
                    if "lmag" in nondetects.columns
                    and np.any(np.isfinite(nondetects["lmag"]))
                    else nondetects[band]
                )
                ax1.errorbar(
                    nondetects.mjd - reference_epoch,
                    y_lim,
                    c=cols.get(b, "k"),
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
            bbox_to_anchor=[0.5, 1],
            loc="lower center",
            frameon=False,
            ncols=ncols,
        )

        plt.show()

        return
