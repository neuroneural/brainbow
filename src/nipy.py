""" Most of the code here is from nipy.slices module"""
import numpy as np

from nibabel.affines import apply_affine

import matplotlib.pyplot as plt
from matplotlib import transforms

from src.utils import is_numlike, is_iterable, get_bounds, get_mask_bounds


def plot_map(
    map=None,
    affine=None,
    cut_coords=None,
    anat=None,
    anat_affine=None,
    slicer="ortho",
    figure=None,
    axes=None,
    title=None,
    threshold=None,
    black_bg=False,
    **imshow_kwargs,
):
    """Plot three cuts of a given activation map (Frontal, Axial, and Lateral)

    Parameters
    ----------
    map : 3D ndarray
        The activation map, as a 3D image.
    affine : 4x4 ndarray
        The affine matrix going from image voxel space to MNI space.
    cut_coords: None, int, or a tuple of floats
        The MNI coordinates of the point where the cut is performed, in
        MNI coordinates and order.
        If slicer is 'ortho', this should be a 3-tuple: (x, y, z)
        For slicer == 'x', 'y', or 'z', then these are the
        coordinates of each cut in the corresponding direction.
        If None or an int is given, then a maximally separated sequence (
        with exactly cut_coords elements if cut_coords is not None) of
        cut coordinates along the slicer axis is computed automatically
    anat : 3D ndarray or False, optional
        The anatomical image to be used as a background. If None, the
        MNI152 T1 1mm template is used. If False, no anat is displayed.
    anat_affine : 4x4 ndarray, optional
        The affine matrix going from the anatomical image voxel space to
        MNI space. This parameter is not used when the default
        anatomical is used, but it is compulsory when using an
        explicite anatomical image.
    slicer: {'ortho', 'x', 'y', 'z'}
        Choose the direction of the cuts. With 'ortho' three cuts are
        performed in orthogonal directions
    figure : integer or matplotlib figure, optional
        Matplotlib figure used or its number. If None is given, a
        new figure is created.
    axes : matplotlib axes or 4 tuple of float: (xmin, ymin, width, height), optional
        The axes, or the coordinates, in matplotlib figure space,
        of the axes used to display the plot. If None, the complete
        figure is used.
    title : string, optional
        The title dispayed on the figure.
    threshold : a number, None, or 'auto'
        If None is given, the maps are not thresholded.
        If a number is given, it is used to threshold the maps:
        values below the threshold are plotted as transparent. If
        auto is given, the threshold is determined magically by
        analysis of the map.
    black_bg: boolean, optional
        If True, the background of the image is set to be black. If
        you whish to save figures with a black background, you
        will need to pass "facecolor='k', edgecolor='k'" to pylab's
        savefig.
    imshow_kwargs: extra keyword arguments, optional
        Extra keyword arguments passed to pylab.imshow

    Notes
    -----
    Arrays should be passed in numpy convention: (x, y, z)
    ordered.

    Use masked arrays to create transparency:

        import numpy as np
        map = np.ma.masked_less(map, 0.5)
        plot_map(map, affine)
    """

    slicer = CustomSlicer.init_with_figure(
        cut_coords=cut_coords,
        figure=figure,
        axes=axes,
        black_bg=black_bg,
    )

    if anat is not None:
        _plot_anat(slicer, anat, anat_affine, title=title)
    if map is not None:
        slicer.plot_map(map, affine, **imshow_kwargs)

    return slicer


def _plot_anat(
    slicer,
    anat,
    anat_affine,
    title=None,
    dim=False,
    cmap="gray",
    **imshow_kwargs,
):
    """Internal function used to plot anatomy"""

    black_bg = slicer._black_bg

    if anat is not False:
        if dim:
            vmin = anat.min()
            vmax = anat.max()
        else:
            vmin = None
            vmax = None
        if dim:
            vmean = 0.5 * (vmin + vmax)
            ptp = 0.5 * (vmax - vmin)
            if not is_numlike(dim):
                dim = 0.6
            if black_bg:
                vmax = vmean + (1 + dim) * ptp
            else:
                vmin = vmean - (1 + dim) * ptp
        slicer.plot_map(
            anat, anat_affine, cmap=cmap, vmin=vmin, vmax=vmax, **imshow_kwargs
        )

    if black_bg:
        # To have a black background in PDF, we need to create a
        # patch in black for the background
        for ax in slicer.axes.values():
            ax.ax.imshow(
                np.zeros((2, 2, 3)), extent=[-5000, 5000, -5000, 5000], zorder=-500
            )

    if title is not None and not title == "":
        slicer.title(title)
    return slicer


class CustomSlicer(object):
    """The main purpose of these class is to have auto adjust of axes size
    to the data with different layout of cuts.
    """

    # This actually encodes the figsize for only one axe
    _default_figsize = [2.2, 2.6]

    def __init__(self, cut_coords, axes=None, black_bg=False):
        """Create 3 linked axes for plotting orthogonal cuts.

        Parameters
        ----------
        cut_coords: 3 tuple of ints
            The cut position, in world space.
        axes: matplotlib axes object, optional
            The axes that will be subdivided in 3.
        black_bg: boolean, optional
            If True, the background of the figure will be put to
            black. If you whish to save figures with a black background,
            you will need to pass "facecolor='k', edgecolor='k'" to
            pylab's savefig.

        """
        self._cut_coords = cut_coords
        if axes is None:
            # axes = pl.axes((0.0, 0.0, 1.0, 1.0))
            axes = plt.axes((0.0, 0.0, 1.0, 1.0))
            axes.axis("off")
        self.frame_axes = axes
        axes.set_zorder(1)
        bb = axes.get_position()
        self.rect = (bb.x0, bb.y0, bb.x1, bb.y1)
        self._black_bg = black_bg
        self._init_axes()

    def _init_axes(self):
        x0, y0, x1, y1 = self.rect
        # Create our axes:
        self.axes = dict()
        for index, direction in enumerate(("y", "x", "z")):
            # ax = pl.axes([0.3 * index * (x1 - x0) + x0, y0, 0.3 * (x1 - x0), y1 - y0])
            ax = plt.axes([0.3 * index * (x1 - x0) + x0, y0, 0.3 * (x1 - x0), y1 - y0])
            ax.axis("off")
            coord = self._cut_coords["xyz".index(direction)]
            cut_ax = CutAxes(ax, direction, coord)
            self.axes[direction] = cut_ax
            ax.set_axes_locator(self._locator)

    def _locator(self, axes, renderer):
        """The locator function used by matplotlib to position axes.
        Here we put the logic used to adjust the size of the axes.
        """
        x0, y0, x1, y1 = self.rect
        width_dict = dict()
        cut_ax_dict = self.axes
        x_ax = cut_ax_dict["x"]
        y_ax = cut_ax_dict["y"]
        z_ax = cut_ax_dict["z"]
        for cut_ax in cut_ax_dict.values():
            bounds = cut_ax.get_object_bounds()
            if not bounds:
                # This happens if the call to _map_show was not
                # succesful. As it happens asyncroniously (during a
                # refresh of the figure) we capture the problem and
                # ignore it: it only adds a non informative traceback
                bounds = [0, 1, 0, 1]
            xmin, xmax, ymin, ymax = bounds
            width_dict[cut_ax.ax] = xmax - xmin
        total_width = float(sum(width_dict.values()))
        for ax, width in width_dict.items():
            width_dict[ax] = width / total_width * (x1 - x0)
        left_dict = dict()
        left_dict[y_ax.ax] = x0
        left_dict[x_ax.ax] = x0 + width_dict[y_ax.ax]
        left_dict[z_ax.ax] = x0 + width_dict[x_ax.ax] + width_dict[y_ax.ax]
        return transforms.Bbox(
            [[left_dict[axes], y0], [left_dict[axes] + width_dict[axes], y1]]
        )

    @classmethod
    def init_with_figure(
        cls,
        cut_coords=None,
        figure=None,
        axes=None,
        black_bg=False,
    ):
        if isinstance(axes, plt.Axes) and figure is None:
            figure = axes.figure

        if not isinstance(figure, plt.Figure):
            # Make sure that we have a figure
            figsize = cls._default_figsize[:]
            # Adjust for the number of axes
            figsize[0] *= len(cut_coords)
            facecolor = "k" if black_bg else "w"

            # figure = pl.figure(figure, figsize=figsize, facecolor=facecolor)
            figure = plt.figure(figure, figsize=figsize, facecolor=facecolor)
        else:
            if isinstance(axes, plt.Axes):
                assert axes.figure is figure, "The axes passed are not " "in the figure"

        if axes is None:
            axes = [0.0, 0.0, 1.0, 1.0]
        if is_iterable(axes):
            axes = figure.add_axes(axes)
        # People forget to turn their axis off, or to set the zorder, and
        # then they cannot see their slicer
        axes.axis("off")
        return cls(cut_coords, axes, black_bg)

    def title(
        self, text, x=0.01, y=0.99, size=15, color=None, bgcolor=None, alpha=1, **kwargs
    ):
        """Write a title to the view.

        Parameters
        ----------
        text: string
            The text of the title
        x: float, optional
            The horizontal position of the title on the frame in
            fraction of the frame width.
        y: float, optional
            The vertical position of the title on the frame in
            fraction of the frame height.
        size: integer, optional
            The size of the title text.
        color: matplotlib color specifier, optional
            The color of the font of the title.
        bgcolor: matplotlib color specifier, optional
            The color of the background of the title.
        alpha: float, optional
            The alpha value for the background.
        kwargs:
            Extra keyword arguments are passed to matplotlib's text
            function.
        """
        if color is None:
            color = "k" if self._black_bg else "w"
        if bgcolor is None:
            bgcolor = "w" if self._black_bg else "k"
        self.frame_axes.text(
            x,
            y,
            text,
            transform=self.frame_axes.transAxes,
            horizontalalignment="left",
            verticalalignment="top",
            size=size,
            color=color,
            bbox=dict(boxstyle="square,pad=.3", ec=bgcolor, fc=bgcolor, alpha=alpha),
            **kwargs,
        )

    def plot_map(self, map, affine, threshold=None, **kwargs):
        """Plot a 3D map in all the views.

        Parameters
        -----------
        map: 3D ndarray
            The 3D map to be plotted. If it is a masked array, only
            the non-masked part will be plotted.
        affine: 4x4 ndarray
            The affine matrix giving the transformation from voxel
            indices to world space.
        threshold : a number, None, or 'auto'
            If None is given, the maps are not thresholded.
            If a number is given, it is used to threshold the maps:
            values below the threshold are plotted as transparent.
        kwargs:
            Extra keyword arguments are passed to imshow.
        """
        if threshold is not None:
            if threshold == 0:
                map = np.ma.masked_equal(map, 0, copy=False)
            else:
                map = np.ma.masked_inside(map, -threshold, threshold, copy=False)

        self._map_show(map, affine, type="imshow", **kwargs)

    def _map_show(self, map, affine, type="imshow", **kwargs):
        data_bounds = get_bounds(map.shape, affine)
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = data_bounds

        xmin_, xmax_, ymin_, ymax_, zmin_, zmax_ = xmin, xmax, ymin, ymax, zmin, zmax
        if hasattr(map, "mask"):
            not_mask = np.logical_not(map.mask)
            xmin_, xmax_, ymin_, ymax_, zmin_, zmax_ = get_mask_bounds(not_mask, affine)
            if kwargs.get("vmin") is None and kwargs.get("vmax") is None:
                # Avoid dealing with masked arrays: they are slow
                if not np.any(not_mask):
                    # Everything is masked
                    vmin = vmax = 0
                else:
                    masked_map = np.asarray(map)[not_mask]
                    vmin = masked_map.min()
                    vmax = masked_map.max()
                if kwargs.get("vmin") is None:
                    kwargs["vmin"] = vmin
                if kwargs.get("max") is None:
                    kwargs["vmax"] = vmax
        else:
            if "vmin" not in kwargs:
                kwargs["vmin"] = map.min()
            if "vmax" not in kwargs:
                kwargs["vmax"] = map.max()
        if "vmin" not in kwargs:
            kwargs["vmin"] = map.min()
        if "vmax" not in kwargs:
            kwargs["vmax"] = map.max()

        bounding_box = (xmin_, xmax_), (ymin_, ymax_), (zmin_, zmax_)

        # For each ax, cut the data and plot it
        for cut_ax in self.axes.values():
            try:
                cut = cut_ax.do_cut(map, affine)
            except IndexError:
                # We are cutting outside the indices of the data
                continue
            cut_ax.draw_cut(cut, data_bounds, bounding_box, type=type, **kwargs)

    def annotate(self, text, mode="minimal", size=12, **kwargs):
        """Add annotation to the plot.

        Parameters
        ----------
        left_right: boolean, optional
            If left_right is True, annotations indicating which side
            is left and which side is right are drawn.
        positions: boolean, optional
            If positions is True, annotations indicating the
            positions of the cuts are drawn.
        size: integer, optional
            The size of the text used.
        kwargs:
            Extra keyword arguments are passed to matplotlib's text
            function.
        """
        if mode != "none":
            kwargs = kwargs.copy()
            if not "color" in kwargs:
                if self._black_bg:
                    kwargs["color"] = "w"
                else:
                    kwargs["color"] = "k"

            bg_color = "k" if self._black_bg else "w"

            if mode in ["minimal", "full"]:
                self.axes["y"].add_annotation(
                    text, mode="component", size=size, bg_color=bg_color, **kwargs
                )
            if mode == "full":
                self.axes["x"].add_annotation(
                    None, mode="coordinates", size=size, bg_color=bg_color, **kwargs
                )
                self.axes["y"].add_annotation(
                    None, mode="coordinates", size=size, bg_color=bg_color, **kwargs
                )
                self.axes["z"].add_annotation(
                    None, mode="coordinates", size=size, bg_color=bg_color, **kwargs
                )


class CutAxes(object):
    """An MPL axis-like object that displays a cut of 3D volumes"""

    def __init__(self, ax, direction, coord):
        """An MPL axis-like object that displays a cut of 3D volumes

        Parameters
        ==========
        ax: a MPL axes instance
            The axes in which the plots will be drawn
        direction: {'x', 'y', 'z'}
            The directions of the cut
        coord: float
            The coordinnate along the direction of the cut
        """
        self.ax = ax
        self.direction = direction
        self.coord = coord
        self._object_bounds = list()

    def do_cut(self, map, affine):
        """Cut the 3D volume into a 2D slice

        Parameters
        ==========
        map: 3D ndarray
            The 3D volume to cut
        affine: 4x4 ndarray
            The affine of the volume
        """
        coords = [0, 0, 0]
        coords["xyz".index(self.direction)] = self.coord
        x_map, y_map, z_map = [
            int(np.round(c))
            for c in apply_affine(np.linalg.inv(affine), np.array(coords))
        ]
        if self.direction == "y":
            cut = np.rot90(map[:, y_map, :])
        elif self.direction == "x":
            cut = np.rot90(map[x_map, :, :])
        elif self.direction == "z":
            cut = np.rot90(map[:, :, z_map])
        else:
            raise ValueError("Invalid value for direction %s" % self.direction)
        return cut

    def draw_cut(self, cut, data_bounds, bounding_box, type="imshow", **kwargs):
        # kwargs massaging
        kwargs["origin"] = "upper"

        if self.direction == "y":
            (xmin, xmax), (_, _), (zmin, zmax) = data_bounds
            (xmin_, xmax_), (_, _), (zmin_, zmax_) = bounding_box
        elif self.direction == "x":
            (_, _), (xmin, xmax), (zmin, zmax) = data_bounds
            (_, _), (xmin_, xmax_), (zmin_, zmax_) = bounding_box
        elif self.direction == "z":
            (xmin, xmax), (zmin, zmax), (_, _) = data_bounds
            (xmin_, xmax_), (zmin_, zmax_), (_, _) = bounding_box
        else:
            raise ValueError("Invalid value for direction %s" % self.direction)
        ax = self.ax
        getattr(ax, type)(cut, extent=(xmin, xmax, zmin, zmax), **kwargs)

        self._object_bounds.append((xmin_, xmax_, zmin_, zmax_))
        ax.axis(self.get_object_bounds())

    def get_object_bounds(self):
        """Return the bounds of the objects on this axes."""
        if len(self._object_bounds) == 0:
            # Nothing plotted yet
            return -0.01, 0.01, -0.01, 0.01
        xmins, xmaxs, ymins, ymaxs = np.array(self._object_bounds).T
        xmax = max(xmaxs.max(), xmins.max())
        xmin = min(xmins.min(), xmaxs.min())
        ymax = max(ymaxs.max(), ymins.max())
        ymin = min(ymins.min(), ymaxs.min())
        return xmin, xmax, ymin, ymax

    def add_annotation(self, text, mode, size, bg_color, **kwargs):
        if mode == "component":
            self.ax.text(
                0.0,
                1.0,
                s=text,
                transform=self.ax.transAxes,
                horizontalalignment="center",
                verticalalignment="center",
                size=size,
                bbox=dict(boxstyle="square,pad=0", ec=bg_color, fc=bg_color, alpha=1),
                **kwargs,
            )
        elif mode == "coordinates":
            self.ax.text(
                0.5,
                1.0,
                s=f"{self.direction}: {self.coord:0.0f}",
                transform=self.ax.transAxes,
                horizontalalignment="center",
                verticalalignment="center",
                size=size,
                bbox=dict(boxstyle="square,pad=0", ec=bg_color, fc=bg_color, alpha=1),
                **kwargs,
            )
