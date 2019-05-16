"""
Copyright 2019 Zachary Phillips, Waller Lab, University of California, Berkeley

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np
from .base import zeros, crop, pad, asbackend, shape
import copy


class Roi():
    """Generic class for roi objects."""

    def __init__(self,
                 shape=None,
                 start=None,
                 end=None,
                 input_shape=None,
                 axis_labels=['t', 'z', 'y', 'x'],
                 units='pixels'):

        # Check that shape and end are not both defined, as these are contridictory
        assert not (shape and end), 'shape and end cannot both be provided.'

        # Initialize variables
        self._start, self._end, self._shape, self._input_shape = None, None, None, None

        # Set shape based on user inputs
        if end is not None:
            # Set start
            self.start = start

            # Use end to set shape
            self.end = end

        elif shape is not None:
            # Set shape
            self.shape = list(shape)

            # Set start value if provided
            self.start = start if start is not None else tuple([0] * len(shape))
            assert len(self.start) == len(self.shape)

            # Warn if end is provided as well
            if end is not None:
                print('WARNING: Ignoring "end" argument in the presence of start and shape')

        # Store axis labels
        self._axis_labels = axis_labels

        # Define input shape
        self.input_shape = tuple(input_shape) if input_shape is not None else None

        # Define units
        self.units = units  # Human-readable unit description

    @property
    def axis_labels(self):
        # Get axis labels
        return tuple([self._axis_labels[dim] for dim in range(self.ndim)])

    @axis_labels.setter
    def axis_labels(self, new_axis_labels):
        if new_axis_labels is not None:
            self._axis_labels = new_axis_labels

    @property
    def ndim(self):
        if self._shape:
            return len(self.shape)
        else:
            return 0

    @property
    def size(self):
        if self._shape:
            return tuple([int(sh) for sh in self._shape])

    @size.setter
    def size(self, new_shape):
        self._shape = list(new_shape)

    @property
    def shape(self):
        if self._shape:
            return tuple([int(sh) for sh in self._shape])

    @shape.setter
    def shape(self, new_shape):
        if new_shape is not None:
            self._shape = list(new_shape)

    @property
    def mask(self, backend=None):
        """Return a mask indicating the coverage of the Roi as an array."""
        if max(self.input_shape) is not np.inf:
            _mask = zeros(self.input_shape, dtype='int8', backend='numpy')
            _mask[self.slice] = 1
            return _mask
        else:
            raise ValueError('input_shape is not defined!')

    @property
    def start(self):
        """Return the start coordinates of the Roi as a tuple."""
        if self._start:
            return tuple([int(st) for st in self._start])

    @start.setter
    def start(self, new_start):
        if new_start is not None:
            self._start = list(new_start)

    @property
    def end(self):
        """Return the end coordinates of the Roi as a tuple."""
        return tuple([int(st + sh) for (st, sh) in zip(self._start, self._shape)])

    @end.setter
    def end(self, new_end):
        """Set the Roi's shape value to reflect the desired end point."""
        self._shape = [int(abs(end - start)) for (start, end) in zip(self.start, new_end)]

    @property
    def center(self, use_roi_coordinates=False):
        return tuple([int(st + sh // 2) for (st, sh) in zip(self._start, self._shape)])

    @property
    def slice(self):
        """Return a python slice object. Can be used for indexing numpy arrays directly."""
        # Check if input_shape is defined.
        _input_shape = self.input_shape if self.input_shape is not None else [np.inf] * self.ndim

        return tuple([slice(int(max(st, 0)), int(min(st + sh, input_shape))) for (st, sh, input_shape) in zip(self._start, self._shape, _input_shape)])

    @property
    def valid(self):
        """Return a new Roi which contains only regions inside input_shape."""
        # Check if input_shape is defined.
        if self.input_shape is not None:
            # Return new ROI
            return Roi(start=[int(max(st, 0)) for st in self._start],
                       end=[int(min(st + sh, input_shape)) for (st, sh, input_shape) in zip(self._start, self._shape, self.input_shape)],
                       input_shape=self.input_shape)
        else:
            return self
    # @property
    # def x_start(self):
    #     """Return the start of the crop region in the x dimension."""
    #     if 'x' in self.axis_labels:
    #         return self.start[self.axis_labels.index('x')]
    #     else:
    #         raise ValueError('Axis x is not defined in axis_labels!')
    #
    # @x_start.setter
    # def x_start(self, new_x_start):
    #     """Set the start parameter in the x dimension."""
    #     if 'x' in self.axis_labels:
    #         self._shape[self.axis_labels.index('x')] += self._start[self.axis_labels.index('x')] - new_x_start
    #         self._start[self.axis_labels.index('x')] = new_x_start
    #     else:
    #         raise ValueError('Axis x is not defined in axis_labels!')
    #
    # @property
    # def x_end(self):
    #     """Return the end of the crop region in the x dimension."""
    #     if 'x' in self.axis_labels:
    #         return self.end[self.axis_labels.index('x')]
    #     else:
    #         raise ValueError('Axis x is not defined in axis_labels!')
    #
    # @x_end.setter
    # def x_end(self, new_x_end):
    #     """Set the end parameter in the x dimension."""
    #     if 'x' in self.axis_labels:
    #         _end = list(self.end)
    #         _end[self.axis_labels.index('x')] = new_x_end
    #         self.end = _end
    #     else:
    #         raise ValueError('Axis x is not defined in axis_labels!')
    #
    # @property
    # def y_start(self):
    #     """Return the start of the crop region in the y dimension."""
    #     if 'y' in self.axis_labels:
    #         return self.start[self.axis_labels.index('y')]
    #     else:
    #         raise ValueError('Axis y is not defined in axis_labels!')
    #
    # @y_start.setter
    # def y_start(self, new_y_start):
    #     """Set the start parameter in the y dimension."""
    #     if 'y' in self.axis_labels:
    #         self._shape[self.axis_labels.index('y')] += self._start[self.axis_labels.index('y')] - new_y_start
    #         self._start[self.axis_labels.index('y')] = new_y_start
    #     else:
    #         raise ValueError('Axis y is not defined in axis_labels!')
    #
    # @property
    # def y_end(self):
    #     """Return the end of the crop region in the y dimension."""
    #     if 'y' in self.axis_labels:
    #         return self.end[self.axis_labels.index('y')]
    #     else:
    #         raise ValueError('Axis y is not defined in axis_labels!')
    #
    # @y_end.setter
    # def y_end(self, new_y_end):
    #     """Set the end parameter in the y dimension."""
    #     if 'y' in self.axis_labels:
    #         _end = list(self.end)
    #         _end[self.axis_labels.index('y')] = new_y_end
    #         self.end = _end
    #     else:
    #         raise ValueError('Axis y is not defined in axis_labels!')
    #
    # @property
    # def z_start(self):
    #     """Return the start of the crop region in the z dimension."""
    #     if 'z' in self.axis_labels:
    #         return self.start[self.axis_labels.index('z')]
    #     else:
    #         raise ValueError('Axis z is not defined in axis_labels!')
    #
    # @z_start.setter
    # def z_start(self, new_z_start):
    #     """Set the start parameter in the z dimension."""
    #     if 'z' in self.axis_labels:
    #         self._shape[self.axis_labels.index('z')] += self._start[self.axis_labels.index('z')] - new_z_start
    #         self._start[self.axis_labels.index('z')] = new_z_start
    #     else:
    #         raise ValueError('Axis z is not defined in axis_labels!')
    #
    # @property
    # def z_end(self):
    #     """Return the end of the crop region in the z dimension."""
    #     if 'z' in self.axis_labels:
    #         return self.end[self.axis_labels.index('z')]
    #     else:
    #         raise ValueError('Axis z is not defined in axis_labels!')
    #
    # @z_end.setter
    # def z_end(self, new_z_end):
    #     """Set the end parameter in the z dimension."""
    #     if 't' in self.axis_labels:
    #         _end = list(self.end)
    #         _end[self.axis_labels.index('z')] = new_z_end
    #         self.end = _end
    #     else:
    #         raise ValueError('Axis z is not defined in axis_labels!')
    #
    # @property
    # def t_start(self):
    #     """Return the start of the crop region in the t dimension."""
    #     if 't' in self.axis_labels:
    #         return self.start[self.axis_labels.index('t')]
    #     else:
    #         raise ValueError('Axis t is not defined in axis_labels!')
    #
    # @t_start.setter
    # def t_start(self, new_t_start):
    #     """Set the start parameter in the t dimension."""
    #     if 't' in self.axis_labels:
    #         self._start[self.axis_labels.index('t')] = new_t_start
    #     else:
    #         raise ValueError('Axis t is not defined in axis_labels!')
    #
    # @property
    # def t_end(self):
    #     """Return the end of the crop region in the t dimension."""
    #     if 't' in self.axis_labels:
    #         return self.end[self.axis_labels.index('t')]
    #     else:
    #         raise ValueError('Axis t is not defined in axis_labels!')
    #
    # @t_end.setter
    # def t_end(self, new_t_end):
    #     """Set the end parameter in the z dimension."""
    #     if 't' in self.axis_labels:
    #         _end = self.end
    #         _end[self.axis_labels.index('t')] = new_t_end
    #         self.end = _end
    #     else:
    #         raise ValueError('Axis t is not defined in axis_labels!')

    def overlaps(self, other):
        """Checks if this roi overlaps another."""
        assert self.__class__.__name__ is other.__class__.__name__, "Second input must be ROI object!"

        # Test overlap
        test_1 = all([start < end for (start, end) in zip(self.start, other.end)])
        test_2 = all([start < end for (start, end) in zip(other.start, self.end)])

        return (test_1 and test_2)

    def copy(self):
        return self.__deepcopy__(None)

    def draw(self, ax=None, c='r', linewidth=2, **kwargs):
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        # Create matplotlib patch
        rect = mpatches.Rectangle(list(reversed(self.start)), self.shape[1], self.shape[0],
                                  edgecolor=c, facecolor=None, linewidth=linewidth,
                                  fill=False, **kwargs)
        ax.add_artist(rect)

    def decimate(self, decimation_factor=2):
        """Returns equliivant ROI for decimated image."""

        # Compute new shapes
        start = [st // decimation_factor for st in self.start]
        shape = [sh // decimation_factor for sh in self.shape]
        input_shape = [ish // decimation_factor for ish in self.input_shape] if self.input_shape is not None else None

        # Return new Roi
        return Roi(start=start, shape=shape, input_shape=input_shape, units=self.units)

    def __add__(self, val):
        """Roi addition."""

        # Create new ROI object to return
        new_roi = copy.deepcopy(self)

        # Operate based on type of val
        if isinstance(val, (list, tuple, np.ndarray)):
            if len(val) == 1:
                new_roi.start = tuple([start + val for start in new_roi._start])
            elif len(val) == 2:
                new_roi.start = tuple([start + offset for (start, offset) in zip(new_roi._start, val)])
        elif self.__class__.__name__ is val.__class__.__name__:
            # Check the units
            assert new_roi.units == val.units, "Intersecting ROI should have the same units."

            # Check the mininum values
            new_roi.start = tuple([min(our_start, their_start) for (our_start, their_start) in zip(self.start, val.start)])

            # Check the maximum valyes
            new_roi.end = tuple([max(our_end, their_end) for (our_end, their_end) in zip(self.end, val.end)])

        else:
            new_roi._start = tuple([start + val for start in new_roi._start])

        return new_roi

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, val):
        """Roi addition."""

        # Operate based on type of val
        if isinstance(val, (list, tuple, np.ndarray)):
            if len(val) == 1:
                self.start = tuple([start + val for start in self._start])
            elif len(val) == 2:
                self.start = tuple([start + offset for (start, offset) in zip(self._start, val)])
        elif self.__class__.__name__ is val.__class__.__name__:
            # Check the units
            assert self.units == val.units, "Intersecting ROI should have the same units."

            # Check the mininum values
            self.start = tuple([min(our_start, their_start) for (our_start, their_start) in zip(self.start, val.start)])

            # Check the maximum valyes
            self.end = tuple([max(our_end, their_end) for (our_end, their_end) in zip(self.end, val.end)])
        else:
            self._start = tuple([start + val for start in self._start])

        return self

    def __sub__(self, val):
        """Roi subtraction."""

        # Create new ROI object to return
        new_roi = copy.deepcopy(self)

        # Operate based on type of val
        if isinstance(val, (list, tuple, np.ndarray)):
            if len(val) == 1:
                new_roi.start = tuple([start - val for start in new_roi._start])
            elif len(val) == 2:
                new_roi.start = tuple([start - offset for (start, offset) in zip(new_roi._start, val)])
        elif self.__class__.__name__ is val.__class__.__name__:
            raise ValueError('Subtraction is not supported for Roi objects inputs.')
        else:
            new_roi._start = tuple([start - val for start in new_roi._start])

        return new_roi

    def __isub__(self, val):
        """Roi subtraction."""

        # Operate based on type of val
        if isinstance(val, (list, tuple, np.ndarray)):
            if len(val) == 1:
                self.start = tuple([start - val for start in self._start])
            elif len(val) == 2:
                self.start = tuple([start - offset for (start, offset) in zip(self._start, val)])
        elif self.__class__.__name__ is val.__class__.__name__:
            raise ValueError('Subtraction is not supported for Roi objects inputs.')
        else:
            self._start = tuple([start - val for start in self._start])

        return self

    def __mul__(self, val):
        """Serial Application of Multiple Rois."""

        # val is inner operator
        roi_new = copy.deepcopy(val)

        if self.__class__.__name__ is val.__class__.__name__:
            # Check the units
            assert self.units == val.units, "Intersecting ROI should have the same units."

            if max(val.input_shape) is not np.inf:
                assert all([idim == sh for (idim, sh) in zip(val.input_shape, self.shape)])

            # Check the mininum values
            roi_new.start = tuple([our_start + their_start for (our_start, their_start) in zip(self.start, val.start)])

            # Check the maximum values
            roi_new.end = tuple([new_start + their_shape for (new_start, their_shape) in zip(roi_new.start, val.shape)])

            # Set new input shape
            roi_new.input_shape = self.input_shape
        else:
            raise ValueError('Multiplication is only supported for two roi objects')

        return roi_new

    def __and__(self, val):
        """Intersection of two ROIs"""

        new_roi = copy.deepcopy(self)

        if self.__class__.__name__ is val.__class__.__name__:
            # Check the units
            assert self.units == val.units, "Intersecting ROI should have the same units."

            # Check the mininum values
            new_roi.start = tuple([max(our_start, their_start) for (our_start, their_start) in zip(self.start, val.start)])

            # Check the maximum valyes
            new_roi.end = tuple([min(our_end, their_end) for (our_end, their_end) in zip(self.end, val.end)])
        else:
            raise ValueError('Intersection is only supported for two roi objects')

        return new_roi

    def __deepcopy__(self, memo_dict):
        return Roi(start=copy.deepcopy(self.start),
                   shape=copy.deepcopy(self.shape),
                   input_shape=copy.deepcopy(self.input_shape),
                   units=copy.deepcopy(self.units))

    def __str__(self):
        if self._start:
            return 'Roi(start=%s, end=%s, shape=%s)' % (str(self.start), str(self.end), str(self.shape))
        else:
            return 'Roi()'

    def __repr__(self):
        if self._start:
            repr = 'Roi: start=%s, end=%s, shape=%s' % (str(self.start), str(self.end), str(self.shape))
            if self.input_shape is not None:
                repr += ', input_shape: %s' % str(self.input_shape)
        else:
            repr = 'Roi(Empty)'

        return repr

    @property
    def __dict__(self):
        return {'start': self.start,
                'shape': self.shape,
                'units': self.units,
                'axis_labels': self.axis_labels,
                'input_shape': self.input_shape}


def selectRoi(img, message='Select Region of Interest', roi=None):
    from skimage.viewer import ImageViewer
    from skimage.viewer.canvastools import RectangleTool
    if roi is None:
        roi = Roi()
    viewer = ImageViewer(img)
    viewer.fig.suptitle(message)
    rect = RectangleTool(viewer)
    viewer.show()
    extents = np.round(rect.extents).astype('int64')

    roi.x_start = extents[0]
    roi.x_end = extents[1]
    roi.y_start = extents[2]
    roi.y_end = extents[3]
    return roi


def copy_roi(x, y, x_roi=None, y_roi=None):
    """Crop from one ROI to another"""
    if x_roi is None and y_roi is None:
        # Why did you call this function...?
        y[:] = x[:]
    elif y_roi is None and x_roi is not None:
        # Input has an ROI which copies directly to the output
        y[:] = x[x_roi.slice]
    elif x_roi is None and y_roi is not None:
        # Output has an ROI which copies relevent information from the input
        x_roi = copy.deepcopy(y_roi) - y_roi.start
        y[y_roi.slice] = x[x_roi.slice]
    else:
        # Perform ROI algebra
        roi_overlap = x_roi * y_roi
        x_roi_crop = roi_overlap - x_roi.start
        y_roi_crop = roi_overlap - y_roi.start

        # Set values in y
        y[y_roi_crop.slice] = x[x_roi_crop.slice]


def crop_roi(x, roi, y=None, out_of_bounds_placeholder=None):
    """Helper function for cropping with a Roi() object"""
    return crop(x, roi.shape, crop_start=roi.start,
                out_of_bounds_placeholder=out_of_bounds_placeholder, y=y,
                center=False)


def pad_roi(x, roi, pad_value=0, y=None):
    """Helper function for padding with a Roi() object"""
    return pad(x, roi.input_shape, crop_start=roi.start, pad_value=pad_value,
               y=y, center=False)


def crop_to_support(x):
    """Crop an array to it's support (non-zero values)"""
    return x[boundingBox(x, return_roi=True).slice]


def boundingBox(x, return_roi=False):
    """Return the bounding box of an array-like.

    This function returns a bounding box of all non-zero values in an array.
    It also treats Nan values as zero by design (ignoring these).

    Parameters
    ----------
    x: array-like
        The array we wish to bound.
    return_roi: bool
        Whether to return a Roi Object. If false, returns in the format:
        (y_start, y_end, x_start, x_end)

    Returns
    -------
    bbox: Roi or list
        coordinates of the bounding box, returned either as a Roi object or as
        a tuple of the format (y_start, y_end, x_start, x_end)

    """

    # Convert to numpy backend
    x = asbackend(x, 'numpy')

    # Get indicies of values
    indicies = np.where(abs(x) > 0)
    bbox = np.min(indicies[0]), np.max(indicies[0]) + 1, np.min(indicies[1]), np.max(indicies[1]) + 1
    if return_roi:
        return Roi(start=(np.min(indicies[0]), np.min(indicies[1])), end=(np.max(indicies[0]) + 1, np.max(indicies[1]) + 1), input_shape=shape(x))
    else:
        # return start, shape
        return (bbox[0], bbox[2]), (bbox[1] - bbox[0], bbox[3] - bbox[2])


def getOverlapRegion(image_pair, roi_pair):
    """Returns a tuple of images containing the overlap region of two ROIs.

    Parameters
    ----------
    image_pair: tuple
        A tuple of two array-like to crop (image_0, image_1)
    roi_pair: bool
        A tuple of two Roi objects to use for crop (roi_0, roi_0)

    Returns
    -------
    tuple:
        Crop of overlap regions of the images (overlap_0, overlap_1)

    """

    # Check dimensions
    assert len(image_pair) == len(roi_pair) == 2, "Length of image and roi pairs should be 2."

    # Unpack variables
    image_0, image_1 = image_pair
    roi_0, roi_1 = roi_pair

    # # Check dimensions
    # assert all([shape(image_0)[i] == roi_0.shape[i] for i in range(len(roi_0.shape))]), "Image 0 shape (%s) != Roi 0 shape (%s)" % (str(shape(image_0)), str(roi_0.shape))
    # assert all([shape(image_1)[i] == roi_1.shape[i] for i in range(len(roi_1.shape))]), "Image 1 shape (%s) != Roi 1 shape (%s)" % (str(shape(image_1)), str(roi_1.shape))

    # Get overlap
    roi_overlap = roi_0 & roi_1
    roi_overlap_0 = roi_overlap - roi_0.start
    roi_overlap_1 = roi_overlap - roi_1.start

    # Return
    return image_0[roi_overlap_0.slice], image_1[roi_overlap_1.slice]
