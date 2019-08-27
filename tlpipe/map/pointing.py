import numpy as np
import scipy as sp
import _mapmaker as _mapmaker_c
from tlpipe.map import algebra as al


class Pointing(object):
    """Class represents the pointing operator.

    The pointing operator converts from the map domain to the time domain in
    its native form and from the time domain to the map domain in its
    transposed form.
    
    Parameters
    ----------
    axis_names: tuple of strings
        The names of the axes in the map domain e.g. ("ra", "dec")
    coords : tuple of 1D arrays
        Tuple must be same length as `axis_names`.  The coordinates as a 
        function of time for each of the map axes.
    map : al.vect object
        The map that we will be gridding onto.  Map axes must include
        `axis_names`.  No modification to the map is made, only the axis
        information is used.
    scheme : string
        Gridding scheme to use.  Choices are 'nearest'.
    """

    def __init__(self, axis_names, coords, map, scheme='nearest'):
        # Sanity check some of the inputs.
        if len(axis_names) != len(coords):
            msg = "Got %d pointing axis names, but got %d coordinate arrays."
            raise ValueError(msg % (len(axis_names), len(coords)))
        n_pointings = coords[0].shape[0]
        for coordinate_array in coords:
            if coordinate_array.shape != (n_pointings,):
                msg = "Coordinate arrays must all be 1D and same length."
                raise ValueError(msg)
        # Save the data we need for the pointing.
        self._axis_names = axis_names
        self._coords = tuple(coords) # A tuple of 1D arrays.
        self._scheme = scheme
        # Get all coordinate information from the map.
        # Figure out which axes we are doing the pointing for, get the shape of
        # those axes, the coordinate centres and the pixel size.
        map_axis_indices = ()
        map_axes_shape = ()
        #map_axes_centre = ()
        #map_axes_delta = ()
        list_map_axis_names = list(map.axes)
        for axis_name in axis_names:
            axis_index = list_map_axis_names.index(axis_name)
            map_axis_indices += (axis_index,)
            map_axes_shape += (map.shape[axis_index],)
            #map_axes_centre += (map.info[axis_name + "_centre"],)
            #map_axes_delta += (map.info[axis_name + "_delta"],)
        self._map_axes = map_axis_indices
        self._map_shape = map_axes_shape
        
        # Store the full pointing matrix in sparse form.
        n_pointings = len(coords[0])
        n_coords = len(axis_names)
        self.dtype = np.float
        # Loop over the time stream and get the weights for each pointing.
        memory_allocated = False
        for ii in xrange(n_pointings):
            coordinate = ()
            for jj in xrange(n_coords):
                coordinate += (self._coords[jj][ii],)
            pixels, weights = map.slice_interpolate_weights(
                self._map_axes, coordinate, scheme)
            # On first iteration need to allocate memory for the sparse matrix
            # storage.
            if not memory_allocated:
                n_points_template = pixels.shape[0]
                self._pixel_inds = sp.zeros((n_pointings, n_coords,
                                              n_points_template), dtype=np.int)
                self._weights = sp.zeros((n_pointings, n_points_template),
                                         dtype=self.dtype)
                memory_allocated = True
            self._pixel_inds[ii,:,:] = pixels.transpose()
            self._weights[ii,:] = weights
    
    def get_sparse(self):
        """Return the arrays representing the pointing matrix in sparse form.

        Returns
        -------
        pixel_inds : array of ints
            Shape is (n_pointings, n_coordinates, n_pixels_per_pointing).
            These give which pixel entries are that are non zero for each
            pointing.  If a pointing is off the map, that row will be zeros.
        weights : array of floats
            Shape is (n_pointings, n_pixels_per_pointing).
            The entry of the pointing matrix corresponding with the matching
            pixel index.  If a pointing is off the map, that row will be zeros.
        """
        return self._pixel_inds, self._weights

    def apply_to_time_axis(self, time_stream, map_out=None):
        """Use this operator to convert a 'time' axis to a coordinate axis.
        
        This functions implements a fast matrix multiplication. It is roughly
        equivalent to using `algebra.partial_dot` except in axis placement.
        This function "replaces" the time axis with the map axes which is
        different from `partial_dot`'s behaviour.  This function is much more
        efficient than using `partial_dot`.
        
        For input
        `map`, the following operations should be equivalent, with the later
        much more efficient.
        
        # XXX This example is wrong and broken.
        >>> a = al.partial_dot(map)
        >>> b = self.apply_to_time_axis(map)
        >>> sp.allclose(a, b)
        True
        """
        
        if not isinstance(time_stream, al.vect):
            raise TypeError("Input data must be an algebra.vect object.")
        # Find the time axis for the input.
        for ii in range(time_stream.ndim):
            if time_stream.axes[ii] == 'time':
                time_axis = ii
                break
        else :
            raise ValueError("Input data vect doesn't have a time axis.")
        # Get some dimensions.
        n_pointings = self._pixel_inds.shape[0]
        n_pixels_template = self._pixel_inds.shape[2]
        n_axes = len(self._axis_names)
        if time_stream.shape[time_axis] != n_pointings:
            msg = ("Time stream data and pointing have different number of"
                   " time points.")
            raise ValueError(msg)
        # Get the shape and axis names of the output.
        out_shape = (time_stream.shape[:time_axis] + self._map_shape
                     + time_stream.shape[time_axis + 1:])
        out_axes = (time_stream.axes[:time_axis] + self._axis_names
                     + time_stream.axes[time_axis + 1:])
        # Allowcate output memory if not passed.
        if map_out is None:
            map_out = sp.zeros(out_shape, dtype=float)
            map_out = al.make_vect(map_out, axis_names=out_axes)
        else :
            if map_out.shape != out_shape:
                raise ValueError("Output array is the wrong shape.")
        # Initialize tuples that will index the input and the output.
        data_index = [slice(None),] * time_stream.ndim + [None]
        out_index = [slice(None),] * map_out.ndim
        # Loop over the time axis and do the dot.
        for ii in xrange(n_pointings):
            data_index[time_axis] = ii
            for kk in xrange(n_axes):
                out_index[time_axis + kk] = self._pixel_inds[ii,kk,:]
            map_out[tuple(out_index)] += (self._weights[ii,:]
                                   * time_stream[tuple(data_index)])
        return map_out

    def get_matrix(self):
        """Gets the matrix representation of the pointing operator."""

        n_pointings = self._pixel_inds.shape[0]
        n_coords = self._pixel_inds.shape[1]
        n_pixels_per_pointing = self._pixel_inds.shape[2]
        # Initialize the output matrix.
        matrix = sp.zeros((n_pointings,) + self._map_shape, dtype=self.dtype)
        matrix = al.make_mat(matrix, axis_names=("time",) + self._axis_names,
                             row_axes=(0,), col_axes=range(1, n_coords + 1))
        # Loop over the time stream and get the weights for each pointing.
        for ii in xrange(n_pointings):
            for jj in xrange( n_pixels_per_pointing):
                matrix[(ii,) + tuple(self._pixel_inds[ii,:,jj])] = \
                        self._weights[ii, jj]
        return matrix

    def noise_to_map_domain(self, Noise, f_ind, ra_ind, map_noise_inv):
        """Convert noise to map space.
        
        For performace and IO reasons this is done with a call to this function
        for each frequency row and each ra row.  All dec rows and all columns
        are handled in this function simultaniousely.

        This function is designed to be thread safe in that if it is called
        from two separate threads but with different `f_ind` or `ra_ind`, there
        should be no race conditions.
        """
        
        Noise._assert_finalized()
        if Noise._frequency_correlations:
            _mapmaker_c.update_map_noise_chan_ra_row(Noise.diagonal_inv,
                    Noise.freq_modes, Noise.time_modes, Noise.freq_mode_update,
                    Noise.time_mode_update, Noise.cross_update, 
                    self._pixel_inds, self._weights, f_ind, ra_ind,
                    map_noise_inv)
        else:
            msg = ("Noise object has no frequency correlations.  Use "
                   " `noise_channel_to_map` instead.")
            raise RuntimeError(msg)

    def noise_channel_to_map(self, Noise, f_ind, map_noise_inv):
        """Convert noise to map space.
        
        Use this function over `noise_to_map_domain` if the noise has no
        frequency correlations.
        """

        Noise._assert_finalized()
        if not Noise._frequency_correlations:
            _mapmaker_c.update_map_noise_independant_chan(Noise.diagonal_inv,
                    Noise.time_modes, Noise.time_mode_update, self._pixel_inds,
                    self._weights, f_ind, map_noise_inv)
        else:
            msg = ("Noise object has frequency correlations.  Use "
                   " `noise_to_map_domain` instead.")
            raise RuntimeError(msg)


