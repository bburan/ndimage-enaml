import logging
log = logging.getLogger(__name__)

from atom.api import Atom, Bool, Dict, Float, Int, List, Str, Typed
from matplotlib import transforms as T
import numpy as np
from raster_geometry import sphere
from scipy import ndimage
from scipy import signal

from .util import get_image


class ChannelConfig(Atom):

    name = Str()
    min_value = Float(0)
    max_value = Float(1)
    visible = Bool(True)


class NDImage(Atom):

    #: Should be a dictionary containing the following keys:
    #: * lower: image origin (lower left)
    #: * voxel_size: size of a single image voxel
    #: * channels: list of channels as a dictionary with 'name' as key
    #: Both `lower` and `voxel_size` should be in same units.
    info = Dict()
    image = Typed(np.ndarray)
    extent = List()
    n_channels = Int()
    source = Str()

    def __init__(self, info, image):
        self.info = info
        self.image = image
        xlb, ylb, zlb = self.info["lower"][:3]

        # Images are in XYZC dimension. We need to calculate the upper extent
        # of the image so we can properly plot it.
        xpx, ypx, zpx = self.image.shape[:3]
        xv, yv, zv = self.info['voxel_size'][:3]
        xub = xlb + xpx * xv
        yub = ylb + ypx * yv
        zub = zlb + zpx * zv
        self.extent = [xlb, xub, ylb, yub, zlb, zub]
        self.n_channels = self.image.shape[-1]

    @property
    def channel_names(self):
        return [c['name'] for c in self.info['channels']]

    def contains(self, x, y):
        contains_x = self.extent[0] <= x <= self.extent[1]
        contains_y = self.extent[2] <= y <= self.extent[3]
        return contains_x and contains_y

    def to_coords(self, x, y, z=None):
        lower = self.info["lower"]
        voxel_size = self.info["voxel_size"]
        if z is None:
            indices = np.c_[x, y, np.full_like(x, lower[-1])]
        else:
            indices = np.c_[x, y, z]
        points = (indices * voxel_size) + lower
        if z is None:
            return points[:, :2].T
        return points.T

    def to_indices(self, x, y, z=None):
        lower = self.info["lower"]
        voxel_size = self.info["voxel_size"]
        if z is None:
            points = np.c_[x, y, np.full_like(x, lower[-1])]
        else:
            points = np.c_[x, y, z]
        indices = (points - lower) / voxel_size
        if z is None:
            return indices[:, :2].T
        return indices.T

    def to_indices_delta(self, v, axis='x'):
        if axis == 'x':
            return v / self.info['voxel_size'][0]
        elif axis == 'y':
            return v / self.info['voxel_size'][1]
        elif axis == 'z':
            return v / self.info['voxel_size'][2]
        else:
            raise ValueError('Unsupported axis')

    def nuclei_template(self, radius=2.5):
        voxel_size = self.info["voxel_size"][0]
        pixel_radius = int(np.round(radius / voxel_size))
        template = sphere(pixel_radius * 3, pixel_radius)
        return template / template.sum()

    def get_image_extent(self, axis='z', norm=False):
        e = np.array(self.extent).reshape((3, 2))
        if norm:
            e = e - e[:, [0]]
        extent = e.ravel().tolist()
        x = extent[0:2]
        y = extent[2:4]
        z = extent[4:6]
        if axis == 'x':
            return tuple(y + z)
        if axis == 'y':
            return tuple(x + z)
        if axis == 'z':
            return tuple(x + y)

    def get_image_transform(self):
        return T.Affine2D().rotate_deg_around(*self.get_image_center(),
                                              self.get_rotation())

    def get_rotated_extent(self):
        '''
        Calculate the new extents of the tile after rotation.

        This assumes that the tile is rotated using scipy.ndimage where the
        resulting array is reshaped to ensure that the input image is contained
        entirely in the output image.
        '''
        e = self.extent[:]
        ll = e[0], e[2]
        lr = e[1], e[2]
        ul = e[0], e[3]
        ur = e[1], e[3]
        coords = np.array([ll, lr, ur, ul, ll])
        t_coords = self.get_image_transform().transform(coords)
        xlb, ylb = t_coords.min(axis=0)
        xub, yub = t_coords.max(axis=0)
        e[:4] = xlb, xub, ylb, yub
        return e

    def get_image_center(self, axis='z', norm=False):
        extent = self.get_image_extent()
        center = np.array(extent).reshape((2, 2)).mean(axis=1)
        return tuple(center)

    def get_rotation(self):
        return self.info.get('rotation', 0)

    def get_image(self, channels=None, z_slice=None, axis='z',
                  norm_percentile=99):
        return get_image(self.image, channel_names=self.channel_names,
                         channels=channels, z_slice=z_slice, axis=axis,
                         norm_percentile=norm_percentile)

    def get_state(self):
        return {"extent": self.extent}

    def set_state(self, state):
        self.extent = state["extent"]

    def map(self, x, y, channel, smooth_radius=2.5, width=5):
        """
        Calculate intensity in the specified channel for the xy coordinates.

        Optionally apply image smoothing and/or a maximum search.
        """
        # get_image returns a Nx3 array where the final dimension is RGB color.
        # We are only requesting one channel, but it is possible that the
        # information in the channel will be split among multiple RGB colors
        # depending on the specific color it is coded as. The sum should never
        # exceed 255.
        image = self.get_image(channel).sum(axis=-1)
        if smooth_radius:
            template = self.nuclei_template(smooth_radius)
            template = template.mean(axis=-1)
            image = signal.convolve2d(image, template, mode="same")

        if width:
            x, y = util.expand_path(x, y, width)

        xi, yi = self.to_indices(x.ravel(), y.ravel())
        i = ndimage.map_coordinates(image, [xi, yi])

        i.shape = x.shape
        if width is not None:
            i = i.max(axis=0)
        return i

    def center(self, dx, dy):
        '''
        Center tile origin with respect to dx and dy

        This is used for attempting to register images using phase cross-correlation
        '''
        extent = np.array(self.extent)
        width, height = extent[1:4:2] - extent[:4:2]
        self.extent = [dx, dx + width, dy, dy + height] + extent[4:]

    @property
    def z_slice_max(self):
        # This is the maximum z-slice in the image
        return self.image.shape[2]

    def get_voxel_size(self, dim):
        return self.info['voxel_size']['xyz'.index(dim)]


class NDImageCollection(Atom):

    tiles = List(NDImage)

    def __init__(self, tiles):
        super().__init__(tiles=tiles)

    def __iter__(self):
        yield from self.tiles

    def __len__(self):
        return len(self.tiles)

    def get_image_extent(self):
        return self._get_extent(lambda t: t.get_image_extent())

    def get_rotated_extent(self):
        return self._get_extent(lambda t: t.get_rotated_extent())

    def _get_extent(self, cb):
        extents = np.vstack([cb(t) for t in self.tiles])
        xmin = extents[:, 0].min()
        xmax = extents[:, 1].max()
        ymin = extents[:, 2].min()
        ymax = extents[:, 3].max()
        return [xmin, xmax, ymin, ymax]
