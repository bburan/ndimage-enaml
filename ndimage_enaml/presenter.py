from copy import deepcopy
import threading
import time

from atom.api import (
    Atom,
    Bool,
    Dict,
    Enum,
    Event,
    Float,
    Instance,
    Int,
    List,
    observe,
    Property,
    set_default,
    Str,
    Tuple,
    Typed,
    Value,
)

from enaml.application import deferred_call

import matplotlib as mp
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib import patheffects as pe

from matplotlib import patches as mpatches
from matplotlib import path as mpath
from matplotlib import transforms as T

import numpy as np

from .model import ChannelConfig, NDImage, NDImageCollection


class NDImagePlot(Atom):

    alpha = Float(0.75)
    highlight = Bool(False)
    zorder = Int(10)

    channel_config = Value()

    display_mode = Enum("projection", "slice")
    display_channels = List()
    visible_channels = Property()
    extent = Tuple()
    z_slice = Property()
    z_slice_lb = Int(0)
    z_slice_ub = Int(1)
    z_slice_thickness = Int(1)
    constrain_z_slice_thickness = Bool(True)
    z_slice_min = Int(0)
    z_slice_max = Int(0)
    shift = Float()

    ndimage = Value()
    artist = Value()
    rectangle = Value()
    axes = Value()
    auto_rotate = Bool(True)
    rotation_transform = Value()
    transform = Value()

    updated = Event()
    needs_redraw = Bool(False)

    z_slice_update = Value()

    def _get_visible_channels(self):
        return [c.name for c in self.channel_config.values() if c.visible]

    def _default_z_slice_update(self):
        return threading.Event()

    def get_state(self):
        return {
            "alpha": self.alpha,
            "zorder": self.zorder,
            "display_mode": self.display_mode,
            "display_channels": self.display_channels,
            "shift": self.shift,
        }

    def set_state(self, state):
        self.alpha = state["alpha"]
        self.zorder = state["zorder"]
        self.display_mode = state["display_mode"]
        self.display_channels = state["display_channels"]
        self.shift = state["shift"]

    def __init__(self, axes, ndimage=None, **kwargs):
        super().__init__(**kwargs)
        self.axes = axes
        self.axes.set_axis_off()
        self.rotation_transform = T.Affine2D()
        self.transform = self.rotation_transform + axes.transData
        self.artist = AxesImage(axes, data=np.array([[]]), origin='lower', transform=self.transform)
        axes.add_artist(self.artist)

        self.rectangle = mp.patches.Rectangle((0, 0), 0, 0, ec='red', fc='None', zorder=5000, transform=self.transform)
        self.rectangle.set_alpha(0)
        self.axes.add_patch(self.rectangle)
        self.axes.axis('equal')

        if ndimage is not None:
            self.ndimage = ndimage

    def _observe_ndimage(self, event):
        if event.get('oldvalue', None) is not None:
            event['oldvalue'].unobserve('extent', self.request_redraw)

        ndimage = event['value']
        self.z_slice_max = ndimage.z_slice_max
        self.shift = ndimage.get_voxel_size('x') * 5

        self.channel_config = {n: ChannelConfig(**c) for n, c in ndimage.channel_config.items()}
        for config in self.channel_config.values():
            config.observe('visible', self.request_redraw)
            config.observe('min_value', self.request_redraw)
            config.observe('max_value', self.request_redraw)
            config.observe('display_color', self.request_redraw)
        ndimage.observe('extent', self.request_redraw)

    def next_z_slice(self, step):
        if self.display_mode == 'projection':
            if step > 0:
                self.z_slice_lb = 0
            else:
                self.z_slice_lb = self.z_slice_max - self.z_slice_thickness
            self.display_mode = 'slice'
        else:
            if 0 <= (self.z_slice_lb + step) <= (self.z_slice_max - self.z_slice_thickness):
                self.z_slice_lb += step
                self.display_mode = 'slice'
            else:
                self.display_mode = 'projection'

    def center_z_substack(self, z):
        thickness = self.z_slice_thickness
        z_max = self.z_slice_max

        self.z_slice_update.set()
        lb = np.clip(z - np.floor(thickness / 2), 0, z_max)
        ub = np.clip(z + np.ceil(thickness / 2), 0, z_max)
        self.z_slice_lb = int(lb)
        self.z_slice_ub = int(ub)
        self.z_slice_update.clear()

    @observe('z_slice_lb', 'z_slice_ub', 'z_slice_thickness')
    def _update_z(self, event):
        # Skip if we're in the process of updating the details
        if self.z_slice_update.is_set():
            return
        self.z_slice_update.set()

        lb, ub = self.z_slice_lb, self.z_slice_ub
        constrain = self.constrain_z_slice_thickness
        thickness = self.z_slice_thickness
        name = event['name']
        z_max = self.z_slice_max

        if thickness < 1:
            thickness = 1

        if constrain and name == 'z_slice_lb':
            ub = lb + thickness
        elif constrain and name == 'z_slice_ub':
            lb = ub - thickness
        elif name == 'z_slice_thickness':
            if thickness > z_max:
                thickness = z_max
            ub = lb + thickness
        if lb == ub:
            ub = lb + 1
        if lb < 0:
            lb = 0
            ub = lb + thickness
        if ub > z_max:
            lb = z_max - thickness
            ub = z_max
        thickness = ub - lb

        # By deferring this update, we ensure that the slider is properly
        # updated rather than staying at the last value forced by the user.
        def update_z():
            nonlocal self
            nonlocal lb
            nonlocal ub
            nonlocal thickness
            self.z_slice_lb = lb
            self.z_slice_ub = ub
            self.z_slice_thickness = thickness
            self.z_slice_update.clear()
        deferred_call(update_z)

    def _observe_highlight(self, event):
        if self.highlight:
            self.rectangle.set_alpha(1)
        else:
            self.rectangle.set_alpha(0)

    def _observe_alpha(self, event):
        self.artist.set_alpha(self.alpha)

    def _observe_zorder(self, event):
        self.artist.set_zorder(self.zorder)

    def _get_z_slice(self):
        return np.s_[self.z_slice_lb:self.z_slice_ub]

    def drag_image(self, dx, dy):
        extent = np.array(self.ndimage.extent)
        extent[0:2] += dx
        extent[2:4] += dy
        self.ndimage.extent = extent.tolist()

    def move_image(self, direction, step_scale=1):
        extent = np.array(self.ndimage.extent)
        step = step_scale * self.shift
        if direction == "up":
            extent[2:4] += step
        elif direction == "down":
            extent[2:4] -= step
        elif direction == "left":
            extent[0:2] -= step
        elif direction == "right":
            extent[0:2] += step
        self.ndimage.extent = extent.tolist()

    @observe("z_slice_lb", "z_slice_ub", "display_mode", "alpha", "highlight")
    def request_redraw(self, event=None):
        self.needs_redraw = True
        deferred_call(self.redraw_if_needed)

    def redraw_if_needed(self):
        if self.needs_redraw:
            self.redraw()
            self.needs_redraw = False

    def get_image(self):
        z_slice = None if self.display_mode == 'projection' else self.z_slice
        channels = [c for c in self.channel_config.values() if c.visible]
        return self.ndimage.get_image(channels=channels, z_slice=z_slice).swapaxes(0, 1)

    def redraw(self, event=None):
        self.artist.set_data(self.get_image())
        xlb, xub, ylb, yub = extent = self.ndimage.get_image_extent()[:4]
        self.artist.set_extent(extent)
        self.rectangle.set_bounds(xlb, ylb, xub-xlb, yub-ylb)
        t = self.ndimage.get_image_transform()
        if self.auto_rotate:
            self.rotation_transform.set_matrix(t.get_matrix())
        self.updated = True

    def contains(self, x, y):
        return self.ndimage.contains(x, y)

    def set_channel_visible(self, channel_name, visible):
        self.channel_config[channel_name].visible = visible

    def set_channel_min_value(self, channel_name, min_value):
        self.channel_config[channel_name].min_value = min_value

    def set_channel_max_value(self, channel_name, max_value):
        self.channel_config[channel_name].max_value = max_value


class FigurePresenter(Atom):
    #: Parent figure of the axes
    figure = Typed(Figure)

    #: Axes on which all artists will be rendered
    axes = Typed(Axes)

    #: Flag indicating whether the figure needs to be redrawn
    needs_redraw = Bool(False)

    def button_press(self, event):
        if event.button == MouseButton.LEFT:
            self.left_button_press(event)
        elif event.button == MouseButton.RIGHT:
            self.right_button_press(event)

    def left_button_press(self, event):
        return

    def right_button_press(self, event):
        return

    def button_release(self, event):
        return

    def scroll(self, event):
        return

    def motion(self, event):
        return

    def key_press(self, event):
        return

    def _default_figure(self):
        figure = Figure(facecolor='black')
        figure.canvas.mpl_connect('key_press_event', lambda e: self.key_press(e))
        figure.canvas.mpl_connect('button_press_event', lambda e: self.button_press(e))
        figure.canvas.mpl_connect('button_release_event', lambda e: self.button_release(e))
        figure.canvas.mpl_connect('scroll_event', lambda e: self.scroll(e))
        figure.canvas.mpl_connect('motion_notify_event', lambda e: self.motion(e))
        return figure

    def _default_axes(self):
        return self.figure.add_axes([0, 0, 1, 1])

    def request_redraw(self, event=None):
        self.needs_redraw = True
        deferred_call(self.redraw_if_needed)

    def redraw(self):
        self.figure.canvas.draw()

    def redraw_if_needed(self):
        if self.needs_redraw:
            self.redraw()
            self.needs_redraw = False


class NDImageCollectionPresenter(FigurePresenter):

    #: Object being presented
    obj = Typed(NDImageCollection)

    #: Currently selected ndimage artist
    current_artist = Value()

    #: Dictionary of all ndimage artists
    ndimage_artists = Dict()

    #: Minimum z-slice across all ndimage in image
    z_min = Property()

    #: Maximum z-slice across all ndimage in image
    z_max = Property()

    #: Starting pan event
    pan_event = Value(None)
    pan_xlim = Value()
    pan_ylim = Value()

    #: True if we actually had a pan event. This allows us to distinguish
    #: between clicks that select a ndimage vs. clicks that are intended to start
    #: a pan.
    pan_performed = Bool(False)

    drag_event = Value(None)
    drag_x = Value()
    drag_y = Value()

    current_artist_index = Int()

    #: Track timestamp of last scroll event recieved to ensure that we don't
    #: zoom too quickly.
    last_scroll_time = Value(time.time())

    #: If True, rotate ndimage so they represent the orientation they were imaged
    #: on the confocal.
    rotate_ndimage = Bool(True)

    def _observe_obj(self, event):
        self.ndimage_artists = {
            t.source: NDImagePlot(self.axes, t, auto_rotate=self.rotate_ndimage) for t in self.obj
        }
        for artist in self.ndimage_artists.values():
            artist.observe('updated', self.request_redraw)

        # Needs to be set to force a change notification that sets the current
        # artist.
        self.current_artist_index = 0

        # This is necessary because `imshow` will override some axis settings.
        # We need to set them back to what we want.
        self.axes.axis('equal')
        self.axes.axis(self.obj.get_image_extent())

    ###################################################################################
    # Code for handling events from Matplotlib
    def motion(self, event):
        if self.pan_event is not None:
            self.motion_pan(event)
        elif self.drag_event is not None:
            self.motion_drag(event)

    def motion_drag(self, event):
        raise NotImplementedError

    def start_pan(self, event):
        self.pan_event = event
        self.pan_performed = False
        self.pan_xlim = self.axes.get_xlim()
        self.pan_ylim = self.axes.get_ylim()

    def motion_pan(self, event):
        if event.xdata is None:
            return
        if self.pan_event is None:
            return
        dx = event.xdata - self.pan_event.xdata
        dy = event.ydata - self.pan_event.ydata
        self.pan_xlim -= dx
        self.pan_ylim -= dy
        self.axes.set_xlim(self.pan_xlim)
        self.axes.set_ylim(self.pan_ylim)
        self.pan_performed = True
        self.request_redraw()

    def end_pan(self, event):
        self.pan_event = None

    def button_release(self, event):
        if event.button == MouseButton.LEFT:
            self.left_button_release(event)
        elif event.button == MouseButton.RIGHT:
            self.right_button_release(event)

    def left_button_release(self, event):
        self.end_pan(event)

    def right_button_release(self, event):
        pass

    def left_button_press(self, event):
        if event.xdata is not None:
            self.start_pan(event)

    def right_button_press(self, event):
        pass

    def scroll_zaxis(self, step):
        self.current_artist.next_z_slice(-1 if step == 'down' else 1)
        self.request_redraw()

    def scroll(self, event):
        if event.xdata is None:
            return

        base_scale = 1.1

        cur_xlim = self.axes.get_xlim()
        cur_ylim = self.axes.get_ylim()
        cur_xrange = cur_xlim[1] - cur_xlim[0]
        cur_yrange = cur_ylim[1] - cur_ylim[0]

        xdata = event.xdata  # get event x location
        ydata = event.ydata  # get event y location
        xfrac = (xdata - cur_xlim[0]) / cur_xrange
        yfrac = (ydata - cur_ylim[0]) / cur_yrange

        if event.button == "up":
            scale_factor = 1 / base_scale
        elif event.button == "down":
            scale_factor = base_scale
        else:
            scale_factor = 1

        # set new limits
        new_xrange = cur_xrange * scale_factor
        new_xlim = [xdata - xfrac * new_xrange, xdata + (1 - xfrac) * new_xrange]

        new_yrange = cur_yrange * scale_factor
        new_ylim = [ydata - yfrac * new_yrange, ydata + (1 - yfrac) * new_yrange]
        self.axes.set_xlim(new_xlim)
        self.axes.set_ylim(new_ylim)
        self.request_redraw()

    def _get_z_min(self):
        return min(a.z_slice_min for a in self.ndimage_artists.values())

    def _get_z_max(self):
        return min(a.z_slice_max for a in self.ndimage_artists.values())

    def set_display_mode(self, display_mode, all_artists=False):
        if all_artists:
            for artist in self.ndimage_artists.values():
                artist.display_mode = display_mode
        elif self.current_artist is not None:
            self.current_artist.display_mode = display_mode

    def set_channel_visible(self, channel_name, visible, all_artists=False):
        if all_artists:
            for artist in self.ndimage_artists.values():
                artist.set_channel_visible(channel_name, visible)
        elif self.current_artist is not None:
            self.current_artist.set_channel_visible(channel_name, visible)

    def set_channel_min_value(self, channel_name, low_value, all_artists=False):
        if all_artists:
            for artist in self.ndimage_artists.values():
                artist.set_channel_min_value(channel_name, low_value)
        elif self.current_artist is not None:
            self.current_artist.set_channel_min_value(channel_name, low_value)

    def set_channel_max_value(self, channel_name, high_value, all_artists=False):
        if all_artists:
            for artist in self.ndimage_artists.values():
                artist.set_channel_max_value(channel_name, high_value)
        elif self.current_artist is not None:
            self.current_artist.set_channel_max_value(channel_name, high_value)

    def set_z_slice_lb(self, z_slice, all_artists=False):
        if all_artists:
            for artist in self.ndimage_artists.values():
                artist.z_slice_lb = z_slice
        elif self.current_artist is not None:
            self.current_artist.z_slice_lb = z_slice

    def set_z_slice_ub(self, z_slice, all_artists=False):
        if all_artists:
            for artist in self.ndimage_artists.values():
                artist.z_slice_ub = z_slice
        elif self.current_artist is not None:
            self.current_artist.z_slice_ub = z_slice

    def _observe_current_artist_index(self, event):
        self.current_artist = list(self.ndimage_artists.values())[self.current_artist_index]


class StatePersistenceMixin(Atom):
    '''
    Mixin that provides basic state persistence
    '''
    #: Flag indicating whether there are unsaved changes
    unsaved_changes = Bool(False)

    #: State as saved to file (used for checking against present state to
    #: determine if there are unsaved changes)
    saved_state = Dict()

    def _default_saved_state(self):
        return self.get_full_state()

    def _observe_saved_state(self, event):
        self.check_for_changes()

    def get_full_state(self):
        return deepcopy({
            'data': self.obj.get_state(),
        })

    def save_state(self):
        state = self.get_full_state()
        self.reader.save_state(self.obj, state)
        self.saved_state = state
        self.update_state()

    def load_state(self):
        try:
            state = self.reader.load_state(self.obj)
            self.obj.set_state(state['data'])
            self.saved_state = state
            self.update_state()
        except IOError:
            pass

    def _observe_saved_state(self, event):
        self.check_for_changes()

    def update_state(self, event=None):
        self.check_for_changes()

    def check_for_changes(self):
        self.unsaved_changes = self.saved_state['data'] != self.get_full_state()['data']
