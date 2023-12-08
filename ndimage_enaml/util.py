from matplotlib import colors
import numpy as np


CHANNEL_CONFIG = {
    'CtBP2': { 'display_color': '#FF0000'},
    'MyosinVIIa': {'display_color': '#0000FF'},
    'GluR2': {'display_color': '#00FF00'},
    'GlueR2': {'display_color': '#00FF00'},
    'PMT': {'display_color': '#FFFFFF'},
    'DAPI': {'display_color': '#FFFFFF'},

    # Channels are tagged as unknown if there's difficulty parsing the channel
    # information from the file.
    'Unknown 1': {'display_color': '#FF0000'},
    'Unknown 2': {'display_color': '#00FF00'},
    'Unknown 3': {'display_color': '#0000FF'},
    'Unknown 4': {'display_color': '#FFFFFF'},
}


LABEL_CONFIG = {
    'artifact': 'maroon',
    'selected': 'white',
    'orphan': 'mediumseagreen',
}


def color_image(image, channel_names=None, channels=None):
    if channel_names is None:
        channel_names = [f'Unknown {i+1}' for i in range(image.shape[-1])]

    from .model import ChannelConfig
    if channels is None:
        channels = channel_names
    elif isinstance(channels, int):
        raise ValueError('Must provide name for channel')
    elif isinstance(channels, str):
        channels = [channels]
    elif len(channels) == 0:
        return np.zeros_like(data)

    # Check that channels are valid and generate config
    channel_config = {}
    for c in channels:
        if isinstance(c, ChannelConfig):
            if not c.visible:
                continue
            if c.name not in channel_names:
                raise ValueError(f'Channel {c.name} does not exist')
            channel_config[c.name] = {
                'min_value': c.min_value,
                'max_value': c.max_value,
                **CHANNEL_CONFIG[c.name],
            }
        elif isinstance(c, dict):
            channel_config[c['name']] = {
                **c,
                **CHANNEL_CONFIG[c['name']],
            }
        elif c not in channel_names:
            raise ValueError(f'Channel {c} does not exist')
        else:
            channel_config[c] = {
                'min_value': 0,
                'max_value': 1,
                **CHANNEL_CONFIG[c],
            }
    new_image = []
    for c, c_name in enumerate(channel_names):
        if c_name in channel_config:
            config = channel_config[c_name]
            rgb = colors.to_rgba(config['display_color'])[:3]
            lb = config['min_value']
            ub = config['max_value']
            d = np.clip((image[..., c] - lb) / (ub - lb), 0, 1)
            d = d[..., np.newaxis] * rgb
            new_image.append(d)

    return np.concatenate([i[np.newaxis] for i in new_image]).max(axis=0)


def get_image(image, *args, **kwargs):
    # Ensure that image is at least 5D (i.e., a stack of 3D multichannel images).
    if image.ndim == 4:
        return _get_image(image[np.newaxis], *args, **kwargs)[0]
    else:
        return _get_image(image, *args, **kwargs)


def _get_image(image, channel_names=None, channels=None, z_slice=None, axis='z',
               norm_percentile=99):

    # Normalize data before slicing because we need to make sure that the
    # normalization remains constant when stepping through the slices and/or
    # substack.
    ai = 'xyz'.index(axis) + 1
    img_max =  np.percentile(image.max(axis=ai), norm_percentile, axis=(0, 1, 2), keepdims=True)
    img_mask = img_max != 0

    # z_slice can either be an integer or a slice object.
    if z_slice is not None:
        # Image is i, x, y z, c where i is index of tile and c is color/channel
        image = image[:, :, :, z_slice, :]
    if image.ndim == 5:
        image = image.max(axis=ai)

    # Now do the normalization
    image = np.divide(image, img_max, where=img_mask).clip(0, 1)
    return color_image(image, channel_names, channels)


def tile_images(images, n_cols=15, padding=2, classifiers=None):
    n = len(images)
    n_rows = int(np.ceil(n / n_cols))

    xs, ys = images.shape[1:3]
    x_size = (xs + padding) * n_cols + padding
    y_size = (ys + padding) * n_rows + padding
    tiled_image = np.full((x_size, y_size, 3), 0.0)
    for i, img in enumerate(images):
        col = i % n_cols
        row = i // n_cols
        xlb = (xs + padding) * col + padding
        ylb = (ys + padding) * row + padding
        tiled_image[xlb:xlb+xs, ylb:ylb+ys] = img

    if classifiers is None:
        classifiers = {}

    for label, indices in classifiers.items():
        color = LABEL_CONFIG.get(label, 'white')
        rgb = colors.to_rgba(color)[:3]
        for i in indices:
            col = i % n_cols
            row = i // n_cols
            xlb = (xs + padding) * col + padding - 1
            ylb = (ys + padding) * row + padding - 1
            tiled_image[xlb, ylb:ylb+ys+1, :] = rgb
            tiled_image[xlb+xs+1, ylb:ylb+ys+1, :] = rgb
            tiled_image[xlb:xlb+xs+1, ylb, :] = rgb
            tiled_image[xlb:xlb+xs+2, ylb+ys+1, :] = rgb

    return tiled_image


def project_image(image, channel_names, padding=2):
    xs, ys, zs, cs = image.shape
    y_size = xs + ys + padding * 2 + padding
    x_size = (xs + ys + padding * 2) * cs + padding
    tiled_image = np.full((x_size, y_size, 3), 0.0)

    for i in range(cs):
        t = image[..., i] / 255
        x_proj = t.max(axis=0)
        y_proj = t.max(axis=1)
        z_proj = t.max(axis=2)
        xo = i * (xs + ys + padding * 2) + padding
        yo = padding
        zxo = xo
        zyo = yo
        xxo = xo + xs + padding
        xyo = yo
        yxo = xo
        yyo = yo + ys + padding
        tiled_image[zxo:zxo+xs, zyo:zyo+ys, i] = z_proj
        tiled_image[xxo:xxo+xs, xyo:xyo+ys, i] = x_proj.T
        tiled_image[yxo:yxo+ys, yyo:yyo+ys, i] = y_proj

    tiled_image = color_image(tiled_image, channel_names)
    return tiled_image.swapaxes(0, 1)


def expand_path(x, y, width):
    v = x + y * 1j
    a = np.angle(np.diff(v)) + np.pi / 2
    a = np.pad(a, (1, 0), mode='edge')
    dx = width * np.cos(a)
    dy = width * np.sin(a)
    x = np.linspace(x - dx, x + dx, 100)
    y = np.linspace(y - dy, y + dy, 100)
    return x, y
