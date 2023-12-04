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


def get_image(image, *args, **kwargs):
    # Ensure that image is at least 5D (i.e., a stack of 3D multichannel images).
    if image.ndim == 4:
        return _get_image(image[np.newaxis], *args, **kwargs)[0]
    else:
        return _get_image(image, *args, **kwargs)


def _get_image(image, channel_names=None, channels=None, z_slice=None, axis='z',
               norm_percentile=99):
    from .model import ChannelConfig
    if channel_names is None:
        channel_names = [f'Unknown {i+1}' for i in range(image.shape[-1])]

    # z_slice can either be an integer or a slice object.
    if z_slice is not None:
        image = image[:, :, :, z_slice, :]
    if image.ndim == 5:
        image = image.max(axis='xyz'.index(axis) + 1)

    data = image

    # Normalize data
    img_max =  np.percentile(image, norm_percentile, axis=(0, 1, 2), keepdims=True)
    img_mask = img_max != 0
    data = np.divide(image, img_max, where=img_mask).clip(0, 1)

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

    image = []
    for c, c_name in enumerate(channel_names):
        if c_name in channel_config:
            config = channel_config[c_name]
            rgb = colors.to_rgba(config['display_color'])[:3]
            print(rgb)

            lb = config['min_value']
            ub = config['max_value']
            d = np.clip((data[..., c] - lb) / (ub - lb), 0, 1)
            d = d[..., np.newaxis] * rgb
            image.append(d)

    return np.concatenate([i[np.newaxis] for i in image]).max(axis=0)


def tile_images(images, n_cols=15, padding=1):
    n = len(images)
    n_rows = int(np.ceil(n / n_cols))

    xs, ys = images.shape[1:3]
    x_size = (xs + padding) * n_cols + padding
    y_size = (ys + padding) * n_rows + padding
    tiled_image = np.full((x_size, y_size, 3), 1.0)
    for i, img in enumerate(images):
        col = i % n_cols
        row = i // n_cols
        xlb = (xs + padding) * col + padding
        ylb = (ys + padding) * row + padding
        tiled_image[xlb:xlb+xs, ylb:ylb+ys] = img
    return tiled_image
