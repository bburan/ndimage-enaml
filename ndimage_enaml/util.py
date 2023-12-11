from matplotlib import colors
import numpy as np


LABEL_CONFIG = {
    'artifact': {
        'display_color': '#FF0000',
    },
    'selected': {
        'display_color': 'white',
        'border_expand': 2,
    },
    'orphan': {
        'display_color': '#00FF00',
    },
}


def color_image(image, channel_config):
    # Add a basic black image that way if no channels are selected, we still
    # return something (this happens when the user uses the GUI and un-checks
    # all the channels).
    blank_shape = list(image.shape[:-1]) + [3]
    new_image = [np.zeros(blank_shape)]
    for config in channel_config:
        if not config.get('visible', True):
            continue
        rgb = colors.to_rgb(config['display_color'])
        lb = config.get('min_value', 0)
        ub = config.get('max_value', 1)
        i = config['i']
        d = np.clip((image[..., i] - lb) / (ub - lb), 0, 1)
        d = d[..., np.newaxis] * rgb
        new_image.append(d)

    return np.concatenate([i[np.newaxis] for i in new_image]).max(axis=0)


def get_image(image, *args, **kwargs):
    # Ensure that image is at least 5D (i.e., a stack of 3D multichannel images).
    if image.ndim == 4:
        return _get_image(image[np.newaxis], *args, **kwargs)[0]
    else:
        return _get_image(image, *args, **kwargs)


def _get_image(image, channel_config, z_slice=None, axis='z',
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
    return color_image(image, channel_config)


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
        config = LABEL_CONFIG.get(label, {})
        color = config.get('display_color', 'white')
        expand = config.get('border_expand', 1)
        rgb = colors.to_rgba(color)[:3]
        for i in indices:
            col = i % n_cols
            row = i // n_cols
            xlb = (xs + padding) * col + padding - expand
            ylb = (ys + padding) * row + padding - expand
            xub = (xs + padding) * col + padding + xs + (expand - 1)
            yub = (ys + padding) * row + padding + ys + (expand - 1)
            tiled_image[xlb, ylb:yub+1, :] = rgb
            tiled_image[xub, ylb:yub+1, :] = rgb
            tiled_image[xlb:xub+1, ylb, :] = rgb
            tiled_image[xlb:xub+1, yub, :] = rgb

    return tiled_image


def project_image(image, channel_config, padding=1):
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

    tiled_image = color_image(tiled_image, channel_config)
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
