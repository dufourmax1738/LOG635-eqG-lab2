import os
import numpy as np
from skimage import io, color, transform, measure, filters


def make_one_hot_key(class_index, num_classes):
    key = np.zeros(num_classes, dtype=int)
    key[class_index] = 1
    return tuple(key.tolist())


def image_db_to_arrays(image_db):
    X = []
    y = []
    for one_hot_key, image_vectors in image_db.items():
        label = int(np.argmax(one_hot_key))
        for image_vector in image_vectors:
            X.append(image_vector)
            y.append(label)
    if not X:
        return None, None
    return np.vstack(X), np.array(y, dtype=int)


def load_image(path, as_gray=True):
    img = io.imread(path)
    if as_gray and img.ndim == 3:
        img = color.rgb2gray(img)
    return img


def extract_mask_and_crop(img, thresh=None):
    # img expected grayscale in [0,1] or [0,255]
    imgf = img.astype(float)
    if imgf.max() > 1.0:
        imgf = imgf / 255.0
    if thresh is None:
        thresh = filters.threshold_otsu(imgf)
    mask = imgf > thresh
    # bounding box
    props = measure.regionprops(measure.label(mask.astype(int)))
    if not props:
        # return whole image as ROI
        return mask, imgf, None
    # take bounding box of all regions (union)
    coords = np.column_stack(np.where(mask))
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    cropped = imgf[r0:r1, c0:c1]
    cropped_mask = mask[r0:r1, c0:c1]
    return cropped_mask, cropped, props


def roi_pixels_feature(img, image_shape=(32,32)):
    mask, cropped, props = extract_mask_and_crop(img)
    # resize cropped ROI to image_shape and flatten
    if cropped is None:
        resized = transform.resize(img, image_shape, anti_aliasing=True)
    else:
        resized = transform.resize(cropped, image_shape, anti_aliasing=True)
    return resized.flatten()


def extra_features(img):
    mask, cropped, props = extract_mask_and_crop(img)
    if props is None:
        return np.array([0, 0.0])
    # number of shapes
    lab = measure.label(mask.astype(int))
    num_shapes = lab.max()
    # eccentricity of the largest shape
    regions = measure.regionprops(lab)
    if not regions:
        ecc = 0.0
    else:
        areas = [r.area for r in regions]
        largest = regions[int(np.argmax(areas))]
        ecc = float(largest.eccentricity)
    return np.array([num_shapes, ecc])


def load_dataset(
    root_dir,
    image_shape=(32,32),
    as_gray=True,
    max_per_class=None,
    debug_samples=0,
    debug_seed=0,
    show_debug_images=False,
    return_image_db=False,
):
    """Parcourt les sous-dossiers de root_dir; chaque sous-dossier est une classe.
    Retourne: X_pixels (N x D), X_extra (N x E), y (N,), class_names list.
    Si return_image_db=True, retourne aussi image_db avec clef one-hot et valeur liste de vecteurs combines.
    """
    Xp = []
    Xe = []
    ys = []
    sample_paths = []
    sample_class_names = []
    class_names = []
    classes = [d for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]
    image_db = {make_one_hot_key(ci, len(classes)): [] for ci in range(len(classes))}
    for ci, cname in enumerate(classes):
        class_names.append(cname)
        folder = os.path.join(root_dir, cname)
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
        if max_per_class:
            files = files[:max_per_class]
        for f in files:
            try:
                img = load_image(f, as_gray=as_gray)
            except Exception:
                continue
            pix = roi_pixels_feature(img, image_shape=image_shape)
            extras = extra_features(img)
            combined = np.hstack([pix, extras])
            Xp.append(pix)
            Xe.append(extras)
            ys.append(ci)
            sample_paths.append(f)
            sample_class_names.append(cname)
            image_db[make_one_hot_key(ci, len(classes))].append(combined)
    if len(Xp) == 0:
        return None, None, None, []
    Xp = np.vstack(Xp)
    Xe = np.vstack(Xe)
    y = np.array(ys, dtype=int)

    if debug_samples and debug_samples > 0:
        rng = np.random.RandomState(debug_seed)
        sample_count = min(int(debug_samples), len(y))
        chosen = rng.choice(len(y), size=sample_count, replace=False)
        print(f'DEBUG load_dataset: {sample_count} echantillon(s) aleatoire(s)')
        for idx in chosen:
            print('-' * 80)
            print(f'Index: {idx}')
            print(f'Classe: {sample_class_names[idx]} (label={y[idx]})')
            print(f'One-hot: {make_one_hot_key(y[idx], len(class_names))}')
            print(f'Fichier: {sample_paths[idx]}')
            print(f'Pixels[0:10]: {np.round(Xp[idx][:10], 4)}')
            print(f'Extra features: {np.round(Xe[idx], 4)}')

            if show_debug_images:
                import matplotlib.pyplot as plt

                img = load_image(sample_paths[idx], as_gray=as_gray)
                plt.figure(figsize=(3, 3))
                cmap = 'gray' if getattr(img, 'ndim', 2) == 2 else None
                plt.imshow(img, cmap=cmap)
                plt.title(sample_class_names[idx])
                plt.axis('off')
                plt.show()

    if return_image_db:
        return Xp, Xe, y, class_names, image_db

    return Xp, Xe, y, class_names
