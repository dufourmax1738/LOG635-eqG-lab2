import os
import numpy as np
from skimage import io, color, transform, measure, filters
from skimage import morphology
from skimage import feature


def make_one_hot_key(class_index, num_classes):
    key = np.zeros(num_classes, dtype=int)
    key[class_index] = 1
    return tuple(key.tolist())


def infer_image_shape_from_vector(vector):
    size = int(vector.size)
    side = int(np.sqrt(size))
    if side * side == size:
        return side, side

    if size % 3 == 0:
        side = int(np.sqrt(size // 3))
        if side * side * 3 == size:
            return side, side, 3

    raise ValueError(f'Impossible de reconstruire une image a partir dun vecteur de taille {size}')


def vector_to_image(vector):
    return np.asarray(vector).reshape(infer_image_shape_from_vector(np.asarray(vector)))


def build_image_db(root_dir, as_gray=True, max_per_class=None):
    classes = [d for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]
    image_db = {make_one_hot_key(ci, len(classes)): [] for ci in range(len(classes))}

    for ci, cname in enumerate(classes):
        folder = os.path.join(root_dir, cname)
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))
        ]
        if max_per_class:
            files = files[:max_per_class]

        for file_path in files:
            try:
                img = load_image(file_path, as_gray=as_gray)
            except Exception:
                continue
            image_db[make_one_hot_key(ci, len(classes))].append(np.asarray(img).flatten())

    return image_db, classes


def image_db_to_arrays(image_db, image_shape=(32,32), return_separate=False):
    Xp = []
    ys = []

    for one_hot_key, image_vectors in image_db.items():
        label = int(np.argmax(one_hot_key))
        for image_vector in image_vectors:
            img = vector_to_image(image_vector)
            pix = roi_pixels_feature(img, image_shape=image_shape)
            Xp.append(pix)
            ys.append(label)

    if not Xp:
        if return_separate:
            return None, None, None
        return None, None

    Xp = np.vstack(Xp)
    y = np.array(ys, dtype=int)

    if return_separate:
        return Xp, y

    return Xp, y


def load_image(path, as_gray=True):
    img = io.imread(path)
    if as_gray and img.ndim == 3:
        img = color.rgb2gray(img)
    return img


def get_canny_edges(img):
    """Return Canny edges for a grayscale image normalized in [0, 1]."""
    imgf = img.astype(float)
    if imgf.max() > 1.0:
        imgf = imgf / 255.0
    imgf = filters.gaussian(imgf, sigma=0.8)
    return feature.canny(imgf, sigma=1.2)


def extract_mask_and_crop(img, thresh=None):
    # img expected grayscale in [0,1] or [0,255]
    imgf = img.astype(float)
    if imgf.max() > 1.0:
        imgf = imgf / 255.0

    # Smooth first to stabilize edge detection.
    imgf = filters.gaussian(imgf, sigma=0.8)

    # Canny contours, then morphological cleanup to recover filled shape regions.
    edges = feature.canny(imgf, sigma=1.2)
    edges = morphology.binary_closing(edges, morphology.disk(1))
    edges = morphology.binary_dilation(edges, morphology.disk(1))

    # Fill enclosed contours (shape interiors) while keeping external background out.
    fill_area = max(64, int(0.5 * edges.size))
    mask = morphology.remove_small_holes(edges, area_threshold=fill_area)

    # Remove tiny artifacts.
    min_area = max(8, int(0.001 * mask.size))
    mask = morphology.remove_small_objects(mask, min_size=min_area)
    mask = morphology.binary_closing(mask, morphology.disk(1))

    # Fallback if edge-based mask is empty.
    if not np.any(mask):
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
    """Construit dabord une image_db one-hot -> vecteurs dimage, puis extrait les features.
    Retourne: X_pixels (N x D), y (N,), class_names list.
    Si return_image_db=True, retourne aussi image_db avec clef one-hot et valeur liste de vecteurs dimage bruts.
    """
    image_db, class_names = build_image_db(root_dir, as_gray=as_gray, max_per_class=max_per_class)
    Xp, y = image_db_to_arrays(image_db, image_shape=image_shape, return_separate=True)

    if Xp is None:
        return None, None, None, []

    sample_paths = []
    sample_class_names = []
    sample_raw_vectors = []
    for ci, cname in enumerate(class_names):
        folder = os.path.join(root_dir, cname)
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))
        ]
        if max_per_class:
            files = files[:max_per_class]
        for f in files:
            try:
                img = load_image(f, as_gray=as_gray)
            except Exception:
                continue
            sample_paths.append(f)
            sample_class_names.append(cname)
            sample_raw_vectors.append(np.asarray(img).flatten())

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
            print(f'Raw[0:10]: {np.round(sample_raw_vectors[idx][:10], 4)}')
            print(f'Pixels[0:10]: {np.round(Xp[idx][:10], 4)}')

            if show_debug_images:
                import matplotlib.pyplot as plt

                img = load_image(sample_paths[idx], as_gray=as_gray)
                edges = get_canny_edges(img)

                plt.figure(figsize=(7, 3))
                plt.subplot(1, 2, 1)
                cmap = 'gray' if getattr(img, 'ndim', 2) == 2 else None
                plt.imshow(img, cmap=cmap)
                plt.title(f'Original - {sample_class_names[idx]}')
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(edges, cmap='gray')
                plt.title('Canny')
                plt.axis('off')
                plt.show()

    if return_image_db:
        return Xp, y, class_names, image_db

    return Xp, y, class_names
