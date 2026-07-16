import os
import numpy as np
from skimage import io, color, transform, measure, filters
from skimage import morphology
from skimage import feature
from abc import ABC, abstractmethod

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

# Polymorphisme des algorithmes d'activation pour les couches du réseau de neurones
class activation_functions(ABC):
    @abstractmethod
    def function(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass

    def backward_delta(self, x, output_error):
        return output_error * self.derivative(x)

# Implementation de la fonction d'activation ReLU
class relu(activation_functions):
    def function(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return (x > 0).astype(float)

# Implementation de la fonction d'activation Softmax
class softmax(activation_functions):
    def function(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def derivative(self, x):
        # Not used directly for output layer when combined with cross-entropy.
        s = self.function(x)
        return s * (1 - s)

    def backward_delta(self, x, output_error):
        # With softmax + cross-entropy, upstream gradient is already simplified.
        return output_error

# Classe représentant un réseau de neurones entièrement connecté avec plusieurs couches cachées
class NeuralNetwork():
    def __init__(self, input_size, hidden_size, output_size, hidden_layers, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.trainable_layers = []

        if hidden_layers < 1:
            raise ValueError('hidden_layers must be >= 1')
        
        # On ajoute des couches cachées avec ReLU et une couche de sortie avec Softmax
        self.trainable_layers.append(Layer(input_size, hidden_size, activation=relu()))
        for _ in range(hidden_layers - 1):
            self.trainable_layers.append(Layer(hidden_size, hidden_size, activation=relu()))
        self.trainable_layers.append(Layer(hidden_size, output_size, activation=softmax()))
        self.learning_rate = learning_rate

    # Passe dans le réseau pour obtenir les prédictions
    def forward(self, X):
        for layer in self.trainable_layers:
            X = layer.forward(X)
        return X

    # Calcul de la perte d'entropie croisée pour les étiquettes vraies et les prédictions
    def entropy_loss(self, y_true, y_pred):
        eps = np.finfo(float).eps

        # Transformation des étiquettes vraies en one-hot si elles sont fournies sous forme d'indices
        if y_true.ndim == 1:
            m = y_true.shape[0]
            correct_class_probs = y_pred[np.arange(m), y_true]
            return -np.mean(np.log(correct_class_probs + eps))

        return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))

    
    def backward(self, X, y_true, y_pred):
        m = y_pred.shape[0]
        y_true_one_hot = np.zeros_like(y_pred)
        y_true_one_hot[np.arange(m), y_true] = 1

        # Gradient of cross-entropy wrt logits when output uses softmax.
        output_error = (y_pred - y_true_one_hot) / m

        error = output_error
        for layer in reversed(self.trainable_layers):
            error = layer.backward(error, self.learning_rate)

    def train(self, X, y, nb_iteration=1000):
        for i in range(nb_iteration):
            y_pred = self.forward(X)
            loss = self.entropy_loss(y, y_pred)
            self.backward(X, y, y_pred)
            if (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}/{nb_iteration}, Loss: {loss:.4f}")

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)


class Layer():
    def __init__(self, input_size, output_size, activation=relu()):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation if activation is not None else relu()

        # He init for ReLU layers, Xavier-like init for output layer.
        if isinstance(self.activation, relu):
            scale = np.sqrt(2.0 / input_size)
        else:
            scale = np.sqrt(1.0 / input_size)

        self.weights = np.random.randn(input_size, output_size) * scale
        self.bias = np.zeros((1, output_size))

    def forward(self, X):
        self.input = X
        self.z = np.dot(X, self.weights) + self.bias
        self.output = self.activation.function(self.z)
        return self.output

    def backward(self, output_error, learning_rate):
        delta = self.activation.backward_delta(self.z, output_error)

        weights_error = np.dot(self.input.T, delta)
        bias_error = np.sum(delta, axis=0, keepdims=True)

        # Return error for the previous layer before updating current weights.
        previous_error = np.dot(delta, self.weights.T)

        # Update weights and biases.
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error

        return previous_error