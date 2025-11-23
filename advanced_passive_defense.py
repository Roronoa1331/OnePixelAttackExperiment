# defense_pipeline.py
# Defense toolkit for One-Pixel attacks and small perturbations.
# - Loads model robustly (compat wrapper for SeparableConv2D)
# - Implements input preprocessing defenses (JPEG, median, bilateral, quantization, gaussian)
# - Provides ensemble/randomized smoothing inference (averaging over randomized preprocessings)
# - Evaluates defenses against a one-pixel attack (standalone, no ART)
# - Saves a brief summary and a plot to disk

import os
import random
from typing import Tuple, List, Optional
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import json

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.preprocessing import image as kimage

# Path to experiment visualization (user-uploaded file) — kept here per request
REPORT_IMAGE_PATH = '/mnt/data/experiment_11_advanced_comparison.png'

# ---------------------- Compatibility wrapper & robust loader ----------------------
class CompatibleSeparableConv2D(tf.keras.layers.SeparableConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        def maybe_deserialize(key, deser):
            v = kwargs.get(key, None)
            if isinstance(v, dict):
                try:
                    kwargs[key] = deser(v)
                except Exception:
                    kwargs.pop(key, None)
        maybe_deserialize('kernel_initializer', initializers.deserialize)
        maybe_deserialize('bias_initializer', initializers.deserialize)
        maybe_deserialize('depthwise_initializer', initializers.deserialize)
        maybe_deserialize('pointwise_initializer', initializers.deserialize)
        maybe_deserialize('kernel_regularizer', regularizers.deserialize)
        maybe_deserialize('bias_regularizer', regularizers.deserialize)
        maybe_deserialize('kernel_constraint', constraints.deserialize)
        super().__init__(*args, **kwargs)


def robust_load_model(h5_path: str):
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Model file not found: {h5_path}")

    custom_objects = {'SeparableConv2D': CompatibleSeparableConv2D}
    try:
        model = load_model(h5_path, custom_objects=custom_objects)
        print('Loaded model with compatibility wrapper')
        return model
    except Exception as e_load:
        print('Primary load failed:', repr(e_load))
        print('Trying fallback: reconstruct Xception-like and load weights by_name...')

    try:
        from tensorflow.keras.applications import Xception
        from tensorflow.keras import layers, models
        base = Xception(weights=None, include_top=False, input_shape=(224,224,3))
        x = base.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu', name='fc_128')(x)
        x = layers.Dropout(0.3)(x)
        preds = layers.Dense(1, activation='sigmoid', name='predictions')(x)
        model = models.Model(inputs=base.input, outputs=preds)
        model.load_weights(h5_path, by_name=True)
        print('Loaded weights by name into reconstructed model')
        return model
    except Exception as e_weights:
        print('Fallback failed:', repr(e_weights))
        raise RuntimeError('Failed loading model; see earlier messages') from e_load

# ---------------------- Preprocessing / Defense functions ----------------------

def jpeg_compress_np(img_np: np.ndarray, quality: int = 75) -> np.ndarray:
    img_uint8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    pil = Image.fromarray(img_uint8)
    tmp = 'tmp_defense.jpg'
    pil.save(tmp, 'JPEG', quality=quality)
    out = np.array(Image.open(tmp).convert('RGB')).astype(np.float32)/255.0
    try:
        os.remove(tmp)
    except Exception:
        pass
    return out


def median_bilateral_np(img_np: np.ndarray, ksize: int = 3, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    img_uint8 = (np.clip(img_np,0,1)*255).astype(np.uint8)
    med = cv2.medianBlur(img_uint8, ksize)
    bil = cv2.bilateralFilter(med, d, sigma_color, sigma_space)
    return bil.astype(np.float32)/255.0


def quantize_np(img_np: np.ndarray, levels: int = 8) -> np.ndarray:
    q = np.round(img_np*(levels-1)) / (levels-1)
    return q


def gaussian_np(img_np: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    out = np.zeros_like(img_np)
    for c in range(3):
        out[:,:,c] = cv2.GaussianBlur((img_np[:,:,c]*255).astype(np.uint8),(0,0),sigma)/255.0
    return np.clip(out,0,1)


def feature_squeeze(img_np: np.ndarray) -> np.ndarray:
    # combine quantization + median
    q = quantize_np(img_np, levels=8)
    s = median_bilateral_np(q, ksize=3)
    return s

# ---------------------- Ensemble / Randomized Smoothing ----------------------

def random_preprocess(img_np: np.ndarray) -> np.ndarray:
    # Randomly choose a small pipeline
    x = img_np.copy()
    # Random JPEG
    if random.random() < 0.7:
        q = random.choice([60,70,80])
        x = jpeg_compress_np(x, quality=q)
    # Random median+bilateral sometimes
    if random.random() < 0.6:
        k = random.choice([3,5])
        d = random.choice([5,9])
        sigma_c = random.uniform(40,100)
        sigma_s = random.uniform(40,100)
        x = median_bilateral_np(x, ksize=k, d=d, sigma_color=sigma_c, sigma_space=sigma_s)
    # Small quantization
    if random.random() < 0.5:
        x = quantize_np(x, levels=random.choice([4,8,16]))
    return np.clip(x,0,1)


def defended_predict(model, img_np: np.ndarray, ensemble_size: int = 8) -> Tuple[int, float, dict]:
    # Preprocess (model likely expects ImageNet normalization) - we assume model trained on [0,1] then normalized
    def preprocess_for_model(x):
        mean = np.array([0.485,0.456,0.406],dtype=np.float32)
        std = np.array([0.229,0.224,0.225],dtype=np.float32)
        return (x - mean)/std

    # original pred
    x_in = np.expand_dims(preprocess_for_model(img_np),0)
    pred_orig = model.predict(x_in, verbose=0)
    if pred_orig.ndim==1 or pred_orig.shape[1]==1:
        prob_orig = float(pred_orig.reshape(-1)[0])
        label_orig = int(prob_orig>0.5)
        conf_orig = prob_orig
    else:
        prob_orig = pred_orig[0]
        label_orig = int(np.argmax(prob_orig))
        conf_orig = float(np.max(prob_orig))

    # ensemble over randomized preprocessings
    probs = []
    for _ in range(ensemble_size):
        x_var = random_preprocess(img_np)
        x_var_in = np.expand_dims(preprocess_for_model(x_var),0)
        p = model.predict(x_var_in, verbose=0)
        if p.ndim==1 or p.shape[1]==1:
            probs.append([1-p[0,0], p[0,0]])
        else:
            probs.append(p[0])
    mean_prob = np.mean(probs, axis=0)
    if mean_prob.shape[0]==2:
        label_ens = int(mean_prob[1]>0.5)
        conf_ens = float(mean_prob[1])
    else:
        label_ens = int(np.argmax(mean_prob))
        conf_ens = float(np.max(mean_prob))

    # detection heuristic: if original confidence much higher than ensemble, suspicious
    detected = (conf_orig - conf_ens) > 0.15
    final_label = label_ens if detected else label_orig
    final_conf = conf_ens if detected else conf_orig

    details = {'orig_label': label_orig, 'orig_conf': conf_orig, 'ens_label': label_ens, 'ens_conf': conf_ens, 'detected': detected}
    return final_label, final_conf, details

# ---------------------- One-pixel attack (embedded, same as earlier) ----------------------

random.seed(0)
np.random.seed(0)

def predict_label_and_prob(model, x: np.ndarray) -> Tuple[int, float]:
    preds = model.predict(x, verbose=0)
    if preds.ndim == 1 or preds.shape[1] == 1:
        p = preds.reshape(-1)[0]
        label = int(p > 0.5)
        return label, float(p)
    else:
        p = preds[0]
        label = int(np.argmax(p))
        return label, float(np.max(p))


def one_pixel_attack_random_search(model,
                                   x_orig: np.ndarray,
                                   true_label: int,
                                   max_iter: int = 100,
                                   num_pixels: int = 1,
                                   candidate_values: Optional[np.ndarray] = None,
                                   targeted: bool = False) -> Tuple[np.ndarray, bool]:
    H = x_orig.shape[1]
    W = x_orig.shape[2]
    C = x_orig.shape[3]
    desired_label = true_label
    if candidate_values is None:
        def sample_value():
            return np.random.rand(C).astype(np.float32)
    else:
        candidates = np.asarray(candidate_values).astype(np.float32)
        def sample_value():
            idx = np.random.randint(0, len(candidates))
            return candidates[idx]
    x_base = x_orig.copy()
    for i in range(max_iter):
        ys = np.random.randint(0, H, size=(num_pixels,))
        xs = np.random.randint(0, W, size=(num_pixels,))
        x_try = x_base.copy()
        for px in range(num_pixels):
            r, c = ys[px], xs[px]
            new_val = sample_value()
            x_try[0, r, c, :] = new_val
        pred_label, _ = predict_label_and_prob(model, x_try)
        if (not targeted and pred_label != desired_label) or (targeted and pred_label == desired_label):
            return x_try, True
    return x_orig, False

# ---------------------- Evaluation pipeline ----------------------

def evaluate_defense_on_samples(model, data_dir: str, defense_fn, num_samples_per_class: int = 3, max_iter_attack: int = 300):
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    results = []
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:num_samples_per_class]
        for fname in files:
            path = os.path.join(class_dir, fname)
            pil = kimage.load_img(path, target_size=(224,224))
            arr = kimage.img_to_array(pil).astype(np.float32)/255.0
            x = np.expand_dims(arr,0)
            orig_label, orig_prob = predict_label_and_prob(model, x)
            # Attack against original
            x_adv, success_before = one_pixel_attack_random_search(model, x, true_label=orig_label, max_iter=max_iter_attack)
            # Evaluate defense on original and adversarial
            final_label_orig, final_conf_orig, d_orig = defense_fn(model, arr)
            final_label_adv, final_conf_adv, d_adv = defense_fn(model, x_adv[0])
            results.append({
                'image': fname,
                'class_dir': class_name,
                'orig_label': int(orig_label),
                'adv_success_before': bool(success_before),
                'defended_label_on_orig': int(final_label_orig),
                'defended_label_on_adv': int(final_label_adv),
                'defense_detected_on_orig': d_orig['detected'],
                'defense_detected_on_adv': d_adv['detected']
            })
            print(f"{fname}: adv_success_before={success_before}, defended_on_adv={final_label_adv}, detected={d_adv['detected']}")
    return results

# ---------------------- Utility: plot summary ----------------------

def plot_defense_results(results: List[dict], out_png: str = 'defense_summary.png'):
    # simple summary: percent of samples where adv was successful before vs after defense
    total = len(results)
    if total==0:
        print('No results to plot')
        return
    succ_before = sum(1 for r in results if r['adv_success_before'])
    succ_after = sum(1 for r in results if r['adv_success_before'] and r['defended_label_on_adv'] == r['orig_label'])
    # succ_after counts cases where defense recovered original label
    labels = ['Before defense', 'After defense (recovered)']
    vals = [succ_before/total, succ_after/total]
    plt.figure(figsize=(6,4))
    plt.bar(labels, vals)
    plt.ylim(0,1)
    plt.ylabel('Fraction')
    plt.title('One-pixel attack: success before vs recovered after defense')
    plt.savefig(out_png, dpi=150)
    plt.show()

# ---------------------- Example CLI run ----------------------
if __name__ == '__main__':
    MODEL_PATH = './xception_transfer_model.h5'
    DATA_DIR = './casting_512x512/'
    model = robust_load_model(MODEL_PATH)
    results = evaluate_defense_on_samples(model, DATA_DIR, defense_fn=defended_predict, num_samples_per_class=3, max_iter_attack=300)
    with open('defense_eval.json','w') as f:
        json.dump(results, f, indent=2)
    plot_defense_results(results)
    # Also show the uploaded experiment image if available
    if os.path.exists(REPORT_IMAGE_PATH):
        display_img = Image.open(REPORT_IMAGE_PATH)
        display_img.show()
    print('Done')
