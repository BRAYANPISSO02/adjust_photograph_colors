import json
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
import skimage.color


def load_image(filepath):
    img = Image.open(filepath)
    img = ImageOps.exif_transpose(img)
    return np.array(img, dtype=np.float32) / 255.0


def save_image(filepath, img_srgb):
    clipped = np.clip(img_srgb * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(clipped).save(filepath, quality=95)


def srgb_to_linear(img):
    a = 0.055
    return np.where(img <= 0.04045, img / 12.92, ((img + a) / (1.0 + a)) ** 2.4)


def linear_to_srgb(img):
    a = 0.055
    img = np.clip(img, 0.0, 1.0)
    return np.where(img <= 0.0031308, img * 12.92, (1.0 + a) * (img ** (1.0 / 2.4)) - a)


def get_patch_centers(corners):
    tl = np.array(corners['tl'], dtype=np.float64)
    tr = np.array(corners['tr'], dtype=np.float64)
    bl = np.array(corners['bl'], dtype=np.float64)
    centers = []
    for r in range(6):
        for c in range(4):
            p = tl + (c / 3.0) * (tr - tl) + (r / 5.0) * (bl - tl)
            centers.append(p)
    return centers


def extract_patch_medians(img_linear, corners, half_size=15):
    centers = get_patch_centers(corners)
    medians = []
    for p in centers:
        x, y = int(round(p[0])), int(round(p[1]))
        roi = img_linear[y - half_size : y + half_size + 1, x - half_size : x + half_size + 1]
        medians.append(np.median(roi, axis=(0, 1)))
    return np.array(medians, dtype=np.float64)


def create_verification_crop(img_srgb, corners, half_size=15):
    img_bgr = cv2.cvtColor((img_srgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    centers = get_patch_centers(corners)
    for p in centers:
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.rectangle(img_bgr, (x - half_size, y - half_size), (x + half_size, y + half_size), (0, 255, 0), 2)
        cv2.circle(img_bgr, (x, y), 3, (0, 0, 255), -1)

    tl = np.array(corners['tl'], dtype=np.float64)
    tr = np.array(corners['tr'], dtype=np.float64)
    bl = np.array(corners['bl'], dtype=np.float64)
    br = tr + bl - tl
    center = (tl + br) / 2.0
    cx, cy = int(center[0]), int(center[1])
    crop = img_bgr[cy - 250 : cy + 250, cx - 200 : cx + 200]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def compute_von_kries_gains(x_linear_patches, y_linear_patches, gray_indices):
    x_gray = x_linear_patches[gray_indices]
    y_gray = y_linear_patches[gray_indices]
    return np.sum(x_gray * y_gray, axis=0) / np.sum(x_gray ** 2, axis=0)


def compute_color_matrix(x_linear_patches, y_linear_patches):
    x = x_linear_patches.T
    y = y_linear_patches.T
    return y @ x.T @ np.linalg.inv(x @ x.T)


def apply_matrix_transform(img_linear, matrix):
    transformed = img_linear @ matrix.T
    return np.clip(transformed, 0.0, 1.0)


def calculate_ciede2000_stats(srgb_patches_a, srgb_patches_b):
    lab_a = skimage.color.rgb2lab(srgb_patches_a)
    lab_b = skimage.color.rgb2lab(srgb_patches_b)
    de = skimage.color.deltaE_ciede2000(lab_a, lab_b)
    return float(np.mean(de)), float(np.max(de))


def build_comparison_image(orig_srgb, vk_srgb, mat_srgb, ref_srgb, target_height=1000):
    images = [orig_srgb, vk_srgb, mat_srgb, ref_srgb]
    titles = ['Original', 'Von Kries', 'Matriz 3x3', 'Referencia']
    resized_list = []

    for img, title in zip(images, titles):
        h, w = img.shape[:2]
        scale = target_height / float(h)
        target_w = int(round(w * scale))
        pil_img = Image.fromarray((img * 255).astype(np.uint8)).resize((target_w, target_height), Image.Resampling.LANCZOS)
        annotated = np.array(pil_img)
        cv2.putText(annotated, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(annotated, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        resized_list.append(annotated)

    return np.hstack(resized_list)


def main():
    fotos_dir = 'fotos'
    out_dir = 'resultados'
    config_path = 'corners_config.json'

    os.makedirs(out_dir, exist_ok=True)

    with open(config_path, 'r', encoding='utf-8') as f:
        corners_config = json.load(f)

    color_names = ['Amarillo', 'Azul', 'Cian', 'Magenta', 'Rojo', 'Verde']
    gray_indices = [r * 4 + 3 for r in range(6)]

    ref_filename = 'Referencia.JPG'
    ref_srgb = load_image(os.path.join(fotos_dir, ref_filename))
    ref_linear = srgb_to_linear(ref_srgb)
    ref_patches_lin = extract_patch_medians(ref_linear, corners_config[ref_filename])
    ref_patches_srgb = linear_to_srgb(ref_patches_lin)

    verification_crops = [create_verification_crop(ref_srgb, corners_config[ref_filename])]
    verification_labels = ['Referencia']

    results = []

    for color in color_names:
        filename = f'{color}.JPG'
        orig_srgb = load_image(os.path.join(fotos_dir, filename))
        orig_linear = srgb_to_linear(orig_srgb)
        orig_patches_lin = extract_patch_medians(orig_linear, corners_config[filename])

        vk_gains = compute_von_kries_gains(orig_patches_lin, ref_patches_lin, gray_indices)
        vk_matrix = np.diag(vk_gains)
        vk_linear = apply_matrix_transform(orig_linear, vk_matrix)
        vk_srgb = linear_to_srgb(vk_linear)

        m_matrix = compute_color_matrix(orig_patches_lin, ref_patches_lin)
        mat_linear = apply_matrix_transform(orig_linear, m_matrix)
        mat_srgb = linear_to_srgb(mat_linear)

        save_image(os.path.join(out_dir, f'{color}_von_kries.jpg'), vk_srgb)
        save_image(os.path.join(out_dir, f'{color}_matriz.jpg'), mat_srgb)

        comp_img = build_comparison_image(orig_srgb, vk_srgb, mat_srgb, ref_srgb)
        Image.fromarray(comp_img).save(os.path.join(out_dir, f'comparativa_{color}.jpg'), quality=92)

        orig_patches_srgb = linear_to_srgb(orig_patches_lin)
        vk_patches_srgb = linear_to_srgb(apply_matrix_transform(orig_patches_lin, vk_matrix))
        mat_patches_srgb = linear_to_srgb(apply_matrix_transform(orig_patches_lin, m_matrix))

        orig_mean, orig_max = calculate_ciede2000_stats(orig_patches_srgb, ref_patches_srgb)
        vk_mean, vk_max = calculate_ciede2000_stats(vk_patches_srgb, ref_patches_srgb)
        mat_mean, mat_max = calculate_ciede2000_stats(mat_patches_srgb, ref_patches_srgb)

        results.append({
            'color': color,
            'orig_mean': orig_mean,
            'orig_max': orig_max,
            'vk_mean': vk_mean,
            'vk_max': vk_max,
            'mat_mean': mat_mean,
            'mat_max': mat_max,
        })

        verification_crops.append(create_verification_crop(orig_srgb, corners_config[filename]))
        verification_labels.append(color)

    annotated_crops = []
    for crop, label in zip(verification_crops, verification_labels):
        labeled = crop.copy()
        cv2.putText(labeled, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(labeled, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        annotated_crops.append(labeled)

    blank = np.zeros_like(annotated_crops[0])
    row1 = np.hstack(annotated_crops[:4])
    row2 = np.hstack(annotated_crops[4:] + [blank])
    grid_img = np.vstack([row1, row2])
    Image.fromarray(grid_img).save(os.path.join(out_dir, 'verificacion_parches.jpg'), quality=95)

    print(f'{"Foto":<12} | {"Original (mean / max)":<23} | {"Von Kries (mean / max)":<23} | {"Matriz 3x3 (mean / max)":<23}')
    print('-' * 88)
    for r in results:
        print(f'{r["color"]:<12} | {r["orig_mean"]:6.2f} / {r["orig_max"]:6.2f}        | {r["vk_mean"]:6.2f} / {r["vk_max"]:6.2f}        | {r["mat_mean"]:6.2f} / {r["mat_max"]:6.2f}')


if __name__ == '__main__':
    main()
