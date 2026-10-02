import json
import os
import cv2
import numpy as np
from PIL import Image, ImageOps


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


def compute_color_matrix(x_patches, y_patches):
    x = x_patches.T
    y = y_patches.T
    return y @ x.T @ np.linalg.inv(x @ x.T)


def apply_color_matrix(img_linear, matrix):
    transformed = img_linear @ matrix.T
    return np.clip(transformed, 0.0, 1.0)


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


def main():
    fotos_dir = 'fotos'
    out_dir = 'resultados'
    config_path = 'corners_config.json'

    os.makedirs(out_dir, exist_ok=True)

    with open(config_path, 'r', encoding='utf-8') as f:
        corners_config = json.load(f)

    color_names = ['Amarillo', 'Azul', 'Cian', 'Magenta', 'Rojo', 'Verde']

    ref_filename = 'Referencia.JPG'
    ref_srgb = load_image(os.path.join(fotos_dir, ref_filename))
    ref_linear = srgb_to_linear(ref_srgb)
    ref_patches_lin = extract_patch_medians(ref_linear, corners_config[ref_filename])

    verification_crops = [create_verification_crop(ref_srgb, corners_config[ref_filename])]
    verification_labels = ['Referencia']

    for color in color_names:
        filename = f'{color}.JPG'
        orig_srgb = load_image(os.path.join(fotos_dir, filename))
        orig_linear = srgb_to_linear(orig_srgb)
        orig_patches_lin = extract_patch_medians(orig_linear, corners_config[filename])

        matrix = compute_color_matrix(orig_patches_lin, ref_patches_lin)
        corrected_linear = apply_color_matrix(orig_linear, matrix)
        corrected_srgb = linear_to_srgb(corrected_linear)

        out_path = os.path.join(out_dir, f'{color}_corregida.jpg')
        save_image(out_path, corrected_srgb)
        print(f'Guardada: {out_path}')

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
    print(f'Guardada verificacion: {os.path.join(out_dir, "verificacion_parches.jpg")}')


if __name__ == '__main__':
    main()
