"""Complete pipeline for Week 6 image-processing and machine-learning lab.

Run from this folder with ``python week6_solution.py``. The script reads the
Fruits-360 subset in ``dataset/``, creates a reproducible synthetic shape set,
trains the required models, and writes results plus an editable DOCX report.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
FRUIT_DIR = ROOT / "dataset"
SHAPE_DIR = ROOT / "dataset_shapes"
SHAPE_UNSEEN_DIR = ROOT / "dataset_shapes_unseen"
RESULTS_DIR = ROOT / "results"
CLASS_NAMES = ["apple", "banana", "orange"]
SHAPE_NAMES = ["circle", "square", "triangle"]
RANDOM_STATE = 42
TEST_SIZE = 0.20
IMAGE_SIZE = (128, 128)


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)


def extract_fruit_features(dataset_dir: Path = FRUIT_DIR):
    """Extract RGB/HSV statistics, HSV histograms, and grayscale entropy."""
    feature_sets = {"rgb_stats": [], "hsv_stats": [], "hsv_hist": [], "all": []}
    labels: list[str] = []
    rows: list[dict] = []
    for label in CLASS_NAMES:
        class_dir = dataset_dir / label
        paths = sorted([*class_dir.glob("*.jpg"), *class_dir.glob("*.jpeg")])
        if len(paths) < 50:
            raise RuntimeError(f"{class_dir} contains {len(paths)} images; at least 50 are required.")
        for path in paths:
            bgr = _read_image(path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            rgb_mean = rgb.mean(axis=(0, 1)).astype(np.float32)
            rgb_std = rgb.std(axis=(0, 1)).astype(np.float32)
            hsv_mean = hsv.mean(axis=(0, 1)).astype(np.float32)
            hsv_std = hsv.std(axis=(0, 1)).astype(np.float32)
            hist_parts = []
            for channel, bins, limits in ((0, 8, (0, 180)), (1, 8, (0, 256)), (2, 8, (0, 256))):
                hist = cv2.calcHist([hsv], [channel], None, [bins], list(limits)).flatten()
                hist_parts.append(hist / max(float(hist.sum()), 1.0))
            hsv_hist = np.concatenate(hist_parts).astype(np.float32)
            counts = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
            probabilities = counts[counts > 0] / counts.sum()
            entropy = np.array([-np.sum(probabilities * np.log2(probabilities))], dtype=np.float32)

            rgb_stats = np.concatenate([rgb_mean, rgb_std])
            hsv_stats = np.concatenate([hsv_mean, hsv_std])
            all_features = np.concatenate([rgb_stats, hsv_stats, hsv_hist, entropy])
            feature_sets["rgb_stats"].append(rgb_stats)
            feature_sets["hsv_stats"].append(hsv_stats)
            feature_sets["hsv_hist"].append(hsv_hist)
            feature_sets["all"].append(all_features)
            labels.append(label)

            row = {"file": str(path.relative_to(ROOT)), "label": label}
            for channel_name, value in zip(("R", "G", "B"), rgb_mean):
                row[f"rgb_mean_{channel_name}"] = float(value)
            for channel_name, value in zip(("R", "G", "B"), rgb_std):
                row[f"rgb_std_{channel_name}"] = float(value)
            for channel_name, value in zip(("H", "S", "V"), hsv_mean):
                row[f"hsv_mean_{channel_name}"] = float(value)
            for channel_name, value in zip(("H", "S", "V"), hsv_std):
                row[f"hsv_std_{channel_name}"] = float(value)
            for channel_name, channel_hist in zip(("H", "S", "V"), hist_parts):
                for bin_index, value in enumerate(channel_hist):
                    row[f"hsv_hist_{channel_name}_{bin_index:02d}"] = float(value)
            row["gray_entropy_bits"] = float(entropy[0])
            rows.append(row)

    feature_sets = {name: np.asarray(values, dtype=np.float32) for name, values in feature_sets.items()}
    return feature_sets, np.asarray(labels), rows


def save_fruit_preprocessing_example(dataset_dir: Path = FRUIT_DIR) -> Path:
    """Save an RGB/HSV/grayscale visualization for one example image."""
    sample = sorted((dataset_dir / "apple").glob("*.jpg"))[0]
    bgr = _read_image(sample)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    panels = [
        ("RGB", rgb),
        ("HSV - Hue", hsv[:, :, 0]),
        ("HSV - Saturation", hsv[:, :, 1]),
        ("HSV - Value", hsv[:, :, 2]),
        ("Grayscale", gray),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 2.6))
    for axis, (title, image) in zip(axes, panels):
        axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    path = RESULTS_DIR / "fruit_preprocessing.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_shape_dataset(
    output_dir: Path = SHAPE_DIR,
    samples_per_class: int = 60,
    seed: int = RANDOM_STATE,
) -> None:
    """Create varied, labeled 128x128 shape images using a fixed random seed."""
    rng = np.random.default_rng(seed)
    for label in SHAPE_NAMES:
        class_dir = output_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        current = len(list(class_dir.glob("*.png")))
        if current >= samples_per_class:
            continue
        for index in range(current, samples_per_class):
            canvas = np.full((160, 160, 3), int(rng.integers(238, 256)), dtype=np.uint8)
            mask = np.zeros((160, 160), dtype=np.uint8)
            center = (int(rng.integers(62, 99)), int(rng.integers(62, 99)))
            radius = int(rng.integers(27, 49))
            angle = float(rng.uniform(0, 360))
            if label == "circle":
                cv2.circle(mask, center, radius, 255, -1, lineType=cv2.LINE_AA)
            elif label == "square":
                half = radius
                points = np.array(
                    [[-half, -half], [half, -half], [half, half], [-half, half]], dtype=np.float32
                )
                rotation = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
                points = np.round(points @ rotation[:, :2].T + np.asarray(center)).astype(np.int32)
                cv2.fillPoly(mask, [points], 255, lineType=cv2.LINE_AA)
            else:
                points = np.array(
                    [[0, -radius], [-radius, radius], [radius, radius]], dtype=np.float32
                )
                rotation = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
                points = np.round(points @ rotation[:, :2].T + np.asarray(center)).astype(np.int32)
                cv2.fillPoly(mask, [points], 255, lineType=cv2.LINE_AA)

            # Pick a dark but nonconstant color so grayscale segmentation remains reliable.
            color = tuple(int(value) for value in rng.integers(25, 155, size=3))
            canvas[mask > 0] = color
            if rng.random() < 0.35:
                canvas = cv2.GaussianBlur(canvas, (3, 3), float(rng.uniform(0.2, 0.8)))
            if rng.random() < 0.30:
                noise = rng.normal(0, 2.0, canvas.shape).astype(np.int16)
                canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            destination = class_dir / f"{label}_{index + 1:03d}.png"
            cv2.imwrite(str(destination), canvas)


def _hog_visualization(gray: np.ndarray) -> np.ndarray:
    """Render a compact orientation histogram sketch for notebook/report display."""
    image = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle %= 180.0
    canvas = np.full((160, 160), 245, dtype=np.uint8)
    cell = 8
    bins = 9
    for row in range(0, 64, cell):
        for col in range(0, 64, cell):
            cell_angle = angle[row : row + cell, col : col + cell]
            cell_mag = magnitude[row : row + cell, col : col + cell]
            hist, edges = np.histogram(cell_angle, bins=bins, range=(0, 180), weights=cell_mag)
            bin_index = int(hist.argmax())
            strength = float(hist[bin_index])
            if strength <= 0:
                continue
            theta = math.radians((edges[bin_index] + edges[bin_index + 1]) / 2.0)
            center = (int((col + cell / 2) * 2.5), int((row + cell / 2) * 2.5))
            length = int(min(7, 2 + strength / 1200.0))
            dx = int(length * math.cos(theta))
            dy = int(length * math.sin(theta))
            cv2.line(canvas, (center[0] - dx, center[1] - dy), (center[0] + dx, center[1] + dy), 35, 1, cv2.LINE_AA)
    return canvas


def _shape_features(image_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read shape image: {image_path}")
    bgr = cv2.resize(bgr, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"No object contour found in {image_path}")
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, width, height = cv2.boundingRect(contour)
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    canny = cv2.Canny(blurred, 60, 150)
    roi_edges = canny[y : y + height, x : x + width]
    edge_density = float(np.count_nonzero(roi_edges) / max(width * height, 1))
    geometry = np.array(
        [
            area,
            perimeter,
            width,
            height,
            width / max(height, 1),
            4.0 * math.pi * area / max(perimeter * perimeter, 1e-8),
            area / max(hull_area, 1e-8),
            area / max(width * height, 1),
            edge_density,
        ],
        dtype=np.float32,
    )
    hog_descriptor = cv2.HOGDescriptor(
        (64, 64), (16, 16), (8, 8), (8, 8), 9
    )
    hog_image = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    hog = hog_descriptor.compute(hog_image).flatten().astype(np.float32)
    contour_view = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_view, [contour], -1, (0, 0, 255), 2)
    stages = {
        "Original": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        "Grayscale": gray,
        "Threshold/Otsu": threshold,
        "Canny": canny,
        "Contour": cv2.cvtColor(contour_view, cv2.COLOR_BGR2RGB),
        "HOG": _hog_visualization(gray),
    }
    return geometry, hog, stages


def extract_shape_features(dataset_dir: Path = SHAPE_DIR):
    geometries, hogs, labels, rows = [], [], [], []
    previews: dict[str, dict[str, np.ndarray]] = {}
    for label in SHAPE_NAMES:
        paths = sorted((dataset_dir / label).glob("*.png"))
        if len(paths) < 50:
            raise RuntimeError(f"{dataset_dir / label} contains {len(paths)} images; at least 50 are required.")
        for path in paths:
            geometry, hog, stages = _shape_features(path)
            geometries.append(geometry)
            hogs.append(hog)
            labels.append(label)
            row = {"file": str(path.relative_to(ROOT)), "label": label, **dict(zip(
                ["area", "perimeter", "width", "height", "aspect_ratio", "circularity", "solidity", "extent", "edge_density"],
                geometry.tolist(),
            ))}
            row.update({f"hog_{index:04d}": float(value) for index, value in enumerate(hog)})
            rows.append(row)
            previews.setdefault(label, stages)
    return (
        {"geometry": np.asarray(geometries), "hog": np.asarray(hogs)},
        np.asarray(labels),
        rows,
        previews,
    )


def _new_model(kind: str):
    if kind == "KNN":
        estimator = KNeighborsClassifier(n_neighbors=5, weights="distance")
    elif kind == "SVM":
        estimator = SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced")
    elif kind == "Random Forest":
        estimator = RandomForestClassifier(
            n_estimators=120, max_features="sqrt", class_weight="balanced", random_state=RANDOM_STATE
        )
    else:
        raise ValueError(f"Unknown model: {kind}")
    return make_pipeline(StandardScaler(), estimator)


def run_experiments(
    feature_sets: dict[str, np.ndarray],
    labels: np.ndarray,
    experiments: Iterable[tuple[str, str, str]],
    class_names: list[str],
    output_prefix: str,
):
    """Train/evaluate listed (experiment, feature key, model) combinations."""
    indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    records = []
    for experiment_name, feature_key, model_name in experiments:
        x = feature_sets[feature_key]
        model = _new_model(model_name)
        model.fit(x[train_indices], labels[train_indices])
        predictions = model.predict(x[test_indices])
        matrix = confusion_matrix(labels[test_indices], predictions, labels=class_names)
        records.append(
            {
                "experiment": experiment_name,
                "feature": feature_key,
                "model": model_name,
                "accuracy": float(accuracy_score(labels[test_indices], predictions)),
                "precision_macro": float(precision_score(labels[test_indices], predictions, labels=class_names, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(labels[test_indices], predictions, labels=class_names, average="macro", zero_division=0)),
                "f1_macro": float(f1_score(labels[test_indices], predictions, labels=class_names, average="macro", zero_division=0)),
                "confusion_matrix": matrix.tolist(),
                "class_names": class_names,
                "y_test": labels[test_indices].tolist(),
                "y_pred": predictions.tolist(),
                "test_indices": test_indices.tolist(),
                "per_class": classification_report(
                    labels[test_indices], predictions, labels=class_names,
                    output_dict=True, zero_division=0,
                ),
                "train_size": int(len(train_indices)),
                "test_size": int(len(test_indices)),
            }
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics_csv(records, RESULTS_DIR / f"{output_prefix}_metrics.csv")
    save_confusion_figure(records, class_names, RESULTS_DIR / f"{output_prefix}_confusion_matrices.png")
    return records


def save_metrics_csv(records: list[dict], path: Path) -> None:
    columns = ["experiment", "feature", "model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "train_size", "test_size"]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({column: record[column] for column in columns})


def save_feature_csv(rows: list[dict], path: Path) -> None:
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_figure(records: list[dict], class_names: list[str], path: Path) -> None:
    columns = min(3, len(records))
    rows = math.ceil(len(records) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.3 * rows), squeeze=False)
    for axis, record in zip(axes.flat, records):
        matrix = np.asarray(record["confusion_matrix"])
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"{record['experiment']}\n{record['model']} — Acc={record['accuracy']:.3f}")
        axis.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
        axis.set_yticks(range(len(class_names)), class_names)
        axis.set_xlabel("Nhãn dự đoán")
        axis.set_ylabel("Nhãn thật")
        for (i, j), value in np.ndenumerate(matrix):
            axis.text(j, i, str(value), ha="center", va="center", color="black")
    for axis in axes.flat[len(records) :]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def display_shape_previews(previews: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    stage_names = ["Original", "Grayscale", "Threshold/Otsu", "Canny", "Contour", "HOG"]
    fig, axes = plt.subplots(len(SHAPE_NAMES), len(stage_names), figsize=(12, 6))
    for row, label in enumerate(SHAPE_NAMES):
        for col, stage in enumerate(stage_names):
            axis = axes[row, col]
            image = previews[label][stage]
            axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
            if row == 0:
                axis.set_title(stage)
            if col == 0:
                axis.set_ylabel(label)
            axis.set_xticks([])
            axis.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def records_to_json(records: list[dict], path: Path) -> None:
    serializable = [
        {key: value for key, value in record.items() if key not in {"y_test", "y_pred", "test_indices"}}
        for record in records
    ]
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def save_fruit_prediction_artifacts(records: list[dict], feature_rows: list[dict]):
    """Save predictions for the held-out fruit images and a labeled contact sheet."""
    record = next(item for item in records if item["experiment"] == "TN4")
    rows = []
    for index, actual, predicted in zip(record["test_indices"], record["y_test"], record["y_pred"]):
        rows.append(
            {
                "file": feature_rows[index]["file"],
                "actual_label": actual,
                "predicted_label": predicted,
                "correct": actual == predicted,
                "model": f"{record['feature']} + {record['model']}",
            }
        )
    path = RESULTS_DIR / "fruit_test_predictions.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    examples = rows[:12]
    columns = 4
    fig, axes = plt.subplots(math.ceil(len(examples) / columns), columns, figsize=(10, 6))
    axes = np.asarray(axes).reshape(-1)
    for axis, row in zip(axes, examples):
        image = cv2.cvtColor(_read_image(ROOT / row["file"]), cv2.COLOR_BGR2RGB)
        axis.imshow(image)
        axis.set_title(f"Thật: {row['actual_label']}\nDự đoán: {row['predicted_label']}", fontsize=9)
        axis.axis("off")
    for axis in axes[len(examples) :]:
        axis.axis("off")
    fig.suptitle("Dự đoán TN4 trên ảnh test chưa dùng để huấn luyện")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fruit_test_predictions.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return rows


def predict_unseen_shape_samples(
    shape_features: dict[str, np.ndarray],
    shape_labels: np.ndarray,
    shape_records: list[dict],
):
    """Generate independent shapes and classify them using the best validation setup."""
    generate_shape_dataset(SHAPE_UNSEEN_DIR, samples_per_class=1, seed=2026)
    paths = [SHAPE_UNSEEN_DIR / label / f"{label}_001.png" for label in SHAPE_NAMES]
    new_geometry, new_hog = [], []
    for path in paths:
        geometry, hog, _ = _shape_features(path)
        new_geometry.append(geometry)
        new_hog.append(hog)
    new_geometry = np.asarray(new_geometry, dtype=np.float32)
    new_hog = np.asarray(new_hog, dtype=np.float32)
    all_geometry = shape_features["geometry"]
    all_hog = shape_features["hog"]
    shape_features["combined"] = np.concatenate([all_geometry, all_hog], axis=1)
    unseen_features = {
        "geometry": new_geometry,
        "hog": new_hog,
        "combined": np.concatenate([new_geometry, new_hog], axis=1),
    }
    best = max(shape_records, key=lambda result: (result["f1_macro"], result["accuracy"]))
    model = _new_model(best["model"])
    model.fit(shape_features[best["feature"]], shape_labels)
    predictions = model.predict(unseen_features[best["feature"]])
    rows = [
        {
            "file": str(path.relative_to(ROOT)),
            "actual_label": label,
            "predicted_label": str(prediction),
            "correct": label == prediction,
            "selected_feature": best["feature"],
            "selected_model": best["model"],
        }
        for path, label, prediction in zip(paths, SHAPE_NAMES, predictions)
    ]
    path = RESULTS_DIR / "shape_unseen_predictions.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, len(rows), figsize=(8, 3))
    for axis, row in zip(axes, rows):
        image = cv2.cvtColor(cv2.imread(str(ROOT / row["file"])), cv2.COLOR_BGR2RGB)
        axis.imshow(image)
        axis.set_title(f"Thật: {row['actual_label']}\nDự đoán: {row['predicted_label']}")
        axis.axis("off")
    fig.suptitle(f"Ảnh hình dạng độc lập — {best['feature']} / {best['model']}")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "shape_unseen_predictions.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return rows, best


def _format_metric(value: float) -> str:
    return f"{value:.3f}"


def _add_results_table(document: Document, records: list[dict]) -> None:
    headers = ["Thí nghiệm", "Đặc trưng", "Mô hình", "Accuracy", "Precision", "Recall", "F1-score"]
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, text in zip(table.rows[0].cells, headers):
        cell.text = text
        for run in cell.paragraphs[0].runs:
            run.bold = True
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), "D9EAF7")
        cell._tc.get_or_add_tcPr().append(shading)
    for record in records:
        cells = table.add_row().cells
        values = [
            record["experiment"],
            record["feature"],
            record["model"],
            _format_metric(record["accuracy"]),
            _format_metric(record["precision_macro"]),
            _format_metric(record["recall_macro"]),
            _format_metric(record["f1_macro"]),
        ]
        for cell, value in zip(cells, values):
            cell.text = str(value)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def _add_compact_table(document: Document, headers: list[str], data: list[list[object]]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = str(value)
        for run in cell.paragraphs[0].runs:
            run.bold = True
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), "D9EAF7")
        cell._tc.get_or_add_tcPr().append(shading)
    for record in data:
        cells = table.add_row().cells
        for cell, value in zip(cells, record):
            cell.text = str(value)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(8)


def create_report(
    fruit_records: list[dict],
    shape_records: list[dict],
    fruit_predictions: list[dict],
    shape_predictions: list[dict],
    fruit_feature_rows: list[dict],
    shape_feature_rows: list[dict],
) -> Path:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)
    styles = doc.styles
    styles["Normal"].font.name = "Times New Roman"
    styles["Normal"].font.size = Pt(11)
    styles["Heading 1"].font.name = "Times New Roman"
    styles["Heading 1"].font.color.rgb = RGBColor(31, 78, 121)
    styles["Heading 2"].font.name = "Times New Roman"
    doc.core_properties.title = "Báo cáo thực hành tuần 6 — Xử lý ảnh và Thị giác Robot"
    doc.core_properties.subject = "Trích chọn đặc trưng ảnh và phân loại bằng học máy"
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("Thực hành tuần 6 | Xử lý ảnh và Thị giác Robot")
    footer.add_run("    •    Trang ")
    page_field = OxmlElement("w:fldSimple")
    page_field.set(qn("w:instr"), "PAGE")
    footer._p.append(page_field)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("BÁO CÁO THỰC HÀNH TUẦN 6\nXỬ LÝ ẢNH VÀ THỊ GIÁC ROBOT")
    run.bold = True
    run.font.size = Pt(18)
    subtitle = doc.add_paragraph("Chủ đề: Trích chọn đặc trưng ảnh và ứng dụng trong các mô hình học máy")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for label in ["Sinh viên: ........................................................", "Lớp: ........................................................", "Giảng viên: ........................................................"]:
        line = doc.add_paragraph(label)
        line.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_page_break()

    doc.add_heading("1. Mục tiêu", level=1)
    doc.add_paragraph(
        "Thực hành tiền xử lý ảnh, trích xuất đặc trưng màu sắc, thống kê, hình học và HOG; "
        "huấn luyện các mô hình phân loại KNN, SVM và Random Forest; đánh giá bằng Accuracy, "
        "Precision, Recall, F1-score và ma trận nhầm lẫn."
    )

    doc.add_heading("2. Dữ liệu và thiết lập", level=1)
    doc.add_paragraph(
        f"Bài 1 sử dụng {fruit_records[0]['train_size'] + fruit_records[0]['test_size']} ảnh Fruits-360: "
        "50 ảnh mỗi lớp táo, chuối và cam; mỗi lớp có 40 ảnh Train và 10 ảnh Test. "
        f"Bài 2 tạo 180 ảnh hình dạng tổng hợp: 60 ảnh mỗi lớp, gồm 48 ảnh Train và 12 ảnh Test. "
        "Cả hai phần dùng chia phân tầng 80/20 với random_state=42; mọi phép chuẩn hóa được fit trong pipeline "
        "chỉ trên dữ liệu Train. Khi suy luận trên ảnh độc lập ở Bài 2, mô hình được huấn luyện lại trên toàn bộ "
        "180 ảnh đã gắn nhãn sau khi hoàn tất đánh giá holdout."
    )
    doc.add_paragraph(
        "Lưu ý: dữ liệu hình dạng trong báo cáo là ảnh tổng hợp để đáp ứng yêu cầu tối thiểu và giúp kiểm soát "
        "kích thước, góc quay, vị trí, màu và nhiễu. Kết quả vì vậy phản ánh bộ dữ liệu tổng hợp này."
    )
    doc.add_paragraph(
        "Môi trường chạy gồm Python, OpenCV, NumPy, Matplotlib, scikit-learn và python-docx. "
        "Các phiên bản cài đặt được khai báo trong requirements.txt; lệnh tái tạo toàn bộ kết quả là "
        "python week6_solution.py hoặc chạy tuần tự các cell trong week6_ml.ipynb."
    )

    doc.add_heading("3. Bài 1 — Phân loại ảnh theo đặc trưng thống kê và màu sắc", level=1)
    doc.add_heading("3.1. Tiền xử lý và đặc trưng", level=2)
    doc.add_paragraph(
        "Mỗi ảnh được đọc bằng OpenCV, resize về 128×128, đổi RGB sang HSV và chuyển sang ảnh xám để tính entropy. "
        "Đặc trưng gồm Mean và Standard Deviation trên ba kênh RGB; Mean và Standard Deviation trên ba kênh HSV; "
        "histogram HSV với 8 bins/kênh, chuẩn hóa tổng bằng 1; và entropy ảnh xám tính theo "
        "H = −Σ pᵢ log₂(pᵢ). Vector TN1 có 6 chiều, TN2 có 6 chiều, TN3 có 24 chiều histogram HSV; "
        "TN4 ghép 37 chiều (RGB stats + HSV stats + histogram HSV + entropy)."
    )
    doc.add_picture(str(RESULTS_DIR / "fruit_preprocessing.png"), width=Inches(6.5))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph("Các cấu hình được chạy theo đề: TN1 RGB Mean+Std/KNN; TN2 HSV Mean+Std/KNN; TN3 Histogram HSV/SVM; TN4 kết hợp tất cả đặc trưng/SVM. KNN dùng k=5, distance weighting; SVM dùng RBF kernel, C=2.")
    doc.add_paragraph("Bảng dưới là trích đoạn của bảng vector đầy đủ; toàn bộ 150 hàng và 37 giá trị đặc trưng/ảnh nằm trong results/fruit_feature_table.csv.")
    fruit_preview = [
        [r["file"], r["label"], f"{r['rgb_mean_R']:.1f}", f"{r['rgb_mean_G']:.1f}", f"{r['rgb_mean_B']:.1f}", f"{r['gray_entropy_bits']:.2f}"]
        for r in fruit_feature_rows[:5]
    ]
    _add_compact_table(doc, ["Tệp ảnh", "Nhãn", "Mean R", "Mean G", "Mean B", "Entropy"], fruit_preview)
    _add_results_table(doc, fruit_records)
    doc.add_picture(str(RESULTS_DIR / "fruit_confusion_matrices.png"), width=Inches(6.5))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph("Ma trận thể hiện số lượng dự đoán nguyên trên tập Test: mỗi hàng là nhãn thật, mỗi cột là nhãn dự đoán.")
    doc.add_paragraph("Hình ảnh dự đoán của TN4 hiển thị các ảnh giữ lại khỏi huấn luyện cùng nhãn thật và nhãn dự đoán; danh sách đủ 30 ảnh có trong results/fruit_test_predictions.csv.")
    doc.add_picture(str(RESULTS_DIR / "fruit_test_predictions.png"), width=Inches(6.2))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("3.2. Nhận xét", level=2)
    best_fruit = max(fruit_records, key=lambda row: row["f1_macro"])
    doc.add_paragraph(
        f"Cấu hình có F1-macro cao nhất trên tập kiểm tra là {best_fruit['experiment']} "
        f"({best_fruit['feature']}, {best_fruit['model']}), F1={best_fruit['f1_macro']:.3f}. "
        "Trong lần chạy này cả bốn cấu hình đều phân loại đúng 30/30 ảnh test. Đây không đảm bảo hiệu năng ngoài mẫu "
        "trên ảnh đời thực: tập con nhỏ, ảnh cùng nguồn Fruits-360, nền sáng tương tự và các lớp có khác biệt màu rõ rệt. "
        "Để đánh giá tổng quát hơn cần thêm ảnh độc lập, ánh sáng/nền đa dạng và kiểm tra chéo."
    )
    tn4 = next(record for record in fruit_records if record["experiment"] == "TN4")
    class_metrics = [
        [label, int(tn4["per_class"][label]["support"]), f"{tn4['per_class'][label]['precision']:.3f}", f"{tn4['per_class'][label]['recall']:.3f}", f"{tn4['per_class'][label]['f1-score']:.3f}"]
        for label in CLASS_NAMES
    ]
    doc.add_paragraph("Kết quả theo từng lớp của TN4:")
    _add_compact_table(doc, ["Lớp", "Support", "Precision", "Recall", "F1-score"], class_metrics)

    doc.add_heading("4. Bài 2 — Phân loại hình dạng bằng đặc trưng hình học và HOG", level=1)
    doc.add_heading("4.1. Tiền xử lý và đặc trưng", level=2)
    doc.add_paragraph(
        "Ảnh được chuyển sang Grayscale, làm mờ Gaussian kernel 5×5, phân ngưỡng Otsu để tách vật thể, "
        "phát hiện biên Canny với ngưỡng 60/150 và tìm Contour ngoài lớn nhất. Đặc trưng hình học gồm diện tích, "
        "chu vi, chiều rộng, chiều cao, tỷ lệ W/H, "
        "độ tròn 4πA/P², độ đặc (diện tích/diện tích convex hull), extent (diện tích/diện tích khung bao), "
        "và mật độ biên trong khung bao. HOG được tính bằng OpenCV trên ảnh xám 64×64 với cell 8×8, "
        "block 16×16, stride 8×8, 9 bins; vector có 1.764 chiều. Bộ kết hợp có 1.773 chiều."
    )
    doc.add_paragraph("Bảng dưới minh họa các chỉ số hình học; bảng CSV kèm theo chứa 9 chỉ số hình học, toàn bộ 1.764 chiều HOG và nhãn cho đủ 180 ảnh.")
    shape_preview = [
        [r["file"], r["label"], f"{r['area']:.0f}", f"{r['perimeter']:.1f}", f"{r['aspect_ratio']:.2f}", f"{r['circularity']:.3f}", f"{r['edge_density']:.3f}"]
        for r in shape_feature_rows[:5]
    ]
    _add_compact_table(doc, ["Tệp ảnh", "Nhãn", "Diện tích", "Chu vi", "W/H", "Độ tròn", "Mật độ biên"], shape_preview)
    doc.add_picture(str(RESULTS_DIR / "shape_preprocessing.png"), width=Inches(6.2))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(
        "So sánh ba nhóm đặc trưng (hình học, HOG, kết hợp hình học+HOG) với ba mô hình (KNN, SVM, Random Forest). "
        "Các đặc trưng được chuẩn hóa trong pipeline; thí nghiệm dùng cùng seed và phép chia dữ liệu."
    )
    _add_results_table(doc, shape_records)
    doc.add_picture(str(RESULTS_DIR / "shape_confusion_matrices.png"), width=Inches(6.5))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    best_shape = max(shape_records, key=lambda row: row["f1_macro"])
    doc.add_paragraph(
        f"Cấu hình tốt nhất trên tập kiểm tra là {best_shape['experiment']} "
        f"({best_shape['feature']}, {best_shape['model']}), Accuracy={best_shape['accuracy']:.3f}, "
        f"F1-macro={best_shape['f1_macro']:.3f}."
    )
    weakest_shape = min(shape_records, key=lambda result: result["f1_macro"])
    worst_matrix = np.asarray(weakest_shape["confusion_matrix"])
    np.fill_diagonal(worst_matrix, 0)
    true_idx, pred_idx = np.unravel_index(int(np.argmax(worst_matrix)), worst_matrix.shape)
    confusion_count = int(worst_matrix[true_idx, pred_idx])
    doc.add_paragraph(
        f"Phân tích lỗi cụ thể: cấu hình có F1 thấp nhất là {weakest_shape['experiment']} "
        f"({weakest_shape['model']}), nhầm {int(np.sum(np.asarray(weakest_shape['confusion_matrix'])) - np.trace(np.asarray(weakest_shape['confusion_matrix'])))} ảnh. "
        f"Lỗi xuất hiện nhiều nhất là hình {SHAPE_NAMES[true_idx]} bị dự đoán thành {SHAPE_NAMES[pred_idx]} "
        f"({confusion_count} ảnh). HOG-KNN dùng khoảng cách trong không gian nhiều chiều; với tập nhỏ, một số vector "
        "gradient gần nhau dù đường bao tổng thể khác, nên kết quả kém hơn đặc trưng hình học trên bộ hình đơn giản này."
    )
    doc.add_heading("4.2. Dự đoán ảnh hình dạng độc lập", level=2)
    doc.add_paragraph(
        f"Sau khi chọn cấu hình theo kết quả holdout, mô hình {shape_predictions[0]['selected_feature']} / "
        f"{shape_predictions[0]['selected_model']} được huấn luyện lại trên 180 ảnh có nhãn và thử trên 3 ảnh mới "
        "được sinh bằng seed khác, không nằm trong tập đã dùng để tính điểm. Kết quả được lưu trong "
        "results/shape_unseen_predictions.csv."
    )
    _add_compact_table(
        doc,
        ["Ảnh mới", "Nhãn thật", "Nhãn dự đoán", "Đúng"],
        [[r["file"], r["actual_label"], r["predicted_label"], "Có" if r["correct"] else "Không"] for r in shape_predictions],
    )
    doc.add_picture(str(RESULTS_DIR / "shape_unseen_predictions.png"), width=Inches(5.8))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("5. Thảo luận", level=1)
    qas = [
        ("Vì sao trích chọn đặc trưng thay vì dùng toàn bộ pixel?", "Đặc trưng rút gọn ảnh thành các đại lượng có ý nghĩa như màu trung bình, phân bố màu, hình dạng và hướng biên. Vector nhỏ hơn giúp huấn luyện nhanh, giảm nhiễu và dễ giải thích; đổi lại có thể mất một phần chi tiết không được đặc trưng hóa."),
        ("RGB và HSV ảnh hưởng thế nào đến phân loại?", "RGB biểu diễn cường độ màu theo ba thành phần gắn với thiết bị và dễ bị ảnh hưởng bởi ánh sáng. HSV tách Hue khỏi Saturation và Value nên thường mô tả sắc màu rõ hơn khi độ sáng thay đổi vừa phải. Cả hai vẫn nhạy với nền trắng, cân bằng trắng và điều kiện chụp."),
        ("Kết hợp nhiều đặc trưng có luôn tăng độ chính xác?", "Không. Trong Bài 1, cả bốn cấu hình đều đạt 1.000 trên tập test nên chưa thấy cải thiện nhờ ghép đặc trưng. Trong Bài 2, hình học đơn lẻ đạt F1=1.000, còn cấu hình kết hợp tốt nhất đạt F1=0.944; HOG-KNN cũng thấp hơn. Đặc trưng bổ sung có thể dư thừa hoặc làm tăng số chiều so với lượng dữ liệu. Cần chọn bằng đánh giá trên dữ liệu chưa thấy."),
        ("Đặc trưng hình học phân biệt tròn, vuông, tam giác ra sao?", "Độ tròn, tỷ lệ W/H, độ đặc, diện tích và chu vi khác nhau giữa ba hình. Tỷ lệ và độ tròn chịu ảnh hưởng bởi góc quay, méo phối cảnh, biên nhiễu và chất lượng phân đoạn."),
        ("HOG mô tả thông tin gì?", "HOG tổng hợp hướng gradient theo histogram ở các ô cục bộ, từ đó mô tả biên và cấu trúc hình dạng. HOG ít phụ thuộc màu nhưng phụ thuộc kích thước chuẩn hóa, độ tương phản và hướng biên."),
        ("Thay đổi kích thước/góc quay ảnh ảnh hưởng thế nào?", "Resize đồng nhất kích thước giúp tạo vector cùng chiều nhưng có thể làm nhòe biên. Đặc trưng diện tích, W/H và HOG có thể đổi khi xoay hoặc thay tỷ lệ; chuẩn hóa hình học, dữ liệu tăng cường hoặc đặc trưng bất biến giúp giảm ảnh hưởng."),
        ("Khi nào nên dùng đặc trưng hình dạng, khi nào dùng HOG?", "Đặc trưng hình học phù hợp khi vật thể được phân đoạn tốt và khác biệt về kích thước, tỷ lệ hoặc độ tròn. HOG hữu ích khi thông tin biên cục bộ quan trọng và hình dạng phức tạp hơn; có thể kết hợp hai nhóm rồi kiểm chứng thực nghiệm."),
    ]
    for question, answer in qas:
        paragraph = doc.add_paragraph()
        run = paragraph.add_run(question + " ")
        run.bold = True
        paragraph.add_run(answer)

    doc.add_heading("6. Kết luận", level=1)
    doc.add_paragraph(
        "Bài thực hành đã hoàn thành quy trình từ chuẩn bị dữ liệu, tiền xử lý, trích xuất đặc trưng, "
        "huấn luyện mô hình đến đánh giá và so sánh. Kết quả được xuất kèm bảng CSV và ma trận nhầm lẫn để "
        "có thể kiểm tra, chỉnh sửa và chạy lại."
    )
    doc.add_heading("Tài liệu tham khảo", level=1)
    doc.add_paragraph("Mihai Oltean, Fruits-360 dataset, https://github.com/fruits-360/fruits-360-100x100 (truy cập 04/11/2026). Giấy phép CC BY-SA 4.0.")
    doc.add_paragraph("Thay đổi: chọn 50 ảnh mỗi lớp từ tập Training; thay đổi kích thước ảnh và tạo dữ liệu hình dạng tổng hợp để thực hiện Bài 2.")
    doc.add_paragraph("Tệp nộp kèm: week6_ml.ipynb (notebook), week6_solution.py (mã nguồn), dataset/ và dataset_shapes/ (dữ liệu), results/ (vector đặc trưng, dự đoán, metrics và hình minh họa).")

    report_path = ROOT / "bao_cao_tuan_6.docx"
    doc.save(report_path)
    return report_path


def run_all():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fruit_features, fruit_labels, fruit_rows = extract_fruit_features()
    save_feature_csv(fruit_rows, RESULTS_DIR / "fruit_feature_table.csv")
    save_fruit_preprocessing_example()
    fruit_experiments = [
        ("TN1", "rgb_stats", "KNN"),
        ("TN2", "hsv_stats", "KNN"),
        ("TN3", "hsv_hist", "SVM"),
        ("TN4", "all", "SVM"),
    ]
    fruit_records = run_experiments(fruit_features, fruit_labels, fruit_experiments, CLASS_NAMES, "fruit")
    records_to_json(fruit_records, RESULTS_DIR / "fruit_results.json")
    fruit_predictions = save_fruit_prediction_artifacts(fruit_records, fruit_rows)

    generate_shape_dataset()
    shape_features, shape_labels, shape_rows, previews = extract_shape_features()
    save_feature_csv(shape_rows, RESULTS_DIR / "shape_feature_table.csv")
    display_shape_previews(previews, RESULTS_DIR / "shape_preprocessing.png")
    shape_experiments = [
        (f"{feature_name} / {model_name}", feature_key, model_name)
        for feature_name, feature_key in (("Hình học", "geometry"), ("HOG", "hog"), ("Hình học + HOG", "combined"))
        for model_name in ("KNN", "SVM", "Random Forest")
    ]
    shape_features["combined"] = np.concatenate([shape_features["geometry"], shape_features["hog"]], axis=1)
    shape_records = run_experiments(shape_features, shape_labels, shape_experiments, SHAPE_NAMES, "shape")
    records_to_json(shape_records, RESULTS_DIR / "shape_results.json")
    shape_predictions, _ = predict_unseen_shape_samples(shape_features, shape_labels, shape_records)
    report_path = create_report(fruit_records, shape_records, fruit_predictions, shape_predictions, fruit_rows, shape_rows)
    return fruit_records, shape_records, fruit_predictions, shape_predictions, report_path


if __name__ == "__main__":
    fruit, shapes, _, _, report = run_all()
    print(f"Fruit experiments: {len(fruit)}; shape experiments: {len(shapes)}")
    print(f"Editable report saved to: {report}")
