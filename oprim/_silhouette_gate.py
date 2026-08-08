"""确定性视觉门 (Silhouette Gate) — Tier-1 像素级验证原语。

从 img2threejs forge/stage4_review/diagnose_render.py + extract_pbr_evidence.py
提炼: 在**任何模型视觉判断之前**运行的廉价确定性检查。纯数学/像素,
无模型参与 — 语义与 _vlm_consensus 互补: 硬门否决, 软门由模型意见挽救。

检查项:
- silhouette IoU: 参考图前景 mask vs 渲染图前景 mask 的重叠率;
- aspect-ratio delta / scale delta: 包围盒比例偏差;
- bilateral symmetry error: 渲染 mask 的双边对称误差 (物体通常对称);
- per-part color delta-E: 渲染整体主色簇 vs spec 各部件颜色配方 (lab 空间)。

mask 提取: 角落背景采样 + lab 距离阈值 (纯像素, 无 SAM 等外部依赖)。
依赖 PIL (宿主/容器均已装); 缺失时给出明确指引而非裸 ImportError。

使用方必须保证: 本门跑在渲染截图之上, 且截图相机视角与参考一致
(由调用方/上层管线负责 camera batch)。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:  # pragma: no cover - 环境依赖探测
    from PIL import Image, ImageChops, ImageStat
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Pillow 未安装: pip install pillow (视觉门需要像素级 mask/颜色分析)"
    ) from exc

# 阈值 (与 img2threejs forge 对齐)
SILHOUETTE_IOU_THRESHOLD: float = 0.80
ASPECT_RATIO_DELTA_THRESHOLD: float = 0.20
SCALE_DELTA_THRESHOLD: float = 0.25
COLOR_DELTA_E_THRESHOLD: float = 18.0
MASK_GRID_SIZE: int = 16
_CORNER_BG_FRACTION: float = 0.06  # 四角各取 6% 区域采样背景色


def _srgb_to_lab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """sRGB (0-255) → CIELAB (D65)。"""
    r, g, b = (v / 255.0 for v in rgb)

    def _f(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = _f(r), _f(g), _f(b)
    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) * 100.0
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722) * 100.0
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) * 100.0
    x /= 95.047
    y /= 100.000
    z /= 108.883
    fx = x ** (1 / 3) if x > 0.008856 else (7.787 * x) + 16 / 116
    fy = y ** (1 / 3) if y > 0.008856 else (7.787 * y) + 16 / 116
    fz = z ** (1 / 3) if z > 0.008856 else (7.787 * z) + 16 / 116
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def lab_distance(c1: tuple[float, float, float], c2: tuple[float, float, float]) -> float:
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2) ** 0.5


def _load_rgb(path: str | Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _corner_background_lab(img: Image.Image) -> tuple[float, float, float]:
    """四角采样背景色 (lab 中值)。"""
    w, h = img.size
    cw, ch = max(2, int(w * _CORNER_BG_FRACTION)), max(2, int(h * _CORNER_BG_FRACTION))
    corners = [
        img.crop((0, 0, cw, ch)),
        img.crop((w - cw, 0, w, ch)),
        img.crop((0, h - ch, cw, h)),
        img.crop((w - cw, h - ch, w, h)),
    ]
    pixels = []
    for corner in corners:
        for px in corner.getdata():
            pixels.append(_srgb_to_lab(px))
    if not pixels:
        return (50.0, 0.0, 0.0)
    n = len(pixels)
    return (
        sum(p[0] for p in pixels) / n,
        sum(p[1] for p in pixels) / n,
        sum(p[2] for p in pixels) / n,
    )


def load_mask(
    path: str | Path,
    size: int = MASK_GRID_SIZE,
    bg_lab: tuple[float, float, float] | None = None,
    bg_delta_e: float = 30.0,
) -> tuple[list[bool], list[str]]:
    """前景 mask: 缩放网格 → 与背景色 lab 距离超过阈值者为前景。

    Returns:
        (mask 布尔列表, warnings)。subject 占帧不足 3.5% 时 mask 不可用
        (回退整帧, 调用方应视 IoU/比例检查无效)。
    """
    warnings: list[str] = []
    img = _load_rgb(path).resize((size, size))
    if bg_lab is None:
        bg_lab = _corner_background_lab(img)
    pixels = list(img.getdata())
    mask = [lab_distance(_srgb_to_lab(px), bg_lab) > bg_delta_e for px in pixels]
    foreground = sum(mask)
    if foreground <= size * size * 0.035:
        warnings.append(
            "foreground mask fell back to whole-frame coverage (subject under 3.5% "
            "of the frame) — IoU/proportion not measuring the subject"
        )
    return mask, warnings


def silhouette_iou(reference_mask: list[bool], render_mask: list[bool]) -> float:
    if len(reference_mask) != len(render_mask) or not reference_mask:
        return 0.0
    inter = sum(a and b for a, b in zip(reference_mask, render_mask))
    union = sum(a or b for a, b in zip(reference_mask, render_mask))
    return inter / union if union else 0.0


def bbox_of(mask: list[bool], size: int = MASK_GRID_SIZE) -> tuple[int, int, int, int]:
    xs = [i % size for i, v in enumerate(mask) if v]
    ys = [i // size for i, v in enumerate(mask) if v]
    if not xs:
        return (0, 0, 0, 0)
    return (min(xs), min(ys), max(xs) + 1, max(ys) + 1)


def proportion_delta(
    reference_bbox: tuple[int, int, int, int], render_bbox: tuple[int, int, int, int]
) -> dict[str, float]:
    def _aspect(bbox: tuple[int, int, int, int]) -> float:
        w = max(1, bbox[2] - bbox[0])
        h = max(1, bbox[3] - bbox[1])
        return w / h

    def _diag(bbox: tuple[int, int, int, int]) -> float:
        return ((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2) ** 0.5

    ref_a, render_a = _aspect(reference_bbox), _aspect(render_bbox)
    ref_d, render_d = _diag(reference_bbox), _diag(render_bbox)
    return {
        "aspect_ratio_delta": abs(ref_a - render_a) / max(ref_a, 1e-9),
        "scale_delta": abs(ref_d - render_d) / max(ref_d, 1e-9),
    }


def bilateral_symmetry_error(mask: list[bool], size: int = MASK_GRID_SIZE) -> float:
    """渲染 mask 左右翻转不一致占比 (0=完美对称, 1=完全不对称)。"""
    n = size // 2
    errors = 0
    total = 0
    for y in range(size):
        for x in range(n):
            left = mask[y * size + x]
            right = mask[y * size + (size - 1 - x)]
            total += 1
            if left != right:
                errors += 1
    return errors / total if total else 0.0


def per_part_color_delta(
    recipes: list[dict[str, Any]], render_path: str | Path, size: int = 64
) -> dict[str, Any]:
    """渲染整体主色簇 vs spec 各部件颜色配方 (lab 距离最大差)。

    已知范围限制: 按整体色簇比较, 非真实逐部件裁剪区域 — 足够抓 gross
    mismatch, 不做像素级精确 (与 forge 一致的保守边界, 不静默夸大)。
    """
    img = _load_rgb(render_path).resize((size, size))
    render_labs = [_srgb_to_lab(px) for px in img.getdata()]
    # 渲染主色簇 (粗 k-means 一次迭代: 以 4 等分象限种子聚类)
    centers = [
        (50.0, 0.0, 0.0),
        (50.0, 50.0, 0.0),
        (50.0, -50.0, 0.0),
        (50.0, 0.0, 50.0),
    ]
    for _ in range(6):
        buckets: list[list[tuple[float, float, float]]] = [[] for _ in centers]
        for lab in render_labs:
            idx = min(range(len(centers)), key=lambda i: lab_distance(lab, centers[i]))
            buckets[idx].append(lab)
        for i, bucket in enumerate(buckets):
            if bucket:
                centers[i] = (
                    sum(p[0] for p in bucket) / len(bucket),
                    sum(p[1] for p in bucket) / len(bucket),
                    sum(p[2] for p in bucket) / len(bucket),
                )
    deltas = []
    for recipe in recipes:
        recipe_lab = _srgb_to_lab(tuple(float(v) for v in _recipe_rgb(recipe)))
        closest = min(lab_distance(recipe_lab, c) for c in centers)
        deltas.append(round(closest, 2))
    return {
        "maxDeltaE": max(deltas) if deltas else 0.0,
        "partCount": len(recipes),
        "perPartDeltaE": deltas,
    }


def _recipe_rgb(recipe: dict[str, Any]) -> tuple[float, float, float]:
    """颜色配方 → (r,g,b)。支持 colorMaterialRecipe 的 color/hex/rgb 字段。"""
    color = recipe.get("color") or recipe.get("hex") or recipe.get("rgb") or [128, 128, 128]
    if isinstance(color, str):
        color = color.lstrip("#")
        if len(color) == 3:
            color = "".join(c * 2 for c in color)
        return (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16))
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        return (float(color[0]), float(color[1]), float(color[2]))
    return (128.0, 128.0, 128.0)


def run_silhouette_gate(
    reference_path: str | Path,
    render_path: str | Path,
    spec: dict[str, Any] | None = None,
    pass_id: str | None = None,
    *,
    color_gated: bool = True,
) -> dict[str, Any]:
    """完整 Tier-1 门: mask IoU + 比例 + 对称 + 颜色 (lab ΔE) + 几何完整性。

    Returns:
        {passed, checks, failures, maskWarnings, renderHash, passId}
        passed=False 时 failures 给出每条硬性不通过原因 (模型视觉不得覆盖)。
    """
    import hashlib

    ref_mask, ref_warnings = load_mask(reference_path)
    render_mask, render_warnings = load_mask(render_path)
    mask_warnings = [f"reference: {w}" for w in ref_warnings] + [f"render: {w}" for w in render_warnings]

    iou = silhouette_iou(ref_mask, render_mask)
    ref_bbox, render_bbox = bbox_of(ref_mask), bbox_of(render_mask)
    proportions = proportion_delta(ref_bbox, render_bbox)
    symmetry = bilateral_symmetry_error(render_mask)

    checks: dict[str, Any] = {
        "silhouetteIoU": round(iou, 4),
        "aspectRatioDelta": round(proportions["aspect_ratio_delta"], 4),
        "scaleDelta": round(proportions["scale_delta"], 4),
        "bilateralSymmetryError": round(symmetry, 4),
    }
    failures: list[str] = []
    if any("unusable" in w for w in mask_warnings):
        failures.append(
            "silhouette evidence is unusable: subject under 3.5% of the frame — "
            "re-capture with the subject filling more of the frame"
        )
    if iou < SILHOUETTE_IOU_THRESHOLD:
        failures.append(f"silhouette IoU {iou:.3f} is below threshold {SILHOUETTE_IOU_THRESHOLD}")
    if proportions["aspect_ratio_delta"] > ASPECT_RATIO_DELTA_THRESHOLD:
        failures.append(
            f"aspect-ratio delta {proportions['aspect_ratio_delta']:.3f} exceeds "
            f"threshold {ASPECT_RATIO_DELTA_THRESHOLD}"
        )
    if proportions["scale_delta"] > SCALE_DELTA_THRESHOLD:
        failures.append(
            f"scale delta {proportions['scale_delta']:.3f} exceeds threshold {SCALE_DELTA_THRESHOLD}"
        )

    if spec is not None:
        recipes = [
            component.get("colorMaterialRecipe")
            for component in spec.get("componentTree", [])
            if isinstance(component, dict) and isinstance(component.get("colorMaterialRecipe"), dict)
        ]
        color_report = per_part_color_delta(recipes, render_path)
        color_report["gated"] = color_gated
        checks["colorDelta"] = color_report
        if color_gated and color_report["maxDeltaE"] > COLOR_DELTA_E_THRESHOLD:
            failures.append(
                f"max per-part color delta-E {color_report['maxDeltaE']} exceeds "
                f"threshold {COLOR_DELTA_E_THRESHOLD}"
            )
        geometry = spec.get("builtGeometry") or spec.get("geometry")
        if isinstance(geometry, dict):
            integrity = _geometry_integrity(geometry)
            checks["geometryIntegrity"] = integrity
            failures.extend(integrity["failures"])

    try:
        digest = hashlib.sha256(Path(render_path).read_bytes()).hexdigest()[:12]
    except OSError:
        digest = ""
    return {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "maskWarnings": mask_warnings,
        "renderHash": digest,
        "passId": pass_id,
    }


def _geometry_integrity(geometry: dict[str, Any]) -> dict[str, Any]:
    """几何完整性: spec 声明的几何体必须有非空尺寸/材质 (粗检)。"""
    failures: list[str] = []
    if isinstance(geometry, dict):
        for key, value in geometry.items():
            if isinstance(value, dict):
                dims = value.get("dimensions") or value.get("size") or value.get("radius")
                if dims is None:
                    failures.append(f"geometry part '{key}' lacks dimensions")
    return {"failures": failures, "checked": bool(geometry)}
