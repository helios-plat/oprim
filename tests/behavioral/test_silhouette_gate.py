"""_silhouette_gate 确定性视觉门测试 (纯像素, 无模型)。"""
import pytest

from oprim import (
    bilateral_symmetry_error,
    bbox_of,
    load_mask,
    per_part_color_delta,
    proportion_delta,
    run_silhouette_gate,
    silhouette_iou,
)
from oprim._silhouette_gate import _srgb_to_lab, lab_distance


def _make_png(path, size=(64, 64), fg=(200, 40, 40), bg=(245, 245, 245)):
    from PIL import Image

    img = Image.new("RGB", size, bg)
    px = img.load()
    # 中间 50% 为前景
    for y in range(size[1] // 4, size[1] * 3 // 4):
        for x in range(size[0] // 4, size[0] * 3 // 4):
            px[x, y] = fg
    img.save(path)
    return path


def test_silhouette_iou_perfect_and_different():
    a = [True, True, False, False]
    assert silhouette_iou(a, list(a)) == 1.0
    assert silhouette_iou(a, [False, False, True, True]) == 0.0


def test_bbox_and_proportions():
    mask = [False] * 16
    for i in (5, 6, 9, 10):
        mask[i] = True
    box = bbox_of(mask, size=4)
    assert box == (1, 1, 3, 3)
    d = proportion_delta((0, 0, 4, 2), box)  # 参考是非方形 → aspect delta > 0
    assert d["aspect_ratio_delta"] > 0
    assert d["scale_delta"] > 0


def test_bilateral_symmetry():
    # 完全对称 mask → 0
    mask = [False] * 16
    for i in (5, 6, 9, 10):
        mask[i] = True
    assert bilateral_symmetry_error(mask, size=4) == 0.0
    # 完全不对称 → >0
    mask2 = [False] * 16
    mask2[0] = True
    assert bilateral_symmetry_error(mask2, size=4) > 0.0


def test_lab_distance_self_zero():
    assert lab_distance((50.0, 0.0, 0.0), (50.0, 0.0, 0.0)) == 0.0
    assert lab_distance(_srgb_to_lab((200, 40, 40)), _srgb_to_lab((200, 40, 40))) < 1e-6
    assert lab_distance(_srgb_to_lab((200, 40, 40)), _srgb_to_lab((245, 245, 245))) > 5


def test_load_mask_isolates_foreground(tmp_path):
    ref = _make_png(tmp_path / "ref.png")
    mask, warnings = load_mask(ref)
    assert sum(mask) > 0
    assert not warnings


def test_gate_passes_for_identical_renders(tmp_path):
    ref = _make_png(tmp_path / "ref.png")
    render = _make_png(tmp_path / "render.png", fg=(200, 45, 45))
    result = run_silhouette_gate(ref, render)
    assert result["passed"] is True
    assert result["checks"]["silhouetteIoU"] >= 0.8


def test_gate_fails_for_wrong_proportions(tmp_path):
    ref = _make_png(tmp_path / "ref.png", size=(64, 64))
    # 前景是横条 (宽 > 高), 与参考方块比例不同 → aspect/scale delta 超阈值
    render = _make_png(tmp_path / "render2.png", size=(64, 32), fg=(200, 45, 45))
    img = __import__("PIL").Image.open(render)
    px = img.load()
    for y in range(14, 18):
        for x in range(0, 64):
            px[x, y] = (200, 45, 45)
    img.save(render)
    result = run_silhouette_gate(ref, render)
    assert result["passed"] is False
    assert any("scale" in f or "aspect" in f for f in result["failures"])


def test_per_part_color_delta_catches_mismatch(tmp_path):
    render = _make_png(tmp_path / "r.png", fg=(30, 30, 200))
    recipes = [{"color": "#C82828"}]  # 红色配方 vs 蓝色渲染
    report = per_part_color_delta(recipes, render)
    assert report["maxDeltaE"] > 18.0
