"""
AI Hub + 캐글 데이터 통합 파이프라인

폴더 구조:
    data/raw/
        kaggle_data/
            train_images/       ← 캐글 학습 이미지
            train_annotations/  ← 캐글 JSON
            test_images/        ← 캐글 테스트 이미지
        aihub_data/
            aihub_images/       ← AI Hub 이미지 (1.Training, 2.Validation)
            aihub_annotations/  ← AI Hub JSON (K-xxx_json 폴더들)
    data/processed/
        kaggle_merged.json      ← 캐글 병합 결과
        aihub_merged.json       ← AI Hub 병합 + 필터링 결과
        combined_train.json     ← 최종 통합 학습 데이터
        class_balance.csv       ← 클래스별 분포

실행:
    python aihub_pipeline.py
"""
from __future__ import annotations

import collections
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ── 경로 탐색 ──────────────────────────────────────────────
def find_project_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "data" / "raw").is_dir():
            return candidate
    raise FileNotFoundError("data/raw 폴더를 찾지 못함")


# ── 공통 유틸 ──────────────────────────────────────────────
def parse_drug_codes(file_name: str) -> list[int]:
    stem = file_name.split("_")[0]
    return [int(c) for c in stem.split("-")[1:]]


def is_valid_bbox(bbox: list, w_img: int, h_img: int) -> bool:
    if not bbox or len(bbox) != 4:
        return False
    x, y, w, h = bbox
    return w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= w_img and y + h <= h_img


# ── 1. 캐글 데이터 병합 ────────────────────────────────────
def merge_kaggle(kaggle_ann_dir: Path, processed_dir: Path) -> dict:
    json_files = sorted(kaggle_ann_dir.rglob("*.json"))
    log.info("캐글 JSON: %d개", len(json_files))

    fname_to_id: dict[str, int] = {}
    images, annotations, categories = [], [], {}
    next_img_id = next_ann_id = 1

    for path in json_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        img = data["images"][0]
        fname = img["file_name"]

        img_id = fname_to_id.get(fname)
        if img_id is None:
            img_id = next_img_id; next_img_id += 1
            fname_to_id[fname] = img_id
            images.append({**img, "id": img_id})

        w_img, h_img = img.get("width", 9999), img.get("height", 9999)
        for ann in data.get("annotations", []):
            if not is_valid_bbox(ann.get("bbox"), w_img, h_img):
                continue
            annotations.append({**ann, "image_id": img_id, "id": next_ann_id})
            next_ann_id += 1

        for cat in data.get("categories", []):
            categories[cat["id"]] = cat

    # 파일명 코드 수 vs annotation 수 불일치 → 제외
    anns_by_img = collections.defaultdict(list)
    for a in annotations:
        anns_by_img[a["image_id"]].append(a["category_id"])

    valid_ids = set()
    n_excluded = 0
    for im in images:
        codes = parse_drug_codes(im["file_name"])
        present = anns_by_img.get(im["id"], [])
        if all(c in present for c in codes) and codes:
            valid_ids.add(im["id"])
        else:
            n_excluded += 1

    images = [im for im in images if im["id"] in valid_ids]
    annotations = [a for a in annotations if a["image_id"] in valid_ids]

    coco = {"images": images, "annotations": annotations, "categories": list(categories.values())}
    (processed_dir / "kaggle_merged.json").write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")

    log.info("캐글 병합 완료: 이미지 %d장 / annotation %d개 / 클래스 %d개 (제외 %d건)",
             len(images), len(annotations), len(categories), n_excluded)
    return coco


# ── 2. AI Hub 데이터 병합 + 캐글 클래스 필터링 ────────────
def merge_aihub(aihub_ann_dir: Path, kaggle_cat_ids: set[int],
                kaggle_cat_names: dict[int, str], processed_dir: Path) -> dict:
    json_files = sorted(aihub_ann_dir.rglob("*.json"))
    log.info("AI Hub JSON: %d개", len(json_files))

    fname_to_id: dict[str, int] = {}
    images, annotations = [], []
    next_img_id = next_ann_id = 1
    n_filtered = 0

    for path in json_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        img = data["images"][0]
        fname = img["file_name"]
        mapping_code = img.get("dl_mapping_code", "")

        if not mapping_code.startswith("K-"):
            n_filtered += 1; continue
        k_num = int(mapping_code.replace("K-", ""))
        if k_num not in kaggle_cat_ids:
            n_filtered += 1; continue

        img_id = fname_to_id.get(fname)
        if img_id is None:
            img_id = next_img_id; next_img_id += 1
            fname_to_id[fname] = img_id
            images.append({**img, "id": img_id, "real_category_id": k_num})

        w_img, h_img = img.get("width", 9999), img.get("height", 9999)
        for ann in data.get("annotations", []):
            if not is_valid_bbox(ann.get("bbox"), w_img, h_img):
                continue
            annotations.append({**ann, "image_id": img_id, "id": next_ann_id, "category_id": k_num})
            next_ann_id += 1

    # 누락 제거
    anns_by_img = collections.defaultdict(list)
    for a in annotations:
        anns_by_img[a["image_id"]].append(a["category_id"])

    valid_ids = set()
    n_excluded = 0
    for im in images:
        codes_in_fname = [int(c) for c in im["file_name"].split("_")[0].split("-")[1:]]
        kaggle_codes = [c for c in codes_in_fname if c in kaggle_cat_ids]
        present = anns_by_img.get(im["id"], [])
        if all(c in present for c in kaggle_codes) and kaggle_codes:
            valid_ids.add(im["id"])
        else:
            n_excluded += 1

    images = [im for im in images if im["id"] in valid_ids]
    annotations = [a for a in annotations if a["image_id"] in valid_ids]

    categories = [
        {"id": cid, "name": kaggle_cat_names[cid], "supercategory": "pill"}
        for cid in sorted({a["category_id"] for a in annotations})
    ]

    coco = {"images": images, "annotations": annotations, "categories": categories}
    (processed_dir / "aihub_merged.json").write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")

    log.info("AI Hub 병합 완료: 이미지 %d장 / annotation %d개 (클래스 필터 %d건, 누락 제외 %d건)",
             len(images), len(annotations), n_filtered, n_excluded)
    return coco


# ── 3. 캐글 + AI Hub 통합 ─────────────────────────────────
def combine(kaggle_coco: dict, aihub_coco: dict, processed_dir: Path) -> dict:
    # image_id 재부여 (중복 방지)
    offset = max(im["id"] for im in kaggle_coco["images"]) + 1
    aihub_images = [{**im, "id": im["id"] + offset} for im in aihub_coco["images"]]
    aihub_anns = [{**a, "image_id": a["image_id"] + offset} for a in aihub_coco["annotations"]]

    # annotation_id 재부여
    all_anns = list(kaggle_coco["annotations"]) + aihub_anns
    for i, a in enumerate(all_anns, 1):
        a["id"] = i

    combined = {
        "images": kaggle_coco["images"] + aihub_images,
        "annotations": all_anns,
        "categories": kaggle_coco["categories"],
    }
    (processed_dir / "combined_train.json").write_text(json.dumps(combined, ensure_ascii=False), encoding="utf-8")

    log.info("통합 완료: 이미지 %d장 / annotation %d개",
             len(combined["images"]), len(combined["annotations"]))
    return combined


# ── 4. 클래스 분포 저장 ───────────────────────────────────
def save_class_balance(coco: dict, processed_dir: Path) -> None:
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    counter = collections.Counter(a["category_id"] for a in coco["annotations"])

    df = (pd.DataFrame([
        {"category_id": cid, "class_name": cat_names.get(cid, "?"), "count": cnt}
        for cid, cnt in counter.items()
    ]).sort_values("count", ascending=False).reset_index(drop=True))

    df.to_csv(processed_dir / "class_balance.csv", index=False, encoding="utf-8-sig")
    ratio = df["count"].iloc[0] / df["count"].iloc[-1]
    log.info("클래스 %d개 / 최다/최소 비율 %.1f:1 (최다=%s %d개 / 최소=%s %d개)",
             len(df), ratio, df.iloc[0]["class_name"], df.iloc[0]["count"],
             df.iloc[-1]["class_name"], df.iloc[-1]["count"])


# ── main ──────────────────────────────────────────────────
def main() -> None:
    root = find_project_root()
    raw = root / "data" / "raw"
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    kaggle_ann_dir = raw / "kaggle_data" / "train_annotations"
    aihub_ann_dir  = raw / "aihub_data"  / "aihub_annotations"
    kaggle_img_dir = raw / "kaggle_data" / "train_images"
    aihub_img_dir  = raw / "aihub_data"  / "aihub_images"

    log.info("PROJECT_ROOT: %s", root)

    # 1. 캐글 병합
    kaggle_coco = merge_kaggle(kaggle_ann_dir, processed)
    kaggle_cat_ids   = {c["id"] for c in kaggle_coco["categories"]}
    kaggle_cat_names = {c["id"]: c["name"] for c in kaggle_coco["categories"]}

    # 2. AI Hub 병합 + 필터링
    aihub_coco = merge_aihub(aihub_ann_dir, kaggle_cat_ids, kaggle_cat_names, processed)

    # 3. 통합
    combined = combine(kaggle_coco, aihub_coco, processed)

    # 4. 클래스 분포
    save_class_balance(combined, processed)

    log.info("완료 — 결과: %s", processed)
    log.info("  kaggle_merged.json  : 캐글 단독")
    log.info("  aihub_merged.json   : AI Hub 단독")
    log.info("  combined_train.json : 통합 학습용")
    log.info("  class_balance.csv   : 클래스 분포")


if __name__ == "__main__":
    main()
