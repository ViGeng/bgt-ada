"""Dataset abstraction for video object detection datasets.

This module provides a common interface for different datasets, handling
dataset-specific features like ignored regions, class mappings, etc.

Supported datasets:
- UA-DETRAC: Traffic surveillance videos with vehicle detection
- MS COCO: Standard object detection benchmark (auto-downloaded via HuggingFace)
- MS COCO (local): File-based COCO for pre-downloaded data
- PASCAL VOC: Classic object detection benchmark (auto-extracted from local tar)
- PASCAL VOC (local): File-based VOC for pre-extracted data

To add a new dataset:
1. Create a new class inheriting from BaseVideoDataset
2. Implement the required abstract methods
3. Register it in DATASET_REGISTRY
"""

import json
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def model_dir_name(model_name: str) -> str:
    """Convert model name to directory-style name.

    Example: 'fasterrcnn_resnet50_fpn_v2' -> 'FASTERRCNN-RESNET50-FPN-V2'
    """
    return model_name.upper().replace("_", "-")


class BaseVideoDataset(ABC):
    """Abstract base class for video object detection datasets."""
    
    def __init__(self, data_root: Path):
        """
        Initialize dataset.
        
        Args:
            data_root: Root path to the dataset
        """
        self.data_root = Path(data_root)
    
    @abstractmethod
    def get_video_names(self, split: str = 'train') -> List[str]:
        """Get list of video names for a split."""
        pass
    
    @abstractmethod
    def get_video_split(self, video_name: str) -> str:
        """Determine which split (train/test) a video belongs to."""
        pass
    
    @abstractmethod
    def load_ground_truth(self, video_name: str) -> Dict[int, List[Dict]]:
        """
        Load ground truth annotations for a video.
        
        Returns:
            Dictionary mapping frame_id -> list of GT objects
            Each object: {'id': int, 'bbox': (left, top, width, height), 'class': str}
        """
        pass
    
    @abstractmethod
    def get_ignored_regions(self, video_name: str) -> List[Tuple[float, float, float, float]]:
        """
        Get ignored regions for a video.
        
        Returns:
            List of bounding boxes (left, top, width, height) that should be ignored
        """
        pass

    @abstractmethod
    def get_detection_file(self, video_name: str, model_name: str) -> Path:
        """Get path to detection results file for a video/sequence + model."""
        pass

    @abstractmethod
    def get_image_path(self, video_name: str, frame_id: int) -> Path:
        """Get path to a frame/image file."""
        pass

    @property
    def sequence_label(self) -> str:
        """Human-readable name for sequences (e.g. 'videos', 'chunks')."""
        return "sequences"

    def iter_frames(self, video_name: str) -> List[Tuple[int, Path]]:
        """Return (frame_id, image_path) pairs for a video/sequence.

        Used by detect.py to process images with consistent frame_id assignment.
        Default implementation uses load_ground_truth keys + get_image_path.
        """
        gt = self.load_ground_truth(video_name)
        return [(fid, self.get_image_path(video_name, fid)) for fid in sorted(gt.keys())]

    def filter_detections_by_ignored_regions(
        self, 
        detections: List[Tuple], 
        ignored_regions: List[Tuple[float, float, float, float]],
        overlap_threshold: float = 0.5
    ) -> List[Tuple]:
        """
        Filter out detections that fall within ignored regions.
        
        Args:
            detections: List of (left, top, width, height, conf, class)
            ignored_regions: List of ignored region boxes
            overlap_threshold: IoU threshold to consider a detection as ignored
            
        Returns:
            Filtered list of detections
        """
        if not ignored_regions:
            return detections
        
        filtered = []
        for det in detections:
            det_bbox = det[:4]
            
            # Check if detection overlaps significantly with any ignored region
            is_ignored = False
            for ignored_box in ignored_regions:
                overlap = self._compute_overlap_ratio(det_bbox, ignored_box)
                if overlap >= overlap_threshold:
                    is_ignored = True
                    break
            
            if not is_ignored:
                filtered.append(det)
        
        return filtered
    
    def filter_gt_by_ignored_regions(
        self,
        gt_objects: List[Dict],
        ignored_regions: List[Tuple[float, float, float, float]],
        overlap_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Filter out GT objects that fall within ignored regions.
        
        Args:
            gt_objects: List of GT dicts with 'bbox' key
            ignored_regions: List of ignored region boxes
            overlap_threshold: Overlap threshold to consider a GT as ignored
            
        Returns:
            Filtered list of GT objects
        """
        if not ignored_regions:
            return gt_objects
        
        filtered = []
        for gt in gt_objects:
            gt_bbox = gt['bbox']
            
            # Check if GT overlaps significantly with any ignored region
            is_ignored = False
            for ignored_box in ignored_regions:
                overlap = self._compute_overlap_ratio(gt_bbox, ignored_box)
                if overlap >= overlap_threshold:
                    is_ignored = True
                    break
            
            if not is_ignored:
                filtered.append(gt)
        
        return filtered
    
    def _compute_overlap_ratio(
        self, 
        bbox: Tuple[float, float, float, float], 
        region: Tuple[float, float, float, float]
    ) -> float:
        """
        Compute the ratio of bbox area that overlaps with region.
        
        Args:
            bbox: (left, top, width, height)
            region: (left, top, width, height)
            
        Returns:
            Overlap ratio (0 to 1) - intersection area / bbox area
        """
        x1_1, y1_1, w1, h1 = bbox
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = region
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Compute intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        bbox_area = w1 * h1
        
        return intersection / bbox_area if bbox_area > 0 else 0.0


class UADETRACDataset(BaseVideoDataset):
    """UA-DETRAC dataset handler with ignored regions support."""

    @property
    def sequence_label(self) -> str:
        return "videos"

    def __init__(self, data_root: Path):
        super().__init__(data_root)
        self._ignored_regions_cache: Dict[str, List[Tuple]] = {}
        self._split_cache: Dict[str, str] = {}
        self._build_split_cache()
    
    def _build_split_cache(self):
        """Build a cache of video_name -> split by checking annotation folders."""
        train_dir = self.data_root / 'DETRAC-Annos' / 'DETRAC-Train-Annotations-XML'
        test_dir = self.data_root / 'DETRAC-Annos' / 'DETRAC-Test-Annotations-XML'
        
        if train_dir.exists():
            for xml_file in train_dir.glob('*.xml'):
                self._split_cache[xml_file.stem] = 'train'
        
        if test_dir.exists():
            for xml_file in test_dir.glob('*.xml'):
                self._split_cache[xml_file.stem] = 'test'
    
    def get_video_names(self, split: str = 'train') -> List[str]:
        """Get list of video names for a split."""
        images_dir = self.data_root / 'DETRAC-Images'
        if not images_dir.exists():
            return []
        
        all_videos = sorted([d.name for d in images_dir.iterdir() if d.is_dir()])
        return [v for v in all_videos if self.get_video_split(v) == split]
    
    def get_video_split(self, video_name: str) -> str:
        """Determine which split a video belongs to by checking annotation files."""
        # Use cache if available
        if video_name in self._split_cache:
            return self._split_cache[video_name]
        
        # Fallback: check if annotation file exists in either folder
        train_xml = self.data_root / 'DETRAC-Annos' / 'DETRAC-Train-Annotations-XML' / f'{video_name}.xml'
        test_xml = self.data_root / 'DETRAC-Annos' / 'DETRAC-Test-Annotations-XML' / f'{video_name}.xml'
        
        if train_xml.exists():
            self._split_cache[video_name] = 'train'
            return 'train'
        elif test_xml.exists():
            self._split_cache[video_name] = 'test'
            return 'test'
        else:
            # Default to 'train' if no annotation found (will fail later with proper error)
            return 'train'
    
    def _get_xml_path(self, video_name: str) -> Path:
        """Get path to XML annotation file."""
        split = self.get_video_split(video_name)
        if split == 'train':
            xml_dir = self.data_root / 'DETRAC-Annos' / 'DETRAC-Train-Annotations-XML'
        else:
            xml_dir = self.data_root / 'DETRAC-Annos' / 'DETRAC-Test-Annotations-XML'
        return xml_dir / f'{video_name}.xml'
    
    def load_ground_truth(self, video_name: str) -> Dict[int, List[Dict]]:
        """Load ground truth annotations for a video."""
        xml_path = self._get_xml_path(video_name)
        
        if not xml_path.exists():
            raise FileNotFoundError(f"GT annotation not found: {xml_path}")
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        gt_data = {}
        
        for frame in root.findall('frame'):
            frame_num = frame.get('num')
            if frame_num is None:
                continue
            frame_id = int(frame_num)
            targets = []
            
            target_list = frame.find('target_list')
            if target_list is not None:
                for target in target_list.findall('target'):
                    target_id_str = target.get('id')
                    if target_id_str is None:
                        continue
                    target_id = int(target_id_str)
                    
                    box = target.find('box')
                    if box is None:
                        continue
                    left = float(box.get('left', '0'))
                    top = float(box.get('top', '0'))
                    width = float(box.get('width', '0'))
                    height = float(box.get('height', '0'))
                    
                    attribute = target.find('attribute')
                    if attribute is None:
                        continue
                    vehicle_type = attribute.get('vehicle_type', 'car')
                    
                    targets.append({
                        'id': target_id,
                        'bbox': (left, top, width, height),
                        'class': vehicle_type
                    })
            
            gt_data[frame_id] = targets
        
        return gt_data
    
    def get_ignored_regions(self, video_name: str) -> List[Tuple[float, float, float, float]]:
        """
        Get ignored regions for a video from XML annotations.
        
        UA-DETRAC defines ignored regions at the sequence level (same for all frames).
        """
        # Check cache first
        if video_name in self._ignored_regions_cache:
            return self._ignored_regions_cache[video_name]
        
        xml_path = self._get_xml_path(video_name)
        
        if not xml_path.exists():
            return []
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        ignored_regions = []
        ignored_region_elem = root.find('ignored_region')
        
        if ignored_region_elem is not None:
            for box in ignored_region_elem.findall('box'):
                left = float(box.get('left', '0'))
                top = float(box.get('top', '0'))
                width = float(box.get('width', '0'))
                height = float(box.get('height', '0'))
                ignored_regions.append((left, top, width, height))
        
        # Cache the result
        self._ignored_regions_cache[video_name] = ignored_regions
        
        return ignored_regions
    
    def get_detection_file(self, video_name: str, model_name: str) -> Path:
        """Get path to detection results file."""
        split = self.get_video_split(video_name)
        mdir = model_dir_name(model_name)
        det_base = self.data_root / "DETRAC-Detections" / f"DETRAC-{split.capitalize()}-Detections"
        return det_base / mdir / f"{video_name}_Det_{mdir}.txt"

    def get_image_path(self, video_name: str, frame_id: int) -> Path:
        """Get path to a frame image."""
        return self.data_root / "DETRAC-Images" / video_name / f"img{frame_id:05d}.jpg"

    def iter_frames(self, video_name: str) -> List[Tuple[int, Path]]:
        """List (frame_id, image_path) pairs by scanning the image directory."""
        img_dir = self.data_root / "DETRAC-Images" / video_name
        if not img_dir.exists():
            return []
        frames = sorted(img_dir.glob("img*.jpg"))
        return [(int(f.stem.replace("img", "")), f) for f in frames]

    def load_ground_truth_filtered(self, video_name: str, 
                                   overlap_threshold: float = 0.5) -> Dict[int, List[Dict]]:
        """
        Load ground truth with ignored region filtering applied.
        
        Args:
            video_name: Video name
            overlap_threshold: Overlap threshold to filter GT in ignored regions
            
        Returns:
            Filtered ground truth dictionary
        """
        gt_data = self.load_ground_truth(video_name)
        ignored_regions = self.get_ignored_regions(video_name)
        
        if not ignored_regions:
            return gt_data
        
        filtered_gt = {}
        for frame_id, gt_objects in gt_data.items():
            filtered_gt[frame_id] = self.filter_gt_by_ignored_regions(
                gt_objects, ignored_regions, overlap_threshold)
        
        return filtered_gt


# ========================================================================
# MS COCO Dataset
# ========================================================================

class COCODataset(BaseVideoDataset):
    """MS COCO dataset handler.

    Images are grouped into chunks (synthetic sequences) so the pipeline
    can process them with the same per-video logic used for UA-DETRAC.

    Expected layout::

        data_root/
            images/
                train2017/
                val2017/
            annotations/
                instances_train2017.json
                instances_val2017.json
            detections/          (generated by detect.py)
                train2017/
                    MODEL_DIR/
                        train2017_chunk_0000_Det_MODEL_DIR.txt
                val2017/
                    ...
    """

    CHUNK_SIZE = 500

    # Pipeline split -> COCO split
    _SPLIT_MAP = {"train": "train2017", "test": "val2017"}
    _REVERSE_MAP = {"train2017": "train", "val2017": "test"}

    @property
    def sequence_label(self) -> str:
        return "chunks"

    def __init__(self, data_root: Path, chunk_size: Optional[int] = None):
        super().__init__(data_root)
        if chunk_size is not None:
            self.CHUNK_SIZE = chunk_size
        self._image_ids: Dict[str, List[int]] = {}        # coco_split -> sorted ids
        self._image_info: Dict[str, Dict[int, dict]] = {} # coco_split -> {id: img_dict}
        self._gt: Dict[str, Dict[int, List[Dict]]] = {}   # coco_split -> {img_id: [gt]}
        self._category_map: Dict[int, str] = {}
        self._chunks: Dict[str, Dict[str, List[int]]] = {}  # coco_split -> {chunk_name: [ids]}
        self._chunk_split: Dict[str, str] = {}  # chunk_name -> coco_split
        self._load_annotations()

    # ---- internal helpers ------------------------------------------------

    def _load_annotations(self):
        anno_dir = self.data_root / "annotations"
        if not anno_dir.exists():
            return
        for coco_split in ["train2017", "val2017"]:
            anno_file = anno_dir / f"instances_{coco_split}.json"
            if not anno_file.exists():
                continue
            with open(anno_file) as f:
                data = json.load(f)
            # categories (only once)
            if not self._category_map:
                for cat in data["categories"]:
                    self._category_map[cat["id"]] = cat["name"]
            # images
            self._image_info[coco_split] = {}
            for img in data["images"]:
                self._image_info[coco_split][img["id"]] = img
            self._image_ids[coco_split] = sorted(self._image_info[coco_split])
            # annotations
            gt: Dict[int, List[Dict]] = {}
            for ann in data["annotations"]:
                if ann.get("iscrowd", 0):
                    continue
                cat_name = self._category_map.get(ann["category_id"], "unknown")
                gt.setdefault(ann["image_id"], []).append({
                    "id": ann["id"],
                    "bbox": tuple(ann["bbox"]),  # COCO format: (x, y, w, h)
                    "class": cat_name,
                })
            self._gt[coco_split] = gt
            # chunks
            ids = self._image_ids[coco_split]
            chunks: Dict[str, List[int]] = {}
            for i in range(0, len(ids), self.CHUNK_SIZE):
                cname = f"{coco_split}_chunk_{i // self.CHUNK_SIZE:04d}"
                chunks[cname] = ids[i:i + self.CHUNK_SIZE]
                self._chunk_split[cname] = coco_split
            self._chunks[coco_split] = chunks

    def _coco_split(self, pipeline_split: str) -> str:
        return self._SPLIT_MAP.get(pipeline_split, pipeline_split)

    def _find_chunk(self, video_name: str):
        """Return (coco_split, image_ids) for a chunk name."""
        coco_split = self._chunk_split.get(video_name)
        if coco_split is None:
            raise ValueError(f"Unknown COCO chunk: {video_name}")
        return coco_split, self._chunks[coco_split][video_name]

    # ---- BaseVideoDataset interface --------------------------------------

    def get_video_names(self, split: str = "train") -> List[str]:
        cs = self._coco_split(split)
        return sorted(self._chunks.get(cs, {}).keys())

    def get_video_split(self, video_name: str) -> str:
        cs = self._chunk_split.get(video_name, "")
        return self._REVERSE_MAP.get(cs, "train")

    def load_ground_truth(self, video_name: str) -> Dict[int, List[Dict]]:
        cs, ids = self._find_chunk(video_name)
        gt_data: Dict[int, List[Dict]] = {}
        for img_id in ids:
            gt_data[img_id] = self._gt.get(cs, {}).get(img_id, [])
        return gt_data

    def get_ignored_regions(self, video_name: str) -> List[Tuple[float, float, float, float]]:
        return []  # COCO uses iscrowd flag instead (filtered during loading)

    def get_detection_file(self, video_name: str, model_name: str) -> Path:
        cs = self._chunk_split.get(video_name, video_name)
        mdir = model_dir_name(model_name)
        return self.data_root / "detections" / cs / mdir / f"{video_name}_Det_{mdir}.txt"

    def get_image_path(self, video_name: str, frame_id: int) -> Path:
        cs = self._chunk_split.get(video_name)
        if cs is None:
            raise ValueError(f"Unknown chunk: {video_name}")
        info = self._image_info[cs][frame_id]
        return self.data_root / "images" / cs / info["file_name"]

    def iter_frames(self, video_name: str) -> List[Tuple[int, Path]]:
        cs, ids = self._find_chunk(video_name)
        return [(img_id, self.get_image_path(video_name, img_id)) for img_id in ids]


# ========================================================================
# PASCAL VOC Dataset
# ========================================================================

# VOC class name -> COCO class name (only where they differ)
_VOC_TO_COCO_NAME = {
    "aeroplane": "airplane",
    "motorbike": "motorcycle",
    "diningtable": "dining table",
    "pottedplant": "potted plant",
    "sofa": "couch",
    "tvmonitor": "tv",
}


class VOCDataset(BaseVideoDataset):
    """PASCAL VOC dataset handler (VOC2007 / VOC2012).

    Images are grouped into chunks (synthetic sequences) similar to COCO.
    Class names are mapped to COCO-compatible names so that detection models
    (which output COCO class names) can be matched against VOC ground truth.

    Expected layout::

        data_root/
            VOCdevkit/
                VOC2007/          (or VOC2012)
                    JPEGImages/
                    Annotations/
                    ImageSets/
                        Main/
                            train.txt
                            val.txt
                            trainval.txt
                            test.txt
            detections/           (generated by detect.py)
                train/
                    MODEL_DIR/
                        train_chunk_0000_Det_MODEL_DIR.txt
                val/
                    ...
    """

    CHUNK_SIZE = 500

    # Pipeline split -> VOC split
    _SPLIT_MAP = {"train": "trainval", "test": "val"}
    _REVERSE_MAP: Dict[str, str] = {}  # built dynamically

    @property
    def sequence_label(self) -> str:
        return "chunks"

    def __init__(self, data_root: Path, chunk_size: Optional[int] = None,
                 year: Optional[str] = None):
        super().__init__(data_root)
        if chunk_size is not None:
            self.CHUNK_SIZE = chunk_size
        self._year = year or self._detect_year()
        self._voc_root = self.data_root / "VOCdevkit" / f"VOC{self._year}"
        self._image_ids: Dict[str, List[str]] = {}    # voc_split -> sorted ids
        self._chunks: Dict[str, Dict[str, List[str]]] = {}
        self._chunk_split: Dict[str, str] = {}  # chunk_name -> voc_split
        self._gt_cache: Dict[str, Dict[str, List[Dict]]] = {}  # voc_split -> {img_id: [gt]}
        self._REVERSE_MAP = {}
        self._load_splits()

    # ---- internal helpers ------------------------------------------------

    def _detect_year(self) -> str:
        devkit = self.data_root / "VOCdevkit"
        if not devkit.exists():
            return "2010"
        for y in ["2012", "2010", "2007"]:
            if (devkit / f"VOC{y}").exists():
                return y
        return "2010"

    def _load_splits(self):
        main_dir = self._voc_root / "ImageSets" / "Main"
        if not main_dir.exists():
            return
        for voc_split in ["train", "val", "trainval", "test"]:
            split_file = main_dir / f"{voc_split}.txt"
            if not split_file.exists():
                continue
            ids = sorted(line.strip() for line in split_file.read_text().splitlines() if line.strip())
            self._image_ids[voc_split] = ids
            # Build chunks
            chunks: Dict[str, List[str]] = {}
            for i in range(0, len(ids), self.CHUNK_SIZE):
                cname = f"{voc_split}_chunk_{i // self.CHUNK_SIZE:04d}"
                chunks[cname] = ids[i:i + self.CHUNK_SIZE]
                self._chunk_split[cname] = voc_split
            self._chunks[voc_split] = chunks
        # Build reverse map from pipeline splits
        for pipe_split, voc_split in self._SPLIT_MAP.items():
            if voc_split in self._chunks:
                self._REVERSE_MAP[voc_split] = pipe_split
        # Fallback: map any remaining VOC splits
        for vs in self._image_ids:
            if vs not in self._REVERSE_MAP:
                self._REVERSE_MAP[vs] = "train" if "train" in vs else "test"

    def _parse_voc_annotation(self, img_id: str) -> List[Dict]:
        """Parse a single VOC XML annotation file."""
        xml_path = self._voc_root / "Annotations" / f"{img_id}.xml"
        if not xml_path.exists():
            return []
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objects = []
        for obj_idx, obj in enumerate(root.findall("object")):
            name_elem = obj.find("name")
            if name_elem is None:
                continue
            raw_name = name_elem.text.strip().lower()
            class_name = _VOC_TO_COCO_NAME.get(raw_name, raw_name)
            difficult = obj.find("difficult")
            if difficult is not None and difficult.text.strip() == "1":
                continue  # skip difficult objects
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            objects.append({
                "id": obj_idx,
                "bbox": (xmin, ymin, xmax - xmin, ymax - ymin),
                "class": class_name,
            })
        return objects

    def _get_gt_for_split(self, voc_split: str) -> Dict[str, List[Dict]]:
        """Load (and cache) all GT for a VOC split."""
        if voc_split not in self._gt_cache:
            gt: Dict[str, List[Dict]] = {}
            for img_id in self._image_ids.get(voc_split, []):
                gt[img_id] = self._parse_voc_annotation(img_id)
            self._gt_cache[voc_split] = gt
        return self._gt_cache[voc_split]

    def _find_chunk(self, video_name: str):
        voc_split = self._chunk_split.get(video_name)
        if voc_split is None:
            raise ValueError(f"Unknown VOC chunk: {video_name}")
        return voc_split, self._chunks[voc_split][video_name]

    # ---- BaseVideoDataset interface --------------------------------------

    def get_video_names(self, split: str = "train") -> List[str]:
        vs = self._SPLIT_MAP.get(split, split)
        return sorted(self._chunks.get(vs, {}).keys())

    def get_video_split(self, video_name: str) -> str:
        vs = self._chunk_split.get(video_name, "")
        return self._REVERSE_MAP.get(vs, "train")

    def load_ground_truth(self, video_name: str) -> Dict[int, List[Dict]]:
        """Return GT keyed by 1-based sequential frame index within the chunk."""
        vs, img_ids = self._find_chunk(video_name)
        all_gt = self._get_gt_for_split(vs)
        return {idx: all_gt.get(img_id, []) for idx, img_id in enumerate(img_ids, 1)}

    def get_ignored_regions(self, video_name: str) -> List[Tuple[float, float, float, float]]:
        return []  # VOC does not define ignored regions

    def get_detection_file(self, video_name: str, model_name: str) -> Path:
        vs = self._chunk_split.get(video_name, video_name)
        mdir = model_dir_name(model_name)
        return self.data_root / "detections" / vs / mdir / f"{video_name}_Det_{mdir}.txt"

    def get_image_path(self, video_name: str, frame_id: int) -> Path:
        """frame_id is 1-based index within the chunk."""
        _, img_ids = self._find_chunk(video_name)
        img_id = img_ids[frame_id - 1]
        return self._voc_root / "JPEGImages" / f"{img_id}.jpg"

    def iter_frames(self, video_name: str) -> List[Tuple[int, Path]]:
        _, img_ids = self._find_chunk(video_name)
        return [
            (idx, self._voc_root / "JPEGImages" / f"{img_id}.jpg")
            for idx, img_id in enumerate(img_ids, 1)
        ]


# ========================================================================
# HuggingFace COCO Dataset
# ========================================================================

class HFCOCODataset(BaseVideoDataset):
    """COCO dataset via HuggingFace ``datasets`` library.

    Automatically downloads COCO from ``detection-datasets/coco`` on first use
    (cached by HF afterwards). Images are extracted to *data_root/images/* on
    demand so the rest of the pipeline can access them by path.

    No manual download step is required.

    Generated layout::

        data_root/
            images/
                train2017/    (extracted lazily)
                val2017/
            detections/       (generated by detect.py)
                train2017/
                    MODEL_DIR/
                val2017/
                    ...
    """

    CHUNK_SIZE = 500
    HF_REPO = "detection-datasets/coco"

    # Pipeline split -> HF split name
    _SPLIT_MAP = {"train": "train", "test": "val"}
    _REVERSE_MAP = {"train": "train", "val": "test"}

    @property
    def sequence_label(self) -> str:
        return "chunks"

    def __init__(self, data_root: Path, chunk_size: Optional[int] = None):
        super().__init__(data_root)
        if chunk_size is not None:
            self.CHUNK_SIZE = chunk_size
        self._hf: Dict[str, Any] = {}            # hf_split -> HF Dataset
        self._cat_names: List[str] = []
        self._image_ids: Dict[str, list] = {}    # hf_split -> [image_id, ...]
        self._id_to_idx: Dict[str, Dict[int, int]] = {}  # hf_split -> {img_id: idx}
        self._chunks: Dict[str, Dict[str, Tuple[int, int]]] = {}
        self._chunk_split: Dict[str, str] = {}   # chunk_name -> hf_split

    # ---- lazy loading ----------------------------------------------------

    def _ensure_split(self, hf_split: str) -> None:
        """Download / load a single HF split on first access."""
        if hf_split in self._hf:
            return
        from datasets import load_dataset  # noqa: delayed import

        from . import log
        log.info(f"Loading COCO {hf_split} from HuggingFace ({self.HF_REPO})")
        ds = load_dataset(self.HF_REPO, split=hf_split)
        self._hf[hf_split] = ds

        if not self._cat_names:
            self._cat_names = list(
                ds.features["objects"]["category"].feature.names
            )

        # Fast column read for image IDs (no image decoding)
        ids = ds["image_id"]
        self._image_ids[hf_split] = ids
        self._id_to_idx[hf_split] = {iid: i for i, iid in enumerate(ids)}

        # Build chunks
        n = len(ds)
        chunks: Dict[str, Tuple[int, int]] = {}
        for i in range(0, n, self.CHUNK_SIZE):
            cname = f"{hf_split}_chunk_{i // self.CHUNK_SIZE:04d}"
            chunks[cname] = (i, min(i + self.CHUNK_SIZE, n))
            self._chunk_split[cname] = hf_split
        self._chunks[hf_split] = chunks

    def _find_chunk(self, video_name: str):
        """Return *(hf_split, start_idx, end_idx)* for a chunk."""
        if video_name not in self._chunk_split:
            # Infer split from chunk name prefix
            for hs in ("train", "val"):
                if video_name.startswith(hs):
                    self._ensure_split(hs)
                    break
            else:
                # Fallback: load both splits
                for hs in ("train", "val"):
                    self._ensure_split(hs)
        hs = self._chunk_split.get(video_name)
        if hs is None:
            raise ValueError(f"Unknown COCO chunk: {video_name}")
        start, end = self._chunks[hs][video_name]
        return hs, start, end

    # ---- image extraction ------------------------------------------------

    def _image_file(self, hf_split: str, image_id: int) -> Path:
        """Return local path for an image, extracting from HF if absent."""
        out_dir = self.data_root / "images" / f"{hf_split}2017"
        path = out_dir / f"{image_id:012d}.jpg"
        if not path.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
            idx = self._id_to_idx[hf_split][image_id]
            self._hf[hf_split][idx]["image"].save(path)
        return path

    # ---- BaseVideoDataset interface --------------------------------------

    def get_video_names(self, split: str = "train") -> List[str]:
        hs = self._SPLIT_MAP.get(split, split)
        self._ensure_split(hs)
        return sorted(self._chunks.get(hs, {}).keys())

    def get_video_split(self, video_name: str) -> str:
        hs = self._chunk_split.get(video_name, "")
        return self._REVERSE_MAP.get(hs, "train")

    def load_ground_truth(self, video_name: str) -> Dict[int, List[Dict]]:
        hs, start, end = self._find_chunk(video_name)
        # Use select + remove image column to avoid decoding images
        batch = self._hf[hs].select(range(start, end)).remove_columns("image")
        gt: Dict[int, List[Dict]] = {}
        for row in batch:
            img_id = row["image_id"]
            objs_data = row["objects"]
            objs = []
            for i, bbox in enumerate(objs_data["bbox"]):
                x1, y1, x2, y2 = bbox  # HF uses xyxy
                objs.append({
                    "id": objs_data["bbox_id"][i],
                    "bbox": (x1, y1, x2 - x1, y2 - y1),  # convert to xywh
                    "class": self._cat_names[objs_data["category"][i]],
                })
            gt[img_id] = objs
        return gt

    def get_ignored_regions(self, video_name: str) -> List[Tuple[float, float, float, float]]:
        return []  # COCO uses iscrowd flag (filtered during HF dataset creation)

    def get_detection_file(self, video_name: str, model_name: str) -> Path:
        hs = self._chunk_split.get(video_name, video_name)
        mdir = model_dir_name(model_name)
        return (self.data_root / "detections" / f"{hs}2017"
                / mdir / f"{video_name}_Det_{mdir}.txt")

    def get_image_path(self, video_name: str, frame_id: int) -> Path:
        hs, _, _ = self._find_chunk(video_name)
        return self._image_file(hs, frame_id)

    def iter_frames(self, video_name: str) -> List[Tuple[int, Path]]:
        hs, start, end = self._find_chunk(video_name)
        ids = self._image_ids[hs][start:end]
        out_dir = self.data_root / "images" / f"{hs}2017"

        # Find which images still need extraction
        frames = []
        missing_ids = []
        for img_id in ids:
            path = out_dir / f"{img_id:012d}.jpg"
            frames.append((img_id, path))
            if not path.exists():
                missing_ids.append(img_id)

        if missing_ids:
            out_dir.mkdir(parents=True, exist_ok=True)
            indices = [self._id_to_idx[hs][iid] for iid in missing_ids]
            batch = self._hf[hs].select(indices)
            for row in batch:
                p = out_dir / f"{row['image_id']:012d}.jpg"
                row["image"].save(p)

        return frames


# ========================================================================
# Auto-extracting VOC Dataset
# ========================================================================

class AutoExtractVOCDataset(VOCDataset):
    """PASCAL VOC dataset with auto-extraction from local tar archives.

    If VOCdevkit is not present under *data_root*, searches for VOC tar files
    (e.g. ``VOCtrainval_03-May-2010.tar``) in the **parent** directory
    (``data/``) and extracts them automatically.

    No manual extraction step is required — just place the tar file(s) in
    the ``data/`` directory and run the pipeline.
    """

    def __init__(self, data_root: Path, chunk_size: Optional[int] = None,
                 year: Optional[str] = None):
        self._auto_extract(Path(data_root))
        super().__init__(data_root, chunk_size=chunk_size, year=year)

    def _auto_extract(self, data_root: Path) -> None:
        """Extract VOC tar files into *data_root* if VOCdevkit is missing."""
        devkit = data_root / "VOCdevkit"
        if devkit.exists():
            return  # already extracted

        import tarfile

        # Look for VOC tar files in the parent directory (e.g. data/)
        tar_dir = data_root.parent
        tar_files = sorted(
            list(tar_dir.glob("VOCtrainval*.tar"))
            + list(tar_dir.glob("VOCtest*.tar"))
        )

        if not tar_files:
            raise FileNotFoundError(
                f"VOCdevkit not found at {devkit} and no VOC tar files "
                f"found in {tar_dir}. Place VOC tar files "
                f"(e.g. VOCtrainval_03-May-2010.tar) in {tar_dir}."
            )

        data_root.mkdir(parents=True, exist_ok=True)
        for tar_path in tar_files:
            marker = data_root / f".{tar_path.name}.done"
            if marker.exists():
                continue
            from . import log
            log.info(f"Extracting {tar_path.name}")
            with tarfile.open(tar_path, "r") as tf:
                tf.extractall(path=str(data_root))
            marker.touch()


# ========================================================================
# Registry
# ========================================================================

DATASET_REGISTRY: Dict[str, type] = {
    "ua-detrac": UADETRACDataset,
    "detrac": UADETRACDataset,
    "coco": HFCOCODataset,
    "ms-coco": HFCOCODataset,
    "coco2017": HFCOCODataset,
    "coco-local": COCODataset,
    "voc": AutoExtractVOCDataset,
    "pascal-voc": AutoExtractVOCDataset,
    "voc2007": AutoExtractVOCDataset,
    "voc2010": AutoExtractVOCDataset,
    "voc2012": AutoExtractVOCDataset,
    "voc-local": VOCDataset,
}


def resolve_base_dataset(dataset_name: str) -> str:
    """Resolve a compound dataset name to its base DATASET_REGISTRY key.

    For example, ``"voc-yolov7n-yolov7large"`` resolves to ``"voc"`` because
    the registry contains ``"voc"`` and the compound name starts with it.

    Returns the name unchanged if it is already a registry key.
    """
    dataset_name = dataset_name.lower()
    if dataset_name in DATASET_REGISTRY:
        return dataset_name
    # Longest-prefix match to avoid ambiguity (e.g. "voc" vs "voc-local")
    best = ""
    for key in DATASET_REGISTRY:
        if dataset_name.startswith(key) and len(key) > len(best):
            best = key
    return best or dataset_name


def get_dataset(dataset_name: str, data_root: Path) -> BaseVideoDataset:
    """
    Factory function to get a dataset instance.

    Supports compound names like ``"voc-yolov7n-yolov7large"`` — the base
    dataset type (``"voc"``) is resolved automatically.

    Args:
        dataset_name: Name of the dataset (e.g., 'ua-detrac', 'coco', 'voc')
        data_root: Root path to the dataset

    Returns:
        Dataset instance
    """
    base = resolve_base_dataset(dataset_name)
    if base not in DATASET_REGISTRY:
        available = ", ".join(sorted(set(DATASET_REGISTRY.keys())))
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    return DATASET_REGISTRY[base](data_root)
