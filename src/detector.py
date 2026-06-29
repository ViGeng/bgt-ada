import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights, RetinaNet_ResNet50_FPN_V2_Weights,
    SSD300_VGG16_Weights, fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2, ssd300_vgg16)

# Configuration
FORCE_DEVICE = None  # Force specific device: "cuda", "cuda:0", "cuda:1", "mps", "cpu", or None for auto-detect

# COCO vehicle classes (car, motorcycle, bus, truck)
# COCO class IDs: 3=car, 4=motorcycle, 6=bus, 8=truck
VEHICLE_CLASSES = {3, 4, 6, 8}

# COCO vehicle class names (for ultralytics name-based filtering)
VEHICLE_CLASS_NAMES = {"car", "motorcycle", "bus", "truck"}

# COCO class names
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_device():
    """Detect and return the best available device."""
    if FORCE_DEVICE is not None:
        return FORCE_DEVICE

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def _is_ultralytics_model(name: str) -> bool:
    """Return True if *name* refers to an ultralytics model (YOLO, RT-DETR, etc.)."""
    n = name.lower()
    return "yolo" in n or "rtdetr" in n


class Detector:
    def __init__(self, model_name, conf_threshold, vehicle_only=True):
        self.device = get_device()
        self.conf_threshold = conf_threshold
        self.model_name = model_name
        self.vehicle_only = vehicle_only
        self._backend = None  # "torchvision" or "ultralytics"

        # Load model based on name
        name_lower = model_name.lower()

        if _is_ultralytics_model(name_lower):
            self._init_ultralytics(model_name)
        elif "mobilenet" in name_lower and "320" in name_lower:
            weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
            self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
            self._init_torchvision(weights)
        elif "mobilenet" in name_lower:
            weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
            self._init_torchvision(weights)
        elif "retinanet" in name_lower:
            weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = retinanet_resnet50_fpn_v2(weights=weights)
            self._init_torchvision(weights)
        elif "ssd" in name_lower:
            weights = SSD300_VGG16_Weights.DEFAULT
            self.model = ssd300_vgg16(weights=weights)
            self._init_torchvision(weights)
        elif "resnet50" in name_lower or "fasterrcnn" in name_lower:
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
            self._init_torchvision(weights)
        else:
            raise ValueError(
                f"Unknown model: {model_name}. Supported: mobilenet, "
                f"mobilenet_320, retinanet, ssd300_vgg16, fasterrcnn_resnet50, "
                f"yolo*/rtdetr* (ultralytics — e.g. yolo11n, yolov8x, rtdetr-l, rtdetr-x)")

        from . import log
        log.kv_group([
            ("Device", self.device.upper()),
            ("Model", f"{model_name} (backend: {self._backend})"),
        ], indent=6)

    # ---- backend init helpers -------------------------------------------

    def _init_torchvision(self, weights):
        """Finalize a torchvision detection model."""
        self._backend = "torchvision"
        self.model.to(self.device)
        self.model.eval()
        self.transform = weights.transforms()

    def _init_ultralytics(self, model_name: str):
        """Load an ultralytics model (YOLO, RT-DETR, etc.)."""
        try:
            from ultralytics import YOLO, RTDETR
        except ImportError:
            raise ImportError(
                f"Model '{model_name}' requires the ultralytics package. "
                "Install it with: pip install ultralytics")

        self._backend = "ultralytics"
        # Append .pt if not already present so ultralytics can resolve it
        weight_name = model_name if model_name.endswith(".pt") else f"{model_name}.pt"
        if "rtdetr" in model_name.lower():
            self.model = RTDETR(weight_name)
        else:
            self.model = YOLO(weight_name)
        self.model.to(self.device)
        self.transform = None  # ultralytics handles its own preprocessing

    # ---- detection API --------------------------------------------------

    def detect(self, image_path):
        """Run detection on a single image, return filtered results."""
        if self._backend == "ultralytics":
            return self._detect_ultralytics([image_path])[0]
        return self._detect_torchvision(image_path)

    def detect_batch(self, image_paths):
        """Run detection on a batch of images, return filtered results for each image."""
        if self._backend == "ultralytics":
            return self._detect_ultralytics(image_paths)
        return self._detect_batch_torchvision(image_paths)

    # ---- torchvision backend --------------------------------------------

    def _detect_torchvision(self, image_path):
        """Run detection on a single image with torchvision."""
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).to(self.device)

        with torch.no_grad():
            predictions = self.model([img_tensor])[0]

        return self._filter_torchvision(predictions)

    def _detect_batch_torchvision(self, image_paths):
        """Run detection on a batch of images with torchvision."""
        img_tensors = []
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img).to(self.device)
            img_tensors.append(img_tensor)

        with torch.no_grad():
            predictions = self.model(img_tensors)

        return [self._filter_torchvision(pred) for pred in predictions]

    def _filter_torchvision(self, predictions):
        """Filter torchvision predictions to (left, top, w, h, conf, class) tuples."""
        detections = []
        boxes = predictions['boxes'].cpu()
        labels = predictions['labels'].cpu()
        scores = predictions['scores'].cpu()

        for box, label, score in zip(boxes, labels, scores):
            label_id = int(label)
            conf = float(score)

            if conf >= self.conf_threshold and (not self.vehicle_only or label_id in VEHICLE_CLASSES):
                x1, y1, x2, y2 = box.tolist()
                left, top = x1, y1
                width, height = x2 - x1, y2 - y1
                class_name = COCO_CLASSES[label_id]
                detections.append((left, top, width, height, conf, class_name))

        return detections

    # ---- ultralytics backend --------------------------------------------

    def _detect_ultralytics(self, image_paths):
        """Run ultralytics detection on one or more images.

        ultralytics natively accepts a list of paths and runs batched
        inference, so we use that directly.
        """
        results = self.model(image_paths, verbose=False, conf=self.conf_threshold)

        batch_detections = []
        for result in results:
            detections = []
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                batch_detections.append(detections)
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                class_name = result.names.get(cls_id, "unknown")

                if self.vehicle_only and class_name not in VEHICLE_CLASS_NAMES:
                    continue

                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                left, top = x1, y1
                width, height = x2 - x1, y2 - y1
                detections.append((left, top, width, height, conf, class_name))

            batch_detections.append(detections)

        return batch_detections
