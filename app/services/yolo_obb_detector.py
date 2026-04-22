"""
YOLO OBB 辅助检测器模块

基于 Ultralytics YOLO 的旋转边界框检测器
用于辅助 MIT DBNet 检测器，提高文本检测率

模型下载：
- ysgyolo_yolo26_2.0.pt: https://www.modelscope.cn/models/hgmzhn/manga-translator-ui/resolve/master/ysgyolo_yolo26_2.0.pt
"""

import os
from typing import List, Tuple, Optional, Any

import cv2
import numpy as np
import torch
from threading import Lock

from app.core.logger import logger
from app.core.paths import MODELS_DIR

# ============ 模型下载 URL ============
MODEL_URLS = {
    "yolo_obb": [
        "https://www.modelscope.cn/models/hgmzhn/manga-translator-ui/resolve/master/ysgyolo_yolo26_2.0.pt",
        "https://github.com/dmMazeBall/manga-image-translator/releases/download/yolo/ysgyolo_yolo26_2.0.pt",
    ]
}

# 模型缓存
_MODEL_LOCK = Lock()
_YOLO_MODEL = None


def _download_file(url: str, path: str) -> None:
    """下载文件"""
    import requests
    from tqdm import tqdm
    
    logger.info(f"下载 {url} -> {path}")
    
    response = requests.get(url, stream=True, allow_redirects=True)
    total = int(response.headers.get("content-length", 0))
    
    with open(path, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(path)) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def _ensure_model() -> str:
    """确保 YOLO OBB 模型存在"""
    model_dir = MODELS_DIR / "detection"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "ysgyolo_yolo26_2.0.pt"
    
    if not model_path.exists():
        # 尝试所有 URL
        for url in MODEL_URLS["yolo_obb"]:
            try:
                _download_file(url, str(model_path))
                break
            except Exception as e:
                logger.warning(f"下载失败 {url}: {e}")
                continue
        
        if not model_path.exists():
            raise RuntimeError("无法下载 YOLO OBB 模型，请手动下载到: " + str(model_path))
    
    return str(model_path)


# ============ 类别映射 ============
DEFAULT_CLASS_ID_TO_LABEL = {
    0: "balloon",      # 气泡
    1: "qipao",        # 旗帜
    2: "fangkuai",     # 方块
    3: "changfangtiao", # 长方条
    4: "kuangwai",     # 框外
    5: "other",        # 其他
}


class Quadrilateral:
    """四边形类，用于表示检测框"""
    
    def __init__(self, pts: np.ndarray, label: str, score: float):
        self.pts = pts.astype(np.float32)  # (4, 2) 四个角点
        self.label = label
        self.score = score
        self.det_label = label
        self.yolo_label = label
        self.is_yolo_box = True
    
    @property
    def aabb(self) -> Tuple[float, float, float, float]:
        """返回轴对齐边界框 (min_x, max_x, min_y, max_y)"""
        return (
            float(np.min(self.pts[:, 0])),
            float(np.max(self.pts[:, 0])),
            float(np.min(self.pts[:, 1])),
            float(np.max(self.pts[:, 1])),
        )
    
    @property
    def area(self) -> float:
        """计算边界框面积"""
        min_x, max_x, min_y, max_y = self.aabb
        return (max_x - min_x) * (max_y - min_y)
    
    def to_bbox(self) -> np.ndarray:
        """转换为 [x1, y1, x2, y2] 格式"""
        min_x, max_x, min_y, max_y = self.aabb
        return np.array([min_x, min_y, max_x, max_y], dtype=np.int64)


class YOLOOBBDetector:
    """YOLO OBB 辅助检测器"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.class_id_to_label = dict(DEFAULT_CLASS_ID_TO_LABEL)
        self.input_size = 1600
    
    def _resolve_device(self, device: str) -> torch.device:
        """解析设备"""
        requested = (device or "cpu").lower()
        if requested.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(device)
            logger.warning("YOLO OBB: 请求 CUDA，但当前不可用，回退到 CPU")
        elif requested.startswith("mps"):
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
            logger.warning("YOLO OBB: 请求 MPS，但当前不可用，回退到 CPU")
        return torch.device("cpu")
    
    def _load_model(self) -> None:
        """加载模型"""
        global _YOLO_MODEL
        
        with _MODEL_LOCK:
            if _YOLO_MODEL is not None:
                self.model = _YOLO_MODEL
                return
            
            model_path = _ensure_model()
            
            try:
                from ultralytics import YOLO
            except ImportError:
                raise ImportError("请安装 ultralytics: pip install ultralytics")
            
            self.torch_device = self._resolve_device(self.device)
            
            self.model = YOLO(model_path, task="obb")
            self.model.to(str(self.torch_device))
            
            _YOLO_MODEL = self.model
            logger.info(f"YOLO OBB 模型加载成功，设备: {self.torch_device}")
    
    def _to_numpy(self, data: Any) -> np.ndarray:
        """转换为 numpy 数组"""
        if isinstance(data, np.ndarray):
            return data
        if hasattr(data, "detach"):
            data = data.detach()
        if hasattr(data, "cpu"):
            data = data.cpu()
        if hasattr(data, "numpy"):
            return data.numpy()
        return np.asarray(data)
    
    def _extract_prediction_arrays(
        self,
        raw_result: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """从 YOLO 结果中提取预测数组"""
        if raw_result is None:
            return (
                np.empty((0, 4, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )
        
        obb = getattr(raw_result, "obb", None)
        if obb is not None:
            boxes_corners = self._to_numpy(getattr(obb, "xyxyxyxy", np.empty((0, 4, 2), dtype=np.float32)))
            scores = self._to_numpy(getattr(obb, "conf", np.empty((0,), dtype=np.float32)))
            class_ids = self._to_numpy(getattr(obb, "cls", np.empty((0,), dtype=np.int32)))
        else:
            boxes = getattr(raw_result, "boxes", None)
            if boxes is None:
                return (
                    np.empty((0, 4, 2), dtype=np.float32),
                    np.empty((0,), dtype=np.float32),
                    np.empty((0,), dtype=np.int32),
                )
            boxes_xyxy = self._to_numpy(getattr(boxes, "xyxy", np.empty((0, 4), dtype=np.float32)))
            if boxes_xyxy.ndim == 1:
                boxes_xyxy = boxes_xyxy.reshape(-1, 4)
            if boxes_xyxy.size == 0:
                return (
                    np.empty((0, 4, 2), dtype=np.float32),
                    np.empty((0,), dtype=np.float32),
                    np.empty((0,), dtype=np.int32),
                )
            boxes_corners = self._xyxy_to_xyxyxyxy(boxes_xyxy.astype(np.float32))
            scores = self._to_numpy(getattr(boxes, "conf", np.empty((len(boxes_xyxy),), dtype=np.float32)))
            class_ids = self._to_numpy(getattr(boxes, "cls", np.empty((len(boxes_xyxy),), dtype=np.int32)))
        
        boxes_corners = np.asarray(boxes_corners, dtype=np.float32)
        if boxes_corners.ndim == 2 and boxes_corners.shape[1] == 8:
            boxes_corners = boxes_corners.reshape(-1, 4, 2)
        elif boxes_corners.ndim != 3 or boxes_corners.shape[1:] != (4, 2):
            return (
                np.empty((0, 4, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )
        
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        class_ids = np.asarray(class_ids, dtype=np.int32).reshape(-1)
        
        if len(boxes_corners) == 0 or len(scores) != len(boxes_corners) or len(class_ids) != len(boxes_corners):
            return (
                np.empty((0, 4, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )
        
        # 过滤无效类别
        valid_class_ids = np.array(list(self.class_id_to_label.keys()), dtype=np.int32)
        valid_cls_mask = np.isin(class_ids, valid_class_ids)
        if not np.all(valid_cls_mask):
            drop_count = int(np.size(valid_cls_mask) - np.sum(valid_cls_mask))
            logger.debug(f"YOLO OBB 过滤无效类别: 移除 {drop_count} 个框")
            boxes_corners = boxes_corners[valid_cls_mask]
            scores = scores[valid_cls_mask]
            class_ids = class_ids[valid_cls_mask]
        
        return boxes_corners, scores, class_ids
    
    def _xyxy_to_xyxyxyxy(self, boxes: np.ndarray) -> np.ndarray:
        """将轴对齐框从 xyxy 转换为四角点"""
        x1 = boxes[:, 0:1]
        y1 = boxes[:, 1:2]
        x2 = boxes[:, 2:3]
        y2 = boxes[:, 3:4]
        pt1 = np.concatenate([x1, y1], axis=1)
        pt2 = np.concatenate([x2, y1], axis=1)
        pt3 = np.concatenate([x2, y2], axis=1)
        pt4 = np.concatenate([x1, y2], axis=1)
        return np.stack([pt1, pt2, pt3, pt4], axis=1)
    
    def _deduplicate_boxes(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        distance_threshold: float = 10.0,
        iou_threshold: float = 0.3,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """后处理去重：移除中心点距离很近或高度重叠的框"""
        if len(boxes) == 0:
            return boxes, scores, class_ids
        
        centers = np.mean(boxes, axis=1)
        keep = []
        sorted_indices = np.argsort(scores)[::-1]
        
        for i in sorted_indices:
            should_keep = True
            for j in keep:
                dist = np.linalg.norm(centers[i] - centers[j])
                if class_ids[i] == class_ids[j] and dist < distance_threshold:
                    should_keep = False
                    break
                
                # 计算 IoU
                box_i_min = np.min(boxes[i], axis=0)
                box_i_max = np.max(boxes[i], axis=0)
                box_j_min = np.min(boxes[j], axis=0)
                box_j_max = np.max(boxes[j], axis=0)
                
                inter_min = np.maximum(box_i_min, box_j_min)
                inter_max = np.minimum(box_i_max, box_j_max)
                inter_wh = np.maximum(0, inter_max - inter_min)
                inter_area = inter_wh[0] * inter_wh[1]
                
                box_i_area = (box_i_max[0] - box_i_min[0]) * (box_i_max[1] - box_i_min[1])
                box_j_area = (box_j_max[0] - box_j_min[0]) * (box_j_max[1] - box_j_min[1])
                union_area = box_i_area + box_j_area - inter_area
                
                if union_area > 0:
                    iou = inter_area / union_area
                    if iou > iou_threshold:
                        should_keep = False
                        break
            
            if should_keep:
                keep.append(i)
        
        return boxes[keep], scores[keep], class_ids[keep]
    
    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.5,
    ) -> Tuple[List[Quadrilateral], np.ndarray]:
        """
        检测文本区域
        
        Args:
            image: BGR 图像
            conf_threshold: 置信度阈值
            iou_threshold: IoU 阈值
            
        Returns:
            textlines: 检测到的文本框列表
            mask: 掩码（YOLO OBB 不输出掩码，返回空数组）
        """
        if self.model is None:
            self._load_model()
        
        if image is None or image.size == 0:
            return [], np.zeros((0, 0), dtype=np.uint8)
        
        h, w = image.shape[:2]
        
        # YOLO 推理
        results = self.model.predict(
            source=image,
            imgsz=int(self.input_size),
            conf=float(conf_threshold),
            iou=float(max(0.01, min(iou_threshold, 0.95))),
            device=str(self.torch_device),
            verbose=False,
        )
        
        if isinstance(results, list):
            raw_result = results[0] if results else None
        else:
            raw_result = next(iter(results), None)
        
        boxes_corners, scores, class_ids = self._extract_prediction_arrays(raw_result)
        
        # 去重
        if len(boxes_corners) > 0:
            boxes_corners, scores, class_ids = self._deduplicate_boxes(
                boxes_corners, scores, class_ids,
                distance_threshold=20.0,
                iou_threshold=0.5,
            )
        
        # 转换为 Quadrilateral 列表
        textlines = []
        for corners, score, class_id in zip(boxes_corners, scores, class_ids):
            # 裁剪到图像范围
            corners[:, 0] = np.clip(corners[:, 0], 0, w)
            corners[:, 1] = np.clip(corners[:, 1], 0, h)
            
            pts = corners.astype(np.int32)
            label = self.class_id_to_label.get(int(class_id), "other")
            
            quad = Quadrilateral(pts, label, float(score))
            textlines.append(quad)
        
        logger.debug(f"YOLO OBB 检测到 {len(textlines)} 个文本框")
        return textlines, np.zeros((h, w), dtype=np.uint8)


# ============ 全局接口 ============

_yolo_detector_instance: YOLOOBBDetector = None


def get_yolo_detector(device: str = None) -> YOLOOBBDetector:
    """获取 YOLO OBB 检测器实例"""
    global _yolo_detector_instance
    
    if _yolo_detector_instance is None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        _yolo_detector_instance = YOLOOBBDetector(device)
    
    return _yolo_detector_instance


def detect_text_regions_yolo_obb(
    image: np.ndarray,
    conf_threshold: float = 0.4,
    iou_threshold: float = 0.5,
    device: str = None,
) -> List[Quadrilateral]:
    """
    使用 YOLO OBB 检测文本区域
    
    Args:
        image: BGR 图像
        conf_threshold: 置信度阈值
        iou_threshold: IoU 阈值
        device: 设备（cuda/cpu）
        
    Returns:
        检测到的文本框列表
    """
    detector = get_yolo_detector(device)
    textlines, _ = detector.detect(image, conf_threshold, iou_threshold)
    return textlines
