"""
MIT (manga-image-translator) 检测器模块

基于 manga-image-translator 的 DBNet 检测器实现
支持输出 bbox + raw_mask（像素级热力图）

模型下载：
- detect-20241225.ckpt: https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/detect-20241225.ckpt
"""

import os
import shutil
from typing import List, Tuple

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from threading import Lock
from torchvision.models import resnet34

from app.core.logger import logger
from app.core.paths import MODELS_DIR

# ============ 模型下载 URL ============
MODEL_URLS = {
    "detect": [
        "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/detect-20241225.ckpt",
        "https://www.modelscope.cn/models/hgmzhn/manga-translator-ui/resolve/master/detect-20241225.ckpt",
    ]
}

# 模型缓存
_MODEL_LOCK = Lock()
_DETECT_MODEL = None


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
    """确保检测模型存在"""
    model_dir = MODELS_DIR / "detection"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "detect-20241225.ckpt"
    
    # 检查根目录是否有模型文件（迁移旧文件）
    root_model = MODELS_DIR / "detect-20241225.ckpt"
    if root_model.exists() and not model_path.exists():
        shutil.move(str(root_model), str(model_path))
    
    if not model_path.exists():
        # 尝试所有 URL
        for url in MODEL_URLS["detect"]:
            try:
                _download_file(url, str(model_path))
                break
            except Exception as e:
                logger.warning(f"下载失败 {url}: {e}")
                continue
        
        if not model_path.exists():
            raise RuntimeError("无法下载检测模型，请手动下载到: " + str(model_path))
    
    return str(model_path)


# ============ DBHead 模块 ============

class DBHead(nn.Module):
    def __init__(self, in_channels, out_channels, k=50):
        super().__init__()
        self.k = k
        self.binarize = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 4, 2, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 4, 2, 1),
        )
        self.binarize.apply(self.weights_init)

        self.thresh = self._init_thresh(in_channels)
        self.thresh.apply(self.weights_init)

    def forward(self, x):
        shrink_maps = self.binarize(x)
        threshold_maps = self.thresh(x)
        if self.training:
            binary_maps = self.step_function(shrink_maps.sigmoid(), threshold_maps)
            y = torch.cat((shrink_maps, threshold_maps, binary_maps), dim=1)
        else:
            y = torch.cat((shrink_maps, threshold_maps), dim=1)
        return y

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels, serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels // 4, smooth=smooth, bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self, in_channels, out_channels, smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=True))
            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


# ============ DBNet 模型架构 ============

class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, stride=1, planes=256):
        super(double_conv, self).__init__()
        self.planes = planes
        self.down = None
        if stride > 1:
            self.down = nn.AvgPool2d(2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.down is not None:
            x = self.down(x)
        x = self.conv(x)
        return x


class double_conv_up(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, planes=256):
        super(double_conv_up, self).__init__()
        self.planes = planes
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class TextDetection(nn.Module):
    """DBNet 文本检测模型（ResNet34 backbone）"""
    
    def __init__(self, pretrained=None):
        super(TextDetection, self).__init__()
        self.backbone = resnet34(weights=None)
        
        self.conv_db = DBHead(64, 0)

        self.conv_mask = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.down_conv1 = double_conv(0, 512, 512, 2)
        self.down_conv2 = double_conv(0, 512, 512, 2)
        self.down_conv3 = double_conv(0, 512, 512, 2)

        self.upconv1 = double_conv_up(0, 512, 256)
        self.upconv2 = double_conv_up(256, 512, 256)
        self.upconv3 = double_conv_up(256, 512, 256)
        self.upconv4 = double_conv_up(256, 512, 256, planes=128)
        self.upconv5 = double_conv_up(256, 256, 128, planes=64)
        self.upconv6 = double_conv_up(128, 128, 64, planes=32)
        self.upconv7 = double_conv_up(64, 64, 64, planes=16)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)  # 64@384

        h4 = self.backbone.layer1(x)   # 64@384
        h8 = self.backbone.layer2(h4)  # 128@192
        h16 = self.backbone.layer3(h8) # 256@96
        h32 = self.backbone.layer4(h16) # 512@48
        h64 = self.down_conv1(h32)      # 512@24
        h128 = self.down_conv2(h64)     # 512@12
        h256 = self.down_conv3(h128)    # 512@6

        up256 = self.upconv1(h256)  # 128@12
        up128 = self.upconv2(torch.cat([up256, h128], dim=1))  # 64@24
        up64 = self.upconv3(torch.cat([up128, h64], dim=1))    # 128@48
        up32 = self.upconv4(torch.cat([up64, h32], dim=1))     # 64@96
        up16 = self.upconv5(torch.cat([up32, h16], dim=1))     # 128@192
        up8 = self.upconv6(torch.cat([up16, h8], dim=1))       # 64@384
        up4 = self.upconv7(torch.cat([up8, h4], dim=1))        # 64@768

        return self.conv_db(up8), self.conv_mask(up4)


# ============ 后处理工具（使用 MIT 原始实现）============

class SegDetectorRepresenter:
    """DBNet 后处理器（MIT 原始实现）"""
    
    def __init__(
        self,
        thresh: float = 0.5,     # MIT 默认 0.5
        box_thresh: float = 0.7,  # MIT 默认 0.7
        max_candidates: int = 1000,
        unclip_ratio: float = 2.3,  # MIT 默认 2.3
        min_size: int = 1,  # 降低最小尺寸阈值（原为3），保留更多小文本
    ):
        self.min_size = min_size
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    def __call__(self, batch, pred, is_output_polygon=False):
        '''
        batch: a dict with 'shape' key
        pred: binary text region segmentation map, with shape (N, C, H, W)
        '''
        pred = pred[:, 0, :, :]
        segmentation = self.binarize(pred)
        boxes_batch = []
        scores_batch = []
        batch_size = pred.shape[0]
        for batch_index in range(batch_size):
            # batch['shape'] 是 [(h, w), ...] 列表
            shape = batch['shape'][batch_index]
            height, width = int(shape[0]), int(shape[1])
            boxes, scores = self.boxes_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch

    def binarize(self, pred):
        return (pred > self.thresh).astype(np.float32)

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (H, W), whose values are binarized as {0, 1}
        '''
        bitmap = _bitmap
        height, width = bitmap.shape
        logger.debug(f'boxes_from_bitmap: pred.shape={pred.shape}, bitmap.shape={bitmap.shape}, dest=({dest_width}, {dest_height})')
        try:
            contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            return np.zeros((0, 4, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
        
        logger.debug(f'boxes_from_bitmap: 找到 {len(contours)} 个轮廓')
        boxes_list = []
        scores_list = []

        num_contours = int(min(len(contours), self.max_candidates))
        # Silence leftover contour-level debug prints from the original port.
        def print(*args, **kwargs):
            return None
        print(f'[DEBUG] 开始处理 {num_contours} 个轮廓', flush=True)
        for index in range(num_contours):
            print(f'[DEBUG] 处理轮廓 {index}', flush=True)
            try:
                contour = contours[index]
                print(f'[DEBUG] 轮廓 {index}: shape={contour.shape}', flush=True)
                # 确保 contour 形状正确
                if len(contour.shape) == 3:
                    contour = contour.squeeze(1)
                if len(contour.shape) != 2 or contour.shape[0] < 4:
                    print(f'[DEBUG] 轮廓 {index}: 跳过（形状不正确）', flush=True)
                    continue
                    
                try:
                    points, sside = self.get_mini_boxes(contour)
                except Exception as e:
                    print(f'[DEBUG] 轮廓 {index}: get_mini_boxes 失败 - {e}', flush=True)
                    continue
                    
                if sside < self.min_size:
                    print(f'[DEBUG] 轮廓 {index}: 跳过（min_size={sside:.1f}）', flush=True)
                    continue
                points = np.array(points)
                
                try:
                    score = self.box_score_fast(pred, contour)
                except Exception as e:
                    print(f'[DEBUG] 轮廓 {index}: box_score_fast 失败 - {e}', flush=True)
                    continue
                    
                if self.box_thresh > score:
                    print(f'[DEBUG] 轮廓 {index}: 跳过（score={score:.3f}）', flush=True)
                    continue

                try:
                    box = self.unclip(points, unclip_ratio=self.unclip_ratio).reshape(-1, 1, 2)
                    box, sside = self.get_mini_boxes(box)
                except Exception as e:
                    print(f'[DEBUG] 轮廓 {index}: unclip 失败 - {e}', flush=True)
                    continue
                    
                if sside < self.min_size + 2:
                    print(f'[DEBUG] 轮廓 {index}: 跳过（unclip后 min_size={sside:.1f}）', flush=True)
                    continue
                
                print(f'[DEBUG] 轮廓 {index}: 保留（score={score:.3f}）', flush=True)
                box = np.array(box)

                # 确保 dest_width 和 dest_height 是整数
                dest_width_int = int(dest_width)
                dest_height_int = int(dest_height)
                box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width_int), 0, dest_width_int)
                box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height_int), 0, dest_height_int)
                startidx = box.sum(axis=1).argmin()
                box = np.roll(box, 4-startidx, 0)
                boxes_list.append(box.astype(np.int64))
                scores_list.append(score)
                print(f'[DEBUG] 轮廓 {index}: 已添加，当前共 {len(boxes_list)} 个框', flush=True)
            except Exception as e:
                print(f'[ERROR] 轮廓 {index}: 处理失败 - {e}', flush=True)
                logger.debug("Contour processing failed", exc_info=True)
                continue
            print(f'[DEBUG] 轮廓 {index}: 处理完成', flush=True)
        
        print(f'[DEBUG] 循环结束，最终保留 {len(boxes_list)} 个框', flush=True)

        if len(boxes_list) == 0:
            return np.zeros((0, 4, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
        
        # 确保返回值形状正确
        boxes_array = np.array(boxes_list, dtype=np.int64)
        scores_array = np.array(scores_list, dtype=np.float32)
        
        return boxes_array, scores_array

    def unclip(self, box, unclip_ratio=1.8):
        import pyclipper
        from shapely.geometry import Polygon
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = int(np.clip(np.floor(box[:, 0].min()), 0, w - 1))
        xmax = int(np.clip(np.ceil(box[:, 0].max()), 0, w - 1))
        ymin = int(np.clip(np.floor(box[:, 1].min()), 0, h - 1))
        ymax = int(np.clip(np.ceil(box[:, 1].max()), 0, h - 1))

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def resize_aspect_ratio(
    img: np.ndarray,
    square_size: int,
    interpolation: int,
    mag_ratio: float = 1.0
):
    """保持宽高比缩放图像，padding 到 256 的倍数"""
    height, width, channel = img.shape

    # 计算目标尺寸
    target_size = mag_ratio * square_size
    ratio = target_size / max(height, width)    

    target_h, target_w = int(round(height * ratio)), int(round(width * ratio))
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # 必须是 256 的倍数（DBNet 要求）
    MULT = 256

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    pad_h = 0
    pad_w = 0
    if target_h % MULT != 0:
        pad_h = int(MULT - target_h % MULT)
        target_h32 = target_h + pad_h
    if target_w % MULT != 0:
        pad_w = int(MULT - target_w % MULT)
        target_w32 = target_w + pad_w
    
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.uint8)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w/2), int(target_h/2))

    return resized, ratio, size_heatmap, pad_w, pad_h


def adjust_result_coordinates(coords: np.ndarray, ratio_w: float, ratio_h: float, ratio_net: float = 1):
    """调整坐标到原图尺寸"""
    if len(coords.shape) == 3:
        coords[:, :, 0] = coords[:, :, 0] * ratio_w * ratio_net
        coords[:, :, 1] = coords[:, :, 1] * ratio_h * ratio_net
    else:
        coords[:, 0] = coords[:, 0] * ratio_w * ratio_net
        coords[:, 1] = coords[:, 1] * ratio_h * ratio_net
    
    return coords


# ============ 检测器类 ============

class MITDetector:
    """MIT DBNet 检测器"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        global _DETECT_MODEL
        
        with _MODEL_LOCK:
            if _DETECT_MODEL is not None:
                self.model = _DETECT_MODEL
                return
            
            model_path = _ensure_model()
            
            # 创建模型
            self.model = TextDetection()
            
            # 加载权重
            sd = torch.load(model_path, map_location='cpu', weights_only=False)
            if 'model' in sd:
                sd = sd['model']
            
            self.model.load_state_dict(sd, strict=True)
            self.model.eval()
            
            if self.device != "cpu":
                self.model = self.model.to(self.device)
            
            _DETECT_MODEL = self.model
            logger.info(f"MIT 检测模型加载成功，设备: {self.device}")
    
    def detect(
        self, 
        image: np.ndarray, 
        detect_size: int = 2048,  # MIT 默认使用 2048
        text_threshold: float = 0.5,  # MIT 默认 0.5
        box_threshold: float = 0.7,   # MIT 默认 0.7
        unclip_ratio: float = 2.3,    # MIT 默认 2.3
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        检测文本区域
        
        Args:
            image: BGR 图像
            detect_size: 检测尺寸（默认1536，MIT使用较大尺寸）
            text_threshold: 文本阈值（越低检测越多）
            box_threshold: 框置信度阈值
            unclip_ratio: 扩展比例
            
        Returns:
            bboxes: 文本区域边界框列表 [[x1,y1,x2,y2], ...]
            raw_mask: 原始掩码（热力图）
        """
        # 预处理：双边滤波
        img_resized, ratio, size_heatmap, pad_w, pad_h = resize_aspect_ratio(
            cv2.bilateralFilter(image, 17, 80, 80), 
            detect_size, 
            cv2.INTER_LINEAR, 
            mag_ratio=1
        )
        
        img_resized_h, img_resized_w = img_resized.shape[:2]
        ratio_w = ratio_h = 1 / ratio
        
        # 归一化: HWC -> NCHW
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float()
        img_tensor = (img_tensor / 127.5 - 1.0)
        
        if self.device != "cpu":
            img_tensor = img_tensor.to(self.device)
        
        # 推理
        with torch.no_grad():
            db, mask = self.model(img_tensor)
            db = torch.sigmoid(db).cpu().numpy()
            mask = mask.cpu().numpy()
        
        # 后处理（使用 MIT 原始后处理器）
        # 添加日志确认参数
        logger.debug(f'后处理参数: thresh={text_threshold}, box_thresh={box_threshold}, unclip_ratio={unclip_ratio}')
        logger.debug(f'db shape: {db.shape}, range: min={db.min():.4f}, max={db.max():.4f}')
        logger.debug(f'batch shape: {img_resized.shape[:2]}')
        
        # 使用命名参数避免参数顺序混淆
        seg_rep = SegDetectorRepresenter(
            thresh=text_threshold,
            box_thresh=box_threshold,
            unclip_ratio=unclip_ratio,
            min_size=1,  # 降低最小尺寸阈值，保留更多小文本
        )
        batch_info = {'shape': [img_resized.shape[:2]]}
        boxes_batch, scores_batch = seg_rep(batch_info, db)
        boxes = boxes_batch[0]  # 取第一个批次的结果
        scores = scores_batch[0]
        logger.debug(f'SegDetectorRepresenter 返回: {len(boxes)} 个框')
        
        # 过滤有效框
        if isinstance(boxes, np.ndarray) and boxes.size > 0:
            idx = boxes.reshape(boxes.shape[0], -1).sum(axis=1) > 0
            boxes = boxes[idx]
            
            # 调整坐标到原图（需要考虑缩放比例）
            boxes = boxes.astype(np.float64)
            boxes[:, :, 0] = boxes[:, :, 0] * float(ratio_w)
            boxes[:, :, 1] = boxes[:, :, 1] * float(ratio_h)
            boxes = np.round(boxes).astype(np.int64)
        else:
            boxes = np.zeros((0, 4, 2), dtype=np.int64)
        
        # 转换为 bbox 格式
        bboxes = []
        for box in boxes:
            x1, y1 = int(box[:, 0].min()), int(box[:, 1].min())
            x2, y2 = int(box[:, 0].max()), int(box[:, 1].max())
            bboxes.append(np.array([x1, y1, x2, y2]))
        
        # 处理 mask
        # mask 输出形状: (N, 1, H, W)，需要正确处理
        if len(mask.shape) == 4:
            mask_single = mask[0, 0, :, :]
        elif len(mask.shape) == 3:
            mask_single = mask[0, :, :]
        elif len(mask.shape) == 2:
            mask_single = mask
        else:
            mask_single = mask
        
        # 缩放回原始尺寸（mask 输出是输入的 1/2）
        h_mask, w_mask = int(mask_single.shape[0]), int(mask_single.shape[1])
        mask_resized = cv2.resize(mask_single, (w_mask * 2, h_mask * 2), interpolation=cv2.INTER_LINEAR)
        
        # 去除 padding（注意：pad_h 和 pad_w 需要分别处理，不是 elif）
        pad_h = int(pad_h)
        pad_w = int(pad_w)
        if pad_h > 0 and pad_h < mask_resized.shape[0]:
            mask_resized = mask_resized[:-pad_h, :]
        if pad_w > 0 and pad_w < mask_resized.shape[1]:
            mask_resized = mask_resized[:, :-pad_w]
        
        # 缩放到原图尺寸
        original_h, original_w = int(image.shape[0]), int(image.shape[1])
        if mask_resized.shape[0] != original_h or mask_resized.shape[1] != original_w:
            mask_resized = cv2.resize(mask_resized, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        # 归一化到 0-255
        raw_mask = np.clip(mask_resized * 255, 0, 255).astype(np.uint8)
        
        logger.debug(f"检测到 {len(bboxes)} 个文本区域")
        return bboxes, raw_mask


# ============ 全局接口 ============

_detector_instance: MITDetector = None


def get_detector(device: str = None) -> MITDetector:
    """获取检测器实例"""
    global _detector_instance
    
    if _detector_instance is None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        _detector_instance = MITDetector(device)
    
    return _detector_instance


def detect_text_regions_with_mask(
    image: np.ndarray,
    detect_size: int = 2048,  # MIT 默认使用 2048
    text_threshold: float = 0.5,  # MIT 默认 0.5
    box_threshold: float = 0.7,   # MIT 默认 0.7
    unclip_ratio: float = 2.3,    # MIT 默认 2.3
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    检测文本区域（带掩码输出）
    
    Args:
        image: BGR 图像
        detect_size: 检测尺寸（默认2048）
        text_threshold: 文本阈值（默认0.5）
        box_threshold: 框置信度阈值（默认0.7）
        unclip_ratio: 扩展比例（默认2.3）
        
    Returns:
        bboxes: 文本区域边界框列表
        raw_mask: 原始掩码（用于 inpainting）
    """
    detector = get_detector()
    return detector.detect(image, detect_size, text_threshold, box_threshold, unclip_ratio)


# ============ 混合检测（MIT + YOLO OBB）============

def _get_box_aabb(box_pts: np.ndarray) -> Tuple[float, float, float, float]:
    """获取边界框的 AABB (min_x, max_x, min_y, max_y)"""
    return (
        float(np.min(box_pts[:, 0])),
        float(np.max(box_pts[:, 0])),
        float(np.min(box_pts[:, 1])),
        float(np.max(box_pts[:, 1])),
    )


def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """计算两个边界框的 IoU"""
    a_min_x, a_max_x, a_min_y, a_max_y = _get_box_aabb(box_a)
    b_min_x, b_max_x, b_min_y, b_max_y = _get_box_aabb(box_b)
    
    # 检查是否有重叠
    if a_max_x < b_min_x or a_min_x > b_max_x or a_max_y < b_min_y or a_min_y > b_max_y:
        return 0.0
    
    # 计算交集
    inter_min_x = max(a_min_x, b_min_x)
    inter_max_x = min(a_max_x, b_max_x)
    inter_min_y = max(a_min_y, b_min_y)
    inter_max_y = min(a_max_y, b_max_y)
    inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
    
    # 计算并集
    a_area = (a_max_x - a_min_x) * (a_max_y - a_min_y)
    b_area = (b_max_x - b_min_x) * (b_max_y - b_min_y)
    union_area = a_area + b_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def _contains_box(outer_pts: np.ndarray, inner_pts: np.ndarray, eps: float = 2.0) -> bool:
    """检查 outer 是否完全包含 inner（基于 AABB）"""
    o_min_x, o_max_x, o_min_y, o_max_y = _get_box_aabb(outer_pts)
    i_min_x, i_max_x, i_min_y, i_max_y = _get_box_aabb(inner_pts)
    
    return (o_min_x <= i_min_x + eps and o_max_x >= i_max_x - eps and
            o_min_y <= i_min_y + eps and o_max_y >= i_max_y - eps)


def merge_detection_boxes(
    mit_bboxes: List[np.ndarray],
    yolo_boxes: List,  # List[Quadrilateral] from yolo_obb_detector
    overlap_threshold: float = 0.3,
) -> List[np.ndarray]:
    """
    合并 MIT 和 YOLO OBB 检测器的框（完全参考 MTU 实现）
    
    合并策略（MTU 原始逻辑）：
    1. YOLO 框完全包含 MIT 框 + 面积条件（>= 2倍） → 替换
    2. YOLO 框与 MIT 框重叠 >= threshold → 删除 YOLO 框
    3. 其他 YOLO 框直接添加
    
    Args:
        mit_bboxes: MIT 检测器返回的边界框列表 [[x1,y1,x2,y2], ...]
        yolo_boxes: YOLO OBB 检测器返回的 Quadrilateral 列表
        overlap_threshold: 重叠率阈值
        
    Returns:
        合并后的边界框列表
    """
    if len(mit_bboxes) == 0 and len(yolo_boxes) == 0:
        return []
    
    if len(yolo_boxes) == 0:
        return mit_bboxes
    
    if len(mit_bboxes) == 0:
        return [box.to_bbox() for box in yolo_boxes if box.label != "other"]
    
    # 标记要移除的 MIT 框索引
    main_boxes_to_remove = set()
    # 标记要移除的 YOLO 框索引
    yolo_boxes_to_remove = set()
    # 要添加的 YOLO 框索引
    yolo_boxes_to_add = set()
    
    # 第一步：处理替换逻辑
    for yolo_idx, yolo_box in enumerate(yolo_boxes):
        yolo_label = yolo_box.label
        if yolo_label == "other":
            continue
        
        yolo_pts = yolo_box.pts
        yolo_min_x = np.min(yolo_pts[:, 0])
        yolo_max_x = np.max(yolo_pts[:, 0])
        yolo_min_y = np.min(yolo_pts[:, 1])
        yolo_max_y = np.max(yolo_pts[:, 1])
        yolo_area = (yolo_max_x - yolo_min_x) * (yolo_max_y - yolo_min_y)
        
        # 计算被这个 YOLO 框包含的 MIT 框
        replaced_mit_indices = set()
        contained_mit_area = 0.0
        # 与其他未被包含的 MIT 框的最大重叠率
        max_overlap_with_others = 0.0
        
        for mit_idx, mit_bbox in enumerate(mit_bboxes):
            x1, y1, x2, y2 = mit_bbox
            mit_area = (x2 - x1) * (y2 - y1)
            
            # 计算 AABB 重叠
            inter_min_x = max(yolo_min_x, x1)
            inter_max_x = min(yolo_max_x, x2)
            inter_min_y = max(yolo_min_y, y1)
            inter_max_y = min(yolo_max_y, y2)
            
            if inter_min_x < inter_max_x and inter_min_y < inter_max_y:
                inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
                
                # 检查是否完全包含
                contains = (yolo_min_x <= x1 and yolo_max_x >= x2 and
                           yolo_min_y <= y1 and yolo_max_y >= y2)
                
                if contains:
                    replaced_mit_indices.add(mit_idx)
                    contained_mit_area += mit_area
                else:
                    # 计算重叠率
                    overlap_ratio = inter_area / min(yolo_area, mit_area)
                    max_overlap_with_others = max(max_overlap_with_others, overlap_ratio)
        
        # 判断是否可以替换
        if len(replaced_mit_indices) > 0:
            # MTU 原始条件：面积 >= 2倍
            if yolo_area >= contained_mit_area * 2.0:
                # 允许更高的重叠阈值
                adjusted_threshold = overlap_threshold + 0.1
                if max_overlap_with_others < adjusted_threshold:
                    # 执行替换
                    main_boxes_to_remove.update(replaced_mit_indices)
                    yolo_boxes_to_add.add(yolo_idx)
                else:
                    # 与其他框重叠过高，删除 YOLO 框
                    yolo_boxes_to_remove.add(yolo_idx)
            else:
                # 面积不够大，删除 YOLO 框
                yolo_boxes_to_remove.add(yolo_idx)
        else:
            # 没有完全包含任何 MIT 框
            if max_overlap_with_others >= overlap_threshold:
                # 重叠过高，删除
                yolo_boxes_to_remove.add(yolo_idx)
            # else: 不重叠，会在后面添加
    
    # 构建最终结果
    result = []
    
    # 添加未被移除的 MIT 框
    for idx, bbox in enumerate(mit_bboxes):
        if idx not in main_boxes_to_remove:
            result.append(bbox)
    
    # 添加 YOLO 框
    for idx, yolo_box in enumerate(yolo_boxes):
        if idx not in yolo_boxes_to_remove and yolo_box.label != "other":
            result.append(yolo_box.to_bbox())
    
    return result


def _deduplicate_bboxes(
    bboxes: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> List[np.ndarray]:
    """
    基于 IoU 对边界框进行去重
    
    Args:
        bboxes: 边界框列表 [[x1,y1,x2,y2], ...]
        iou_threshold: IoU 阈值，超过此值的框将被去重
        
    Returns:
        去重后的边界框列表
    """
    if len(bboxes) <= 1:
        return bboxes
    
    # 计算每个框的面积
    areas = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        areas.append((x2 - x1) * (y2 - y1))
    
    # 按面积降序排序（保留大面积的框）
    indices = sorted(range(len(bboxes)), key=lambda i: areas[i], reverse=True)
    
    keep = []
    for i in indices:
        should_keep = True
        bbox_i = bboxes[i]
        x1_i, y1_i, x2_i, y2_i = bbox_i
        area_i = areas[i]
        
        for j in keep:
            bbox_j = bboxes[j]
            x1_j, y1_j, x2_j, y2_j = bbox_j
            area_j = areas[j]
            
            # 计算 IoU
            inter_x1 = max(x1_i, x1_j)
            inter_y1 = max(y1_i, y1_j)
            inter_x2 = min(x2_i, x2_j)
            inter_y2 = min(y2_i, y2_j)
            
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                union_area = area_i + area_j - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                
                if iou > iou_threshold:
                    should_keep = False
                    break
        
        if should_keep:
            keep.append(i)
    
    return [bboxes[i] for i in sorted(keep)]


def detect_text_regions_hybrid(
    image: np.ndarray,
    detect_size: int = 2048,
    text_threshold: float = 0.3,
    box_threshold: float = 0.5,
    unclip_ratio: float = 2.3,
    use_yolo_obb: bool = True,
    yolo_obb_conf: float = 0.4,
    overlap_threshold: float = 0.1,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    混合检测：MIT DBNet + YOLO OBB
    
    Args:
        image: BGR 图像
        detect_size: MIT 检测尺寸
        text_threshold: MIT 文本阈值
        box_threshold: MIT 框阈值
        unclip_ratio: MIT 扩展比例
        use_yolo_obb: 是否启用 YOLO OBB 辅助检测
        yolo_obb_conf: YOLO OBB 置信度阈值
        overlap_threshold: 重叠率阈值
        
    Returns:
        bboxes: 合并后的边界框列表
        raw_mask: 原始掩码
    """
    # 1. MIT 检测
    mit_bboxes, raw_mask = detect_text_regions_with_mask(
        image, detect_size, text_threshold, box_threshold, unclip_ratio
    )
    
    logger.debug(f"MIT 检测到 {len(mit_bboxes)} 个文本区域")
    
    # 如果不启用 YOLO OBB，直接返回
    if not use_yolo_obb:
        return mit_bboxes, raw_mask
    
    # 2. YOLO OBB 检测
    try:
        from app.services.yolo_obb_detector import get_yolo_detector
        
        yolo_detector = get_yolo_detector()
        yolo_boxes, _ = yolo_detector.detect(image, conf_threshold=yolo_obb_conf)
        
        logger.debug(f"YOLO OBB 检测到 {len(yolo_boxes)} 个文本区域")
        
        # 3. 合并检测结果
        combined_bboxes = merge_detection_boxes(
            mit_bboxes, yolo_boxes, overlap_threshold
        )
        
        replaced_count = len(mit_bboxes) + len(yolo_boxes) - len(combined_bboxes)
        logger.debug(f"混合检测: MIT={len(mit_bboxes)}, YOLO={len(yolo_boxes)}, "
                    f"替换={replaced_count}, 总计={len(combined_bboxes)}")
        
        return combined_bboxes, raw_mask
        
    except Exception as e:
        logger.warning(f"YOLO OBB 检测失败，回退到 MIT 检测: {e}")
        return mit_bboxes, raw_mask
