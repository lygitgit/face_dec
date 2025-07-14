from ultralytics import SAM
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

class SAMDetector:
    def __init__(self, weight_path='./weights/segment/sam_b.pt'):
        # 加载 SAM 模型
        self.model = SAM(weight_path)

    def segment_target(self, image, box=None):
        """
        使用框或文本进行分割（可以组合）
        box: [x1, y1, x2, y2]
        text: str 类型，例如 'face'
        """
        kwargs = {}
        if box is not None:
            kwargs['bboxes'] = [box]

        results = self.model(image, **kwargs)
        if results and results[0].masks is not None:
            return results[0].masks.data[0].cpu().numpy()
        else:
            raise ValueError("未能获得有效的分割结果")

    def visualize_mask(self, image, mask, alpha=0.5):
        """
        将 mask 可视化叠加在原图上
        """
        image_np = np.array(image.convert("RGB"))
        mask_colored = np.zeros_like(image_np)
        mask_colored[mask > 0] = [0, 255, 0]  # 绿色

        blended = cv2.addWeighted(image_np, 1 - alpha, mask_colored, alpha, 0)
        plt.imshow(blended)
        plt.axis('off')
        plt.show()

# 示例使用
if __name__ == '__main__':
    detector = SAMDetector()

    # 加载图像
    img_path = './test/bus.jpg'
    image = Image.open(img_path)

    # 简化版的SAM，只有框提示
    box = [100, 400, 160, 475]
    mask = detector.segment_target(image, box=box)

    # 显示结果
    detector.visualize_mask(image, mask)
