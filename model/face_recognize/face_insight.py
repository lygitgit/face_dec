from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo

class FaceEmbedderInsightFace:
    def __init__(self, det_size=(640, 640), model_name='buffalo_l', device='cuda'):
        """
        使用 InsightFace 提取人脸特征
        :param det_size: 人脸检测图像大小
        :param model_name: 模型名称（如 buffalo_l）
        :param device: 设备 (cuda / cpu)
        """
        self.app = FaceAnalysis(name=model_name, root='weights/ReID/insightface', providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=det_size)
        self.device = device

    def preprocess(self, img: Image.Image) -> np.ndarray:
        return np.array(img.convert("RGB"))

    def extract(self, img: Image.Image, return_boxes: bool = False) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        自动人脸检测 + 特征提取
        :return: 特征向量 [N, 512] 和人脸框 [x1, y1, x2, y2]
        """
        img_np = self.preprocess(img)
        faces = self.app.get(img_np)

        if not faces:
            return np.zeros((0, 512)), []

        embeddings = np.array([f.normed_embedding for f in faces])
        boxes = [tuple(map(int, f.bbox)) for f in faces] if return_boxes else []

        return embeddings, boxes

    def extract_with_locations(self, img: Image.Image, known_face_locations: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        使用指定的人脸框提取特征（不调用内部人脸检测器）
        :param known_face_locations: [(top, left, bottom, right)]
        """
        img_np = self.preprocess(img)
        faces = []

        for (top, left, bottom, right) in known_face_locations:
            face = img_np[top:bottom, left:right]
            if face.shape[0] == 0 or face.shape[1] == 0:
                continue
            objs = self.app.get(face)
            if objs and len(objs) > 0:
                embedding = objs[0].embedding  # 512维特征
                
                faces.append(embedding)

        if not faces:
            return np.zeros((0, 512))

        return np.vstack(faces)

if __name__ == "__main__":
    from PIL import Image

    embedder = FaceEmbedderInsightFace()
    img = Image.open("/home/lenovo/New/UnionProject/test/bus.jpg")

    # 自动检测
    feats, boxes = embedder.extract(img, return_boxes=True)
    print("检测到人脸数:", len(feats))

    # 指定框提取
    feats2 = embedder.extract_with_locations(img, boxes)
    print("指定框提取特征 shape:", feats2.shape)