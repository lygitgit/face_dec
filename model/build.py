from model import *

__all__ = ['build_detector', 'build_tracker', 'build_mesh_generater', 'build_face_ReID', 'build_pose_generator']

def build_detector():
    return YOLOFace()

def build_face_ReID(model_name):
    if model_name=='vit_8_112':
        return FaceEmbedder()
    elif model_name=='faceReID_FR':
        return FaceEmbedderFR()
    elif model_name=='faceReID_OF':
        return FaceEmbedderFacenet()

def build_tracker(args, frame_rate=30):
    return BoTSORT(args, frame_rate=30)

def build_mesh_generater():
    return Face3DMeshGenerator()

def build_pose_generator():
    return YOLOPose()