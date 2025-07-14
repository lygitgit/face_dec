import cv2
import sys
import gradio as gr
import numpy as np
import argparse
from main import VideoWrapper, parse_args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='test/video_test.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=1920, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.09, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='IOU threshold for NMS')
    parser.add_argument('--device', default='gpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
    parser.add_argument('--hide-labels-name', default=False, action='store_true', help='hide labels')

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.05, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.4, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.7, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=40000, help='filter out tiny boxes')

    parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true",
                        help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"model/fast_reid/configs/MOT17/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--ReID-emb-name", dest="ReID_emb_name", default="vit_8_112",
                        type=str, help="reid model name")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"weights/ReID/mot17_sbs_S50.pth",
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')

    # YOLOX args
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("--save_result", action="store_true",help="whether to save the inference result of image/video")
    parser.add_argument('--use_external_detect', default=False, action='store_true', help='use external detector')
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")

    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true",help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.")

    return parser.parse_args()


args = parse_args()
args.ablation = False

# 通过Gradio文件选择器选择视频
VIDEO_PATH = None
cap = None
frame_count = 0
estimate_pose = False
stop_flag = False 

def set_video_path(video_file):
    global VIDEO_PATH, cap, frame_count, wrapper

    if video_file is None:
        return "未选择视频"

    # 释放上一个视频资源
    if cap is not None:
        cap.release()

    VIDEO_PATH = video_file.name
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_count = 0

    # 重建 wrapper 实例，避免使用旧状态
    wrapper = VideoWrapper(args)

    if cap.isOpened():
        return f"已加载视频：{VIDEO_PATH}"
    else:
        return "无法打开视频"
    
# 读取视频文件或摄像头
# cap = cv2.VideoCapture(args.source if args.source.isnumeric() else args.source)
# if not cap.isOpened():
#     print(f"无法打开视频源: {args.source}")
#     sys.exit(1)
    
wrapper = VideoWrapper(args)
# print(f"视频FPS: {cap.get(cv2.CAP_PROP_FPS):.2f}")


def video_generator():
    global frame_count, estimate_pose, stop_flag
    stop_flag = False
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while cap.isOpened() and not stop_flag:
        ret, frame = cap.read()
        if not ret:
            print("视频读取完毕")
            break

        frame_with_boxes, tracking_results, inter_res = wrapper.process_frame(frame, frame_count, estimate_pose=estimate_pose)
        frame_count += 1

        # ==== 处理目标 patch ====
        patch_list = []
        for res_line in tracking_results:
            parts = res_line.strip().split(',')
            if len(parts) >= 5:
                x, y, w, h = map(float, parts[1:5])
                x, y, w, h = int(x), int(y), int(w), int(h)
                patch = frame[int(y - 0.2 * h): int(y + 1.2 * h), int(x - 0.2 * w): int(x + 1.2 * w)]
                if patch.size > 0:
                    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                    patch_rgb = cv2.resize(patch_rgb, (128, 128))
                    # 在patch底部写上ID
                    id_text = f"ID:{parts[0]}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    text_size, _ = cv2.getTextSize(id_text, font, font_scale, thickness)
                    text_x = max(0, (patch_rgb.shape[1] - text_size[0]) // 2)
                    text_y = patch_rgb.shape[0] - 5
                    cv2.putText(
                        patch_rgb,
                        id_text,
                        (text_x, text_y),
                        font,
                        font_scale,
                        (0, 255, 0),
                        thickness,
                        cv2.LINE_AA
                    )
                    patch_list.append(patch_rgb)

        # 转换主frame
        frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
        result_text = f"检测目标数: {len(patch_list)}"

        yield frame_rgb, result_text, patch_list


def stop_processing():
    global stop_flag
    stop_flag = True
    return gr.update(), gr.update(), gr.update()


# === 使用 Interface 替代 Blocks ===
with gr.Blocks() as demo:
    gr.Markdown("## 视频目标检测与姿态估计可视化")
    # estimate_pose = gr.Checkbox(label="启用姿态估计", value=False)

    video_input = gr.File(label="上传视频", file_types=[".mp4", ".avi"])
    video_status = gr.Textbox(label="视频状态")

    # 上传视频后自动设置路径
    video_input.change(fn=set_video_path, inputs=[video_input], outputs=[video_status])

    video_display = gr.Image(label="视频帧", streaming=True)
    result_text = gr.Textbox(label="检测信息")
    gallery = gr.Gallery(label="检测目标 Patch", columns=4, rows=2)
    
    start_button = gr.Button("开始处理")
    stop_button = gr.Button("停止处理")

    # 当点击按钮时，启动 video_generator，并传入 estimate_pose 控件的值
    start_button.click(fn=video_generator, 
                       inputs=[], 
                       outputs=[video_display, result_text, gallery])
    stop_button.click(fn=stop_processing, outputs=[video_display, result_text, gallery])

if __name__ == "__main__":
    demo.launch()
