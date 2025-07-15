import gradio as gr
from ultralytics import YOLO
import numpy as np

# 初始化YOLO检测器
detector = YOLO('weights/detector/model_11n.pt')

def process_frame(frame, detection_enabled_state):
    """
    处理来自摄像头的单个帧。如果detection_enabled_state为True，则进行YOLO检测。
    """
    if frame is None:
        return None # Return None if no frame, Gradio will handle it

    if detection_enabled_state:
        # YOLO的predict方法可以直接接受numpy数组图像
        results = detector.predict(frame, verbose=False) # verbose=False 减少控制台输出
        # results[0].plot() 返回一个带有边界框和标签的numpy数组图像
        return results[0].plot()
    else:
        return frame # Return original frame if detection is off

def toggle_detection(current_state):
    """
    切换检测状态并更新按钮文本。
    """
    new_state = not current_state
    new_button_text = "关闭人脸检测" if new_state else "开启人脸检测"
    return new_state, gr.Button(new_button_text)

with gr.Blocks() as demo:
    gr.Markdown("## YOLO实时摄像头人脸检测")
    
    # 存储检测状态的变量
    detection_enabled = gr.State(False) # 初始状态为关闭

    with gr.Row():
        with gr.Column():
            input_video = gr.Image(sources=["webcam"], type="numpy", label="摄像头输入", streaming=True)
            # 按钮用于开启/关闭检测
            toggle_button = gr.Button("开启人脸检测")
            
        with gr.Column():
            output_video = gr.Image(type="numpy", label="检测结果", streaming=True)
    
    # 按钮点击事件：切换检测状态，并更新按钮文本
    toggle_button.click(
        toggle_detection,
        inputs=[detection_enabled],
        outputs=[detection_enabled, toggle_button]
    )

    # 实时流处理：将摄像头帧和检测状态传递给处理函数
    # 这里的inputs要包含所有需要传入process_frame的组件
    input_video.stream(
        process_frame,
        inputs=[input_video, detection_enabled],
        outputs=output_video,
        stream_every=0.1
    )

demo.launch()

# 保留原有代码逻辑的注释部分
# img = Image.open("/home/lenovo/gradio/UnionProject/test/bus.jpg")
# results = detector.predict(img)
# results[0].show()