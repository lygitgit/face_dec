import gradio as gr
from ultralytics import YOLO
import numpy as np # 导入numpy，因为gr.Image(type="numpy")可能需要

# 初始化YOLO检测器
detector = YOLO('weights/detector/model_11n.pt')

def process_frame(frame):
    """
    处理来自摄像头的单个帧，进行YOLO检测并返回带有检测结果的图像。
    """
    if frame is None:
        return None
    # YOLO的predict方法可以直接接受numpy数组图像
    results = detector.predict(frame, verbose=False) # verbose=False 减少控制台输出
    # results[0].plot() 返回一个带有边界框和标签的numpy数组图像
    return results[0].plot()

with gr.Blocks() as demo:
    gr.Markdown("## YOLO实时摄像头检测")
    with gr.Row():
        with gr.Column():
            # sources=["webcam"] 启用摄像头输入
            # streaming=True 表示这是一个实时流输入
            # type="numpy" 表示输入图像为numpy数组
            input_video = gr.Image(sources=["webcam"], type="numpy", label="摄像头输入", streaming=True)
            
        with gr.Column():
            # streaming=True 表示这是一个实时流输出
            output_video = gr.Image(type="numpy", label="检测结果", streaming=True)
    
    # 将input_video的流式数据传递给process_frame函数，并将结果输出到output_video
    # stream_every 参数控制处理帧的频率
    input_video.stream(process_frame, inputs=input_video, outputs=output_video, stream_every=0.1)

demo.launch()

# 保留原有代码逻辑的注释部分
# img = Image.open("/home/lenovo/gradio/UnionProject/test/bus.jpg")
# results = detector.predict(img)
# results[0].show()