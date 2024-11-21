import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Ini menginisialisasi CUDA driver

def load_engine(engine_path):
    """Load a TensorRT engine."""
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    """Allocate input/output buffers for the engine."""
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):
    """Perform inference using TensorRT."""
    [cuda.memcpy_htod_async(inp['device'], inp['host'], stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out['host'], out['device'], stream) for out in outputs]
    stream.synchronize()
    return [out['host'] for out in outputs]

def preprocess_image(image, input_shape):
    """Preprocess image for TensorRT inference."""
    h, w = input_shape
    resized = cv2.resize(image, (w, h))
    img = resized.transpose((2, 0, 1))  # HWC to CHW
    img = img / 255.0  # Normalize to [0, 1]
    return np.ascontiguousarray(img, dtype=np.float32)

def detect_trt(engine_path, video_path, input_shape=(640, 640)):
    # Load TensorRT engine
    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # Video writer (optional)
    output_path = "output_video_trt.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Run inference
    FPS = []
    inference_times = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        input_img = preprocess_image(frame, input_shape)
        inputs[0]['host'] = input_img.ravel()

        # Inference
        start_time = time.time()
        outputs_data = do_inference(context, bindings, inputs, outputs, stream)
        inference_time = (time.time() - start_time) * 1000  # In milliseconds
        inference_times.append(inference_time)
        FPS.append(1000 / inference_time)

        # TODO: Post-process outputs (implement YOLOv7-specific decoding here)
        # For now, we assume the output is ready to use.

        # Draw FPS and inference time on frame
        text = f"FPS: {int(FPS[-1])}, Inference Time: {inference_time:.1f} ms"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Write frame to output video
        out.write(frame)

        # Display the frame (optional)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Print average FPS and inference time
    print(f"Average FPS: {sum(FPS) / len(FPS):.2f}")
    print(f"Average Inference Time: {sum(inference_times) / len(inference_times):.2f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, required=True, help="Path to TensorRT engine file")
    parser.add_argument('--video', type=str, required=True, help="Path to input video file")
    args = parser.parse_args()

    detect_trt(args.engine, args.video)
