import cv2
import pyvirtualcam
import numpy as np
import torch
from engine import CustomSegmentationWithYolo


class Streaming(CustomSegmentationWithYolo):
    def __init__(self, in_source=None, out_source=None, fps=None, blur_strength=None, cam_fps=15, background="none"):
        super().__init__(erode_size=5, erode_intensity=2)
        self.input_source = in_source
        self.output_source = out_source
        self.fps = fps
        self.blur_strength = blur_strength
        self.running = False
        self.original_fps = cam_fps
        self.background = background
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        print(f"Device selected/found for inference : {self.device}")

    def update_streaming_config(self, in_source=None, out_source=None, fps=None, blur_strength=None, background="none"):
        self.input_source = in_source
        self.output_source = out_source
        self.fps = fps
        self.blur_strength = blur_strength
        self.background = background

    def update_cam_fps(self, fps):
        self.original_fps = fps

    def update_running_status(self, running_status=False):
        self.running = running_status
    
    def stream_video(self):
        self.running = True
        print(f"Retrieving feed from source({self.input_source}), FPS: {self.fps}, Blur Strength: {self.blur_strength}")
        cap = cv2.VideoCapture(int(self.input_source))
        frame_idx = 0

        if not cap.isOpened():
            print(f"Error: Unable to access camera with index {self.input_source}.")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        try:
            self.original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        except Exception as e:
            print(f"Webcam({self.input_source}) live fps not available. Setting fps to {self.original_fps}. Exception info: {e}")

        frame_interval = max(1, int(self.original_fps / self.fps)) if self.fps else 1

        with pyvirtualcam.Camera(width=width, height=height, fps=self.fps) as cam:
            print(f"Virtual camera running at {width}x{height} {self.fps}fps")

            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from camera.")
                    break

                result_frame = frame  # Ensure `result_frame` is always defined

                if frame_idx % frame_interval == 0:
                    results = self.model.predict(
                        source=frame, save=False, save_txt=False, stream=True, retina_masks=True, verbose=False, device=self.device
                    )
                    mask = self.generate_mask_from_result(results)

                    if mask is not None:
                        if self.background == "blur":
                            result_frame = self.apply_blur_with_mask(frame, mask, blur_strength=self.blur_strength)
                        elif self.background == "none":
                            result_frame = self.apply_black_background(frame, mask)
                        elif self.background == "default":
                            result_frame = self.apply_custom_background(frame, mask)

            # Ensure `result_frame` is valid before sending
            if result_frame is not None:
                cam.send(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
            else:
                print("Warning: result_frame is None, sending original frame.")
                cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            cam.sleep_until_next_frame()

        cap.release()


    def list_available_devices(self):
        devices = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                devices.append({"id": i, "name": f"Camera {i}"})
                cap.release()
        return devices


if __name__ == "__main__":
    streaming = Streaming()
    print(streaming.list_available_devices())
