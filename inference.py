import argparse
import os
import sys
import time
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import mediapipe as mp
import numpy as np
import cv2
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKL
from src.scheduler.scheduler_ddim import DDIMScheduler
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid import Pose2VideoPipeline, Pose2VideoPipeline_Stream
from src.utils.util import save_videos_grid, crop_face
from src.models.motion_encoder.encoder import MotEncoder
from src.liveportrait.motion_extractor import MotionExtractor
from src.models.pose_guider import PoseGuider
from decord import VideoReader
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# MJPEG HTTP streamer
# ══════════════════════════════════════════════════════════════════════════════

class MJPEGStreamer:
    """
    Serves rendered frames as an MJPEG stream over HTTP so they can be watched
    in any browser — no display server required.

    Open  http://<host>:<port>/  in your browser to watch.
    """

    def __init__(self, port: int = 8080, jpeg_quality: int = 85):
        self.port    = port
        self.quality = jpeg_quality
        self._frame: bytes = b""
        self._lock   = threading.Lock()
        self._stop   = threading.Event()

        streamer = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                if self.path == "/":
                    body = (
                        b"<html><body style='margin:0;background:#000'>"
                        b"<img src='/stream' style='max-width:100%;height:auto'>"
                        b"</body></html>"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=frame"
                    )
                    self.end_headers()
                    try:
                        while not streamer._stop.is_set():
                            with streamer._lock:
                                frame = streamer._frame
                            if frame:
                                self.wfile.write(
                                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                    + frame + b"\r\n"
                                )
                            time.sleep(0.01)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.send_response(404)
                    self.end_headers()

        self._server = HTTPServer(("0.0.0.0", port), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"[stream] Preview → http://localhost:{port}/  (or your server IP)")

    def push(self, rgb_np: np.ndarray):
        """Encode an RGB uint8 frame and make it available to connected clients."""
        ok, buf = cv2.imencode(
            ".jpg", cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, self.quality],
        )
        if ok:
            with self._lock:
                self._frame = buf.tobytes()

    def stop(self):
        self._stop.set()
        self._server.shutdown()


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/prompts/personalive_offline.yaml')
    parser.add_argument("--name", type=str, default='personalive')
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=100, help="Max frames to process (offline mode only)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_xformers", type=bool, default=True)
    parser.add_argument("--stream_gen", type=bool, default=True,
                        help="Use streaming generation to reduce VRAM usage (offline only).")
    parser.add_argument("--reference_image", type=str, required=True,
                        help="Path to reference image.")
    parser.add_argument("--driving_video", type=str, default='',
                        help="Path to driving video, or '0' (integer) for webcam input via V4L2.")
    # Streaming output
    parser.add_argument("--stream", action="store_true",
                        help="Stream output via MJPEG HTTP server instead of (or in addition to) saving.")
    parser.add_argument("--stream_port", type=int, default=8080,
                        help="Port for the MJPEG HTTP stream (default: 8080).")
    parser.add_argument("--stream_quality", type=int, default=85,
                        help="JPEG quality for the MJPEG stream (default: 85).")
    # Webcam capture options
    parser.add_argument("--webcam_width", type=int, default=640,
                        help="Webcam capture width (default: 640).")
    parser.add_argument("--webcam_height", type=int, default=480,
                        help="Webcam capture height (default: 480).")
    parser.add_argument("--webcam_fps", type=int, default=30,
                        help="Webcam capture FPS target (default: 30).")
    # chunk_size is fixed at 4 to match the model's temporal_window_size expectation.
    # Exposed as an arg for clarity but should not be changed unless you retrain.
    parser.add_argument("--chunk_size", type=int, default=4,
                        help="Frames per inference chunk — must match temporal_window_size (default: 4).")
    args = parser.parse_args()
    return args


# ══════════════════════════════════════════════════════════════════════════════
# Model loading (shared between modes)
# ══════════════════════════════════════════════════════════════════════════════

def load_models(args, config, weight_dtype, device):
    infer_config = OmegaConf.load(config.inference_config)

    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(device=device, dtype=weight_dtype)

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device=device)

    vae = AutoencoderKL.from_pretrained(config.vae_path).to(device, dtype=weight_dtype)
    motion_encoder = MotEncoder().to(dtype=weight_dtype, device=device).eval()
    pose_guider = PoseGuider().to(device=device, dtype=weight_dtype)
    pose_encoder = MotionExtractor(num_kp=21).to(device=device, dtype=weight_dtype).eval()
    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device=device)

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    # Load weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"), strict=False
    )
    reference_unet.load_state_dict(
        torch.load(
            config.denoising_unet_path.replace('denoising_unet', 'reference_unet'),
            map_location="cpu",
        ),
        strict=True,
    )
    motion_encoder.load_state_dict(
        torch.load(
            config.denoising_unet_path.replace('denoising_unet', 'motion_encoder'),
            map_location="cpu",
        ),
        strict=True,
    )
    pose_guider.load_state_dict(
        torch.load(
            config.denoising_unet_path.replace('denoising_unet', 'pose_guider'),
            map_location="cpu",
        ),
        strict=True,
    )
    denoising_unet.load_state_dict(
        torch.load(
            config.denoising_unet_path.replace('denoising_unet', 'temporal_module'),
            map_location="cpu",
        ),
        strict=False,
    )
    pose_encoder.load_state_dict(
        torch.load(
            config.denoising_unet_path.replace('denoising_unet', 'motion_extractor'),
            map_location="cpu",
        ),
        strict=False,
    )

    if args.use_xformers:
        if is_xformers_available():
            try:
                reference_unet.enable_xformers_memory_efficient_attention()
                denoising_unet.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print("Failed to enable xformers:", e)
        else:
            print("xformers not available.")

    return dict(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        motion_encoder=motion_encoder,
        pose_encoder=pose_encoder,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )


def build_pipeline(args, models, stream_gen: bool):
    PipelineClass = Pose2VideoPipeline_Stream if stream_gen else Pose2VideoPipeline
    pipe = PipelineClass(**models)
    pipe = pipe.to(args.device)
    return pipe


# ══════════════════════════════════════════════════════════════════════════════
# Offline mode  (video file → video file, optional stream preview)
# ══════════════════════════════════════════════════════════════════════════════

def run_offline(args, config, pipe, face_mesh, streamer):
    width, height = args.W, args.H
    pose_transform = transforms.Compose(
        [transforms.Resize((height, width)), transforms.ToTensor()]
    )

    date_str = datetime.now().strftime("%Y%m%d")
    save_dir_name = f"{date_str}--{args.name}"
    save_vid_dir = os.path.join('results', save_dir_name, 'concat_vid')
    os.makedirs(save_vid_dir, exist_ok=True)
    save_split_vid_dir = os.path.join('results', save_dir_name, 'split_vid')
    os.makedirs(save_split_vid_dir, exist_ok=True)

    ref_image_path  = args.reference_image
    pose_video_path = args.driving_video

    video_name  = os.path.basename(pose_video_path).split(".")[0]
    source_name = os.path.basename(ref_image_path).split(".")[0]
    vid_name    = f"{source_name}_{video_name}.mp4"
    save_vid_path = os.path.join(save_vid_dir, vid_name)
    print(f"Output → {save_vid_path}")

    # Reference image
    if ref_image_path.endswith('.mp4'):
        src_vid = VideoReader(ref_image_path)
        ref_img = Image.fromarray(src_vid[0].asnumpy()).convert("RGB")
    else:
        ref_img = Image.open(ref_image_path).convert("RGB")

    # Driving video
    control     = VideoReader(pose_video_path)
    video_length = min(len(control) // 4 * 4, args.L)
    control      = control.get_batch(list(range(video_length))).asnumpy()

    ref_image_pil = ref_img.copy()
    ref_patch     = crop_face(ref_image_pil, face_mesh)
    ref_face_pil  = Image.fromarray(ref_patch).convert("RGB")

    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)

    dri_faces       = []
    ori_pose_images = []
    for pose_frame in tqdm(control[:video_length], desc='Cropping faces'):
        pose_pil = Image.fromarray(pose_frame).convert("RGB")
        ori_pose_images.append(pose_pil)
        face = crop_face(pose_pil, face_mesh)
        dri_faces.append(Image.fromarray(face).convert("RGB"))

    face_tensor_list     = []
    ori_pose_tensor_list = []
    ref_tensor_list      = []
    for idx, pose_pil in enumerate(ori_pose_images):
        face_tensor_list.append(pose_transform(dri_faces[idx]))
        ori_pose_tensor_list.append(pose_transform(pose_pil))
        ref_tensor_list.append(pose_transform(ref_image_pil))

    ref_tensor      = torch.stack(ref_tensor_list, dim=0).transpose(0, 1).unsqueeze(0)
    face_tensor     = torch.stack(face_tensor_list, dim=0).transpose(0, 1).unsqueeze(0)
    ori_pose_tensor = torch.stack(ori_pose_tensor_list, dim=0).transpose(0, 1).unsqueeze(0)

    gen_video = pipe(
        ori_pose_images,
        ref_image_pil,
        dri_faces,
        ref_face_pil,
        width,
        height,
        len(dri_faces),
        num_inference_steps=4,
        guidance_scale=1.0,
        generator=generator,
        temporal_window_size=4,
        temporal_adaptive_step=4,
    ).videos

    # Stream generated frames if requested
    if streamer is not None:
        # gen_video shape: (1, C, F, H, W), values in [0, 1]
        frames = gen_video[0].permute(1, 2, 3, 0).cpu().numpy()  # (F, H, W, C)
        frames = (frames * 255).clip(0, 255).astype(np.uint8)
        print(f"[stream] Pushing {len(frames)} frames to MJPEG stream…")
        for frame in frames:
            streamer.push(frame)
            time.sleep(1 / 25)

    video = torch.cat([ref_tensor, face_tensor, ori_pose_tensor, gen_video], dim=0)
    save_videos_grid(video, save_vid_path, n_rows=4, fps=25)

    split_path = save_vid_path.replace(save_vid_dir, save_split_vid_dir)
    save_videos_grid(gen_video, split_path, n_rows=1, fps=25, crf=18,
                     audio_source=pose_video_path)
    print("Done.")


# ══════════════════════════════════════════════════════════════════════════════
# Webcam / real-time mode
# ══════════════════════════════════════════════════════════════════════════════

def open_webcam(device_index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    """Open the webcam with V4L2 backend and apply capture settings."""
    cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        # Fallback: let OpenCV pick the backend
        cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam device {device_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    # Use MJPEG on V4L2 for higher throughput
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[webcam] Opened device {device_index}: {actual_w}×{actual_h} @ {actual_fps:.1f} fps")
    return cap


def run_webcam(args, config, pipe, face_mesh, streamer):
    """
    Real-time loop: read `chunk_size` frames from the webcam, run inference,
    push to the MJPEG streamer and/or display locally.
    """
    width, height   = args.W, args.H
    chunk_size      = args.chunk_size
    pose_transform  = transforms.Compose(
        [transforms.Resize((height, width)), transforms.ToTensor()]
    )

    cap = open_webcam(0, args.webcam_width, args.webcam_height, args.webcam_fps)

    # Load & prepare reference image
    ref_image_pil = Image.open(args.reference_image).convert("RGB")
    ref_patch     = crop_face(ref_image_pil, face_mesh)
    ref_face_pil  = Image.fromarray(ref_patch).convert("RGB")

    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)

    show_window = False
    try:
        cv2.namedWindow("PersonaLive – press q to quit", cv2.WINDOW_NORMAL)
        show_window = True
    except Exception:
        pass  # headless environment

    # These must match the model's training config — do not change.
    TEMPORAL_WINDOW_SIZE   = 4
    TEMPORAL_ADAPTIVE_STEP = 4
    # Minimum frames the Stream pipeline needs for its internal padding to work correctly.
    # = (TEMPORAL_ADAPTIVE_STEP - 1) * TEMPORAL_WINDOW_SIZE + TEMPORAL_WINDOW_SIZE
    MIN_FRAMES = (TEMPORAL_ADAPTIVE_STEP - 1) * TEMPORAL_WINDOW_SIZE + TEMPORAL_WINDOW_SIZE  # 16

    frame_buffer = []
    print("[webcam] Starting real-time inference. Press Ctrl-C to stop.")

    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                print("[webcam] Frame capture failed, retrying…")
                time.sleep(0.01)
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            frame_buffer.append(pil)

            if len(frame_buffer) < chunk_size:
                continue

            # Consume the oldest chunk_size frames.
            chunk        = frame_buffer[:chunk_size]
            frame_buffer = frame_buffer[chunk_size:]

            dri_faces       = []
            ori_pose_images = []
            for pil_frame in chunk:
                ori_pose_images.append(pil_frame)
                try:
                    face = crop_face(pil_frame, face_mesh)
                    dri_faces.append(Image.fromarray(face).convert("RGB"))
                except Exception:
                    dri_faces.append(pil_frame.resize((256, 256)))

            # The Stream pipeline's internal padding assumes at least
            # (temporal_adaptive_step-1)*temporal_window_size + temporal_window_size = 16
            # input frames. With only 4 webcam frames the padding slices clamp and
            # starve later windows. Tile the chunk to MIN_FRAMES before calling.
            # Only the first chunk_size output frames are meaningful; the rest are
            # tiled duplicates and get discarded after decoding.
            real_len = len(ori_pose_images)
            repeats  = (MIN_FRAMES + real_len - 1) // real_len  # ceil div
            ori_pose_images_in = (ori_pose_images * repeats)[:MIN_FRAMES]
            dri_faces_in       = (dri_faces       * repeats)[:MIN_FRAMES]

            gen_video = pipe(
                ori_pose_images_in,
                ref_image_pil,
                dri_faces_in,
                ref_face_pil,
                width,
                height,
                MIN_FRAMES,
                num_inference_steps=4,
                guidance_scale=1.0,
                generator=generator,
                temporal_window_size=TEMPORAL_WINDOW_SIZE,
                temporal_adaptive_step=TEMPORAL_ADAPTIVE_STEP,
            ).videos  # (1, C, MIN_FRAMES, H, W)

            # Keep only the first real_len frames (the actual webcam chunk).
            gen_video = gen_video[:, :, :real_len]

            # Extract frames and push
            frames = gen_video[0].permute(1, 2, 3, 0).cpu().numpy()
            frames = (frames * 255).clip(0, 255).astype(np.uint8)

            for frame_rgb in frames:
                if streamer is not None:
                    streamer.push(frame_rgb)

                if show_window:
                    cv2.imshow(
                        "PersonaLive – press q to quit",
                        cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    )
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\n[webcam] Stopping…")
    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    device = args.device
    config = OmegaConf.load(args.config)

    weight_dtype = torch.float16 if config.weight_dtype == "fp16" else torch.float32

    # Decide mode
    use_webcam = (args.driving_video == "0" or args.driving_video == "")
    if use_webcam and not args.driving_video:
        # No driving video at all → default to offline config's test_cases if any,
        # otherwise fall through to webcam.
        test_cases = OmegaConf.to_container(config).get("test_cases", {})
        if test_cases:
            use_webcam = False

    print(f"[main] Mode: {'webcam (real-time)' if use_webcam else 'offline (video file)'}")
    print(f"[main] Device: {device}  |  dtype: {weight_dtype}")

    # Face mesh (MediaPipe)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    # Build pipeline
    print("[main] Loading models…")
    models = load_models(args, config, weight_dtype, device)
    # For webcam mode, always use streaming generation to keep VRAM in check
    stream_gen = True if use_webcam else args.stream_gen
    pipe = build_pipeline(args, models, stream_gen)
    print("[main] Models loaded.")

    # Optional MJPEG streamer
    streamer = None
    if args.stream:
        streamer = MJPEGStreamer(port=args.stream_port, jpeg_quality=args.stream_quality)

    try:
        if use_webcam:
            run_webcam(args, config, pipe, face_mesh, streamer)
        else:
            run_offline(args, config, pipe, face_mesh, streamer)
    finally:
        if streamer is not None:
            streamer.stop()


if __name__ == "__main__":
    main()
