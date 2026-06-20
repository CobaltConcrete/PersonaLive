"""
PersonaLive — accelerated inference entry point.

This script extends the unified `inference.py` (webcam / offline, xformers-only)
with a second, faster backend: the TensorRT wrapper at `src/wrapper_trt.py`
(`PersonaLive` class), per handover.md Priority 1.

    --acceleration xformers   → delegates to inference.py's existing
                                 Pose2VideoPipeline_Stream / Pose2VideoPipeline path
                                 (diffusers UNet + xformers attention). Unchanged
                                 behavior, just re-exposed here for a single entry point.

    --acceleration tensorrt   → uses src/wrapper_trt.py's PersonaLive class directly.
                                 This class does NOT use Pose2VideoPipeline_Stream at all —
                                 it runs keypoints/pose/motion/denoising through a
                                 pre-built TensorRT engine and manages its own temporal
                                 state (self.first_frame, self.motion_bank, self.kps_ref, …).
                                 Frame tiling (the MIN_FRAMES=16 workaround used for the
                                 non-TRT webcam path) is NOT needed here — PersonaLive
                                 natively consumes chunk_size=4 frames per call.

IMPORTANT — why the TensorRT path runs in its own subprocess:
    `wrapper_trt.PersonaLive.__init__` first loads several models onto the GPU via
    plain PyTorch (`.to(device)`), which silently creates PyTorch's own CUDA
    context. It then constructs `EngineModel`, whose pycuda backend calls
    `cuda.Device(idx).make_context()` — this pushes a *second*, independent
    driver-level CUDA context on top of the first. From that point on, GPU memory
    allocated under the first context (the earlier torch model weights) and
    kernels launched under the second context (pycuda's) are no longer
    guaranteed consistent, and torch ops elsewhere in the same process can fail
    with `CUDA error: invalid argument`.

    The already-working server path (`inference_online.py` → `webcam/vid2vid_trt.py`)
    avoids this by constructing `PersonaLive` inside its own freshly spawned
    `multiprocessing.Process`, isolated from anything else touching CUDA. This
    script mirrors that pattern: the TensorRT pipeline always runs inside a
    dedicated spawned worker, and we talk to it over plain-numpy multiprocessing
    queues (never CUDA tensors — those can't safely cross a process boundary
    here anyway).

Neither inference.py nor src/pipelines/pipeline_pose2vid.py is modified by this file.
"""

import argparse
import os
import sys
import time
import queue as pyqueue
import multiprocessing as mp
from datetime import datetime

import numpy as np
import cv2
import torch
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm
from decord import VideoReader

# Re-use the existing unified script for model loading / the xformers pipeline /
# the MJPEG streamer / webcam capture helpers. We never modify inference.py.
import inference as base_inference

from src.utils.util import save_videos_grid


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="PersonaLive inference with selectable acceleration backend."
    )
    parser.add_argument(
        "--acceleration", type=str, default="tensorrt",
        choices=["xformers", "tensorrt"],
        help="Inference backend. 'tensorrt' uses src/wrapper_trt.py's PersonaLive "
             "(requires a pre-built engine, see --trt_config). 'xformers' delegates "
             "to inference.py's diffusers-based pipeline.",
    )
    # xformers-path config (same as inference.py's --config)
    parser.add_argument("--config", type=str, default='configs/prompts/personalive_offline.yaml',
                        help="Model/inference config for --acceleration xformers.")
    # tensorrt-path config (same default as the existing online server, config.py)
    parser.add_argument("--trt_config", type=str, default='configs/prompts/personalive_online.yaml',
                        help="Config yaml consumed by src/wrapper_trt.py's PersonaLive "
                             "(must define tensorrt_target_model, pose_guider_path, "
                             "motion_encoder_path, pose_encoder_path, reference_unet_weight_path, "
                             "vae_model_path, image_encoder_path, temporal_window_size, "
                             "temporal_adaptive_step, batch_size, dtype, etc.) "
                             "Used only for --acceleration tensorrt.")

    parser.add_argument("--name", type=str, default='personalive_accelerated')
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=100, help="Max frames to process (offline mode only)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_xformers", type=bool, default=True,
                        help="(xformers backend only) enable xformers memory-efficient attention.")
    parser.add_argument("--stream_gen", type=bool, default=True,
                        help="(xformers backend only) use streaming generation to reduce VRAM.")
    parser.add_argument("--reference_image", type=str, required=True,
                        help="Path to reference image.")
    parser.add_argument("--driving_video", type=str, default='',
                        help="Path to driving video, or '0' (integer) for webcam input via V4L2.")
    # Streaming output
    parser.add_argument("--stream", action="store_true",
                        help="Stream output via MJPEG HTTP server instead of (or in addition to) saving.")
    parser.add_argument("--stream_port", type=int, default=8080)
    parser.add_argument("--stream_quality", type=int, default=85)
    # Webcam capture options
    parser.add_argument("--webcam_width", type=int, default=640)
    parser.add_argument("--webcam_height", type=int, default=480)
    parser.add_argument("--webcam_fps", type=int, default=30)
    # NOTE: chunk_size is intentionally NOT exposed as a CLI arg (handover.md
    # "Clean up" priority #4) — it must always be 4 to match temporal_window_size /
    # the TRT engine's expected batch shape. We set args.chunk_size = 4 in main()
    # purely so the imported inference.run_webcam() (xformers path) keeps working.
    args = parser.parse_args()
    return args


# ══════════════════════════════════════════════════════════════════════════════
# TensorRT backend — shared worker process
# ══════════════════════════════════════════════════════════════════════════════

class _TRTArgs:
    """Minimal namespace satisfying wrapper_trt.PersonaLive's `args.config_path` access."""
    def __init__(self, config_path):
        self.config_path = config_path


def _resolve_cuda_device(device) -> torch.device:
    """Return a torch.device with a concrete integer index.

    wrapper_trt.PersonaLive's EngineModel calls `cuda.Device(self.device.index)`
    (pycuda), which requires an actual int. A bare "cuda" string maps to
    `torch.device("cuda")` whose `.index` is None. Bare "cuda" implicitly means
    the current device, so we resolve that explicitly. This MUST be called
    inside the TensorRT worker process (see module docstring) — never in the
    parent, to avoid triggering any CUDA init there before the worker exists.
    """
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def _frame_to_tensor(rgb_np: np.ndarray, width: int, height: int) -> torch.Tensor:
    """RGB uint8 HxWxC -> CPU float32 tensor (1, C, H, W) normalized to [-1, 1].

    Mirrors webcam/vid2vid_trt.py's accept_new_params normalization so the TRT
    wrapper sees input in the same range/layout it was built and tested against.
    """
    rgb = cv2.resize(rgb_np, (width, height), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(rgb).float() / 255.0
    t = t * 2.0 - 1.0
    t = t.permute(2, 0, 1).unsqueeze(0)
    return t


def _trt_worker(trt_config_path, reference_image_path, width, height,
                 device_str, input_queue, output_queue, ready_event, stop_event, error_queue):
    """
    Entire lifetime of PersonaLive lives in this process. See the module
    docstring for why: pycuda's EngineModel.make_context() must not collide
    with a CUDA context PyTorch already initialized elsewhere.

    Protocol:
      - input_queue receives either:
          * a list of `chunk_size` HxWxC uint8 RGB numpy frames, or
          * None (sentinel) → exit
      - output_queue receives a (chunk_size, H, W, C) uint8 RGB numpy array
        per input chunk.
      - any exception is put on error_queue (as a string) and ready_event is
        set so the parent can stop waiting and surface it.
    """
    try:
        torch.set_grad_enabled(False)
        from src.wrapper_trt import PersonaLive

        trt_device = _resolve_cuda_device(device_str)
        trt_args = _TRTArgs(trt_config_path)
        pipeline = PersonaLive(trt_args, trt_device)

        ref_image_pil = Image.open(reference_image_path).convert("RGB")
        pipeline.fuse_reference(ref_image_pil)
        ready_event.set()

        while not stop_event.is_set():
            try:
                frames_np = input_queue.get(timeout=0.5)
            except pyqueue.Empty:
                continue
            if frames_np is None:
                break

            tensors = [_frame_to_tensor(f, width, height).to(trt_device) for f in frames_np]
            images = torch.cat(tensors, dim=0)

            with torch.no_grad():
                video = pipeline.process_input(images)  # numpy (chunk_size, H, W, C), normalized [0,1]

            frames_out = (np.asarray(video, dtype=np.float32) * 255.0).clip(0, 255).astype(np.uint8)
            output_queue.put(frames_out)

    except Exception as e:
        import traceback
        error_queue.put(f"{e}\n{traceback.format_exc()}")
        ready_event.set()  # unblock the parent so it can read the error and exit


def _start_trt_worker(args, device, chunk_size):
    """Spawn the worker process and block until it's ready (models loaded,
    reference fused) or it reports an init failure."""
    ctx = mp.get_context("spawn")
    input_queue  = ctx.Queue(maxsize=8)
    output_queue = ctx.Queue(maxsize=8)
    error_queue  = ctx.Queue()
    ready_event  = ctx.Event()
    stop_event   = ctx.Event()

    worker = ctx.Process(
        target=_trt_worker,
        args=(args.trt_config, args.reference_image, args.W, args.H,
              device, input_queue, output_queue, ready_event, stop_event, error_queue),
        daemon=True,
    )
    print("[trt] Starting TensorRT worker process…")
    worker.start()
    ready_event.wait()

    if not error_queue.empty():
        msg = error_queue.get()
        worker.join(timeout=1.0)
        raise RuntimeError(f"[trt] Worker failed during init:\n{msg}")

    print("[trt] Worker ready.")
    return worker, input_queue, output_queue, error_queue, stop_event


def _stop_trt_worker(worker, input_queue, stop_event):
    stop_event.set()
    try:
        input_queue.put_nowait(None)
    except Exception:
        pass
    worker.join(timeout=2.0)
    if worker.is_alive():
        worker.terminate()


def _check_worker_alive(worker, error_queue):
    if not error_queue.empty():
        raise RuntimeError(f"[trt] Worker crashed:\n{error_queue.get()}")
    if not worker.is_alive():
        raise RuntimeError("[trt] Worker process exited unexpectedly.")


# ══════════════════════════════════════════════════════════════════════════════
# TensorRT backend — webcam (real-time)
# ══════════════════════════════════════════════════════════════════════════════

def run_webcam_trt(args, device, streamer):
    """Real-time webcam loop using src/wrapper_trt.py's PersonaLive, hosted in a
    dedicated subprocess (see module docstring). No Pose2VideoPipeline_Stream
    involved, no MIN_FRAMES tiling — PersonaLive natively consumes
    chunk_size=4 frames per process_input() call and tracks its own temporal
    state across calls.
    """
    width, height = args.W, args.H
    chunk_size = 4

    worker, input_queue, output_queue, error_queue, stop_event = _start_trt_worker(
        args, device, chunk_size
    )

    cap = base_inference.open_webcam(0, args.webcam_width, args.webcam_height, args.webcam_fps)

    show_window = False
    try:
        cv2.namedWindow("PersonaLive (TensorRT) – press q to quit", cv2.WINDOW_NORMAL)
        show_window = True
    except Exception:
        pass  # headless environment

    frame_buffer = []
    fps_t0, fps_count = time.time(), 0
    print("[trt] Starting real-time inference (TensorRT). Press Ctrl-C to stop.")

    try:
        while True:
            _check_worker_alive(worker, error_queue)

            ret, bgr = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
            frame_buffer.append(rgb)
            if len(frame_buffer) < chunk_size:
                continue

            # Consume the oldest chunk_size frames — drain, no overlap
            # (handover.md §6: overlap re-processes the same frames forever).
            chunk        = frame_buffer[:chunk_size]
            frame_buffer = frame_buffer[chunk_size:]

            try:
                input_queue.put(chunk, timeout=0.5)
            except pyqueue.Full:
                continue  # worker is behind; drop this chunk rather than stall capture

            try:
                frames_out = output_queue.get(timeout=10.0)
            except pyqueue.Empty:
                continue

            for frame_rgb in frames_out:
                if streamer is not None:
                    streamer.push(frame_rgb)
                if show_window:
                    cv2.imshow(
                        "PersonaLive (TensorRT) – press q to quit",
                        cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                    )
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        raise KeyboardInterrupt

            fps_count += chunk_size
            elapsed = time.time() - fps_t0
            if elapsed >= 2.0:
                print(f"[trt] ~{fps_count / elapsed:.1f} fps")
                fps_t0, fps_count = time.time(), 0

    except KeyboardInterrupt:
        print("\n[trt] Stopping…")
    finally:
        _stop_trt_worker(worker, input_queue, stop_event)
        cap.release()
        if show_window:
            cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
# TensorRT backend — offline (video file)
# ══════════════════════════════════════════════════════════════════════════════

def run_offline_trt(args, device, streamer):
    """Batch-process a driving video through PersonaLive (hosted in a dedicated
    subprocess; see module docstring) in chunk_size=4 windows.

    PersonaLive tracks temporal state (self.first_frame, self.motion_bank, …)
    across successive process_input() calls, so feeding it sequential 4-frame
    chunks from the same video reproduces the same continuity the webcam path
    relies on — no frame tiling required (that workaround is specific to
    Pose2VideoPipeline_Stream's internal padding, which this backend bypasses).
    """
    width, height = args.W, args.H
    chunk_size = 4

    control = VideoReader(args.driving_video)
    video_length = min(len(control) // chunk_size * chunk_size, args.L)
    if video_length <= 0:
        raise ValueError(
            f"Driving video has too few frames for chunk_size={chunk_size} "
            f"(or --L is set too low)."
        )
    frames_np_full = control.get_batch(list(range(video_length))).asnumpy()  # (F, H, W, C) uint8

    worker, input_queue, output_queue, error_queue, stop_event = _start_trt_worker(
        args, device, chunk_size
    )

    out_frames = []  # list of (H, W, C) float32 arrays in [0, 1]
    print(f"[trt] Running TensorRT inference over {video_length} frames "
          f"in chunks of {chunk_size}…")

    try:
        for start in tqdm(range(0, video_length, chunk_size), desc="TRT chunks"):
            _check_worker_alive(worker, error_queue)

            chunk_np = [
                cv2.resize(f, (width, height), interpolation=cv2.INTER_LINEAR)
                for f in frames_np_full[start:start + chunk_size]
            ]

            input_queue.put(chunk_np)
            frames_out = output_queue.get(timeout=60.0)

            for frame in frames_out:
                out_frames.append(frame.astype(np.float32) / 255.0)
                if streamer is not None:
                    streamer.push(frame)
    finally:
        _stop_trt_worker(worker, input_queue, stop_event)

    gen = np.stack(out_frames, axis=0)                                     # (F, H, W, C) in [0,1]
    gen_video = torch.from_numpy(gen).permute(3, 0, 1, 2).unsqueeze(0)     # (1, C, F, H, W)

    date_str = datetime.now().strftime("%Y%m%d")
    save_dir_name = f"{date_str}--{args.name}"
    save_split_vid_dir = os.path.join('results', save_dir_name, 'split_vid')
    os.makedirs(save_split_vid_dir, exist_ok=True)

    video_name = os.path.basename(args.driving_video).split(".")[0]
    source_name = os.path.basename(args.reference_image).split(".")[0]
    save_vid_path = os.path.join(save_split_vid_dir, f"{source_name}_{video_name}_trt.mp4")

    save_videos_grid(gen_video, save_vid_path, n_rows=1, fps=25, crf=18,
                     audio_source=args.driving_video)
    print(f"[trt] Saved → {save_vid_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    device = args.device
    args.chunk_size = 4  # hardcoded; see parse_args note and handover.md priority #4

    # NOTE: no torch.cuda calls happen here in the parent process for the
    # tensorrt path — that's deliberate. See module docstring: any CUDA
    # context the parent creates before the worker process exists is exactly
    # the kind of state that caused the original "invalid argument" crash.
    torch.set_grad_enabled(False)

    use_webcam = (args.driving_video == "0" or args.driving_video == "")
    print(f"[main] Mode: {'webcam (real-time)' if use_webcam else 'offline (video file)'}  "
          f"|  Acceleration: {args.acceleration}  |  Device: {device}")

    streamer = None
    if args.stream:
        streamer = base_inference.MJPEGStreamer(port=args.stream_port, jpeg_quality=args.stream_quality)

    try:
        if args.acceleration == "tensorrt":
            if use_webcam:
                run_webcam_trt(args, device, streamer)
            else:
                run_offline_trt(args, device, streamer)
        else:
            # --acceleration xformers: delegate entirely to inference.py's existing,
            # already-working diffusers + xformers path, in this same process
            # (no subprocess needed — there's no pycuda context to collide with).
            config = OmegaConf.load(args.config)
            weight_dtype = torch.float16 if config.weight_dtype == "fp16" else torch.float32

            import mediapipe as mp
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

            print("[main] Loading models…")
            models = base_inference.load_models(args, config, weight_dtype, device)
            stream_gen = True if use_webcam else args.stream_gen
            pipe = base_inference.build_pipeline(args, models, stream_gen)
            print("[main] Models loaded.")

            if use_webcam:
                base_inference.run_webcam(args, config, pipe, face_mesh, streamer)
            else:
                base_inference.run_offline(args, config, pipe, face_mesh, streamer)
    finally:
        if streamer is not None:
            streamer.stop()


if __name__ == "__main__":
    main()
