import asyncio
import os
import subprocess
import yaml
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import soundfile as sf
import numpy as np
import cv2


@dataclass
class MuseTalkConfig:
    repo_dir: Path
    infer_script: Path
    sample_rate: int = 16000
    channels: int = 1
    cmd_template: Optional[str] = None  # e.g. "python {script} --image {image} --audio {audio} --out {out}"


class MuseTalkRunner:
    """
    Thin adapter around MuseTalk 1.5 inference.

    By default uses subprocess to call the official inference script. Adjust the
    command line and arguments to match the v1.5 interface (image path, audio path,
    output path, etc.). For real-time, you might run in chunked mode or maintain
    a warm model process with IPC (advanced).
    """

    def __init__(self, cfg: MuseTalkConfig):
        self.cfg = cfg

    def synthesize_batch(self, image_path: Path, audio_wav_path: Path, out_video_path: Path) -> None:
        repo_dir = self.cfg.repo_dir
        script = (repo_dir / self.cfg.infer_script).resolve()
        out_video_path.parent.mkdir(parents=True, exist_ok=True)

        if script.exists():
            # MuseTalk v1.5 の inference.py は image/audio の直指定ではなく、YAMLのinference_configを取る。
            # 一時YAMLを作成し、それを使って推論を実行する。
            tmp_cfg_dir = repo_dir / "configs" / "inference"
            tmp_cfg_dir.mkdir(parents=True, exist_ok=True)
            tmp_cfg = tmp_cfg_dir / "live_avatar_tmp.yaml"
            cfg = {
                "task_0": {
                    "video_path": str(image_path),
                    "audio_path": str(audio_wav_path),
                }
            }
            with open(tmp_cfg, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

            result_dir = repo_dir / "results" / "live_avatar"
            result_dir.mkdir(parents=True, exist_ok=True)
            # 出力ファイル名はスクリプト内部で result_dir/v15/{name} に置かれるため、
            # 固定名にして後で所望の場所へ移動する。
            temp_name = "_tmp_out.mp4"
            # 事前に座標を保存しておき、mmpose依存を回避（簡易bbox）。
            try:
                import cv2
                img = cv2.imread(str(image_path))
                ih, iw = img.shape[:2]
                cx, cy = iw // 2, ih // 2
                bw, bh = int(iw * 0.6), int(ih * 0.6)
                x1 = max(0, cx - bw // 2)
                y1 = max(0, cy - bh // 2)
                x2 = min(iw, cx + bw // 2)
                y2 = min(ih, cy + bh // 2)
                saved_coord_path = repo_dir / "results" / f"{image_path.stem}.pkl"
                with open(saved_coord_path, "wb") as f:
                    pickle.dump([(x1, y1, x2, y2)], f)
            except Exception:
                pass

            cmd = [
                "python", "-m", "scripts.inference",
                "--inference_config", str(tmp_cfg.relative_to(repo_dir)),
                "--result_dir", str(result_dir.relative_to(repo_dir)),
                "--unet_model_path", "models/musetalkV15/unet.pth",
                "--unet_config", "models/musetalkV15/musetalk.json",
                "--version", "v15",
                "--fps", "25",
                "--output_vid_name", temp_name,
                "--use_saved_coord",
            ]
            try:
                subprocess.run(" ".join(cmd), check=True, cwd=str(repo_dir), shell=True)
                # 生成場所: results/live_avatar/v15/_tmp_out.mp4
                gen_path = result_dir / "v15" / temp_name
                if gen_path.exists():
                    shutil.move(str(gen_path), str(out_video_path))
                    return
            except subprocess.CalledProcessError:
                pass

        # フォールバック: MuseTalk未設定時はダミー動画を生成（無音 or 音声なし）
        # 音声長から動画尺を決定して、静止画に軽いズームをかけた動画を生成
        try:
            data, sr = sf.read(str(audio_wav_path), dtype="int16")
            duration = len(data) / float(sr)
        except Exception:
            duration = 3.0

        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Image not readable: {image_path}")
        h, w = img.shape[:2]
        fps = 25
        total = max(1, int(duration * fps))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(out_video_path), fourcc, fps, (w, h))
        for i in range(total):
            scale = 1.0 + 0.03 * (i / total)
            nh, nw = int(h * scale), int(w * scale)
            resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            y0 = (nh - h) // 2
            x0 = (nw - w) // 2
            crop = resized[y0:y0 + h, x0:x0 + w]
            vw.write(crop)
        vw.release()
        return

    async def synthesize_live_chunk(self, image_path: Path, pcm_chunk: bytes) -> Optional[bytes]:
        """
        Placeholder for real-time chunk synthesis. A practical approach is to
        accumulate a short window of audio, write to a temp wav, and call a
        fast/low-latency path in MuseTalk that emits frames incrementally.

        Returns video frame(s) bytes or None. Implement this by switching to a
        long-lived process with IPC if MuseTalk supports it.
        """
        # Not implemented in skeleton; returning None to indicate no frames ready.
        await asyncio.sleep(0)
        return None
