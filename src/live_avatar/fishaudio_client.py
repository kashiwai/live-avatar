import asyncio
import json
import os
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import aiohttp
import math
import websockets
from websockets.legacy.client import WebSocketClientProtocol
import numpy as np
from scipy.signal import resample_poly


@dataclass
class FishAudioConfig:
    api_key: str
    base_url: str
    voice_id: Optional[str] = None
    sample_rate: int = 16000  # ここは最終的にMuseTalkへ渡すターゲットSR
    channels: int = 1
    incoming_sample_rate: int = 24000  # FishAudioからの受信想定SR（要調整）
    chunk_ms: int = 40
    keepalive_sec: int = 20
    audio_format: str = "pcm_s16le"  # 受信想定フォーマット（要調整）


class FishAudioLiveClient:
    """
    Minimal FishAudio LIVE client (skeleton).

    You must set the correct endpoint, auth, and message schema per FishAudio docs.
    This client exposes an async generator `stream_tts(text)` that yields raw PCM bytes.
    """

    def __init__(self, cfg: FishAudioConfig):
        self.cfg = cfg
        self.ws: Optional[WebSocketClientProtocol] = None

    async def __aenter__(self):
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
        # Replace with the actual LIVE WebSocket endpoint
        self.ws = await websockets.connect(
            self.cfg.base_url,
            extra_headers=headers,
            ping_interval=self.cfg.keepalive_sec,
            ping_timeout=self.cfg.keepalive_sec + 5,
            max_size=2**22,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.ws is not None:
            await self.ws.close()
            self.ws = None

    def _resample_mono_s16le(self, pcm_bytes: bytes, from_sr: int, to_sr: int) -> bytes:
        if from_sr == to_sr:
            return pcm_bytes
        x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        # resample_poly for quality+速度
        gcd = np.gcd(from_sr, to_sr)
        up = to_sr // gcd
        down = from_sr // gcd
        y = resample_poly(x, up, down)
        y = np.clip(y, -32768, 32767).astype(np.int16)
        return y.tobytes()

    async def stream_tts(self, text: str) -> AsyncIterator[bytes]:
        if self.ws is None:
            raise RuntimeError("WebSocket is not connected")

        # TODO: Confirm FishAudio's start synthesis payload
        start_payload = {
            "type": "start",
            "voice_id": self.cfg.voice_id,
            "text": text,
            # 受信側（FishAudioの出力条件）。MuseTalk用のターゲットSRはローカルで変換。
            "sample_rate": self.cfg.incoming_sample_rate,
            "channels": 1,
            "format": self.cfg.audio_format,
            "chunk_ms": self.cfg.chunk_ms,
        }
        await self.ws.send(json.dumps(start_payload))

        # 固定長チャンクにパック
        bytes_per_sample = 2  # s16le
        frame_bytes = int(self.cfg.incoming_sample_rate * self.cfg.channels * bytes_per_sample * (self.cfg.chunk_ms / 1000.0))
        buf = bytearray()

        try:
            async for message in self.ws:
                # Expecting either JSON control or binary audio frames
                if isinstance(message, (bytes, bytearray)):
                    buf.extend(message)
                    # 固定長パック
                    while len(buf) >= frame_bytes:
                        raw = bytes(buf[:frame_bytes])
                        del buf[:frame_bytes]
                        # 必要ならSR変換
                        out = self._resample_mono_s16le(raw, self.cfg.incoming_sample_rate, self.cfg.sample_rate)
                        yield out
                else:
                    # Parse JSON control messages (progress, end, error)
                    try:
                        data = json.loads(message)
                    except Exception:
                        continue
                    if data.get("type") == "end":
                        break
                    if data.get("type") == "error":
                        raise RuntimeError(f"FishAudio error: {data}")
        finally:
            # Optionally send stop
            stop_payload = {"type": "stop"}
            try:
                await self.ws.send(json.dumps(stop_payload))
            except Exception:
                pass


async def tts_to_wav_bytes_http(text: str, cfg: FishAudioConfig) -> bytes:
    """
    Optional non-live/batch TTS via HTTP endpoint (placeholder).
    Adjust URL and payload according to FishAudio's HTTP TTS API.
    Returns WAV bytes.
    """
    url = os.environ.get("FISH_AUDIO_TTS_URL", "https://api.fishaudio.example/tts")
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
    payload = {
        "voice_id": cfg.voice_id,
        "text": text,
        "sample_rate": cfg.sample_rate,
        "channels": cfg.channels,
        "format": "wav",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                return await resp.read()
    except Exception:
        # オフライン/未設定時のフォールバック: 簡易サイン波でWAVを生成
        sr = cfg.sample_rate
        dur = max(1.5, min(6.0, 0.12 * len(text)))
        t = np.arange(int(sr * dur)) / sr
        tone = 0.15 * np.sin(2 * math.pi * 220 * t)
        data = (np.clip(tone, -1.0, 1.0) * 32767).astype(np.int16)
        import io, soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, data, sr, subtype="PCM_16", format="WAV")
        return buf.getvalue()
