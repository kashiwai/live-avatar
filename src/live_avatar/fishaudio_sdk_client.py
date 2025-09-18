import os
from dataclasses import dataclass
from typing import AsyncIterator, Optional, Any, AsyncGenerator
import inspect


@dataclass
class FishAudioSDKConfig:
    api_key: str
    voice_id: Optional[str] = None
    sample_rate: int = 16000
    channels: int = 1


class FishAudioSDKClient:
    """
    FishAudio 公式SDKを利用したLIVEクライアントの薄いラッパ。

    注意: 実際のSDKクラス名や呼び出しはFishAudioの最新ドキュメントに合わせて調整してください。
    このクラスは import 時にSDKを探し、見つからない場合は使用方法を案内して例外を投げます。
    """

    def __init__(self, cfg: FishAudioSDKConfig):
        self.cfg = cfg
        self._client = None

        # 代表的なモジュール名の探索。まずは公式配布名に合わせて優先。
        sdk = None
        for name in ("fish_audio_sdk", "fish_audio", "fishaudio", "fishaudio_sdk"):
            try:
                sdk = __import__(name)
                break
            except Exception:
                continue
        if sdk is None:
            raise ImportError(
                "FishAudio公式SDKが見つかりません。`pip install <公式SDKパッケージ>` を実行し、"
                "import 名を `src/live_avatar/fishaudio_sdk_client.py` 内で正しく指定してください。"
            )

        # Session 初期化（docs準拠）
        Session = getattr(sdk, "Session", None)
        api_key = (
            self.cfg.api_key
            or os.environ.get("FISH_API_KEY")
            or os.environ.get("FISH_AUDIO_API_KEY")
            or os.environ.get("FISH_AUDIO_KEY")
        )
        if not api_key:
            raise ValueError("FishAudioのAPIキーが未設定です。FISH_API_KEY もしくは FISH_AUDIO_API_KEY を設定してください。")
        base_url = os.environ.get("FISH_AUDIO_BASE_HTTP_URL") or os.environ.get("FISH_AUDIO_BASE_URL")
        self._session = None
        if Session is not None:
            try:
                if base_url:
                    # keyword apikey/base_url 形式
                    self._session = Session(apikey=api_key, base_url=base_url)
                else:
                    # 位置引数形式
                    try:
                        self._session = Session(api_key)
                    except TypeError:
                        self._session = Session(apikey=api_key)
            except Exception as e:
                # セッションがなくても後段でapi_keyを直接渡すフォールバック
                self._session = None

        # 公式SDKのWebSocketセッション（非同期）を使用
        ws = getattr(sdk, "websocket")
        self._AsyncWebSocketSession = getattr(ws, "AsyncWebSocketSession")
        self._TTSRequest = getattr(sdk.schemas, "TTSRequest")
        self._Backends = getattr(sdk.schemas, "Backends") if hasattr(sdk.schemas, "Backends") else None
        self._backend_default = "speech-1.5"
        self._ws_session = None

    async def __aenter__(self):
        # 例: await self._client.connect()
        if hasattr(self._client, "connect"):
            maybe = self._client.connect()
            if hasattr(maybe, "__await__"):
                await maybe
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # 例: await self._client.close()
        if hasattr(self._client, "close"):
            maybe = self._client.close()
            if hasattr(maybe, "__await__"):
                await maybe

    async def stream_tts(self, text: str) -> AsyncIterator[bytes]:
        """
        SDKのストリーミングTTSを呼び出し、PCM/WAVチャンクを順次yieldします。
        具体的なイベント名や返却型はSDKドキュメントに合わせて修正してください。
        """
        # AsyncWebSocketSession を開き、.tts(...) でストリーミングPCMを受信
        # chunk_length は100..300の範囲（ms）
        chunk_len = max(100, min(300, int(os.environ.get("FISH_AUDIO_CHUNK_MS", "200"))))
        # FishAudioはreference_idに話者/モデルIDを指定
        req = self._TTSRequest(
            text="",
            format="pcm",
            sample_rate=self.cfg.sample_rate,
            chunk_length=chunk_len,
            reference_id=self.cfg.voice_id,
            normalize=True,
            latency="balanced",
            temperature=0.6,
            top_p=0.95,
        )

        async def _text_stream() -> AsyncGenerator[str, None]:
            yield text

        # base_url は HTTPベース（例: https://api.fish.audio）
        base_http = os.environ.get("FISH_AUDIO_BASE_HTTP_URL") or os.environ.get("FISH_AUDIO_BASE_URL") or "https://api.fish.audio"
        async with self._AsyncWebSocketSession(self.cfg.api_key, base_url=base_http) as ws:
            async for audio in ws.tts(request=req, text_stream=_text_stream(), backend=self._backend_default):
                if isinstance(audio, (bytes, bytearray)):
                    yield bytes(audio)
                else:
                    # 念のため辞書形式も対応
                    try:
                        buf = audio.get("audio")
                        if buf:
                            yield buf
                    except Exception:
                        continue
