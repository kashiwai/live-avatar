import argparse
import asyncio
import io
import os
from pathlib import Path
from typing import Optional

import click
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

from .fishaudio_client import FishAudioConfig, FishAudioLiveClient, tts_to_wav_bytes_http
from .fishaudio_sdk_client import FishAudioSDKConfig, FishAudioSDKClient
from .musetalk_runner import MuseTalkConfig, MuseTalkRunner
from .llm_agent import OpenAIConfig, OpenAIChatAgent


def _load_env():
    load_dotenv()


def _get_muse_cfg() -> MuseTalkConfig:
    repo = Path(os.environ.get("MUSE_TALK_REPO", "external/MuseTalk"))
    script = Path(os.environ.get("MUSE_TALK_INFER_SCRIPT", "scripts/inference.py"))
    sr = int(os.environ.get("MUSE_TALK_SR", "16000"))
    ch = int(os.environ.get("MUSE_TALK_CHANNELS", "1"))
    cmd_template = os.environ.get("MUSE_TALK_CMD_TEMPLATE")
    return MuseTalkConfig(repo_dir=repo, infer_script=script, sample_rate=sr, channels=ch, cmd_template=cmd_template)


def _get_fish_cfg() -> FishAudioConfig:
    return FishAudioConfig(
        api_key=(os.environ.get("FISH_API_KEY") or os.environ.get("FISH_AUDIO_API_KEY") or ""),
        base_url=os.environ.get("FISH_AUDIO_BASE_URL", ""),
        voice_id=os.environ.get("FISH_AUDIO_VOICE_ID", None),
        sample_rate=int(os.environ.get("MUSE_TALK_SR", "16000")),
        channels=int(os.environ.get("MUSE_TALK_CHANNELS", "1")),
        incoming_sample_rate=int(os.environ.get("FISH_AUDIO_INCOMING_SR", "24000")),
        chunk_ms=int(os.environ.get("FISH_AUDIO_CHUNK_MS", "40")),
        keepalive_sec=int(os.environ.get("FISH_AUDIO_KEEPALIVE_SEC", "20")),
        audio_format=os.environ.get("FISH_AUDIO_FORMAT", "pcm_s16le"),
    )


def wav_bytes_to_file(wav_bytes: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(wav_bytes)


@click.command()
@click.option("--mode", type=click.Choice(["batch", "live"]), required=True, help="Run mode")
@click.option("--image", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="Reference portrait image")
@click.option("--text", type=str, default=None, help="Input text (batch mode or LLM user input)")
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), default=Path("out/output.mp4"), help="Output video path (batch mode)")
@click.option("--use-openai/--no-use-openai", default=True, help="Use OpenAI (gpt-4o) to generate the response")
@click.option(
    "--persona",
    type=str,
    default=(
        "あなたは日本語で話す、関西弁の“大御所お笑い芸人風”アシスタントです。"
        "明るくハイテンポで、軽快なツッコミと軽いボケを適度に交え、相手を立てつつ礼節を保ってください。"
        "回答は簡潔に要点から述べます。実在の特定人物の固有の決め台詞や独自表現は模倣しません。"
        "文末は柔らかく、関西弁の相槌や感嘆（例: ほんまに？ ええやん、なんでやねん 等）を控えめに使います。"
        "過度にネガティブ・攻撃的にならず、場を明るくするトーンを心がけてください。"
    ),
    help="System prompt for LLM",
)
@click.option("--use-fishaudio-sdk/--no-use-fishaudio-sdk", default=True, help="FishAudio公式SDKを使用する")
def cli(mode: str, image: Path, text: Optional[str], out_path: Path, use_openai: bool, persona: str, use_fishaudio_sdk: bool):
    """Entry point for live/batch demos."""
    _load_env()

    muse_cfg = _get_muse_cfg()
    fish_cfg = _get_fish_cfg()
    runner = MuseTalkRunner(muse_cfg)

    if mode == "batch":
        # Determine final_text: either provided by user, or generated via OpenAI from user's input
        final_text = text
        if use_openai:
            if not text:
                raise click.BadParameter("--use-openai を指定した場合、--text にはユーザー入力（LLMへの指示）を指定してください")
            oa = OpenAIChatAgent(OpenAIConfig(
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
                temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0.6")),
            ))
            final_text = oa.chat(persona, text)

        if not final_text:
            raise click.BadParameter("TTSへ送るテキストが空です")

        # バッチはHTTP TTS経由（SDKがHTTP TTSを提供するなら差し替え可能）
        wav_bytes = asyncio.run(tts_to_wav_bytes_http(final_text, fish_cfg))
        tmp_wav = Path("out/tmp_tts.wav")
        wav_bytes_to_file(wav_bytes, tmp_wav)
        runner.synthesize_batch(image, tmp_wav, out_path)
        click.echo(f"Wrote video: {out_path}")
    else:
        asyncio.run(run_live(image, fish_cfg, runner, use_openai, persona, use_fishaudio_sdk))


async def run_live(image: Path, fish_cfg: FishAudioConfig, runner: MuseTalkRunner, use_openai: bool, persona: str, use_fishaudio_sdk: bool):
    """
    Experimental: open LIVE connection, prompt for text lines, stream audio, and attempt
    low-latency updates. The default runner only supports batch; extend synthesize_live_chunk
    for true real-time frames.
    """
    click.echo("LIVE mode: テキストを1行ずつ入力してください（空行で終了）。")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break
        text_to_say = line
        if use_openai:
            # LLMで応答を生成（簡易: 一括テキスト）。LIVEでの逐次化はFishAudioの仕様確定後に最適化。
            oa = OpenAIChatAgent(OpenAIConfig(
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
                temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0.6")),
            ))
            text_to_say = oa.chat(persona, line)
        if use_fishaudio_sdk:
            sdk_cfg = FishAudioSDKConfig(
                api_key=(os.environ.get("FISH_API_KEY") or os.environ.get("FISH_AUDIO_API_KEY") or ""),
                voice_id=os.environ.get("FISH_AUDIO_VOICE_ID", None),
                sample_rate=fish_cfg.sample_rate,
                channels=fish_cfg.channels,
            )
            async with FishAudioSDKClient(sdk_cfg) as fa:
                pcm_acc = bytearray()
                async for pcm in fa.stream_tts(text_to_say):
                    pcm_acc.extend(pcm)
                # 1ターン分の音声が揃ったら、暫定としてバッチ推論に回して動画化
                if pcm_acc:
                    tmp_wav = Path("out/live_tmp.wav")
                    arr = np.frombuffer(bytes(pcm_acc), dtype=np.int16)
                    sf.write(str(tmp_wav), arr, fish_cfg.sample_rate, subtype="PCM_16")
                    out_file = Path(f"out/live_{int(__import__('time').time())}.mp4")
                    runner.synthesize_batch(image, tmp_wav, out_file)
                    click.echo(f"Live clip saved: {out_file}")
        else:
            async with FishAudioLiveClient(fish_cfg) as fa:
                pcm_acc = bytearray()
                async for pcm in fa.stream_tts(text_to_say):
                    pcm_acc.extend(pcm)
                if pcm_acc:
                    tmp_wav = Path("out/live_tmp.wav")
                    arr = np.frombuffer(bytes(pcm_acc), dtype=np.int16)
                    sf.write(str(tmp_wav), arr, fish_cfg.sample_rate, subtype="PCM_16")
                    out_file = Path(f"out/live_{int(__import__('time').time())}.mp4")
                    runner.synthesize_batch(image, tmp_wav, out_file)
                    click.echo(f"Live clip saved: {out_file}")
    click.echo("LIVE session ended")


if __name__ == "__main__":
    cli()
