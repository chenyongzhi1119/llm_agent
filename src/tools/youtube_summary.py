"""YouTube/Podcast summary tool - extract subtitles and summarize"""

import subprocess
import json

from src.tools.base import Tool
from config.settings import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
from openai import OpenAI


class YouTubeSummaryTool(Tool):
    name = "youtube_summary"
    description = "Summarize a YouTube video from its URL. Returns overview and key points."
    parameters = {
        "url": {"description": "YouTube URL", "required": True},
    }

    def execute(self, **kwargs) -> str:
        url = kwargs.get("url", "")
        if not url:
            return "Error: please provide a YouTube URL"

        # Step 1: Extract subtitles using yt-dlp
        subtitles = self._extract_subtitles(url)
        if subtitles.startswith("Error"):
            return subtitles

        # Step 2: Truncate if too long (keep within LLM context)
        max_chars = 8000
        if len(subtitles) > max_chars:
            subtitles = subtitles[:max_chars] + "\n... (truncated)"

        # Step 3: Summarize with LLM
        return self._summarize(subtitles)

    def _extract_subtitles(self, url: str) -> str:
        """Extract subtitles/auto-captions from YouTube video"""
        try:
            # Try to get subtitle info first
            result = subprocess.run(
                [
                    "yt-dlp",
                    "--skip-download",
                    "--write-auto-sub",
                    "--sub-lang", "en,zh",
                    "--sub-format", "json3",
                    "--dump-json",
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return f"Error: yt-dlp failed: {result.stderr[:200]}"

            video_info = json.loads(result.stdout)
            title = video_info.get("title", "Unknown")

            # Try to extract subtitles text
            subtitles_text = self._download_subtitles(url)
            if subtitles_text:
                return f"Video title: {title}\n\nSubtitles:\n{subtitles_text}"

            # Fallback: use video description
            description = video_info.get("description", "")
            if description:
                return f"Video title: {title}\n\nDescription:\n{description}"

            return f"Error: no subtitles or description found for this video"

        except subprocess.TimeoutExpired:
            return "Error: yt-dlp timed out"
        except Exception as e:
            return f"Error extracting video info: {e}"

    def _download_subtitles(self, url: str) -> str:
        """Download and parse subtitles"""
        import tempfile
        import os
        import glob

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "sub")
            result = subprocess.run(
                [
                    "yt-dlp",
                    "--skip-download",
                    "--write-auto-sub",
                    "--write-sub",
                    "--sub-lang", "en,zh",
                    "--sub-format", "vtt",
                    "--output", output_path,
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Find any downloaded subtitle files
            vtt_files = glob.glob(os.path.join(tmpdir, "*.vtt"))
            if not vtt_files:
                return ""

            # Parse VTT file to plain text
            with open(vtt_files[0], "r", encoding="utf-8") as f:
                lines = f.readlines()

            text_lines = []
            seen = set()
            for line in lines:
                line = line.strip()
                # Skip VTT headers, timestamps, and empty lines
                if not line or "-->" in line or line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
                    continue
                # Skip HTML tags
                clean = line.replace("<c>", "").replace("</c>", "")
                if clean.startswith("<"):
                    continue
                # Deduplicate
                if clean not in seen:
                    seen.add(clean)
                    text_lines.append(clean)

            return "\n".join(text_lines)

    def _summarize(self, content: str) -> str:
        """Summarize content using LLM"""
        try:
            client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a summarization assistant. Summarize the following video content concisely. Include: 1) A brief overview (2-3 sentences), 2) Key points (bullet list), 3) Main takeaways.",
                    },
                    {"role": "user", "content": content},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content or "Failed to generate summary"
        except Exception as e:
            return f"Summarization error: {e}"
