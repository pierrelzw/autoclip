# AutoClip

本地运行的 CLI 工具，自动去除口播视频中的口误、填充词和停顿。

## 功能

- **自动检测并移除**：填充词（um/uh/嗯/啊）、重复/口吃、错误起句、长停顿
- **本地运行**：ASR 使用 faster-whisper，LLM 默认使用 Ollama，零月费
- **可配置**：置信度阈值、移除类别、LLM 提供商均可调整
- **预览模式**：先分析后导出，避免不必要的重编码
- **支持 YouTube**：直接输入 URL 自动下载处理

## 依赖

- Python 3.11+
- FFmpeg + ffprobe（系统安装）
- Ollama（默认 LLM，可选 OpenAI API）

## 安装

```bash
# 克隆项目
git clone <repo-url> && cd autoclip

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装（开发模式）
pip install -e ".[dev]"
```

### 准备 Ollama

```bash
# 安装 Ollama: https://ollama.com
ollama serve                    # 启动服务
ollama pull qwen2.5:7b          # 拉取默认模型
```

> **注意**：首次运行时会自动下载 Whisper large-v3 模型（~3GB），请确保网络通畅。
> 如果系统设置了 HTTP 代理，模型下载可能失败。可临时清除代理：
> ```bash
> http_proxy="" https_proxy="" autoclip clean video.mp4 --preview
> ```

## 使用

```bash
# 预览分析（不导出）
autoclip clean video.mp4 --preview

# 清理并导出
autoclip clean video.mp4 -o ./output

# 只移除填充词和重复
autoclip clean video.mp4 --categories filler,repeat

# 提高置信度阈值（更保守）
autoclip clean video.mp4 --threshold 0.9

# 使用 OpenAI API
autoclip clean video.mp4 --llm openai

# YouTube URL
autoclip clean "https://youtube.com/watch?v=xxx"
```

### 命令选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-o, --output DIR` | 输出目录 | `./output` |
| `--threshold FLOAT` | 置信度阈值 (0.0-1.0) | `0.7` |
| `--categories LIST` | 移除类别 (逗号分隔) | `filler,repeat,false-start,pause` |
| `--llm PROVIDER` | LLM 提供商 (`ollama`/`openai`) | `ollama` |
| `--preview` | 只分析不导出 | - |
| `-v, --verbose` | 详细日志 | - |

### 输出

- `video_clean.mp4` — 清理后的视频
- `video_clean.json` — 分析报告（移除详情、时长统计）

## 配置

支持 YAML 配置文件，优先级：CLI 参数 > 项目 `autoclip.yaml` > 用户 `~/.config/autoclip/config.yaml` > 默认值。

参考 [autoclip.example.yaml](autoclip.example.yaml) 查看所有配置项。

## 口误分类

| CLI 类别 | 内部分类 | 检测方式 |
|----------|----------|----------|
| `filler` | filler | 词表匹配 (confidence=1.0) |
| `repeat` | stutter, repeat | LLM 分类 |
| `false-start` | false_start | LLM 分类 (阈值下限 0.85) |
| `pause` | long_pause | 启发式 (gap≥500ms, confidence=1.0) |

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 代码检查
ruff check src/ tests/      # Lint
mypy src/                    # 类型检查
pytest                       # 测试 (113 tests, 92% coverage)

# 或一次性检查
make check
```

## 技术栈

- **CLI**: Click + Rich
- **数据模型**: Pydantic v2
- **ASR**: faster-whisper (word_timestamps + Silero VAD)
- **LLM**: Ollama / OpenAI (via OpenAI SDK)
- **媒体**: FFmpeg + ffprobe + yt-dlp
- **测试**: pytest + mypy (strict) + ruff

## License

MIT
