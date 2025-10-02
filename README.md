# Sub2Trans - Advanced Subtitle to Transcript Converter

A powerful Python tool that converts subtitle files into well-formatted transcripts with AI-enhanced processing, multi-platform video support, and comprehensive configuration options.

## ✨ Features

### 🤖 **Multi-AI Provider Support**
- **LM Studio** (local) - Free, runs locally with your own models
- **OpenAI** (ChatGPT) - Cloud-based, requires API key
- **Anthropic** (Claude) - Cloud-based, requires API key  
- **Grok** (X.AI) - Cloud-based, requires API key
- **Google** (Gemini) - Cloud-based, requires API key

### 📝 **Comprehensive Subtitle Format Support**
- **SRT** (SubRip Subtitle) - Most common format
- **VTT** (WebVTT) - Web video format
- **ASS/SSA** (Advanced SubStation Alpha) - Anime and styled subtitles
- **SBV** (SubViewer) - YouTube format
- **TTML/DFXP** (Timed Text Markup Language) - Broadcast format
- **SAMI** (Synchronized Accessible Media Interchange) - Microsoft format
- **LRC** (Lyric File) - Music synchronization

### 🎥 **Multi-Platform Video Integration**
- **YouTube** - Clickable timestamps with `?t=120s` format
- **Vimeo** - Clickable timestamps with `#t=120.521` format
- **Twitch** - Clickable timestamps with `?t=120s` format
- **Dailymotion** - Clickable timestamps with `?start=120` format
- **Facebook** - Clickable timestamps with `?t=120` format
- **Rumble** - Clickable timestamps with `?t=120` format
- **Odysee** - Clickable timestamps with `?t=120` format
- **TikTok, Instagram, Twitter, LinkedIn** - Base URLs (no timestamp support)

### ⚙️ **Configuration System**
- **Global Configuration**: `~/sub2trans_config.json` (accessible from anywhere)
- **Interactive Setup**: `--setup-config` for guided configuration
- **Default Output Format**: Set your preferred format once
- **AI Provider Management**: Configure multiple providers with API keys
- **Processing Settings**: Customize paragraph grouping and processing options

### 📊 **Advanced Processing**
- **Smart Paragraph Grouping**: AI-powered paragraph detection
- **Selective Headlines**: Generates headlines only for major sections
- **Progress Tracking**: Visual progress bars for long files
- **Batch Processing**: Process multiple files at once
- **Multiple Output Formats**: Markdown, HTML, PDF, DOCX, ODT, RTF

### 🐍 **Python-Native Conversion**
- **No External Dependencies**: No Pandoc or system tools required
- **Pure Python**: All conversion happens in Python
- **Optional Libraries**: Install only what you need
- **Automatic Detection**: Checks which formats are available
- **Graceful Fallback**: Falls back to markdown if conversion fails
- **Cross-Platform**: Works on any system with Python

## 🚀 Installation

### Basic Installation
```bash
# Clone the repository
git clone https://github.com/jaminmc/sub2trans.git
cd sub2trans

# Windows users can also download the ZIP file and extract it
```

### Core Dependencies (Required)
```bash
# Install core dependencies
pip install tqdm requests

# Windows users: If you get permission errors, use:
# pip install --user tqdm requests
```

### Optional Dependencies (For Output Formats)
```bash
# For HTML output
pip install markdown

# For PDF output (recommended)
pip install weasyprint

# For PDF output (alternative)
pip install reportlab

# For DOCX output (Microsoft Word)
pip install python-docx

# For ODT output (OpenDocument)
pip install odfpy

# Install all optional dependencies at once
pip install markdown weasyprint python-docx odfpy
```

### AI Provider Dependencies
```bash
# For local AI (LM Studio)
# Download from: https://lmstudio.ai/

# For cloud AI providers, you only need API keys:
# - OpenAI: Get API key from https://platform.openai.com/
# - Anthropic: Get API key from https://console.anthropic.com/
# - Grok: Get API key from https://console.x.ai/
# - Google: Get API key from https://aistudio.google.com/
```

### Windows-Specific Notes
- **Path Handling**: All paths use `pathlib.Path` for cross-platform compatibility
- **Configuration**: Stored in `%USERPROFILE%\sub2trans_config.json` on Windows
- **Python Path**: Make sure Python is in your PATH or use `python` instead of `python3`
- **Permissions**: Use `--user` flag if you get permission errors: `pip install --user package_name`

### Format Availability by Dependencies

| Output Format | Required Library | Always Available |
|---------------|------------------|------------------|
| **Markdown** | None | ✅ Yes |
| **RTF** | None | ✅ Yes |
| **HTML** | `pip install markdown` | ❌ No |
| **PDF** | `pip install weasyprint` or `pip install reportlab` | ❌ No |
| **DOCX** | `pip install python-docx` | ❌ No |
| **ODT** | `pip install odfpy` | ❌ No |

### Quick Installation Guide
```bash
# 1. Install core dependencies
pip install tqdm requests

# 2. Check what formats are available
python sub2trans.py --list-formats

# 3. Install additional formats as needed
pip install markdown weasyprint python-docx

# 4. Check formats again
python sub2trans.py --list-formats

# 5. Setup configuration
python sub2trans.py --setup-config
```

### Check Available Formats
```bash
# See which formats are available with your current setup
python sub2trans.py --list-formats
```

### No External Dependencies Required
- **Markdown & RTF**: Always available (no dependencies)
- **Python-Native**: All conversion happens in Python
- **Optional Libraries**: Install only what you need
- **Graceful Fallback**: Falls back to markdown if conversion fails

## 📖 Usage

### Quick Start
```bash
# Convert subtitle to transcript
python sub2trans.py input.srt

# With video timestamps
python sub2trans.py input.srt -u "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Convert to PDF
python sub2trans.py input.srt -f pdf
```

### Windows Usage
```cmd
# Windows Command Prompt
python sub2trans.py input.srt

# If python3 doesn't work, try:
python sub2trans.py input.srt

# PowerShell
python sub2trans.py input.srt -f pdf

# Alternative if Python is not in PATH:
C:\Python39\python.exe sub2trans.py input.srt
```

### Configuration Setup
```bash
# Interactive configuration setup
python sub2trans.py --setup-config

# View current configuration
python sub2trans.py --show-config

# Use custom config file
python sub2trans.py input.srt --config /path/to/config.json
```

### AI Provider Selection
```bash
# Use OpenAI ChatGPT
python sub2trans.py input.srt --ai-provider openai

# Use Anthropic Claude
python sub2trans.py input.srt --ai-provider anthropic

# Use Grok
python sub2trans.py input.srt --ai-provider grok

# Use Google Gemini
python sub2trans.py input.srt --ai-provider google

# Use LM Studio (default)
python sub2trans.py input.srt --ai-provider lm_studio
```

### Subtitle Format Examples
```bash
# SRT files
python sub2trans.py input.srt

# VTT files
python sub2trans.py input.vtt

# ASS/SSA files (anime subtitles)
python sub2trans.py anime.ass

# SBV files (YouTube subtitles)
python sub2trans.py youtube.sbv

# TTML files (broadcast subtitles)
python sub2trans.py broadcast.ttml

# SAMI files (multilingual content)
python sub2trans.py multilingual.smi

# LRC files (lyrics)
python sub2trans.py song.lrc
```

### Output Format Examples
```bash
# Markdown (default, no dependencies)
python sub2trans.py input.srt

# HTML (requires: pip install markdown)
python sub2trans.py input.srt -f html

# PDF (requires: pip install weasyprint)
python sub2trans.py input.srt -f pdf

# Microsoft Word (requires: pip install python-docx)
python sub2trans.py input.srt -f docx

# OpenDocument (requires: pip install odfpy)
python sub2trans.py input.srt -f odt

# Rich Text Format (no dependencies)
python sub2trans.py input.srt -f rtf
```

### Check Available Formats
```bash
# See which formats are available with your installed libraries
python sub2trans.py --list-formats
```

### Video Platform Examples
```bash
# YouTube
python sub2trans.py input.srt -u "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Vimeo
python sub2trans.py input.srt -u "https://vimeo.com/123456789"

# Twitch
python sub2trans.py input.srt -u "https://www.twitch.tv/videos/123456789"

# Dailymotion
python sub2trans.py input.srt -u "https://www.dailymotion.com/video/x123456"

# Facebook
python sub2trans.py input.srt -u "https://www.facebook.com/watch/?v=123456789"

# Rumble
python sub2trans.py input.srt -u "https://rumble.com/v123456789"

# Odysee
python sub2trans.py input.srt -u "https://odysee.com/@channel/video-title"
```

### Advanced Options
```bash
# Custom paragraph grouping
python sub2trans.py input.srt --max-gap 5.0 --min-length 30

# Disable headlines
python sub2trans.py input.srt --no-headlines

# Verbose output
python sub2trans.py input.srt --verbose

# Batch processing
python sub2trans.py --batch

# Preview mode
python sub2trans.py input.srt --preview
```

## ⚙️ Configuration

### Configuration File Location
- **Default**: `~/sub2trans_config.json` (global, accessible from anywhere)
- **Custom**: Use `--config /path/to/config.json`
- **Global Access**: Works from any directory on your system

### Interactive Setup
```bash
python sub2trans.py --setup-config
```

This will guide you through:
- Setting default AI provider
- Configuring API keys for cloud providers
- Setting default output format
- Configuring processing options

### Configuration File Structure
```json
{
  "ai_providers": {
    "lm_studio": {
      "enabled": true,
      "base_url": "http://localhost:1234",
      "default_model": "qwen/qwen3-4b-2507",
      "timeout": 120,
      "max_retries": 3
    },
    "openai": {
      "enabled": false,
      "api_key": "",
      "base_url": "https://api.openai.com/v1",
      "default_model": "gpt-3.5-turbo",
      "timeout": 60,
      "max_retries": 3
    },
    "anthropic": {
      "enabled": false,
      "api_key": "",
      "base_url": "https://api.anthropic.com",
      "default_model": "claude-3-haiku-20240307",
      "timeout": 60,
      "max_retries": 3
    },
    "grok": {
      "enabled": false,
      "api_key": "",
      "base_url": "https://api.x.ai/v1",
      "default_model": "grok-beta",
      "timeout": 60,
      "max_retries": 3
    },
    "google": {
      "enabled": false,
      "api_key": "",
      "base_url": "https://generativelanguage.googleapis.com/v1beta",
      "default_model": "gemini-pro",
      "timeout": 60,
      "max_retries": 3
    }
  },
  "default_provider": "lm_studio",
  "processing": {
    "max_gap_seconds": 3.0,
    "min_paragraph_length": 10,
    "wait_for_model": false,
    "no_headlines": false,
    "verbose": false
  },
  "output": {
    "default_format": "markdown",
    "pandoc_path": "pandoc",
    "auto_detect_format": true,
    "preferred_formats": ["markdown", "html", "pdf", "docx"]
  }
}
```

## 📋 Command Line Options

### Core Options
- `input_file`: Path to input subtitle file (required unless using `--batch` or `--list-models`)
- `-o, --output`: Output file path (default: same as input with appropriate extension)
- `-t, --title`: Custom title for the document
- `--preview`: Preview the first few paragraphs without saving

### AI Configuration
- `--ai-provider`: AI provider to use (lm_studio, openai, anthropic, grok, google)
- `--model`: Model name to use (overrides config)
- `--lm-studio-url`: LM Studio API URL (default: http://localhost:1234)
- `--wait-for-model`: Wait for model to load before processing
- `--no-ai`: Disable AI processing and use fallback methods only
- `--list-models`: List available models in LM Studio and exit

### Configuration Management
- `--config`: Path to configuration file (default: ~/.sub2trans_config.json)
- `--setup-config`: Setup configuration file interactively
- `--show-config`: Show current configuration and exit

### Output Control
- `-f, --format`: Output format (markdown, html, pdf, docx, odt, rtf)
- `--no-headlines`: Disable headline generation (output raw transcript)
- `--pandoc-path`: Path to pandoc executable (default: pandoc)
- `-v, --verbose`: Enable verbose output
- `-q, --quiet`: Suppress non-essential output

### Processing Options
- `--max-gap`: Maximum gap in seconds between subtitles to group into paragraphs (default: 3.0)
- `--min-length`: Minimum paragraph length in characters (default: 10)
- `--batch`: Process all subtitle files in current directory

### Video Integration
- `-u, --video-url`: Video URL to create clickable timestamps

## 🎯 How It Works

### 1. Subtitle Parsing
- **Multi-Format Support**: Automatically detects and parses various subtitle formats
- **Encoding Handling**: Supports UTF-8, Latin-1, and other encodings
- **Time Format Conversion**: Converts different time formats to standard seconds
- **Content Cleaning**: Removes formatting codes and normalizes text

### 2. AI-Enhanced Processing
**AI-Powered (with configured provider):**
- **Smart Analysis**: Uses AI to detect natural conversation breaks
- **Content Understanding**: Analyzes context for better paragraph grouping
- **Headline Generation**: Creates meaningful section titles
- **Progress Tracking**: Visual progress bars for long files

**Rule-Based Fallback:**
- **Timing Analysis**: Groups subtitles based on time gaps
- **Speaker Detection**: Identifies potential speaker changes
- **Topic Analysis**: Detects topic shifts using keyword analysis
- **Pattern Recognition**: Uses linguistic patterns for grouping

### 3. Video Integration
- **Platform Detection**: Automatically identifies video platform from URL
- **Timestamp Generation**: Creates platform-specific timestamp URLs
- **Clickable Links**: Generates clickable timestamp markers in output
- **Fallback Support**: Handles platforms without timestamp support

### 4. Output Formatting
- **Format Detection**: Automatically detects output format from file extension
- **Pandoc Integration**: Uses Pandoc for advanced format conversion
- **Clean Markdown**: Generates well-structured Markdown output
- **Timestamp Integration**: Embeds clickable timestamps in output

## 📄 Example Output

### With Headlines and Video Integration
```markdown
# Interview with Dr. Keppel

## Introduction *(0:00)*
[**0:00**](https://www.youtube.com/watch?v=dQw4w9WgXcQ?t=0s) I am rolling on the camera. You can't relate. That's funny. Cut. Just give us audio. We'll get him to say his name.

## Opening Discussion *(1:12)*
[**1:12**](https://www.youtube.com/watch?v=dQw4w9WgXcQ?t=72s) So, we'll go over a list of questions and elaborate or make them as short as you need to.

## Questions & Discussion *(3:21)*
[**3:21**](https://www.youtube.com/watch?v=dQw4w9WgXcQ?t=201s) But at that time, I didn't really know a whole lot about the difference between food supplements and man-made chemical supplements, right?
```

### Without Headlines (`--no-headlines`)
```markdown
# Interview with Dr. Keppel

**0:00** I am rolling on the camera. You can't relate. That's funny. Cut. Just give us audio. We'll get him to say his name.

**1:12** So, we'll go over a list of questions and elaborate or make them as short as you need to.

**3:21** But at that time, I didn't really know a whole lot about the difference between food supplements and man-made chemical supplements, right?
```

## 🔧 Performance Tips

### For Large Files
- Use `--verbose` to monitor progress
- Consider `--no-headlines` for faster processing
- Use `--wait-for-model` when switching AI models

### For Batch Processing
- Use `--batch` for multiple files
- Combine with `--quiet` for script automation
- Monitor with `--verbose` for debugging

### AI Provider Selection
- **LM Studio**: Free, local processing, good for privacy
- **OpenAI**: Fast, reliable, requires API key
- **Anthropic**: High quality, good for complex content
- **Grok**: Good for creative content, requires API key
- **Google**: Fast and efficient, requires API key

## 🐛 Troubleshooting

### Common Issues
- **AI not available**: Check provider configuration and API keys
- **Format conversion fails**: Install required Python libraries (see `--list-formats`)
- **Slow processing**: Try `--no-headlines` or use a faster AI provider
- **Video timestamps not working**: Check URL format and platform support
- **Missing output formats**: Install optional libraries (e.g., `pip install weasyprint` for PDF)

### Missing Dependencies Issues
- **"No conversion libraries available"**: Install required libraries (see `--list-formats`)
- **"HTML conversion failed"**: Run `pip install markdown`
- **"PDF conversion failed"**: Run `pip install weasyprint` (recommended) or `pip install reportlab`
- **"DOCX conversion failed"**: Run `pip install python-docx`
- **"ODT conversion failed"**: Run `pip install odfpy`
- **Format not available**: Check `python sub2trans.py --list-formats` to see what's installed

### Windows-Specific Issues
- **Permission denied**: Use `pip install --user package_name` instead of `pip install package_name`
- **Python not found**: Make sure Python is in your PATH, or use full path to python.exe
- **Path issues**: All paths are handled automatically with `pathlib.Path`
- **Configuration location**: Check `%USERPROFILE%\sub2trans_config.json` for config file
- **WeasyPrint issues**: On Windows, you may need to install additional dependencies for PDF generation

### Getting Help
```bash
# See all options
python sub2trans.py --help

# View current configuration
python sub2trans.py --show-config

# Test with preview
python sub2trans.py input.srt --preview

# List available models (LM Studio only)
python sub2trans.py --list-models
```

## 📁 File Structure

```
sub2trans/
├── sub2trans.py              # Main converter script
├── README.md                 # This documentation
├── .gitignore               # Git ignore file
├── sub2trans_config.json    # Configuration file (auto-created)
└── output.md                # Generated transcript file
```

## 📦 Requirements

### Core Dependencies
- **Python 3.6+**
- **tqdm**: `pip install tqdm`
- **requests**: `pip install requests`

### Optional Dependencies (Python-Native)
- **markdown**: For HTML output (`pip install markdown`)
- **weasyprint**: For PDF output (`pip install weasyprint`)
- **python-docx**: For DOCX output (`pip install python-docx`)
- **odfpy**: For ODT output (`pip install odfpy`)
- **LM Studio**: For local AI processing
- **API Keys**: For cloud AI providers (OpenAI, Anthropic, Grok, Google)

### No External Dependencies
- **No Pandoc required**: All conversion happens in Python
- **No system dependencies**: Pure Python implementation
- **Optional libraries**: Install only what you need

## 🚀 Quick Start Examples

### Basic Conversion
```bash
python sub2trans.py input.srt
```

### With Video Timestamps
```bash
python sub2trans.py input.srt -u "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

### Convert to PDF
```bash
python sub2trans.py input.srt -f pdf
```

### Use OpenAI ChatGPT
```bash
python sub2trans.py input.srt --ai-provider openai
```

### Setup Configuration
```bash
# Interactive setup (creates ~/sub2trans_config.json)
python sub2trans.py --setup-config

# View current configuration
python sub2trans.py --show-config

# Check available output formats
python sub2trans.py --list-formats
```

## 📄 License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.

### Key Points:
- **Commercial Use**: ✅ Allowed
- **Modification**: ✅ Allowed  
- **Distribution**: ✅ Allowed
- **Private Use**: ✅ Allowed
- **Attribution**: ✅ Required
- **Liability**: ❌ No warranty provided

### Quick Summary:
You are free to use, modify, and distribute this software for any purpose, including commercial use, as long as you include the original copyright notice and license text. This is a simple, permissive license that's very developer-friendly.

---

## 🎯 Why Sub2Trans?

### **🚀 Modern Python-Native Approach**
- **No External Dependencies**: No need to install Pandoc or system tools
- **Pure Python**: All conversion happens in Python for maximum compatibility
- **Optional Libraries**: Install only the formats you need
- **Cross-Platform**: Works on Windows, macOS, Linux without system dependencies

### **🤖 Multi-AI Provider Support**
- **Local AI**: Use LM Studio with your own models (free)
- **Cloud AI**: Integrate with OpenAI, Anthropic, Grok, Google
- **Flexible**: Switch between providers based on your needs
- **Cost-Effective**: Choose between free local or paid cloud options

### **📝 Comprehensive Format Support**
- **8 Subtitle Formats**: SRT, VTT, ASS, SSA, SBV, TTML, SAMI, LRC
- **6 Output Formats**: Markdown, HTML, PDF, DOCX, ODT, RTF
- **10+ Video Platforms**: YouTube, Vimeo, Twitch, Facebook, and more
- **Smart Detection**: Automatically detects formats and platforms

### **⚙️ Professional Configuration**
- **Global Settings**: Configuration accessible from anywhere (`~/sub2trans_config.json`)
- **Interactive Setup**: Guided configuration with `--setup-config`
- **Default Formats**: Set your preferred output format once
- **AI Management**: Configure multiple AI providers with API keys

---

**Sub2Trans** - Transform your subtitles into professional transcripts with AI-powered intelligence and multi-platform support.