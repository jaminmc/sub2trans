# Sub2Trans - Subtitle to Transcript Converter

A powerful Python tool that converts SRT/VTT subtitle files into well-formatted transcripts with AI-enhanced paragraph grouping, intelligent headline generation, and video timestamp integration.

## Features

- **🤖 AI-Enhanced Processing**: Uses LM Studio integration for intelligent paragraph detection and headline generation
- **📝 Smart Paragraph Grouping**: Automatically groups consecutive subtitles into logical paragraphs using AI analysis
- **⏰ Timestamp Markers**: Each paragraph includes clickable timestamps that link to video positions
- **📋 Selective Headlines**: Generates headlines only for major sections (introduction, topic changes, conclusion)
- **🎥 Video Integration**: Creates clickable timestamps for YouTube and Vimeo videos
- **📊 Progress Tracking**: Visual progress bars for long file processing
- **🔄 Batch Processing**: Process multiple SRT files at once
- **⚙️ Flexible Options**: Verbose/quiet modes, headline control, and model selection

## Installation

### Basic Installation
```bash
git clone <repository-url>
cd sub2trans
```

### Dependencies
- **Python 3.6+**
- **tqdm** (for progress bars): `pip install tqdm`
- **requests** (for LM Studio): `pip install requests`

### LM Studio Setup (Optional)
For AI-enhanced processing, install and run [LM Studio](https://lmstudio.ai/):
1. Download and install LM Studio
2. Load a compatible model (recommended: `qwen/qwen3-4b-2507`)
3. Start the local server (usually `http://localhost:1234`)

## Usage

### Basic Usage

```bash
# Convert SRT to Markdown (creates input.md)
python sub2trans.py input.srt

# Save to specific file
python sub2trans.py input.srt -o output.md

# Add custom title
python sub2trans.py input.srt -t "My Video Transcript"
```

### AI-Enhanced Processing

```bash
# Use AI for better paragraph detection (requires LM Studio)
python sub2trans.py input.srt --model "qwen/qwen3-4b-2507"

# Wait for model to load
python sub2trans.py input.srt --wait-for-model

# Disable AI processing (use fallback rules)
python sub2trans.py input.srt --no-ai
```

### Video Integration

```bash
# Create clickable YouTube timestamps
python sub2trans.py input.srt -u "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Create clickable Vimeo timestamps
python sub2trans.py input.srt -u "https://vimeo.com/123456789"
```

### Batch Processing

```bash
# Process all SRT files in current directory
python sub2trans.py --batch

# Batch with specific settings
python sub2trans.py --batch --model "qwen/qwen3-4b-2507" --verbose
```

### Output Control

```bash
# Disable headlines (raw transcript)
python sub2trans.py input.srt --no-headlines

# Verbose output for debugging
python sub2trans.py input.srt --verbose

# Quiet mode for scripts
python sub2trans.py input.srt --quiet
```

### Advanced Options

```bash
# Custom paragraph grouping
python sub2trans.py input.srt --max-gap 5.0 --min-length 30

# Preview mode
python sub2trans.py input.srt --preview

# List available models
python sub2trans.py --list-models
```

## Command Line Options

### Core Options
- `input_file`: Path to the input SRT file (required unless using `--batch` or `--list-models`)
- `-o, --output`: Output Markdown file path (default: same as input with .md extension)
- `-t, --title`: Custom title for the document
- `--preview`: Preview the first few paragraphs without saving

### AI Processing
- `--model`: Model name to use with LM Studio (default: `qwen/qwen3-4b-2507`)
- `--lm-studio-url`: LM Studio API URL (default: `http://localhost:1234`)
- `--wait-for-model`: Wait for model to load before processing
- `--no-ai`: Disable AI processing and use fallback methods only
- `--list-models`: List available models in LM Studio and exit

### Output Control
- `--no-headlines`: Disable headline generation (output raw transcript)
- `-v, --verbose`: Enable verbose output
- `-q, --quiet`: Suppress non-essential output

### Processing Options
- `--max-gap`: Maximum gap in seconds between subtitles to group into paragraphs (default: 3.0)
- `--min-length`: Minimum paragraph length in characters (default: 10)
- `--batch`: Process all SRT files in current directory

### Video Integration
- `-u, --video-url`: Video URL (YouTube or Vimeo) to create clickable timestamps

## How It Works

### 1. SRT Parsing
- Parses SRT files with proper timestamp extraction
- Handles various encoding formats (UTF-8, Latin-1)
- Extracts subtitle text and timing information

### 2. AI-Enhanced Paragraph Grouping
The tool uses multiple approaches for paragraph detection:

**AI-Powered (with LM Studio):**
- **Smart Analysis**: Uses AI to detect natural conversation breaks
- **Content Understanding**: Analyzes context for better grouping
- **Progress Tracking**: Visual progress bars for long files

**Rule-Based Fallback:**
- **Timing gaps**: Subtitles with gaps longer than threshold start new paragraphs
- **Speaker changes**: Detects potential speaker changes based on text patterns
- **Topic changes**: Identifies topic shifts using keyword analysis

### 3. Selective Headline Generation
Generates headlines only for major sections:
- **Introduction**: First paragraph
- **Topic Changes**: When conversation shifts to new subjects
- **Conclusion**: Last paragraph
- **AI Detection**: Uses AI to identify significant topic transitions

### 4. Video Integration
- **YouTube**: Creates clickable timestamps (`?t=120s`)
- **Vimeo**: Creates clickable timestamps (`#t=120.521`)
- **Automatic Detection**: Identifies platform from URL

### 5. Markdown Formatting
- Creates clean, readable Markdown with proper structure
- Adds clickable timestamp markers for each paragraph
- Uses proper heading hierarchy
- Includes line breaks between paragraphs

## Example Output

### With Headlines (Default)
```markdown
# Interview with Dr. Keppel

## Introduction *(0:00)*
**0:00** I am rolling on the camera. You can't relate. That's funny. Cut. Just give us audio. We'll get him to say his name.

## Opening Discussion *(1:12)*
**1:12** So, we'll go over a list of questions and elaborate or make them as short as you need to.

## Questions & Discussion *(3:21)*
**3:21** But at that time, I didn't really know a whole lot about the difference between food supplements and man-made chemical supplements, right?
```

### With Video Integration
```markdown
# Interview with Dr. Keppel

## Introduction *(0:00)*
[**0:00**](https://www.youtube.com/watch?v=dQw4w9WgXcQ?t=0s) I am rolling on the camera. You can't relate. That's funny. Cut. Just give us audio. We'll get him to say his name.

## Opening Discussion *(1:12)*
[**1:12**](https://www.youtube.com/watch?v=dQw4w9WgXcQ?t=72s) So, we'll go over a list of questions and elaborate or make them as short as you need to.
```

### Without Headlines (`--no-headlines`)
```markdown
# Interview with Dr. Keppel

**0:00** I am rolling on the camera. You can't relate. That's funny. Cut. Just give us audio. We'll get him to say his name.

**1:12** So, we'll go over a list of questions and elaborate or make them as short as you need to.

**3:21** But at that time, I didn't really know a whole lot about the difference between food supplements and man-made chemical supplements, right?
```

## Performance Tips

### For Large Files
- Use `--verbose` to monitor progress
- Consider `--no-headlines` for faster processing
- Use `--wait-for-model` when switching models

### For Batch Processing
- Use `--batch` for multiple files
- Combine with `--quiet` for script automation
- Monitor with `--verbose` for debugging

### AI Model Selection
- **qwen/qwen3-4b-2507**: Fast, reliable (default)
- **gpt-oss-20b**: More powerful but slower
- Use `--list-models` to see available options

## Troubleshooting

### Common Issues
- **Model not loading**: Use `--wait-for-model` or check LM Studio
- **No AI processing**: Ensure LM Studio is running and model is loaded
- **Slow processing**: Try `--no-headlines` or use a faster model

### Getting Help
```bash
# See all options
python sub2trans.py --help

# List available models
python sub2trans.py --list-models

# Test with preview
python sub2trans.py input.srt --preview
```

## File Structure

```
sub2trans/
├── sub2trans.py       # Main converter script
├── README.md          # This documentation
├── .gitignore         # Git ignore file
└── output.md          # Generated Markdown file
```

## Requirements

- **Python 3.6+**
- **tqdm**: `pip install tqdm`
- **requests**: `pip install requests`
- **LM Studio** (optional, for AI features)

## License

This tool is provided as-is for converting SRT subtitle files to Markdown format.

