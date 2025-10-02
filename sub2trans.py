#!/usr/bin/env python3
"""
SRT to Markdown Converter

Converts SRT subtitle files to well-formatted Markdown with:
- Paragraph grouping based on timing and content
- Timestamp markers for paragraph starts
- Automatic headline and subheadline generation
- Clean, readable formatting
"""

import re
import argparse
import sys
import json
import requests
import time
import subprocess
import tempfile
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
from urllib.parse import urlparse, parse_qs

# Global verbosity setting
VERBOSE = False

def log(message: str, level: str = "INFO"):
    """Log message based on verbosity level"""
    if VERBOSE or level == "ERROR":
        print(message)

def log_progress(message: str):
    """Log progress message (always shown)"""
    print(message)

def handle_error(error: Exception, context: str = "", fallback_action: str = None) -> bool:
    """Handle errors gracefully with fallback options"""
    error_msg = f"Error in {context}: {str(error)}" if context else str(error)
    log(error_msg, "ERROR")
    
    if fallback_action:
        log(f"Attempting fallback: {fallback_action}")
        return True
    return False


class VideoURLHandler:
    """Handles video URL processing and timestamp generation"""
    
    def __init__(self, video_url: str = None):
        self.video_url = video_url
        self.platform = self._detect_platform()
        self.base_url = self._get_base_url()
    
    def _detect_platform(self) -> Optional[str]:
        """Detect video platform from URL"""
        if not self.video_url:
            return None
        
        url = self.video_url.lower()
        
        if 'youtube.com' in url or 'youtu.be' in url:
            return 'youtube'
        elif 'vimeo.com' in url:
            return 'vimeo'
        else:
            return 'unknown'
    
    def _get_base_url(self) -> Optional[str]:
        """Get the base video URL without existing timestamps"""
        if not self.video_url or self.platform == 'unknown':
            return None
        
        if self.platform == 'youtube':
            # Clean YouTube URL
            if 'youtu.be' in self.video_url:
                video_id = self.video_url.split('/')[-1].split('?')[0]
                return f"https://www.youtube.com/watch?v={video_id}"
            else:
                # Remove existing timestamp parameters
                parsed = urlparse(self.video_url)
                if parsed.query:
                    params = parse_qs(parsed.query)
                    if 'v' in params:
                        return f"https://www.youtube.com/watch?v={params['v'][0]}"
                return self.video_url
        
        elif self.platform == 'vimeo':
            # Clean Vimeo URL
            parsed = urlparse(self.video_url)
            return f"https://vimeo.com{parsed.path}"
        
        return self.video_url
    
    def get_timestamp_url(self, seconds: float) -> Optional[str]:
        """Generate timestamped URL for the given seconds"""
        if not self.base_url:
            return None
        
        if self.platform == 'youtube':
            return f"{self.base_url}&t={int(seconds)}s"
        elif self.platform == 'vimeo':
            return f"{self.base_url}#t={seconds:.3f}"
        else:
            return None
    
    def is_valid(self) -> bool:
        """Check if the video URL is valid and supported"""
        return self.platform in ['youtube', 'vimeo']


class LMStudioClient:
    """Client for communicating with LM Studio API"""
    
    def __init__(self, base_url: str = "http://localhost:1234", model: str = "qwen/qwen3-4b-2507"):
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def generate_text(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7, retries: int = 3) -> str:
        """Generate text using LM Studio API with retry mechanism"""
        for attempt in range(retries):
            try:
                payload = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False
                }
                
                response = self.session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
                    if content:
                        return content
                    else:
                        log(f"Empty response from model (attempt {attempt + 1}/{retries})")
                else:
                    log(f"LM Studio API error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.Timeout:
                log(f"Request timeout (attempt {attempt + 1}/{retries})")
            except requests.exceptions.ConnectionError:
                log(f"Connection error (attempt {attempt + 1}/{retries})")
            except Exception as e:
                log(f"Unexpected error: {e}")
            
            if attempt < retries - 1:
                log(f"Retrying in 2 seconds...")
                time.sleep(2)
        
        log("All retry attempts failed")
        return ""
    
    def is_available(self) -> bool:
        """Check if LM Studio is available"""
        try:
            response = self.session.get(f"{self.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def wait_for_model_ready(self, timeout: int = 60) -> bool:
        """Wait for the model to be ready to accept requests"""
        print("Waiting for model to load...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Try a simple test request
                test_response = self.generate_text("Hello", max_tokens=5, temperature=0.1)
                if test_response and test_response.strip():
                    print("✓ Model is ready")
                    return True
            except:
                pass
            
            print(".", end="", flush=True)
            time.sleep(2)
        
        print(f"\n⚠ Model not ready after {timeout} seconds")
        return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from LM Studio"""
        try:
            response = self.session.get(f"{self.base_url}/v1/models", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                models = []
                for model in models_data.get('data', []):
                    model_id = model.get('id', '')
                    if model_id:
                        models.append(model_id)
                return models
            return []
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available"""
        available_models = self.get_available_models()
        return model_name in available_models
    
    def get_model_info(self) -> Dict:
        """Get detailed information about the current model"""
        try:
            response = self.session.get(f"{self.base_url}/v1/models", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                for model in models_data.get('data', []):
                    if model.get('id') == self.model:
                        return model
            return {}
        except Exception as e:
            print(f"Error fetching model info: {e}")
            return {}
    
    def get_context_window(self) -> int:
        """Get the context window size for the current model"""
        model_info = self.get_model_info()
        # Try different possible fields for context window
        context_fields = ['context_length', 'max_tokens', 'max_context_length', 'context_window']
        for field in context_fields:
            if field in model_info:
                return int(model_info[field])
        
        # Default fallbacks based on common model sizes
        if 'gpt-oss-20b' in self.model:
            return 8192
        elif 'gpt-oss-7b' in self.model:
            return 4096
        elif 'llama-2' in self.model:
            return 4096
        elif 'mistral' in self.model:
            return 32768
        else:
            return 4096  # Conservative default
    
    def estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count (4 chars per token average)"""
        return len(text) // 4


def select_lm_studio_model(lm_client: LMStudioClient, preferred_model: str = "qwen/qwen3-4b-2507") -> str:
    """Select an available model from LM Studio, with fallback options"""
    if not lm_client.is_available():
        return None
    
    available_models = lm_client.get_available_models()
    if not available_models:
        print("No models found in LM Studio")
        return None
    
    # Check if preferred model is available
    if preferred_model in available_models:
        print(f"✓ Using preferred model: {preferred_model}")
        # Wait for model to be ready
        if lm_client.wait_for_model_ready():
            return preferred_model
        else:
            print("⚠ Model failed to load, trying fallback...")
    else:
        print(f"⚠ Preferred model '{preferred_model}' not available")
        print("Available models:")
        for i, model in enumerate(available_models, 1):
            print(f"  {i}. {model}")
    
    # Try to find a reasonable fallback
    fallback_models = [
        "qwen/qwen3-4b-2507", "qwen2.5-0.5b-instruct-mlx", "qwen3-coder-30b-a3b-instruct-mlx",
        "gpt-oss-20b", "gpt-oss-7b", "gpt-oss-3b", "gpt-oss-1b",
        "llama-2-70b", "llama-2-13b", "llama-2-7b",
        "codellama-34b", "codellama-13b", "codellama-7b",
        "mistral-7b", "mixtral-8x7b"
    ]
    
    for fallback in fallback_models:
        if fallback in available_models:
            print(f"✓ Trying fallback model: {fallback}")
            lm_client.model = fallback
            if lm_client.wait_for_model_ready():
                return fallback
            else:
                print(f"⚠ {fallback} failed to load, trying next...")
    
    # If no fallback found, use the first available model
    selected_model = available_models[0]
    print(f"✓ Trying first available model: {selected_model}")
    lm_client.model = selected_model
    if lm_client.wait_for_model_ready():
        return selected_model
    
    print("⚠ No models could be loaded successfully")
    return None


@dataclass
class SubtitleBlock:
    """Represents a single subtitle block from SRT file"""
    index: int
    start_time: str
    end_time: str
    text: str
    start_seconds: float
    end_seconds: float


@dataclass
class Paragraph:
    """Represents a grouped paragraph with metadata"""
    start_time: str
    start_seconds: float
    text: str
    subtitle_blocks: List[SubtitleBlock]


class SubtitleParser:
    """Parses SRT and VTT subtitle files"""
    
    def __init__(self):
        # SRT time format: 00:00:00,000
        self.srt_time_pattern = re.compile(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})')
        # VTT time format: 00:00:00.000
        self.vtt_time_pattern = re.compile(r'(\d{2}):(\d{2}):(\d{2})\.(\d{3})')
    
    def parse_time_to_seconds(self, time_str: str) -> float:
        """Convert SRT or VTT time format to seconds"""
        # Try SRT format first (comma separator)
        match = self.srt_time_pattern.match(time_str)
        if match:
            hours, minutes, seconds, milliseconds = map(int, match.groups())
            return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
        
        # Try VTT format (dot separator)
        match = self.vtt_time_pattern.match(time_str)
        if match:
            hours, minutes, seconds, milliseconds = map(int, match.groups())
            return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
        
        return 0.0
    
    def detect_format(self, file_path: str) -> str:
        """Detect subtitle format (SRT or VTT) based on file extension and content"""
        # Check file extension first
        if file_path.lower().endswith('.vtt'):
            return 'vtt'
        elif file_path.lower().endswith('.srt'):
            return 'srt'
        
        # If no extension or unknown, check content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line == 'WEBVTT':
                    return 'vtt'
                elif first_line.isdigit():
                    return 'srt'
        except:
            pass
        
        # Default to SRT for backward compatibility
        return 'srt'
    
    def parse_srt_file(self, file_path: str) -> List[SubtitleBlock]:
        """Parse SRT or VTT file and return list of subtitle blocks with enhanced error handling"""
        blocks = []
        
        try:
            # Detect file format
            file_format = self.detect_format(file_path)
            log(f"Detected format: {file_format.upper()}")
            
            # Try multiple encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    log(f"Successfully read file with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                raise ValueError("Could not read file with any supported encoding")
            
            # Parse based on format
            if file_format == 'vtt':
                subtitle_blocks = self._parse_vtt_content(content)
            else:
                subtitle_blocks = self._parse_srt_content(content)
            
            log(f"Found {len(subtitle_blocks)} subtitle blocks")
            
            for i, block in enumerate(subtitle_blocks):
                try:
                    lines = block.strip().split('\n')
                    if len(lines) < 3:
                        log(f"Skipping malformed block {i+1}: insufficient lines")
                        continue
                    
                    # Handle VTT format differences
                    if file_format == 'vtt':
                        # VTT blocks: timestamp line, text lines (no index)
                        time_line = lines[0]
                        text = '\n'.join(lines[1:])
                        index = i + 1  # Generate index for VTT
                    else:
                        # SRT blocks: index, timestamp line, text lines
                        index = int(lines[0])
                        time_line = lines[1]
                        text = '\n'.join(lines[2:])
                    
                    # Parse time range
                    if ' --> ' not in time_line:
                        log(f"Skipping block {i+1}: invalid timestamp format")
                        continue
                    
                    start_time, end_time = time_line.split(' --> ')
                    start_seconds = self.parse_time_to_seconds(start_time)
                    end_seconds = self.parse_time_to_seconds(end_time)
                    
                    blocks.append(SubtitleBlock(
                        index=index,
                        start_time=start_time,
                        end_time=end_time,
                        text=text.strip(),
                        start_seconds=start_seconds,
                        end_seconds=end_seconds
                    ))
                    
                except (ValueError, IndexError) as e:
                    log(f"Error parsing block {i+1}: {e}")
                    continue
                except Exception as e:
                    log(f"Unexpected error in block {i+1}: {e}")
                    continue
            
            log(f"Successfully parsed {len(blocks)} subtitle blocks")
            return blocks
            
        except FileNotFoundError:
            log(f"File not found: {file_path}", "ERROR")
            raise
        except Exception as e:
            log(f"Error parsing SRT file: {e}", "ERROR")
            raise
    
    def _parse_srt_content(self, content: str) -> List[str]:
        """Parse SRT content into subtitle blocks"""
        return content.strip().split('\n\n')
    
    def _parse_vtt_content(self, content: str) -> List[str]:
        """Parse VTT content into subtitle blocks"""
        lines = content.strip().split('\n')
        blocks = []
        current_block = []
        
        for line in lines:
            line = line.strip()
            
            # Skip WEBVTT header and empty lines
            if line == 'WEBVTT' or line == '':
                continue
            
            # Skip style and note blocks
            if line.startswith('NOTE') or line.startswith('STYLE'):
                continue
            
            # Check if this is a timestamp line (contains -->)
            if ' --> ' in line:
                # If we have a current block, save it
                if current_block:
                    blocks.append('\n'.join(current_block))
                    current_block = []
                # Start new block with timestamp
                current_block = [line]
            else:
                # This is text content
                if current_block:  # Only add if we're in a block
                    current_block.append(line)
        
        # Add the last block if it exists
        if current_block:
            blocks.append('\n'.join(current_block))
        
        return blocks


class ParagraphGrouper:
    """Groups subtitle blocks into logical paragraphs using AI assistance"""
    
    def __init__(self, max_gap_seconds: float = 3.0, min_paragraph_length: int = 10, lm_client: Optional[LMStudioClient] = None):
        self.max_gap_seconds = max_gap_seconds
        self.min_paragraph_length = min_paragraph_length
        self.lm_client = lm_client
    
    def group_into_paragraphs(self, blocks: List[SubtitleBlock]) -> List[Paragraph]:
        """Group subtitle blocks into paragraphs based on timing and content"""
        if not blocks:
            return []
        
        # Use AI for paragraph detection if available
        if self.lm_client and self.lm_client.is_available():
            log("Using AI-enhanced paragraph grouping...")
            return self._group_with_ai_chunking(blocks)
        
        # For very long files, use smart content analysis
        if len(blocks) > 1000:
            print("Using smart content analysis for long file...")
            return self._group_with_smart_chunking(blocks)
        
        return self._group_regular_paragraphs(blocks)
    
    def _group_regular_paragraphs(self, blocks: List[SubtitleBlock]) -> List[Paragraph]:
        """Regular paragraph grouping based on timing and content with sentence boundary respect"""
        import re
        
        paragraphs = []
        current_paragraph_blocks = [blocks[0]]
        
        for i in range(1, len(blocks)):
            current_block = blocks[i]
            previous_block = blocks[i-1]
            
            # Calculate gap between subtitles
            gap = current_block.start_seconds - previous_block.end_seconds
            
            # Check if we should start a new paragraph
            should_break = (
                gap > self.max_gap_seconds or
                self._is_speaker_change(previous_block.text, current_block.text) or
                self._is_topic_change(previous_block.text, current_block.text)
            )
            
            if should_break:
                # Check if this is a good sentence boundary
                if self._is_good_sentence_boundary(blocks, i):
                    # Finalize current paragraph
                    if current_paragraph_blocks:
                        paragraph = self._create_paragraph(current_paragraph_blocks)
                        if paragraph and len(paragraph.text) >= self.min_paragraph_length:
                            paragraphs.append(paragraph)
                        current_paragraph_blocks = [current_block]
                    else:
                        current_paragraph_blocks = [current_block]
                else:
                    # Not a good boundary, continue building current paragraph
                    current_paragraph_blocks.append(current_block)
            else:
                current_paragraph_blocks.append(current_block)
        
        # Add final paragraph
        if current_paragraph_blocks:
            paragraph = self._create_paragraph(current_paragraph_blocks)
            if paragraph and len(paragraph.text) >= self.min_paragraph_length:
                paragraphs.append(paragraph)
        
        return paragraphs
    
    def _group_with_ai_chunking(self, blocks: List[SubtitleBlock]) -> List[Paragraph]:
        """Use AI-assisted chunking for very long subtitle files"""
        log("Using AI-assisted paragraph grouping for long file...")
        
        # Process in smaller chunks for better analysis
        chunk_size = 100
        all_paragraphs = []
        processed_blocks = set()
        
        # Calculate number of chunks for progress bar
        num_chunks = len(range(0, len(blocks), chunk_size // 2))
        
        with tqdm(total=num_chunks, desc="Processing chunks", unit="chunk") as pbar:
            for start_idx in range(0, len(blocks), chunk_size // 2):  # 50% overlap
                end_idx = min(start_idx + chunk_size, len(blocks))
                chunk = blocks[start_idx:end_idx]
                
                if not chunk:
                    pbar.update(1)
                    continue
                
                log(f"Processing blocks {start_idx+1}-{end_idx} with AI...")
                
                # Use AI analysis for this chunk
                chunk_paragraphs = self._ai_find_natural_boundaries(chunk, start_idx)
                
                # Only add paragraphs that haven't been processed yet
                for paragraph in chunk_paragraphs:
                    paragraph_start_idx = start_idx + blocks.index(paragraph.subtitle_blocks[0])
                    if paragraph_start_idx not in processed_blocks:
                        all_paragraphs.append(paragraph)
                        for block in paragraph.subtitle_blocks:
                            block_idx = blocks.index(block)
                            processed_blocks.add(block_idx)
                
                pbar.update(1)
        
        # Sort paragraphs by their start time
        all_paragraphs.sort(key=lambda p: p.start_seconds)
        
        log(f"Created {len(all_paragraphs)} paragraphs using AI")
        return all_paragraphs
    
    def _group_with_smart_chunking(self, blocks: List[SubtitleBlock]) -> List[Paragraph]:
        """Use fallback rule-based chunking for very long subtitle files"""
        print("Using fallback paragraph grouping for long file...")
        
        # Process in smaller chunks for better analysis
        chunk_size = 100
        all_paragraphs = []
        processed_blocks = set()
        
        for start_idx in range(0, len(blocks), chunk_size // 2):  # 50% overlap
            end_idx = min(start_idx + chunk_size, len(blocks))
            chunk = blocks[start_idx:end_idx]
            
            if not chunk:
                continue
            
            print(f"Processing blocks {start_idx+1}-{end_idx} with fallback rules...")
            
            # Use fallback analysis for this chunk
            chunk_paragraphs = self._ai_find_natural_boundaries(chunk, start_idx)
            
            # Only add paragraphs that haven't been processed yet
            for paragraph in chunk_paragraphs:
                paragraph_start_idx = start_idx + blocks.index(paragraph.subtitle_blocks[0])
                if paragraph_start_idx not in processed_blocks:
                    all_paragraphs.append(paragraph)
                    for block in paragraph.subtitle_blocks:
                        block_idx = blocks.index(block)
                        processed_blocks.add(block_idx)
        
        # Sort paragraphs by their start time
        all_paragraphs.sort(key=lambda p: p.start_seconds)
        
        print(f"Created {len(all_paragraphs)} paragraphs using fallback rules")
        return all_paragraphs
    
    def _ai_find_natural_boundaries(self, blocks: List[SubtitleBlock], start_offset: int = 0) -> List[Paragraph]:
        """Use AI to find natural paragraph boundaries"""
        print(f"  Using AI analysis for {len(blocks)} blocks")
        
        if not self.lm_client or not self.lm_client.is_available():
            print("  AI not available, using fallback rules")
            return self._fallback_find_boundaries(blocks)
        
        # Prepare text for AI analysis
        text_lines = []
        for i, block in enumerate(blocks):
            text_lines.append(f"{i}: {block.text}")
        
        # Create an improved prompt for the AI that respects sentence boundaries
        prompt = f"""Analyze this transcript and find natural paragraph breaks. Consider:
1. Complete sentences - don't break mid-sentence
2. Natural conversation flow
3. Topic changes
4. Speaker changes
5. Pauses in speech

Transcript:
{chr(10).join(text_lines)}

Return ONLY the line numbers where paragraphs should end (after complete sentences), separated by commas:"""
        
        try:
            response = self.lm_client.generate_text(prompt, max_tokens=100, temperature=0.1)
            if response and response.strip():
                print(f"  AI response: {response.strip()}")
                # Parse the response to get break points
                break_indices = []
                for part in response.strip().split(','):
                    try:
                        break_point = int(part.strip())
                        if 0 <= break_point < len(blocks):
                            break_indices.append(break_point)
                    except ValueError:
                        continue
                
                if break_indices:
                    print(f"  AI found {len(break_indices)} break points: {break_indices[:5]}")
                    # Post-process to ensure we don't break mid-sentence
                    validated_breaks = self._validate_sentence_boundaries(blocks, break_indices)
                    return self._create_paragraphs_from_breaks(blocks, validated_breaks)
        except Exception as e:
            print(f"  AI paragraph detection failed: {e}")
        
        # Fallback to rule-based approach
        print("  Falling back to rule-based analysis")
        return self._fallback_find_boundaries(blocks)
    
    def _fallback_find_boundaries(self, blocks: List[SubtitleBlock]) -> List[Paragraph]:
        """Fallback rule-based paragraph detection with sentence boundary respect"""
        import re
        
        break_indices = []
        
        for i in range(1, len(blocks)):
            current_block = blocks[i]
            previous_block = blocks[i-1]
            
            should_break = False
            
            # 1. Questions (new paragraph after questions)
            if previous_block.text.strip().endswith('?'):
                should_break = True
            
            # 2. Time gaps (more than 3 seconds)
            elif current_block.start_seconds - previous_block.end_seconds > 3.0:
                should_break = True
            
            # 3. Short responses followed by longer content
            elif (len(previous_block.text.strip()) < 15 and 
                  len(current_block.text.strip()) > 40):
                should_break = True
            
            # 4. Conversation starters
            elif any(starter in current_block.text.lower() for starter in [
                'so,', 'well,', 'now,', 'okay,', 'right,', 'and so', 'and then',
                'let me', 'i think', 'i believe', 'i would', 'i can'
            ]):
                should_break = True
            
            # 5. Topic change indicators
            elif any(indicator in current_block.text.lower() for indicator in [
                'speaking of', 'on the topic of', 'regarding', 'about',
                'in terms of', 'when it comes to', 'as for'
            ]):
                should_break = True
            
            # 6. Every 8-12 blocks as fallback (aim for 5-10 paragraphs per 100 blocks)
            elif i % 10 == 0:
                should_break = True
            
            if should_break:
                # Validate that this is a good sentence boundary
                if self._is_good_sentence_boundary(blocks, i):
                    break_indices.append(i)
                else:
                    # Try to find a better boundary nearby
                    better_break = self._find_nearby_sentence_boundary(blocks, i)
                    if better_break is not None and better_break not in break_indices:
                        break_indices.append(better_break)
        
        # Limit to reasonable number of breaks
        if len(break_indices) > len(blocks) // 8:
            # Too many breaks, keep only every few
            break_indices = break_indices[::max(1, len(break_indices) // 8)]
        elif len(break_indices) < 3:
            # Too few breaks, add some
            break_indices = list(range(10, len(blocks), 12))
        
        print(f"  Found {len(break_indices)} natural break points: {break_indices[:5]}")
        return self._create_paragraphs_from_breaks(blocks, break_indices)
    
    def _is_good_sentence_boundary(self, blocks: List[SubtitleBlock], break_idx: int) -> bool:
        """Check if a break point is at a good sentence boundary"""
        import re
        
        if break_idx >= len(blocks):
            return False
            
        current_block = blocks[break_idx]
        next_block = blocks[break_idx + 1] if break_idx + 1 < len(blocks) else None
        
        current_text = current_block.text.strip()
        next_text = next_block.text.strip() if next_block else ""
        
        # Good boundary if current block ends with sentence punctuation
        if re.search(r'[.!?]$', current_text):
            return True
        
        # Good boundary if next block starts with capital letter
        if next_text and re.match(r'^[A-Z]', next_text):
            return True
        
        return False
    
    def _find_nearby_sentence_boundary(self, blocks: List[SubtitleBlock], break_idx: int) -> Optional[int]:
        """Find a nearby sentence boundary within 2 blocks"""
        import re
        
        # Look ahead up to 2 blocks
        for i in range(break_idx + 1, min(break_idx + 3, len(blocks))):
            block_text = blocks[i].text.strip()
            
            # Check if this block ends with sentence punctuation
            if re.search(r'[.!?]$', block_text):
                return i
            
            # Check if next block starts with capital letter
            if i + 1 < len(blocks):
                next_text = blocks[i + 1].text.strip()
                if re.match(r'^[A-Z]', next_text):
                    return i
        
        # If no good boundary found, use the original
        return break_idx
    
    def _create_paragraphs_from_breaks(self, blocks: List[SubtitleBlock], break_indices: List[int]) -> List[Paragraph]:
        """Create paragraphs from break indices"""
        paragraphs = []
        current_blocks = []
        
        for i, block in enumerate(blocks):
            current_blocks.append(block)
            
            # Check if this is a break point
            if i in break_indices or i == len(blocks) - 1:
                if current_blocks:
                    paragraph = self._create_paragraph(current_blocks)
                    if paragraph and len(paragraph.text) >= self.min_paragraph_length:
                        paragraphs.append(paragraph)
                    current_blocks = []
        
        print(f"  Created {len(paragraphs)} paragraphs from {len(blocks)} blocks")
        return paragraphs
    
    def _validate_sentence_boundaries(self, blocks: List[SubtitleBlock], break_indices: List[int]) -> List[int]:
        """Validate and adjust break points to respect sentence boundaries"""
        import re
        
        validated_breaks = []
        
        for break_idx in break_indices:
            if break_idx >= len(blocks):
                continue
                
            # Check if the break point is at a natural sentence boundary
            current_block = blocks[break_idx]
            next_block = blocks[break_idx + 1] if break_idx + 1 < len(blocks) else None
            
            # Look for sentence endings in current block
            current_text = current_block.text.strip()
            next_text = next_block.text.strip() if next_block else ""
            
            # Check if current block ends with sentence punctuation
            if re.search(r'[.!?]$', current_text):
                validated_breaks.append(break_idx)
                continue
            
            # Check if next block starts with capital letter (new sentence)
            if next_text and re.match(r'^[A-Z]', next_text):
                validated_breaks.append(break_idx)
                continue
            
            # If not a natural boundary, try to find the next sentence boundary
            adjusted_break = self._find_next_sentence_boundary(blocks, break_idx)
            if adjusted_break is not None and adjusted_break not in validated_breaks:
                validated_breaks.append(adjusted_break)
        
        return sorted(set(validated_breaks))
    
    def _find_next_sentence_boundary(self, blocks: List[SubtitleBlock], start_idx: int) -> Optional[int]:
        """Find the next natural sentence boundary after start_idx"""
        import re
        
        # Look ahead up to 3 blocks for a sentence boundary
        for i in range(start_idx + 1, min(start_idx + 4, len(blocks))):
            block_text = blocks[i].text.strip()
            
            # Check if this block ends with sentence punctuation
            if re.search(r'[.!?]$', block_text):
                return i
            
            # Check if next block starts with capital letter
            if i + 1 < len(blocks):
                next_text = blocks[i + 1].text.strip()
                if re.match(r'^[A-Z]', next_text):
                    return i
        
        # If no natural boundary found, use the original break point
        return start_idx
    
    def _group_chunk_regular(self, blocks: List[SubtitleBlock]) -> List[Paragraph]:
        """Regular grouping for a chunk of blocks"""
        paragraphs = []
        current_paragraph_blocks = [blocks[0]] if blocks else []
        
        for i in range(1, len(blocks)):
            current_block = blocks[i]
            previous_block = blocks[i-1]
            
            gap = current_block.start_seconds - previous_block.end_seconds
            
            should_break = (
                gap > self.max_gap_seconds or
                self._is_speaker_change(previous_block.text, current_block.text) or
                self._is_topic_change(previous_block.text, current_block.text)
            )
            
            if should_break:
                if current_paragraph_blocks:
                    paragraph = self._create_paragraph(current_paragraph_blocks)
                    if paragraph and len(paragraph.text) >= self.min_paragraph_length:
                        paragraphs.append(paragraph)
                    current_paragraph_blocks = [current_block]
                else:
                    current_paragraph_blocks = [current_block]
            else:
                current_paragraph_blocks.append(current_block)
        
        if current_paragraph_blocks:
            paragraph = self._create_paragraph(current_paragraph_blocks)
            if paragraph and len(paragraph.text) >= self.min_paragraph_length:
                paragraphs.append(paragraph)
        
        return paragraphs
    
    def _create_paragraph(self, blocks: List[SubtitleBlock]) -> Optional[Paragraph]:
        """Create a paragraph from subtitle blocks"""
        if not blocks:
            return None
        
        # Combine text from all blocks
        text_parts = []
        for block in blocks:
            # Clean up text (remove HTML tags, normalize whitespace)
            clean_text = re.sub(r'<[^>]+>', '', block.text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            if clean_text:
                text_parts.append(clean_text)
        
        if not text_parts:
            return None
        
        combined_text = ' '.join(text_parts)
        
        return Paragraph(
            start_time=blocks[0].start_time,
            start_seconds=blocks[0].start_seconds,
            text=combined_text,
            subtitle_blocks=blocks
        )
    
    def _is_speaker_change(self, text1: str, text2: str) -> bool:
        """Detect potential speaker changes"""
        # Look for patterns that might indicate speaker changes
        speaker_indicators = [
            r'^[A-Z][a-z]+:',  # "John:"
            r'^[A-Z][A-Z]+:',  # "JOHN:"
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+:',  # "John Smith:"
        ]
        
        for pattern in speaker_indicators:
            if re.search(pattern, text2):
                return True
        
        return False
    
    def _is_topic_change(self, text1: str, text2: str) -> bool:
        """Detect potential topic changes"""
        # Simple heuristics for topic changes
        topic_indicators = [
            r'^(So|Now|Next|Then|Also|However|But|And|Well)',
            r'^(Let\'s|Let us)',
            r'^(Moving on|Speaking of)',
        ]
        
        for pattern in topic_indicators:
            if re.search(pattern, text2, re.IGNORECASE):
                return True
        
        return False


class HeadlineGenerator:
    """Generates headlines and subheadlines using AI analysis"""
    
    def __init__(self, lm_client: Optional[LMStudioClient] = None):
        self.lm_client = lm_client
        self.headline_keywords = [
            'introduction', 'overview', 'summary', 'conclusion',
            'background', 'context', 'setup', 'preparation',
            'main', 'primary', 'key', 'important', 'critical',
            'discussion', 'analysis', 'review', 'examination',
            'next', 'following', 'subsequent', 'additional'
        ]
    
    def generate_headlines(self, paragraphs: List[Paragraph]) -> List[Tuple[int, str, str]]:
        """Generate headlines for paragraphs using AI analysis"""
        headlines = []
        
        # Use AI for headline generation if available
        if self.lm_client and self.lm_client.is_available():
            headlines = self._generate_ai_headlines(paragraphs)
        else:
            # Fallback to rule-based generation - only for major sections
            headlines = []
            
            # Always include first paragraph as introduction
            if paragraphs:
                headlines.append((0, "Introduction", self._get_timestamp_marker(paragraphs[0].start_time)))
            
            # Look for major topic changes
            for i in range(1, len(paragraphs)):
                current_para = paragraphs[i]
                previous_para = paragraphs[i-1]
                
                topic_change = self._rule_based_topic_change(previous_para, current_para)
                if topic_change:
                    headlines.append((i, topic_change, self._get_timestamp_marker(current_para.start_time)))
            
            # Always include last paragraph as conclusion if there are multiple paragraphs
            if len(paragraphs) > 1:
                last_idx = len(paragraphs) - 1
                headlines.append((last_idx, "Conclusion", self._get_timestamp_marker(paragraphs[last_idx].start_time)))
        
        return headlines
    
    def _generate_ai_headlines(self, paragraphs: List[Paragraph]) -> List[Tuple[int, str, str]]:
        """Generate headlines using AI analysis - only for major sections"""
        headlines = []
        
        # Only generate headlines for significant sections, not every paragraph
        # Look for topic changes, introductions, conclusions, and major transitions
        
        # Always include first paragraph as introduction
        if paragraphs:
            headlines.append((0, "Introduction", self._get_timestamp_marker(paragraphs[0].start_time)))
        
        # Analyze for major topic changes
        for i in range(1, len(paragraphs)):
            current_para = paragraphs[i]
            previous_para = paragraphs[i-1]
            
            # Check for major topic changes using AI
            topic_change = self._detect_topic_change(previous_para, current_para)
            if topic_change:
                headlines.append((i, topic_change, self._get_timestamp_marker(current_para.start_time)))
        
        # Always include last paragraph as conclusion if there are multiple paragraphs
        if len(paragraphs) > 1:
            last_idx = len(paragraphs) - 1
            headlines.append((last_idx, "Conclusion", self._get_timestamp_marker(paragraphs[last_idx].start_time)))
        
        return headlines
    
    def _detect_topic_change(self, prev_para: Paragraph, current_para: Paragraph) -> Optional[str]:
        """Detect if there's a significant topic change between paragraphs"""
        if not self.lm_client or not self.lm_client.is_available():
            return self._rule_based_topic_change(prev_para, current_para)
        
        # Use AI to detect topic changes
        prompt = f"""Analyze these two consecutive paragraphs from a transcript. 
        Determine if there's a significant topic change that warrants a section headline.
        
        Previous paragraph: {prev_para.text[:300]}
        Current paragraph: {current_para.text[:300]}
        
        If there's a major topic change, provide a brief headline (2-4 words) for the current paragraph.
        If not, respond with "NO_HEADLINE".
        
        Response:"""
        
        try:
            response = self.lm_client.generate_text(prompt, max_tokens=50, temperature=0.1)
            if response and response.strip() and response.strip() != "NO_HEADLINE":
                return response.strip()
        except Exception as e:
            print(f"AI topic change detection failed: {e}")
        
        return self._rule_based_topic_change(prev_para, current_para)
    
    def _rule_based_topic_change(self, prev_para: Paragraph, current_para: Paragraph) -> Optional[str]:
        """Rule-based topic change detection"""
        prev_text = prev_para.text.lower()
        current_text = current_para.text.lower()
        
        # Look for topic change indicators
        topic_indicators = [
            'now let\'s talk about', 'speaking of', 'on the topic of',
            'regarding', 'about', 'in terms of', 'when it comes to',
            'moving on to', 'next', 'another', 'different',
            'budget', 'timeline', 'cost', 'planning', 'implementation'
        ]
        
        for indicator in topic_indicators:
            if indicator in current_text:
                # Extract the topic
                if 'budget' in current_text:
                    return "Budget Discussion"
                elif 'timeline' in current_text:
                    return "Timeline Planning"
                elif 'cost' in current_text:
                    return "Cost Analysis"
                elif 'planning' in current_text:
                    return "Project Planning"
                else:
                    return "New Topic"
        
        return None
    
    def _analyze_paragraph_for_headline(self, paragraph: Paragraph, index: int, total: int) -> Optional[str]:
        """Analyze paragraph content to generate appropriate headline"""
        text = paragraph.text.lower()
        
        # Check for explicit topic indicators
        topic_indicators = {
            'introduction': ['introduction', 'intro', 'welcome', 'hello', 'hi'],
            'overview': ['overview', 'summary', 'recap', 'briefly'],
            'background': ['background', 'context', 'history', 'previously'],
            'main topic': ['main', 'primary', 'key', 'important', 'critical'],
            'discussion': ['discussion', 'analysis', 'review', 'examine'],
            'conclusion': ['conclusion', 'summary', 'wrap up', 'finish', 'end'],
            'next steps': ['next', 'following', 'subsequent', 'additional', 'more']
        }
        
        for topic, keywords in topic_indicators.items():
            if any(keyword in text for keyword in keywords):
                return topic.title()
        
        # Check for question patterns
        if '?' in paragraph.text:
            return "Questions & Discussion"
        
        # Check for list patterns
        if any(char in paragraph.text for char in ['•', '-', '*', '1.', '2.', '3.']):
            return "Key Points"
        
        # Check for time-based indicators
        if any(word in text for word in ['first', 'initially', 'start', 'begin']):
            return "Introduction"
        
        if any(word in text for word in ['finally', 'last', 'end', 'conclude']):
            return "Conclusion"
        
        # Default based on position
        if index == 0:
            return "Introduction"
        elif index == total - 1:
            return "Conclusion"
        elif index < total // 3:
            return "Opening Discussion"
        elif index > 2 * total // 3:
            return "Closing Discussion"
        else:
            return "Main Discussion"
    
    def _get_timestamp_marker(self, start_time: str) -> str:
        """Convert SRT timestamp to readable format"""
        # Convert "00:01:23,456" to "1:23"
        time_parts = start_time.split(':')
        if len(time_parts) >= 3:
            hours = int(time_parts[0])
            minutes = int(time_parts[1])
            if hours > 0:
                return f"{hours}:{minutes:02d}"
            else:
                return f"{minutes}:{time_parts[2].split(',')[0]}"
        return start_time


class PandocConverter:
    """Handles conversion to various output formats using pandoc"""
    
    def __init__(self, pandoc_path: str = "pandoc"):
        self.pandoc_path = pandoc_path
        self.supported_formats = {
            'html': 'html',
            'pdf': 'pdf',
            'docx': 'docx',
            'odt': 'odt',
            'rtf': 'rtf'
        }
    
    def is_available(self) -> bool:
        """Check if pandoc is available"""
        try:
            result = subprocess.run([self.pandoc_path, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False
    
    def convert(self, markdown_content: str, output_format: str, output_file: str) -> bool:
        """Convert markdown content to specified format"""
        if output_format not in self.supported_formats:
            log(f"Unsupported format: {output_format}", "ERROR")
            return False
        
        if not self.is_available():
            log("Pandoc is not available. Please install pandoc to use format conversion.", "ERROR")
            return False
        
        try:
            # Create temporary markdown file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as temp_md:
                temp_md.write(markdown_content)
                temp_md_path = temp_md.name
            
            # Build pandoc command
            cmd = [
                self.pandoc_path,
                temp_md_path,
                '-o', output_file,
                '-f', 'markdown',
                '-t', self.supported_formats[output_format]
            ]
            
            # Add format-specific options
            if output_format == 'pdf':
                cmd.extend(['--pdf-engine=xelatex', '--variable=geometry:margin=1in'])
            elif output_format == 'html':
                cmd.extend(['--standalone', '--css=style.css'])
            elif output_format in ['docx', 'odt']:
                cmd.extend(['--reference-doc=template.docx'] if output_format == 'docx' else [])
            
            # Run pandoc
            log(f"Converting to {output_format.upper()} using pandoc...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            # Clean up temporary file
            try:
                os.unlink(temp_md_path)
            except:
                pass
            
            if result.returncode == 0:
                log(f"Successfully converted to {output_format.upper()}")
                return True
            else:
                log(f"Pandoc conversion failed: {result.stderr}", "ERROR")
                return False
                
        except subprocess.TimeoutExpired:
            log("Pandoc conversion timed out", "ERROR")
            return False
        except Exception as e:
            log(f"Error during pandoc conversion: {e}", "ERROR")
            return False


class MarkdownFormatter:
    """Formats paragraphs and headlines into Markdown"""
    
    def __init__(self, lm_client: Optional[LMStudioClient] = None, video_handler: Optional[VideoURLHandler] = None, no_headlines: bool = False):
        self.headline_generator = HeadlineGenerator(lm_client)
        self.video_handler = video_handler
        self.no_headlines = no_headlines
    
    def format_to_markdown(self, paragraphs: List[Paragraph], title: str = None) -> str:
        """Convert paragraphs to formatted Markdown"""
        if not paragraphs:
            return "# No content found\n"
        
        # Start building markdown
        markdown_parts = []
        
        # Add title
        if title:
            markdown_parts.append(f"# {title}\n")
        else:
            markdown_parts.append("# Transcript\n")
        
        # Generate headlines only if not disabled
        headlines = []
        if not self.no_headlines:
            headlines = self.headline_generator.generate_headlines(paragraphs)
        
        # Add content with or without headlines
        current_headline_index = 0
        
        for i, paragraph in enumerate(paragraphs):
            # Check if we need to add a headline (only if headlines are enabled)
            if (not self.no_headlines and 
                current_headline_index < len(headlines) and 
                headlines[current_headline_index][0] == i):
                _, headline, timestamp = headlines[current_headline_index]
                markdown_parts.append(f"## {headline} *({timestamp})*\n")
                current_headline_index += 1
            
            # Add paragraph with timestamp
            timestamp_marker = self._format_timestamp(paragraph.start_time)
            if self.video_handler and self.video_handler.is_valid():
                # Create clickable timestamp link
                timestamp_url = self.video_handler.get_timestamp_url(paragraph.start_seconds)
                if timestamp_url:
                    timestamp_marker = f"[{timestamp_marker}]({timestamp_url})"
                else:
                    timestamp_marker = f"**{timestamp_marker}**"
            else:
                timestamp_marker = f"**{timestamp_marker}**"
            
            markdown_parts.append(f"{timestamp_marker} {paragraph.text}\n\n")
        
        return ''.join(markdown_parts)
    
    def _slugify(self, text: str) -> str:
        """Convert text to URL-friendly slug"""
        return re.sub(r'[^a-zA-Z0-9\s-]', '', text.lower()).replace(' ', '-')
    
    def _format_timestamp(self, start_time: str) -> str:
        """Format timestamp for display"""
        # Convert "00:01:23,456" to "1:23"
        time_parts = start_time.split(':')
        if len(time_parts) >= 3:
            hours = int(time_parts[0])
            minutes = int(time_parts[1])
            seconds = time_parts[2].split(',')[0]
            if hours > 0:
                return f"{hours}:{minutes:02d}:{seconds}"
            else:
                return f"{minutes}:{seconds}"
        return start_time


class SRTToMarkdownConverter:
    """Main converter class that orchestrates the conversion process"""
    
    def __init__(self, max_gap_seconds: float = 3.0, min_paragraph_length: int = 10, lm_studio_url: str = "http://localhost:1234", model: str = "qwen/qwen3-4b-2507", video_url: str = None, wait_for_model: bool = False, no_headlines: bool = False, output_format: str = "markdown", pandoc_path: str = "pandoc"):
        self.parser = SubtitleParser()
        self.lm_client = LMStudioClient(lm_studio_url, model)
        self.video_handler = VideoURLHandler(video_url) if video_url else None
        
        # Select the best available model
        if self.lm_client.is_available():
            if wait_for_model:
                print("Waiting for model to be ready...")
                if not self.lm_client.wait_for_model_ready():
                    print("⚠ Model not ready, using fallback processing")
                    self.lm_client = None
            else:
                selected_model = select_lm_studio_model(self.lm_client, model)
                if selected_model:
                    self.lm_client.model = selected_model
                else:
                    print("⚠ No suitable model found, disabling AI processing")
                    self.lm_client = None
        else:
            print("⚠ LM Studio not available, using fallback processing")
            self.lm_client = None
        
        self.grouper = ParagraphGrouper(max_gap_seconds, min_paragraph_length, self.lm_client)
        self.formatter = MarkdownFormatter(self.lm_client, self.video_handler, no_headlines)
        self.pandoc_converter = PandocConverter(pandoc_path)
        self.output_format = output_format
    
    def _detect_format_from_extension(self, filename: str) -> str:
        """Detect output format from file extension"""
        if not filename:
            return self.output_format
        
        # Get file extension
        if '.' in filename:
            extension = filename.lower().split('.')[-1]
        else:
            return self.output_format
        
        # Map extensions to formats
        extension_map = {
            'md': 'markdown',
            'markdown': 'markdown',
            'html': 'html',
            'htm': 'html',
            'pdf': 'pdf',
            'docx': 'docx',
            'doc': 'docx',
            'odt': 'odt',
            'rtf': 'rtf'
        }
        
        return extension_map.get(extension, self.output_format)
    
    def convert(self, input_file: str, output_file: str = None, title: str = None) -> str:
        """Convert SRT file to specified format"""
        print(f"Parsing SRT file: {input_file}")
        blocks = self.parser.parse_srt_file(input_file)
        
        if not blocks:
            print("No subtitle blocks found in the file.")
            return ""
        
        print(f"Found {len(blocks)} subtitle blocks")
        
        # Smart format detection from output file extension
        if output_file:
            detected_format = self._detect_format_from_extension(output_file)
            if detected_format != self.output_format:
                print(f"Detected format from filename: {detected_format}")
                self.output_format = detected_format
        
        # Generate default output filename if not provided
        if output_file is None:
            # Remove subtitle extension and add appropriate extension
            if input_file.lower().endswith('.srt'):
                base_name = input_file[:-4]
            elif input_file.lower().endswith('.vtt'):
                base_name = input_file[:-4]
            else:
                base_name = input_file
            
            # Add appropriate extension based on output format
            extensions = {
                'markdown': '.md',
                'html': '.html',
                'pdf': '.pdf',
                'docx': '.docx',
                'odt': '.odt',
                'rtf': '.rtf'
            }
            extension = extensions.get(self.output_format, '.md')
            output_file = base_name + extension
            print(f"Output file: {output_file}")
        
        # Check LM Studio availability
        if self.lm_client and self.lm_client.is_available():
            context_window = self.lm_client.get_context_window()
            print(f"✓ LM Studio is available - using AI-enhanced processing with {self.lm_client.model}")
            print(f"  Model context window: {context_window} tokens")
        else:
            print("⚠ LM Studio not available - using fallback processing")
            if not self.lm_client:
                print("  No suitable model found in LM Studio")
            else:
                print("  Make sure LM Studio is running with a compatible model loaded")
        
        # Check video URL
        if self.video_handler:
            if self.video_handler.is_valid():
                print(f"✓ Video URL configured - timestamps will link to {self.video_handler.platform} video")
            else:
                print("⚠ Invalid video URL - timestamps will not be clickable")
                print(f"  Supported platforms: YouTube, Vimeo")
                print(f"  Provided URL: {self.video_handler.video_url}")
        
        print("Grouping into paragraphs...")
        try:
            paragraphs = self.grouper.group_into_paragraphs(blocks)
            print(f"Created {len(paragraphs)} paragraphs")
            
            if not paragraphs:
                print("⚠ No paragraphs created from subtitle blocks")
                return ""
        except Exception as e:
            print(f"Error during paragraph grouping: {e}")
            return ""
        
        print("Generating headlines...")
        try:
            markdown_content = self.formatter.format_to_markdown(paragraphs, title)
            
            if not markdown_content.strip():
                print("⚠ Generated markdown content is empty")
                return ""
        except Exception as e:
            print(f"Error during markdown formatting: {e}")
            return ""
        
        try:
            if self.output_format == 'markdown':
                # Direct markdown output
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                print(f"Markdown saved to: {output_file}")
            else:
                # Convert using pandoc
                if self.pandoc_converter.is_available():
                    if self.pandoc_converter.convert(markdown_content, self.output_format, output_file):
                        print(f"{self.output_format.upper()} saved to: {output_file}")
                    else:
                        print(f"Failed to convert to {self.output_format.upper()}")
                        return ""
                else:
                    print(f"Pandoc not available. Saving as markdown instead.")
                    # Fallback to markdown
                    markdown_file = output_file.rsplit('.', 1)[0] + '.md'
                    with open(markdown_file, 'w', encoding='utf-8') as f:
                        f.write(markdown_content)
                    print(f"Markdown saved to: {markdown_file}")
                    return markdown_content
        except Exception as e:
            print(f"Error writing output file: {e}")
            return ""
        
        return markdown_content


def main():
    """Command-line interface for the converter"""
    parser = argparse.ArgumentParser(
        description="Convert SRT/VTT subtitle files to well-formatted transcripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sub2trans.py input.srt                    # Creates input.md
  python sub2trans.py input.vtt                    # Creates input.md (VTT support)
  python sub2trans.py input.srt -o output.md      # Creates output.md
  python sub2trans.py input.vtt -o output.pdf      # Smart format detection (PDF)
  python sub2trans.py input.srt -t "My Video Transcript"  # Creates input.md with title
  python sub2trans.py input.srt --max-gap 5.0 --min-length 30
  python sub2trans.py input.srt --lm-studio-url http://localhost:1234 --model qwen/qwen3-4b-2507
  python sub2trans.py input.srt --no-ai
  python sub2trans.py input.srt -u "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  python sub2trans.py input.srt -u "https://vimeo.com/123456789" -o transcript.md
  python sub2trans.py --list-models                    # List available models
  python sub2trans.py input.srt --model "llama-2-7b"  # Use specific model
  python sub2trans.py input.srt --wait-for-model      # Wait for model to load
  python sub2trans.py input.srt -f pdf                # Convert to PDF
  python sub2trans.py input.srt -f docx               # Convert to Word document
  python sub2trans.py input.srt -f html               # Convert to HTML
  python sub2trans.py input.srt -o output.pdf         # Smart format detection (PDF)
  python sub2trans.py input.srt -o output.docx        # Smart format detection (Word)
  python sub2trans.py input.srt -o output.html        # Smart format detection (HTML)

LM Studio Integration:
  This tool can use LM Studio for enhanced processing:
  - Better paragraph grouping for long transcripts
  - AI-generated headlines and section titles
  - Improved content organization
  - Automatic model selection and fallback
  
  The tool will automatically find and use the best available model.
  Use --list-models to see what's available in your LM Studio instance.

Video URL Integration:
  Use -u/--video-url to create clickable timestamps that link to the video:
  - YouTube: Creates links with ?t=120s format
  - Vimeo: Creates links with #t=120.521 format
  - Timestamps will be clickable in the generated Markdown
        """
    )
    
    parser.add_argument('input_file', nargs='?', help='Input SRT or VTT file path')
    parser.add_argument('-o', '--output', help='Output file path (default: input file with .md extension). Smart format detection: .pdf, .docx, .html, .odt, .rtf')
    parser.add_argument('-t', '--title', help='Title for the document')
    parser.add_argument('--max-gap', type=float, default=3.0,
                       help='Maximum gap in seconds between subtitles to group into paragraphs (default: 3.0)')
    parser.add_argument('--min-length', type=int, default=10,
                       help='Minimum paragraph length in characters (default: 10)')
    parser.add_argument('--preview', action='store_true',
                       help='Preview the first few paragraphs without saving')
    parser.add_argument('--lm-studio-url', default='http://localhost:1234',
                       help='LM Studio API URL (default: http://localhost:1234)')
    parser.add_argument('--model', default='qwen/qwen3-4b-2507',
                       help='Model name to use with LM Studio (default: qwen/qwen3-4b-2507)')
    parser.add_argument('--wait-for-model', action='store_true',
                       help='Wait for model to load before processing (useful when switching models)')
    parser.add_argument('--no-ai', action='store_true',
                       help='Disable AI processing and use fallback methods only')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress non-essential output')
    parser.add_argument('--batch', action='store_true',
                       help='Process all SRT files in current directory')
    parser.add_argument('--no-headlines', action='store_true',
                       help='Disable headline generation (output raw transcript)')
    parser.add_argument('-f', '--format', choices=['markdown', 'html', 'pdf', 'docx', 'odt', 'rtf'], 
                       default='markdown', help='Output format (default: markdown)')
    parser.add_argument('--pandoc-path', default='pandoc',
                       help='Path to pandoc executable (default: pandoc)')
    parser.add_argument('-u', '--video-url', 
                       help='Video URL (YouTube or Vimeo) to create clickable timestamps')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models in LM Studio and exit')
    
    args = parser.parse_args()
    
    # Set global verbosity
    global VERBOSE
    VERBOSE = args.verbose
    
    # Handle batch processing
    if args.batch:
        import glob
        subtitle_files = glob.glob("*.srt") + glob.glob("*.vtt")
        if not subtitle_files:
            print("No SRT or VTT files found in current directory")
            sys.exit(1)

        print(f"Found {len(subtitle_files)} subtitle files to process")
        for subtitle_file in subtitle_files:
            print(f"\nProcessing {subtitle_file}...")
            try:
                converter = SRTToMarkdownConverter(args.max_gap, args.min_length, args.lm_studio_url, args.model, args.video_url, args.wait_for_model, args.no_headlines, args.format, args.pandoc_path)
                converter.convert(subtitle_file, title=args.title)
                print(f"✓ {subtitle_file} processed successfully")
            except Exception as e:
                print(f"✗ Error processing {subtitle_file}: {e}")
        sys.exit(0)
    
    # Validate arguments
    if not args.list_models and not args.input_file:
        parser.error("input_file is required unless using --list-models")
    
    # Handle --list-models option
    if args.list_models:
        lm_client = LMStudioClient(args.lm_studio_url, args.model)
        if lm_client.is_available():
            models = lm_client.get_available_models()
            if models:
                print("Available models in LM Studio:")
                for i, model in enumerate(models, 1):
                    print(f"  {i}. {model}")
            else:
                print("No models found in LM Studio")
        else:
            print("LM Studio is not available")
            print(f"Make sure LM Studio is running at {args.lm_studio_url}")
        sys.exit(0)
    
    # Create converter
    if args.no_ai:
        # Create a dummy LM client that's not available
        class DummyLMClient:
            def is_available(self): return False
        dummy_client = DummyLMClient()
        converter = SRTToMarkdownConverter(args.max_gap, args.min_length, "dummy", "dummy", args.video_url, False, args.no_headlines, args.format, args.pandoc_path)
        converter.lm_client = dummy_client
        converter.grouper.lm_client = dummy_client
        converter.formatter.headline_generator.lm_client = dummy_client
    else:
        converter = SRTToMarkdownConverter(args.max_gap, args.min_length, args.lm_studio_url, args.model, args.video_url, args.wait_for_model, args.no_headlines, args.format, args.pandoc_path)
    
    try:
        if args.preview:
            # Preview mode - show first few paragraphs
            try:
                blocks = converter.parser.parse_srt_file(args.input_file)
                if not blocks:
                    print("No subtitle blocks found in the file")
                    sys.exit(1)
                
                paragraphs = converter.grouper.group_into_paragraphs(blocks)
                if not paragraphs:
                    print("No paragraphs created from subtitle blocks")
                    sys.exit(1)
                
                print(f"\nPreview of first 3 paragraphs:\n")
                for i, para in enumerate(paragraphs[:3]):
                    timestamp = converter.formatter._format_timestamp(para.start_time)
                    print(f"**{timestamp}** {para.text}\n")
                
                if len(paragraphs) > 3:
                    print(f"... and {len(paragraphs) - 3} more paragraphs")
            except Exception as e:
                print(f"Error in preview mode: {e}")
                if VERBOSE:
                    import traceback
                    traceback.print_exc()
                sys.exit(1)
        else:
            # Full conversion
            try:
                result = converter.convert(args.input_file, args.output, args.title)
                if not result:
                    print("Conversion failed - no output generated")
                    sys.exit(1)
            except Exception as e:
                print(f"Error during conversion: {e}")
                if VERBOSE:
                    import traceback
                    traceback.print_exc()
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        sys.exit(1)
    except PermissionError as e:
        print(f"Permission denied: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        if VERBOSE:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

