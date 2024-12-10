import os
import textwrap
import scipy.io.wavfile
import ffmpeg
import numpy as np

def clear_screen():
    """Clears the console screen."""
    os.system("cls" if os.name == "nt" else "clear")

def print_transcript(text):
    """Prints formatted transcript text."""
    wrapper = textwrap.TextWrapper(width=60)
    for line in wrapper.wrap(text="".join(text)):
        print(line)
        
def format_time(s):
    """Convert seconds (float) to SRT time format."""
    hours = int(s // 3600)
    minutes = int((s % 3600) // 60)
    seconds = int(s % 60)
    milliseconds = int((s - int(s)) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def create_srt_file(segments, output_file):
    """Create SRT subtitle file from segments."""
    with open(output_file, 'w', encoding='utf-8') as srt_file:
        segment_number = 1
        for segment in segments:
            start_time = format_time(float(segment['start']))
            end_time = format_time(float(segment['end']))
            text = segment['text']

            srt_file.write(f"{segment_number}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"{text}\n\n")

            segment_number += 1

def resample(file: str, sr: int = 16000):
    """Resample audio file to target sample rate."""
    try:
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    
    np_buffer = np.frombuffer(out, dtype=np.int16)
    resampled_file = f"{file.split('.')[0]}_resampled.wav"
    scipy.io.wavfile.write(resampled_file, sr, np_buffer.astype(np.int16))
    return resampled_file

def calculate_energy(audio_chunk):
    """Calculate the energy of an audio chunk."""
    return np.sqrt(np.mean(np.square(audio_chunk)))

def get_audio_duration(file_path):
    """Get duration of audio file in seconds."""
    probe = ffmpeg.probe(file_path)
    return float(probe['streams'][0]['duration'])

def convert_audio_to_mono(file_path):
    """Convert audio file to mono."""
    output_path = f"{os.path.splitext(file_path)[0]}_mono.wav"
    try:
        (
            ffmpeg
            .input(file_path)
            .output(output_path, ac=1)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        return output_path
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to convert audio to mono: {e.stderr.decode()}")