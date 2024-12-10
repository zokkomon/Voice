import torch
import threading
import numpy as np
from typing import List, Optional
import logging
import time
import pyaudio
import wave
import os
import shutil
from transcriber import WhisperModel
from vad import VoiceActivityDetector
import utils

# logging.basicConfig(level=logging.INFO)

class LiveTranscriber:
    RATE = 16000

    def __init__(self, task: str = "transcribe", device: str = "cpu", language: str = "en", model: str = "turbo",
                 initial_prompt: Optional[str] = None, vad_parameters: dict = None, srt_file_path="output.srt",
                 use_vad=True, log_transcription=True, save_output_recording=False, output_recording_filename="./output_recording.wav"):
        
        # Model configuration
        self.model = model
        self.language = "en" if self.model.endswith("en") else language
        self.task = task
        self.initial_prompt = initial_prompt
        self.vad_parameters = vad_parameters or {"threshold": 0.5}
        self.no_speech_thresh = 0.45
        
        # Audio processing state
        self.frames_np = None
        self.frames = b""
        self.timestamp_offset = 0.0
        self.frames_offset = 0.0
        
        # Transcription state
        self.text = []
        self.current_out = ''
        self.prev_out = ''
        self.t_start = None
        self.exit = False
        self.transcript = []
        self.last_segment = None
        self.last_received_segment = None
        
        # Thresholds and configuration
        self.show_prev_out_thresh = 5   # Show previous output for 5 seconds if paused
        self.add_pause_thresh = 3       # Add a blank to segment list as a pause for 3 seconds
        self.send_last_n_segments = 10
        self.no_voice_activity_chunks = 0
        self.single_model = False
        self.pick_previous_segments = 2
        self.same_output_threshold = 0
        self.audio_threshold = 0.01
        self.silence_threshold = 2.0
        
        # Threading and synchronization
        self.lock = threading.Lock()
        self.last_audio_time = time.time()
        self.last_response_received = time.time()

        # Device and compute configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            major, _ = torch.cuda.get_device_capability(self.device)
            self.compute_type = "float16" if major >= 7 else "float32"
        else:
            self.compute_type = "int8"

        # VAD configuration
        self.use_vad = use_vad
        if self.use_vad:
            self.vad_detector = VoiceActivityDetector(frame_rate=self.RATE)

        # Output configuration
        self.srt_file_path = srt_file_path
        self.log_transcription = log_transcription
        self.save_output_recording = save_output_recording
        self.output_recording_filename = output_recording_filename

        # Audio configuration
        self.chunk = 4096
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.record_seconds = 60000
        
        # PyAudio initialization
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            output=True,
            frames_per_buffer=self.chunk
        )

        self.lock = threading.Lock()
        self.frames_lock = threading.Lock()  # Add a separate lock for frames


    def add_frames(self, frame_np):
        """Add incoming audio frames for processing."""
        with self.lock:  # Use lock as context manager
            if self.frames_np is not None and self.frames_np.shape[0] > 45 * self.RATE:
                self.frames_offset += 30.0
                self.frames_np = self.frames_np[int(30 * self.RATE):]
                if self.timestamp_offset < self.frames_offset:
                    self.timestamp_offset = self.frames_offset
            if self.frames_np is None:
                self.frames_np = frame_np.copy()
            else:
                self.frames_np = np.concatenate((self.frames_np, frame_np), axis=0)


    def clip_audio_if_no_valid_segment(self):
        """Clip audio if no valid segment found."""
        if self.frames_np[int((self.timestamp_offset - self.frames_offset) * self.RATE):].shape[0] > 25 * self.RATE:
            duration = self.frames_np.shape[0] / self.RATE
            self.timestamp_offset = self.frames_offset + duration - 5

    def get_audio_chunk_for_processing(self):
        """Get the chunk of audio data for transcription."""
        samples_take = max(0, (self.timestamp_offset - self.frames_offset) * self.RATE)
        input_bytes = self.frames_np[int(samples_take):].copy()
        duration = input_bytes.shape[0] / self.RATE
        return input_bytes, duration

    def prepare_segments(self, last_segment=None):
        """Prepare the transcript segments."""
        segments = []
        if len(self.transcript) >= self.send_last_n_segments:
            segments = self.transcript[-self.send_last_n_segments:].copy()
        else:
            segments = self.transcript.copy()
        if last_segment is not None:
            segments = segments + [last_segment]
        return segments

    def process_segments(self, segments):
        """Processes transcript segments."""
        text = []
        for i, seg in enumerate(segments):
            if not text or text[-1] != seg["text"]:
                text.append(seg["text"])
                if i == len(segments) - 1:
                    self.last_segment = seg
                elif (self.transcript or
                        float(seg['start']) >= float(self.transcript[-1]['end'])):
                    self.transcript.append(seg)
        
        if self.last_received_segment is None or self.last_received_segment != segments[-1]["text"]:
            self.last_response_received = time.time()
            self.last_received_segment = segments[-1]["text"]

        if self.log_transcription:
            text = text[-3:]
            utils.clear_screen()
            utils.print_transcript(text)

    def format_segment(self, start: float, end: float, text: str):
        """Format a transcript segment."""
        return {"start": start, "end": end, "text": text}

    def create_model(self, device):
        """Create a whisper model for transcription."""
        return WhisperModel(
            self.model,
            device=device,
            compute_type=self.compute_type,
            local_files_only=False,
        )

    def voice_activity(self, frame_np: np.ndarray):
        """Check for voice activity using VAD if enabled."""
        if not self.vad_detector(frame_np):
            self.no_voice_activity_chunks += 1
            if self.no_voice_activity_chunks > 3:
                return None
            return None
        self.no_voice_activity_chunks = 0
        return True

    def set_language(self, info):
        """Update the language if it is not set and detected with high confidence."""
        if info and info.language_probability > 0.5:
            self.language = info.language
            # logging.info(f"Detected language {self.language} with probability {info.language_probability}")

    def transcribe_audio(self, audio_chunk: np.ndarray) -> Optional[str]:
        """Transcribe a chunk of audio."""
        result, info = self.create_model(self.device).transcribe(
            audio_chunk,
            initial_prompt=self.initial_prompt,
            language=self.language,
            task=self.task,
            vad_filter=self.use_vad,
            vad_parameters=self.vad_parameters if self.use_vad else None
        )

        if self.language is None and info is not None:
            self.set_language(info)

        return result

    def get_previous_output(self):
        """Return previous output if no new transcription available."""
        segments = []
        if self.t_start is None:
            self.t_start = time.time()
        if time.time() - self.t_start < self.show_prev_out_thresh:
            segments = self.prepare_segments()

        if len(self.text) and self.text[-1] != '':
            if time.time() - self.t_start > self.add_pause_thresh:
                self.text.append('')
        return segments

    def update_segments(self, segments, duration):
        """Update the list of transcript segments."""
        offset = None
        self.current_out = ''
        last_segment = None

        if len(segments) > 1:
            for s in segments[:-1]:
                text_ = s.text
                self.text.append(text_)
                start, end = self.timestamp_offset + s.start, self.timestamp_offset + min(duration, s.end)
                if start >= end:
                    continue
                if s.no_speech_prob > self.no_speech_thresh:
                    continue
                self.transcript.append(self.format_segment(start, end, text_))
                offset = min(duration, s.end)

        if segments[-1].no_speech_prob <= self.no_speech_thresh:
            self.current_out += segments[-1].text
            last_segment = self.format_segment(
                self.timestamp_offset + segments[-1].start,
                self.timestamp_offset + min(duration, segments[-1].end),
                self.current_out
            )

        if self.current_out.strip() == self.prev_out.strip() and self.current_out != '':
            self.same_output_threshold += 1
        else:
            self.same_output_threshold = 0

        if self.same_output_threshold > 5:
            if not len(self.text) or self.text[-1].strip().lower() != self.current_out.strip().lower():
                self.text.append(self.current_out)
                self.transcript.append(self.format_segment(
                    self.timestamp_offset,
                    self.timestamp_offset + duration,
                    self.current_out
                ))
            self.current_out = ''
            offset = duration
            self.same_output_threshold = 0
            last_segment = None
        else:
            self.prev_out = self.current_out

        if offset is not None:
            self.timestamp_offset += offset

        return last_segment

    def handle_transcription_output(self, result, duration):
        """Handle the transcription output and update segments."""
        segments = []
        if len(result):
            self.t_start = None
            last_segment = self.update_segments(result, duration)
            segments = self.prepare_segments(last_segment)
        else:
            segments = self.get_previous_output()
        return segments

    def speech_to_text(self):
        """Main loop to process speech to text."""
        while True:
            if self.exit:
                # logging.info("Exiting speech-to-text thread")
                break

            if self.frames_np is None:
                continue

            self.clip_audio_if_no_valid_segment()

            input_bytes, duration = self.get_audio_chunk_for_processing()
            if duration < 1.0:
                time.sleep(0.1)
                continue
            try:
                input_sample = input_bytes.copy()

                if self.use_vad and not self.voice_activity(input_sample):
                    self.timestamp_offset += duration
                    time.sleep(0.25)
                    continue

                result = self.transcribe_audio(input_sample)
                if result is None or self.language is None:
                    self.timestamp_offset += duration
                    time.sleep(0.25)
                    continue

                segments = self.handle_transcription_output(result, duration)
                self.process_segments(segments)

            except Exception as e:
                # logging.error(f"[ERROR]: Failed to transcribe audio chunk: {e}")
                time.sleep(0.01)

    def check_silence(self, audio_chunk):
        """Check if the audio chunk is silence."""
        energy = utils.calculate_energy(audio_chunk)
        return energy < self.audio_threshold

    def reset_stream(self):
        """Reset the audio stream."""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            output=True,
            frames_per_buffer=self.chunk
        )

    def cleanup(self):
        """Cleanup resources."""
        self.exit = True
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

    def write_srt_file(self, output_path="output.srt"):
        """Write transcription to SRT file."""
        if self.last_segment:
            self.transcript.append(self.last_segment)
        utils.create_srt_file(self.transcript, output_path)

    def play_file(self, filename):
        """Play an audio file."""
        with wave.open(filename, "rb") as wavfile:
            self.stream = self.p.open(
                format=self.p.get_format_from_width(wavfile.getsampwidth()),
                channels=wavfile.getnchannels(),
                rate=wavfile.getframerate(),
                input=True,
                output=True,
                frames_per_buffer=self.chunk,
            )
            self.stream.close()

    def save_chunk(self, n_audio_file):
        """Save audio chunk to file."""
        t = threading.Thread(
            target=self.write_audio_frames_to_file,
            args=(self.frames[:], f"chunks/{n_audio_file}.wav",),
        )
        t.start()

    def write_audio_frames_to_file(self, frames, file_name):
        """Write audio frames to WAV file."""
        with wave.open(file_name, "wb") as wavfile:
            wavfile.setnchannels(self.channels)
            wavfile.setsampwidth(2)
            wavfile.setframerate(self.rate)
            wavfile.writeframes(frames)

    def write_output_recording(self, n_audio_file):
        """Combine and write all audio chunks to final output file."""
        input_files = [
            f"chunks/{i}.wav"
            for i in range(n_audio_file)
            if os.path.exists(f"chunks/{i}.wav")
        ]
        with wave.open(self.output_recording_filename, "wb") as wavfile:
            wavfile.setnchannels(self.channels)
            wavfile.setsampwidth(2)
            wavfile.setframerate(self.rate)
            for in_file in input_files:
                with wave.open(in_file, "rb") as wav_in:
                    while True:
                        data = wav_in.readframes(self.chunk)
                        if data == b"":
                            break
                        wavfile.writeframes(data)
                os.remove(in_file)
        
        if os.path.exists("chunks"):
            shutil.rmtree("chunks")

    @staticmethod
    def bytes_to_float_array(audio_bytes):
        """Convert audio bytes to float array."""
        raw_data = np.frombuffer(buffer=audio_bytes, dtype=np.int16)
        return raw_data.astype(np.float32) / 32768.0

    def finalize_recording(self, n_audio_file):
        """Finalize the recording process."""
        if self.save_output_recording and len(self.frames):
            self.write_audio_frames_to_file(
                self.frames[:], f"chunks/{n_audio_file}.wav"
            )
            n_audio_file += 1
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        if self.save_output_recording:
            self.write_output_recording(n_audio_file)

    def record(self):
        """Main recording loop."""
        n_audio_file = 0
        if self.save_output_recording:
            if os.path.exists("chunks"):
                shutil.rmtree("chunks")
            os.makedirs("chunks")
        
        try:
            while not self.exit:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                with self.frames_lock:  # Use separate lock for frames
                    self.frames += data

                # Convert to numpy array for processing
                audio_array = self.bytes_to_float_array(data)
                
                # Check for silence
                if self.check_silence(audio_array):
                    if time.time() - self.last_audio_time > self.silence_threshold:
                        continue
                else:
                    self.last_audio_time = time.time()

                # Process audio
                self.add_frames(audio_array)  # Lock handling is inside add_frames now

                # Save frames if more than a minute
                with self.frames_lock:  # Use separate lock for frames
                    if len(self.frames) > 60 * self.rate:
                        if self.save_output_recording:
                            self.save_chunk(n_audio_file)
                            n_audio_file += 1
                        self.frames = b""

        except KeyboardInterrupt:
            self.finalize_recording(n_audio_file)
        finally:
            self.cleanup()

    def process(self, audio=None, save_file=None):
        """Process audio from file or microphone."""
        if audio is not None:
            resampled_file = utils.resample(audio)
            self.play_file(resampled_file)
        else:
            self.record()


def main():
    """Example usage of LiveTranscriber."""
    # Initialize transcriber
    transcriber = LiveTranscriber(
        model="turbo",
        language="en",
        use_vad=True,
        save_output_recording=True,
        output_recording_filename="output.wav"
    )

    try:
        # Start transcription thread
        transcription_thread = threading.Thread(target=transcriber.speech_to_text)
        transcription_thread.start()

        # Start recording
        print("Recording... Press Ctrl+C to stop")
        transcriber.record()

    except KeyboardInterrupt:
        print("\nStopping recording...")
    finally:
        # Cleanup
        transcriber.cleanup()
        transcription_thread.join()
        
        # Save transcription to SRT file
        transcriber.write_srt_file()
        print("Transcription saved to output.srt")

if __name__ == "__main__":
    main()