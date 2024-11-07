import whisperx
import os
import datetime
import torch

class AudioProcessor:
    def __init__(self, audio_file, device="cpu", batch_size=64, compute_type="int8"):
        self.audio_file = audio_file
        self.device = device
        self.batch_size = batch_size
        self.compute_type = compute_type
        self.model = None
        self.diarize_model = None

    def load_models(self):
        print("Model loading started", datetime.datetime.now())
        self.model = whisperx.load_model("tiny", self.device, compute_type=self.compute_type)
        self.diarize_model = whisperx.DiarizationPipeline(
            use_auth_token="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",
            device=torch.device('cpu'),  # Use CPU for Ray compatibility
            model_name="pyannote/speaker-diarization"
        )

    def process_audio(self):
        # Transcribe and perform diarization
        print("Transcription started", datetime.datetime.now())
        audio = whisperx.load_audio(self.audio_file)
        transcription = self.model.transcribe(audio)
        diarization_result = self.diarize_model(audio)

        return {
            "transcription": transcription,
            "diarization": diarization_result
        }
