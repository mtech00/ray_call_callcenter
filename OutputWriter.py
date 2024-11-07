import uuid
import os
import datetime

class OutputWriter:
    def __init__(self, output_directory="output"):
        self.output_directory = output_directory
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

    def write(self, result):
        # Generate a unique filename using UUID
        filename = f"{self.output_directory}/transcription_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4()}.txt"

        # Convert transcription and diarization results to strings if they are not already
        transcription = str(result["transcription"])
        diarization = str(result["diarization"])

        with open(filename, 'w') as f:
            f.write("Transcription:\n")
            f.write(transcription)  # Ensure transcription is in string format
            f.write("\n\nDiarization:\n")
            f.write(diarization)  # Ensure diarization is in string format
        print(f"Result saved to {filename}")
