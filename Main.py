import os
import ray
from AudioProcessor import AudioProcessor
from OutputWriter import OutputWriter

def get_audio_files(directory):
    """Returns all .wav files in the specified directory."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]

@ray.remote
def process_audio_file(audio_file):
    print(f"Processing on node: {ray.util.get_node_ip_address()}")
    processor = AudioProcessor(audio_file, device="cpu")  # Use CPU only for Ray cluster compatibility
    processor.load_models()
    return processor.process_audio()

def main():
    # Set up Ray runtime environment
    runtime_env = {"pip": ["whisperx", "torch", "ffmpeg-python"]}
    ray.init(runtime_env=runtime_env, address="auto")  # Connect to the Ray cluster

    # Specify the audio directory
    audio_directory = "/root"  # Update this to actual audio data path
    audio_files = get_audio_files(audio_directory)

    if not audio_files:
        print("No audio files found in the specified directory.")
        return

    # Process files in parallel using Ray tasks
    results = ray.get([process_audio_file.remote(audio) for audio in audio_files])

    # Example OutputWriter handling (if needed)
    output_writer = OutputWriter()
    for result in results:
        output_writer.write(result)

    ray.shutdown()

if __name__ == "__main__":
    main()
