from pathlib import Path

import speech_recognition as sr
import soundfile as sf
import torch
from speechbrain.inference.speaker import SpeakerRecognition


class SpeakerVerifier:
    """
    ASRA Speaker Verification

    Current responsibility:
    - Record owner's voice
    - Generate speaker embedding
    - Save owner voice profile

    This module does not execute commands.
    """

    def __init__(self):
        print("[SpeakerVerifier] Loading speaker verification model...")

        # Load pretrained SpeechBrain speaker-recognition model.
        self.verifier = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb"
        )

        # --------------------------------------------------
        # Find ASRA project root
        #
        # app/voice/speaker_verification.py
        #       ↑
        # parents[0] -> voice
        # parents[1] -> app
        # parents[2] -> Asra-AI
        # --------------------------------------------------

        self.project_root = Path(__file__).resolve().parents[2]

        # --------------------------------------------------
        # Security data directory
        # --------------------------------------------------

        self.security_dir = (
            self.project_root
            / "data"
            / "security"
        )

        self.security_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Owner voice recording
        self.voice_sample_path = (
            self.security_dir
            / "owner_voice.wav"
        )

        # Owner speaker embedding
        self.embedding_path = (
            self.security_dir
            / "owner_embedding.pt"
        )

        print("[SpeakerVerifier] Model loaded successfully.")

    # ======================================================
    # RECORD OWNER VOICE
    # ======================================================

    def record_owner_voice(self):
        """
        Record the owner's voice from the microphone.

        Returns:
            True  -> recording successful
            False -> recording failed
        """

        recognizer = sr.Recognizer()

        print("\n[Voice Enrollment]")
        print("Please speak normally for about 5 seconds.")
        print("Say something like:")
        print('"Hey ASRA, this is my voice profile."')
        print("\nRecording...")

        try:

            with sr.Microphone() as source:

                # Adjust microphone according to
                # current background noise.
                recognizer.adjust_for_ambient_noise(
                    source,
                    duration=1
                )

                # Record voice.
                audio = recognizer.listen(
                    source,
                    timeout=10,
                    phrase_time_limit=6
                )

            # Save recorded audio.
            with open(
                self.voice_sample_path,
                "wb"
            ) as file:

                file.write(
                    audio.get_wav_data()
                )

            print(
                "[Voice Enrollment] "
                "Voice sample saved."
            )

            return True

        except sr.WaitTimeoutError:

            print(
                "[Voice Enrollment] "
                "No speech detected."
            )

            return False

        except OSError as error:

            print(
                f"[Voice Enrollment] "
                f"Microphone error: {error}"
            )

            return False

        except Exception as error:

            print(
                f"[Voice Enrollment] "
                f"Recording error: {error}"
            )

            return False

    # ======================================================
    # CREATE OWNER EMBEDDING
    # ======================================================

    def create_owner_embedding(self):
        """
        Convert owner's recorded voice into a
        speaker embedding.

        Returns:
            True  -> embedding created
            False -> failed
        """

        # Check voice recording.
        if not self.voice_sample_path.exists():

            print(
                "[SpeakerVerifier] "
                "Voice sample not found."
            )

            return False

        print(
            "[SpeakerVerifier] "
            "Creating owner voice embedding..."
        )

        try:

            # ------------------------------------------------
            # STEP 1
            # Read WAV file directly using soundfile.
            #
            # We intentionally do NOT use:
            #
            # self.verifier.load_audio(...)
            #
            # because that was causing the Windows
            # path duplication problem.
            # ------------------------------------------------

            audio_data, sample_rate = sf.read(
                str(self.voice_sample_path),
                dtype="float32"
            )

            print(
                f"[SpeakerVerifier] "
                f"Audio sample rate: {sample_rate} Hz"
            )

            # ------------------------------------------------
            # STEP 2
            # Convert stereo audio to mono.
            # ------------------------------------------------

            if audio_data.ndim > 1:

                audio_data = audio_data.mean(
                    axis=1
                )

            # ------------------------------------------------
            # STEP 3
            # Convert audio to PyTorch tensor.
            # ------------------------------------------------

            waveform = torch.tensor(
                audio_data,
                dtype=torch.float32
            )

            # ------------------------------------------------
            # STEP 4
            # SpeechBrain expects:
            #
            # [batch, time]
            #
            # Add batch dimension.
            # ------------------------------------------------

            waveform = waveform.unsqueeze(0)

            # ------------------------------------------------
            # STEP 5
            # Generate speaker embedding.
            # ------------------------------------------------

            embedding = self.verifier.encode_batch(
                waveform,
                normalize=True
            )

            # ------------------------------------------------
            # STEP 6
            # Move embedding to CPU.
            # ------------------------------------------------

            embedding = embedding.detach().cpu()

            # ------------------------------------------------
            # STEP 7
            # Save owner embedding.
            # ------------------------------------------------

            torch.save(
                embedding,
                self.embedding_path
            )

            print(
                "[SpeakerVerifier] "
                "Owner voice embedding saved."
            )

            print(
                f"[SpeakerVerifier] "
                f"Profile saved at: "
                f"{self.embedding_path}"
            )

            return True

        except Exception as error:

            print(
                f"[SpeakerVerifier] "
                f"Embedding creation failed: {error}"
            )

            return False

    # ======================================================
    # OWNER ENROLLMENT
    # ======================================================

    def enroll_owner(self):
        """
        Complete owner voice enrollment.

        Existing owner profile will not be automatically
        overwritten.
        """

        # --------------------------------------------------
        # Prevent accidental replacement of owner profile.
        # --------------------------------------------------

        if self.embedding_path.exists():

            print(
                "\n[SpeakerVerifier] "
                "An owner voice profile already exists."
            )

            print(
                "[SpeakerVerifier] "
                "Enrollment will not overwrite it "
                "automatically."
            )

            return False

        # --------------------------------------------------
        # Record owner's voice.
        # --------------------------------------------------

        recorded = self.record_owner_voice()

        if not recorded:

            return False

        # --------------------------------------------------
        # Create speaker embedding.
        # --------------------------------------------------

        return self.create_owner_embedding()


# ==========================================================
# STANDALONE TEST
# ==========================================================

if __name__ == "__main__":

    print("========================================")
    print("      ASRA Speaker Verification")
    print("          Owner Enrollment")
    print("========================================")

    verifier = SpeakerVerifier()

    success = verifier.enroll_owner()

    if success:

        print("\n========================================")
        print("Owner voice enrollment successful.")
        print("ASRA owner profile has been created.")
        print("========================================")

    else:

        print("\n========================================")
        print("Owner voice enrollment was not completed.")
        print("========================================")