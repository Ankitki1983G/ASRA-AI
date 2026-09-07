from pathlib import Path

import speech_recognition as sr
import soundfile as sf
import torch
import torch.nn.functional as F
from speechbrain.inference.speaker import SpeakerRecognition


class SpeakerVerifier:
    """
    ASRA Speaker Verification

    Current responsibilities:
    1. Owner voice enrollment
    2. Owner speaker embedding creation
    3. New voice recording
    4. Speaker embedding comparison
    5. Threshold-based speaker authentication

    This module does NOT execute ASRA commands.
    """

    # Provisional threshold for speaker authentication.
    # This value will be validated with more voice samples later.
    VERIFICATION_THRESHOLD = 0.50

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
        # parents[2] -> project root
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

        # Owner reference recording.
        self.voice_sample_path = (
            self.security_dir
            / "owner_voice.wav"
        )

        # Owner speaker embedding.
        self.embedding_path = (
            self.security_dir
            / "owner_embedding.pt"
        )

        # Temporary verification recording.
        self.verification_audio_path = (
            self.security_dir
            / "verification_voice.wav"
        )

        print("[SpeakerVerifier] Model loaded successfully.")

    # ======================================================
    # RECORD OWNER VOICE
    # ======================================================

    def record_owner_voice(self):
        """
        Record the owner's reference voice.

        Returns:
            True  -> successful
            False -> failed
        """

        recognizer = sr.Recognizer()

        print("\n[Voice Enrollment]")
        print("Please speak normally for about 5 seconds.")
        print("Example:")
        print('"Hey ASRA, this is my voice profile."')
        print("\nRecording...")

        try:
            with sr.Microphone() as source:

                recognizer.adjust_for_ambient_noise(
                    source,
                    duration=1
                )

                audio = recognizer.listen(
                    source,
                    timeout=10,
                    phrase_time_limit=6
                )

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
    # LOAD AUDIO
    # ======================================================

    def load_audio_file(self, audio_path):
        """
        Load WAV audio safely and convert it into
        a PyTorch waveform tensor.

        Returns:
            waveform, sample_rate
        """

        audio_data, sample_rate = sf.read(
            str(audio_path),
            dtype="float32"
        )

        # Convert stereo → mono.
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        waveform = torch.tensor(
            audio_data,
            dtype=torch.float32
        )

        # SpeechBrain expects [batch, time].
        waveform = waveform.unsqueeze(0)

        return waveform, sample_rate

    # ======================================================
    # CREATE OWNER EMBEDDING
    # ======================================================

    def create_owner_embedding(self):
        """
        Create and save the owner's speaker embedding.

        Returns:
            True  -> successful
            False -> failed
        """

        if not self.voice_sample_path.exists():

            print(
                "[SpeakerVerifier] "
                "Owner voice sample not found."
            )

            return False

        print(
            "[SpeakerVerifier] "
            "Creating owner voice embedding..."
        )

        try:

            waveform, sample_rate = self.load_audio_file(
                self.voice_sample_path
            )

            print(
                f"[SpeakerVerifier] "
                f"Audio sample rate: {sample_rate} Hz"
            )

            embedding = self.verifier.encode_batch(
                waveform,
                normalize=True
            )

            embedding = embedding.detach().cpu()

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
    # ENROLL OWNER
    # ======================================================

    def enroll_owner(self):
        """
        Create the owner's voice profile.

        Existing profile will NOT be overwritten automatically.
        """

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

        recorded = self.record_owner_voice()

        if not recorded:
            return False

        return self.create_owner_embedding()

    # ======================================================
    # RECORD VERIFICATION VOICE
    # ======================================================

    def record_verification_voice(self):
        """
        Record a new voice for speaker verification.

        Returns:
            True  -> successful
            False -> failed
        """

        recognizer = sr.Recognizer()

        print("\n[Speaker Verification]")
        print("Speak normally for about 5 seconds.")
        print("Please say:")
        print('"Hey ASRA, verify my voice."')
        print("\nRecording...")

        try:

            with sr.Microphone() as source:

                recognizer.adjust_for_ambient_noise(
                    source,
                    duration=1
                )

                audio = recognizer.listen(
                    source,
                    timeout=10,
                    phrase_time_limit=6
                )

            with open(
                self.verification_audio_path,
                "wb"
            ) as file:

                file.write(
                    audio.get_wav_data()
                )

            print(
                "[Speaker Verification] "
                "Voice sample saved."
            )

            return True

        except sr.WaitTimeoutError:

            print(
                "[Speaker Verification] "
                "No speech detected."
            )

            return False

        except OSError as error:

            print(
                f"[Speaker Verification] "
                f"Microphone error: {error}"
            )

            return False

        except Exception as error:

            print(
                f"[Speaker Verification] "
                f"Recording error: {error}"
            )

            return False

    # ======================================================
    # VERIFY SPEAKER
    # ======================================================

    def verify_speaker(self):
        """
        Compare the newly recorded voice against
        the enrolled owner voice.

        Returns:
            True  -> speaker is authorized
            False -> speaker is rejected
            None  -> verification could not be completed
        """

        if not self.embedding_path.exists():

            print(
                "[SpeakerVerifier] "
                "Owner voice profile does not exist."
            )

            print(
                "Please enroll the owner first."
            )

            return None

        recorded = self.record_verification_voice()

        if not recorded:
            return None

        print(
            "[SpeakerVerifier] "
            "Creating verification embedding..."
        )

        try:

            # ----------------------------------------------
            # Load owner's saved embedding.
            # ----------------------------------------------

            owner_embedding = torch.load(
                self.embedding_path,
                map_location="cpu",
                weights_only=True
            )

            # ----------------------------------------------
            # Load newly recorded voice.
            # ----------------------------------------------

            waveform, sample_rate = self.load_audio_file(
                self.verification_audio_path
            )

            print(
                f"[SpeakerVerifier] "
                f"Verification sample rate: "
                f"{sample_rate} Hz"
            )

            # ----------------------------------------------
            # Generate new speaker embedding.
            # ----------------------------------------------

            new_embedding = self.verifier.encode_batch(
                waveform,
                normalize=True
            )

            new_embedding = new_embedding.detach().cpu()

            # ----------------------------------------------
            # Cosine similarity.
            # ----------------------------------------------

            similarity = F.cosine_similarity(
                owner_embedding.flatten(),
                new_embedding.flatten(),
                dim=0
            )

            score = similarity.item()

            # ----------------------------------------------
            # Speaker authentication decision.
            # ----------------------------------------------

            is_authorized = (
                score >= self.VERIFICATION_THRESHOLD
            )

            print(
                "\n========================================"
            )

            print(
                "[SpeakerVerifier] "
                f"Similarity Score: {score:.4f}"
            )

            if is_authorized:

                print(
                    "[SpeakerVerifier] "
                    "Speaker Status: AUTHORIZED"
                )

            else:

                print(
                    "[SpeakerVerifier] "
                    "Speaker Status: REJECTED"
                )

            print(
                "========================================"
            )

            return is_authorized

        except Exception as error:

            print(
                f"[SpeakerVerifier] "
                f"Verification failed: {error}"
            )

            return None


# ==========================================================
# STANDALONE TEST
# ==========================================================

if __name__ == "__main__":

    print("========================================")
    print("      ASRA Speaker Verification")
    print("========================================")

    verifier = SpeakerVerifier()

    if verifier.embedding_path.exists():

        print(
            "\n[System] Existing owner profile found."
        )

        print(
            "[System] Starting speaker verification test."
        )

        authorized = verifier.verify_speaker()

        if authorized is not None:

            print(
                "\n[System] "
                "Verification test completed."
            )

            if authorized:

                print(
                    "[System] "
                    "Owner authenticated successfully."
                )

            else:

                print(
                    "[System] "
                    "Speaker rejected."
                )

    else:

        print(
            "\n[System] "
            "No owner profile found."
        )

        print(
            "[System] Starting owner enrollment."
        )

        success = verifier.enroll_owner()

        if success:

            print(
                "\n========================================"
            )

            print(
                "Owner voice enrollment successful."
            )

            print(
                "ASRA owner profile has been created."
            )

            print(
                "========================================"
            )

        else:

            print(
                "\n========================================"
            )

            print(
                "Owner voice enrollment was not completed."
            )

            print(
                "========================================"
            )