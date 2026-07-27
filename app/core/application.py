"""
Application Controller

This module controls the complete lifecycle of ASTRA AI.
"""
from app.utils.logger import get_logger
from app.config.settings import APP_NAME, VERSION
from app.voice.engine import VoiceEngine

class Application:
    """
    Main Application Controller
    """

    def __init__(self):
        self.logger = get_logger()
        self.app_name = APP_NAME
        self.version = VERSION
        self.voice = VoiceEngine()
       

    def run(self):
       self.logger.info("=" * 50)
       self.logger.info(f"Welcome to {self.app_name}")
       self.logger.info(f"Version : {self.version}")
       self.logger.info("Application Started Successfully...")
       self.logger.info("=" * 50)
       self.logger.info("Voice Engine Initialized")
       self.voice.listen()