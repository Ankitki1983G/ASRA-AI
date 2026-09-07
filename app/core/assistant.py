"""
ASRA Assistant Core Controller
"""

from app.voice.engine import VoiceEngine
from app.command.processor import CommandProcessor
from app.registry.app_registry import AppRegistry
from app.command.fuzzy import FuzzyMatcher
from app.launcher.application_launcher import ApplicationLauncher
from app.scanner.app_scanner import AppScanner
from app.resolver.alias_resolver import AliasResolver


class AsraAssistant:
    """
    Central controller of ASRA AI.
    """

    def __init__(self):

        # -----------------------------
        # Voice Engine
        # -----------------------------

        self.voice_engine = VoiceEngine()

        # -----------------------------
        # Command Processor
        # -----------------------------

        self.command_processor = CommandProcessor()

        # -----------------------------
        # Application Registry
        # -----------------------------

        self.registry = AppRegistry()

        # -----------------------------
        # Alias Resolver
        # -----------------------------

        self.alias_resolver = AliasResolver(
            self.registry
        )

        # -----------------------------
        # Fuzzy Matcher
        # -----------------------------

        self.fuzzy_matcher = FuzzyMatcher()

        # -----------------------------
        # Application Scanner
        # -----------------------------

        self.scanner = AppScanner()

        # -----------------------------
        # Application Launcher
        # -----------------------------

        self.launcher = ApplicationLauncher()

        print(
            "[INFO] ASRA Assistant Initialized Successfully"
        )

    def execute_command(self, text: str):
        """
        Process and execute a text command.
        """

        if not text:

            print(
                "[INFO] No command received."
            )

            return

        # -----------------------------
        # Command Processor
        # -----------------------------

        command = self.command_processor.process(
            text
        )

        print(
            f"[Assistant] {command}"
        )

        # -----------------------------
        # Intent
        # -----------------------------

        if command.get("intent") != "open_application":

            print(
                "[INFO] Unsupported command."
            )

            return

        # -----------------------------
        # Application Name
        # -----------------------------

        raw_name = command.get(
            "target",
            ""
        ).lower().strip()

        if not raw_name:

            print(
                "[ERROR] Application name missing."
            )

            return

        # -----------------------------
        # Alias Resolver
        # -----------------------------

        resolved_name = self.alias_resolver.resolve(
            raw_name
        )

        if resolved_name:

            app_name = resolved_name

            print(
                f"[INFO] Alias Resolved: "
                f"{raw_name} -> {app_name}"
            )

        else:

            app_name = raw_name

        # -----------------------------
        # Registry Search
        # -----------------------------

        app_path = self.registry.get_application_path(
            app_name
        )

        if app_path:

            print(
                f"[INFO] Found in Registry: "
                f"{app_name}"
            )

            success = self.launcher.launch(
                app_path
            )

            if success:

                return

        print(
            "[INFO] Registry path not found."
        )

        # -----------------------------
        # Scanner Fallback
        # -----------------------------

        print(
            "[INFO] Scanning installed applications..."
        )

        applications = self.scanner.scan()

        if not applications:

            print(
                "[ERROR] No applications detected."
            )

            return

        # -----------------------------
        # Exact Match First
        # -----------------------------

        exact_match = None

        for app in applications:

            if app.get("name", "").lower().strip() == app_name:

                exact_match = app

                break

        # -----------------------------
        # Fuzzy Matching
        # -----------------------------

        if exact_match:

            matched_app = exact_match

            print(
                f"[INFO] Exact Match Found: "
                f"{matched_app['name']}"
            )

        else:

            app_names = [
                app["name"]
                for app in applications
            ]

            best_match = self.fuzzy_matcher.find_best_match(
                app_name,
                app_names
            )

            if not best_match:

                print(
                    f"[ERROR] Application "
                    f"'{app_name}' not found."
                )

                return

            print(
                f"[INFO] Best Match Found: "
                f"{best_match}"
            )

            matched_app = None

            for app in applications:

                if app["name"] == best_match:

                    matched_app = app

                    break

            if not matched_app:

                print(
                    "[ERROR] Matched application data not found."
                )

                return

        # -----------------------------
        # Get Application Path
        # -----------------------------

        path = matched_app.get(
            "path"
        )

        if not path:

            print(
                "[ERROR] Application path missing."
            )

            return

        # -----------------------------
        # Application Type
        # -----------------------------

        app_type = matched_app.get(
            "type",
            "desktop"
        )

        print(
            f"[INFO] Launching: {path}"
        )

        print(
            f"[INFO] Application Type: "
            f"{app_type}"
        )

        # -----------------------------
        # Launch Application
        # -----------------------------

        success = self.launcher.launch(
            path,
            app_type
        )

        if not success:

            print(
                "[ERROR] Failed to launch application."
            )

    def run_text(self, text: str):
        """
        Execute ASRA using text input.
        """

        self.execute_command(
            text
        )

    def run(self):
        """
        Execute ASRA using voice input.
        """

        print(
            "[ASRA] Standby mode..."
        )

        text = self.voice_engine.listen()

        if not text:

            print(
                "[INFO] No command received."
            )

            return

        # -----------------------------
        # Wake Word Detection
        # -----------------------------

        if "hey asra" not in text.lower():

            print(
                "[ASRA] Wake word not detected."
            )

            return

        print(
            "[ASRA] Wake word detected."
        )

        # -----------------------------
        # Execute Command
        # -----------------------------

        self.execute_command(
            text
        )
