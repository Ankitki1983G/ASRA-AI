"""
Command Processing Module

This module receives recognised speech text and converts it
into structured commands that other ASRA AI modules can use.
"""


class CommandProcessor:

    def __init__(self):
        pass

    def process(self, text: str) -> dict:
        """
        Process the user's command.
        """

        normalized_text = self.normalize(text)

        action = self.detect_action(normalized_text)

        target = self.extract_target(
            normalized_text,
            action
        )

        command = self.build_command(
            action,
            target
        )

        print(f"[Command Processor] {command}")

        return command

    # --------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------

    def normalize(self, text: str) -> str:
        """
        Normalize recognized speech text.
        """

        normalized_text = text.strip().lower()

        # Remove extra spaces
        normalized_text = " ".join(
            normalized_text.split()
        )

        return normalized_text

    # --------------------------------------------------
    # ACTION DETECTION
    # --------------------------------------------------

    def detect_action(self, text: str) -> str:
        """
        Detect the action requested by the user.
        """

        open_keywords = [
            "open",
            "launch",
            "start",
            "run",
            "khol",
            "kholo",
            "open karo",
            "open the",
            "hey asra open",
            "hey asra kholo",
        ]

        search_keywords = [
            "search",
            "find",
            "look for",
        ]

        for keyword in open_keywords:

            if keyword in text:
                return "open_application"

        for keyword in search_keywords:

            if keyword in text:
                return "web_search"

        return "unknown"

    # --------------------------------------------------
    # INTENT DETECTION
    # --------------------------------------------------

    def detect_intent(self, text: str) -> dict:
        """
        Detect the user's intent from normalized text.
        """

        text = self.normalize(text)

        action = self.detect_action(text)

        target = self.extract_target(
            text,
            action
        )

        return self.build_command(
            action,
            target
        )

    # --------------------------------------------------
    # TARGET EXTRACTION
    # --------------------------------------------------

    def extract_target(
        self,
        text: str,
        action: str
    ) -> str:
        """
        Extract the actual application or search target.
        """

        if action == "open_application":

            remove_words = [
                "hey asra",
                "open",
                "launch",
                "start",
                "run",
                "please",
                "the",
                "khol",
                "kholo",
                "khol do",
                "kholna",
                "open karo",
                "launch karo",
                "start karo",
                "run karo",
                "please open",
                "please launch",
                "please start",
                "karo",
                "do",
                "to",
            ]

        elif action == "web_search":

            remove_words = [
                "search",
                "find",
                "look for",
                "please",
                "search karo",
                "find karo",
                "karo",
                "do",
                "to",
            ]

        else:

            return ""

        target = text

        # Remove phrases first
        # Longer phrases should be removed before
        # smaller words.

        remove_words = sorted(
            remove_words,
            key=len,
            reverse=True
        )

        for word in remove_words:

            target = target.replace(
                word,
                " "
            )

        # Clean multiple spaces

        target = " ".join(
            target.split()
        )

        # Remove unwanted trailing conversational words

        trailing_words = [
            "to",
            "karo",
            "do",
            "please",
            "hai",
            "na",
        ]

        words = target.split()

        while words and words[-1] in trailing_words:

            words.pop()

        target = " ".join(words)

        return target.strip()

    # --------------------------------------------------
    # BUILD COMMAND
    # --------------------------------------------------

    def build_command(
        self,
        action: str,
        target: str
    ) -> dict:
        """
        Build a structured command dictionary.
        """

        if action == "open_application":

            return {
                "intent": action,
                "target": target
            }

        elif action == "web_search":

            return {
                "intent": action,
                "query": target
            }

        return {
            "intent": "unknown",
            "text": target
        }