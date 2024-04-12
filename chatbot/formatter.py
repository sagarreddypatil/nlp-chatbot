from .conversation import Conversation, ChatbotMessage
import arrow
import re

class Formatter():
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.stop_pattern = re.compile(r"\n\[|\n.*\[.+\]<.*>|\n-+|\n\\[A-Za-z]+{|\n<|\n.*\\")

    def format_time(self, timestamp: int) -> str:
        return arrow.get(timestamp).humanize()

    def format(self, convo: Conversation, trail=True) -> str:
        out = self.preamble + "\n"
        message: ChatbotMessage = None

        for message in convo.queue:
            out += f"[{self.format_time(message.timestamp)}]<{message.sender}>{message.message}\n"

        if trail:
            out += f"[{self.format_time(arrow.utcnow())}]<{self.name}>"

        return out
