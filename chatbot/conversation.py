import datetime
import os
from typing import NamedTuple, Optional, Tuple


class ChatbotMessage(NamedTuple):
    sender: str
    message: str


class Conversation(object):
    def __init__(self, id: str, logdir: Optional[str] = None):
        self.id = id
        self.__queue: list(ChatbotMessage) = []
        self.start_offset = 0
        self.logdir = logdir

    def add_message(self, message: ChatbotMessage):
        self.__queue.append(message)

    def get_last_message(self, sender: str = None) -> Tuple[int, ChatbotMessage]:
        if sender is None:
            return self.__queue[-1]

        for i in range(len(self.__queue) - 1, -1, -1):
            if self.__queue[i].sender == sender:
                return i, self.__queue[i]

    def dequeue(self):
        self.start_offset += 1

    def get_queue(self):
        return self.__queue[self.start_offset :]

    def amend(self, idx: int, message: ChatbotMessage):
        self.__queue[idx] = message

    queue = property(fget=get_queue)

    def dump(self):
        if self.logdir is None:
            return

        if os.path.isfile(self.logdir):
            raise ValueError("logdir is a file")

        if not os.path.isdir(self.logdir):
            os.mkdir(self.logdir)

        uid = f"{self.id}_{datetime.now().isoformat()}"
        with open(os.path.join(self.logdir, f"{uid}.txt"), "w") as f:
            f.write(self.summary(full=True))

    def reset(self):
        self.dump()

        self.start_offset = 0
        self.__queue = []

    def summary(self, full=False) -> str:
        out = ""
        msgs = self.__queue if full else self.queue

        for message in msgs:
            out += f"{message.sender}: {message.message}\n"

        return out
