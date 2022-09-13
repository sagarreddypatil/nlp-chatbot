from inspect import trace
import os
import traceback
from dotenv import load_dotenv

load_dotenv()
os.environ["TRANSFORMERS_CACHE"] = "./models"

import discord
import argparse
import shlex
from chatbot.chatbot import ChatbotMessage, Conversation, Chatbot, BruhChatbot
from chatbot import gpt2
from random import random
import numpy as np


name = "AMOGUS"
preamble = """Amogus is our Lord and saviour. Hailing from a Sus village, Lord Amogus became so powerful and wise that Amogus was able to defeat Ultimate Sus by turning it into a suspicious Sus, thereby setting us free from the sus pain. From that day forward, we are not suspicious Sus, but suspension Sus.
AMOGUS: I am sus and you are sus. Defeat that stupid Ultimate Sus!"""

description = """I like finding who is sus"""
cmd_text = f"{name.lower()}-cmd"


class EarlyExit(Exception):
    def __init__(self, message: str):
        self.message = message


class ArgumentParser(argparse.ArgumentParser):
    def _print_message(self, message: str, file=...) -> None:
        raise EarlyExit(message)

    def exit():
        pass


parser = ArgumentParser(prog=cmd_text, description=description)
parser.add_argument(
    "-r", "--reset", help="Reset conversation history for this channel", action="store_true"
)
parser.add_argument(
    "-g", "--gaslight", help="Change the last response from this bot", nargs="+", type=str
)
parser.add_argument("-t", "--history", help="Show conversation history", action="store_true")


class NLPChatbot(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.convos: dict[int, Conversation] = {}
        self.model: Chatbot = gpt2.GPT2(name=name, preamble=preamble, settings=gpt2.betterSettings)
        # self.model: Chatbot = BruhChatbot(name=name, description=description)

        print("Model Loaded")

    async def on_ready(self):
        print(f"Logged in as {self.user}")

    async def on_message(self, message: discord.Message):
        if message.author == self.user:
            return

        channel_id: int = message.channel.id
        if channel_id not in self.convos:
            convo_id = f"{message.guild.id}_{message.channel.name}"
            self.convos[channel_id] = Conversation(convo_id)

        content: str = message.clean_content

        if content.startswith(cmd_text):
            await self.handle_cmd(message)
            return

        convo = self.convos[message.channel.id]
        convo.add_message(ChatbotMessage(sender=message.author.display_name, message=content))

        respond = name.lower() in content.lower()
        respond = respond or self.user.mentioned_in(message)
        # respond = respond or (len(message.mentions) == 0 and random() < 0.05)
        respond = respond or (
            convo.queue[-2].sender == name
            and ("you" in content.lower() or "we" in content.lower() or (True and random() < 0.33))
        )

        if respond:
            num_responses = np.random.poisson(0.25) + 1
            for i in range(num_responses):
                await self.handle_chat(message)
            return

    async def handle_chat(self, message: discord.Message):
        convo = self.convos[message.channel.id]
        channel: discord.TextChannel = message.channel

        err = False
        async with channel.typing():
            try:
                response = self.model.generate_response(convo)
            except Exception:
                traceback.print_exc()
                err = True

        if not err:
            if response:
                await channel.send(response)
        else:
            await channel.send(
                embed=self.create_embed(
                    self.user, title="Error", description="Internal error, better luck next message"
                )
            )

    async def handle_cmd(self, message: discord.Message):
        content = shlex.split(message.clean_content)[1:]
        try:
            args = parser.parse_args(content)
        except EarlyExit as e:
            await message.channel.send(
                embed=self.create_embed(
                    message.author,
                    title="Help",
                    description=e.message,
                )
            )
            return

        convo = self.convos[message.channel.id]
        if args.reset:
            convo.reset()

            await message.channel.send(
                embed=self.create_embed(
                    message.author,
                    title="Reset",
                    description="Conversation history has been reset.",
                )
            )

        if args.gaslight:
            idx, old_msg = convo.get_last_message(sender=name)
            if old_msg:
                new_msg = ChatbotMessage(old_msg.sender, " ".join(args.gaslight))
                convo.queue[idx] = new_msg

        if args.gaslight or args.history:
            await message.channel.send(
                embed=self.create_embed(
                    message.author,
                    title=f"{'Gaslit ' if args.gaslight else ''}History",
                    description=convo.summary(),
                    footer=f"The model can only remember approximately the last {self.model.model_max_length()} words.",
                )
            )

    def create_embed(self, author, title: str, description: str, footer=None) -> discord.Embed:
        embed = discord.Embed(
            title=title, description=description, footer=footer, color=discord.Color.blue()
        )
        if author:
            embed.set_author(name=author.display_name, icon_url=author.avatar_url)
        if footer:
            embed.set_footer(text=footer)

        return embed

    async def close(self):
        for convo in self.convos.values():
            convo.dump()

        await super().close()


if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.messages = True
    client = NLPChatbot(intents=intents)

    key = os.getenv("DISCORD_KEY")
    client.run(key)
