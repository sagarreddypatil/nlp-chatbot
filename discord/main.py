import discord
import argparse
import shlex
from dotenv import load_dotenv
import os
from chatbot.chatbot import ChatbotMessage, Conversation, Chatbot, BruhChatbot
from chatbot import gpt2
from random import random
import numpy as np

load_dotenv()

name = "AMOGUS"
description = """I like finding who is sus"""
key = os.getenv("DISCORD_KEY")

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
        self.model: Chatbot = gpt2.GPT2Large(
            name=name, description=description, settings=gpt2.betterSettings
        )
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
            self.model.init_conversation(self.convos[channel_id])

        content: str = message.content

        if content.startswith(cmd_text):
            await self.handle_cmd(message)
            return

        convo = self.convos[message.channel.id]
        convo.add_message(
            ChatbotMessage(sender=message.author.display_name, message=message.content)
        )

        respond = name.lower() in message.content.lower()
        respond = respond or self.user.mentioned_in(message)
        respond = respond or (len(message.mentions) == 0 and random() < 0.05)
        respond = respond or (
            convo.queue[-2].sender == name and ("you" in message.content.lower() or random() < 0.33)
        )

        if respond:
            num_responses = np.random.poisson(0.25) + 1
            for i in range(num_responses):
                await self.handle_chat(message)
            return

    async def handle_chat(self, message: discord.Message):
        convo = self.convos[message.channel.id]
        channel: discord.TextChannel = message.channel
        async with channel.typing():
            response = self.model.generate_response(convo)
            await channel.send(response)

    async def handle_cmd(self, message: discord.Message):
        content = shlex.split(message.content)[1:]
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
            self.model.init_conversation(convo)

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


intents = discord.Intents.default()
intents.messages = True
client = NLPChatbot(intents=intents)
client.run(key)
