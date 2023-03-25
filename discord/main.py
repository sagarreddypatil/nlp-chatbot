import time
import discord
import argparse
import shlex
from chatbot.chatbot import ChatbotMessage, Conversation, Chatbot, BruhChatbot
from chatbot import transformer
from random import random
import numpy as np
import logging

import sys
import os
from dotenv import load_dotenv

load_dotenv()
# os.environ["TRANSFORMERS_CACHE"] = "./models"


log_formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(name)s: %(message)s")

log_handler_file = logging.FileHandler("discord.log", encoding="utf-8", mode="w")
log_handler_file.setFormatter(log_formatter)

log_handler_console = logging.StreamHandler(sys.stdout)
log_handler_console.setFormatter(log_formatter)

logger = logging.getLogger(__name__)
logger.addHandler(log_handler_file)
logger.addHandler(log_handler_console)
logger.setLevel(logging.INFO)

discord_logger = logging.getLogger("discord")
discord_logger.setLevel(logging.INFO)
logging.getLogger("discord.http").setLevel(logging.INFO)
discord_logger.addHandler(log_handler_file)
discord_logger.addHandler(log_handler_console)

chat_logdir = "chatlogs/"


name = "AMOGUS"
preamble = """
Hailing from a Sus village, Lord AMOGUS became so powerful and wise that he was able to defeat Ultimate Sus by turning it into a suspicious Sus, thereby setting us free from the sus pain.
"I am sus and you are sus. Defeat that stupid Ultimate Sus!" - AMOGUS

The following excerpt is a conversation between a superintelligent AI secretly taking the form of AMOGUS, on Discord.
The AI, and conversely AMOGUS, was created by the user binarypasta
-----"""

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
parser.add_argument("-r", "--reset", help="Reset conversation history for this channel", action="store_true")
parser.add_argument("-g", "--gaslight", help="Change the last response from this bot", nargs="+", type=str)
parser.add_argument("-t", "--history", help="Show conversation history", action="store_true")


class NLPChatbot(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.convos: dict[int, Conversation] = {}
        logger.info("Loading Model")
        self.model: Chatbot = transformer.Transformer(name=name, preamble=preamble, settings=transformer.llama13b)
        # self.model: Chatbot = BruhChatbot(name=name, preamble=preamble)

        logger.info("Model Loaded")

    async def on_ready(self):
        logger.info(f"Logged in as {self.user}")

    async def on_message(self, message: discord.Message):
        if message.author == self.user:
            return

        channel_id: int = message.channel.id
        content: str = message.clean_content
        dm = isinstance(message.channel, discord.DMChannel)

        if channel_id not in self.convos:
            # check if dm or server channel
            if dm:
                convo_id = f"dm_{message.author.name}_{message.author.discriminator}"
            else:
                convo_id = f"{message.guild.id}_{message.channel.name}"

            self.convos[channel_id] = Conversation(f"{convo_id}_{int(time.time())}", logdir=chat_logdir)

        if content.startswith(cmd_text):
            await self.handle_cmd(message)
            return

        convo = self.convos[message.channel.id]
        convo.add_message(ChatbotMessage(sender=message.author.display_name, message=content))

        respond = name.lower() in content.lower()
        respond = respond or self.user.mentioned_in(message)
        # respond = respond or (len(message.mentions) == 0 and random() < 0.05)
        respond = respond or (len(convo.queue) >= 2 and convo.queue[-2].sender == name and ("you" in content.lower() or "we" in content.lower() or (True and random() < 0.33)))
        respond = respond or dm

        if respond:
            await self.handle_chat(message)
            return

    # async def generate_response(self, convo: Conversation) -> str:
    #     return self.model.generate_response(convo)

    async def handle_chat(self, message: discord.Message):
        convo = self.convos[message.channel.id]
        channel: discord.TextChannel = message.channel

        # message: discord.Message = await channel.send("Thinking...")

        def update(response: str):
            pass
            # self.loop.create_task(message.edit(content=response))

        err = False
        final_response = ""
        async with channel.typing():
            try:
                # final_response = await self.generate_response(convo)
                final_response = self.model.generate_response(convo, update)
            except Exception as exc:
                logger.exception(exc)
                err = True

        # await message.delete()

        if err:
            await channel.send(embed=self.create_embed(self.user, title="Error", description="Internal error, better luck next message"))
        elif final_response:
            await channel.send(final_response)

        # err = False
        # async with channel.typing():
        #     try:
        #         response = await self.generate_response(convo)
        #     except Exception as exc:
        #         logger.exception(exc)
        #         err = True

        # if not err:
        #     if response:
        #         await channel.send(response)
        # else:
        #     await channel.send(embed=self.create_embed(self.user, title="Error", description="Internal error, better luck next message"))

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
            convo.dump()
            del self.convos[message.channel.id]

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
                convo.amend(idx, new_msg)

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
        embed = discord.Embed(title=title, description=description, color=discord.Color.blue())
        if author:
            embed.set_author(name=author.display_name, icon_url=author.avatar.url)
        if footer:
            embed.set_footer(text=footer)

        return embed

    async def close(self):
        for convo in self.convos.values():
            convo.dump()

        await super().close()


if __name__ == "__main__":
    try:
        intents = discord.Intents.default()
        intents.messages = True
        intents.message_content = True
        client = NLPChatbot(intents=intents)

        key = os.getenv("DISCORD_KEY")
        client.run(key, log_handler=None)
    except Exception as exc:
        logger.exception(exc)
        raise exc
