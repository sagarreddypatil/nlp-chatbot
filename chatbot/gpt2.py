import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from .chatbot import *


class GPT2LargeSettings(NamedTuple):
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float


crackheadSettings = GPT2LargeSettings(
    temperature=0.7, top_p=0.9, top_k=None, repetition_penalty=2.0
)

betterSettings = GPT2LargeSettings(temperature=1.0, top_p=0.9, top_k=None, repetition_penalty=1.33)


# Sike it's actually distilgpt2
class GPT2Large(Chatbot):
    def _init_model(self, settings: GPT2LargeSettings = betterSettings):
        self.settings = settings
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not self.force_cpu else "cpu"
        )

        self.model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(self.device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-large")
        self.model.eval()

        self.eos = self.tokenizer.encode("\n")[0]

        self.seperator = ":"

    def model_max_length(self) -> str:
        return str(self.tokenizer.model_max_length)


    def _generate(self, convo: Conversation) -> str:
        input_text = self._generate_model_input(convo) + f"{self.name}{self.seperator}"
        while len(self.tokenizer.encode(input_text)) >= self.tokenizer.model_max_length:
            convo.do_fifo()
            input_text = self._generate_model_input(convo) + f"{self.name}{self.seperator}"

        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        outputs = self.model.generate(
            input_ids.to(self.device),
            max_length=self.tokenizer.model_max_length,
            temperature=self.settings.temperature,
            top_p=self.settings.top_p,
            repetition_penalty=self.settings.repetition_penalty,
            eos_token_id=self.eos,
            pad_token_id=50256,
        )
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("\n==============================")
        print(output)
        print("==============================\n")

        output = output.split(f"{self.name}{self.seperator.strip()}")[-1].replace("\n", "").strip()

        return output


if __name__ == "__main__":
    test(settings=betterSettings)
