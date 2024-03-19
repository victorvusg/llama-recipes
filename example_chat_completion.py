# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 1,
    top_p: float = 1,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs: List[Dialog] = [
        [
            {"role": "user", "content": "Hi, I am Dr Joe, how can I help you today?"},
            {"role": "assistant", "content": "Hi Dr Joe, I've been experiencing some sensitivity in my teeth, especially when I drink hot or cold beverages."},
            {"role": "user", "content": "I'm sorry to hear that. How long have you been experiencing this sensitivity?"},
        ],
        [
            {"role": "user", "content": "I'm sorry to hear that. How long have you been experiencing this sensitivity?"},
            {"role": "assistant", "content": "It started a few weeks ago, but it seems to be getting worse."},
            {"role": "user", "content": "Have there been any changes in your diet or oral hygiene routine recently?"},
        ],
        [
            {"role": "user", "content": "Have there been any changes in your diet or oral hygiene routine recently?"},
            {"role": "assistant", "content": "Not really, though I've probably been drinking more coffee than usual."},
            {"role": "user", "content": "Okay, let's take a look. I'll do a thorough examination of your teeth and gums. [After examination] It looks like you have some enamel wear, which is likely causing the sensitivity. This can happen from acidic foods and drinks like coffee."},
        ],
        [
            {"role": "user", "content": "Okay, let's take a look. I'll do a thorough examination of your teeth and gums. [After examination] It looks like you have some enamel wear, which is likely causing the sensitivity. This can happen from acidic foods and drinks like coffee."},
            {"role": "assistant", "content": "Is there something I can do to reduce the sensitivity?"},
            {"role": "user", "content": "Yes, there are a few things we can do. First, I recommend using a toothpaste designed for sensitive teeth. It can help protect the enamel and reduce sensitivity. Also, try to limit acidic drinks like coffee, and when you do have them, don't brush immediately after – wait for at least 30 minutes."},
        ],
        
    ]
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
