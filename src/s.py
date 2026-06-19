from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich import box
import random

console = Console()

RESPONSES = {
    "oiiiii betuuuu kkrh": [
        "bsssssss tumhara kaam khud to krte ni kuch huh 😒"
    ],
    "hii": [
        "hiiiiii betuuuuuuu 🫶",
        "hellooooo ji ✨"
    ],
    "khana khaya": [
        "haan ji, tumne khaya? 😤",
        "nhiii, khila do 😔"
    ]
}

def get_reply(msg):
    msg = msg.lower().strip()

    for trigger, replies in RESPONSES.items():
        if trigger in msg:
            return random.choice(replies)

    return random.choice([
        "hmmmmmmmm 🤨",
        "acha jiii 😌",
        "kya baat haiii 😭",
        "tum ajeeb ho 😒"
    ])

console.clear()

title = Text("💖 Simmi Chat 💖", style="bold magenta")
console.print(
    Panel(
        title,
        border_style="bright_magenta",
        box=box.DOUBLE,
        subtitle="Type 'exit' to quit"
    )
)

while True:
    user = Prompt.ask("\n[bold cyan]You[/bold cyan]")

    if user.lower() == "exit":
        console.print("\n[bold red]Byeeeeeeeeee 👋[/bold red]")
        break

    reply = get_reply(user)

    console.print(
        Panel(
            f"[bold pink1]{reply}[/bold pink1]",
            title="💗 Simmi",
            border_style="magenta",
            box=box.ROUNDED
        )
    )