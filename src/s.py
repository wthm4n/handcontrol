import os
import random
import logging
from pathlib import Path

import discord
from discord import app_commands, FFmpegPCMAudio
from discord.ext import commands

try:
    from yt_dlp import YoutubeDL
except ImportError:
    from youtube_dl import YoutubeDL

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('bot.log', encoding='utf-8'),
        logging.StreamHandler(),
    ],
)

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)


@bot.event
async def on_ready():
    logging.info(f'Bot ready: {bot.user} (ID: {bot.user.id})')


@bot.event
async def on_command(ctx):
    logging.info(
        f'Command invoked: {ctx.command} by {ctx.author} in '
        f'{ctx.guild.name if ctx.guild else "DM"}/{ctx.channel}'
    )


@bot.event
async def on_command_completion(ctx):
    logging.info(f'Command completed: {ctx.command} by {ctx.author}')


@bot.event
async def on_command_error(ctx, error):
    logging.error(f'Command error: {ctx.command} by {ctx.author}: {error}')
    if isinstance(error, commands.CommandNotFound):
        return
    await ctx.send(f'Error: {error}')


@bot.command(name='ping')
async def ping(ctx):
    await ctx.send('Pong!')


@bot.command(name='echo')
async def echo(ctx, *, message: str):
    await ctx.send(message)


@bot.command(name='join')
async def join_voice(ctx):
    if ctx.author.voice is None:
        await ctx.send('You need to be in a voice channel first.')
        return
    channel = ctx.author.voice.channel
    if ctx.voice_client is not None:
        await ctx.voice_client.move_to(channel)
    else:
        await channel.connect()
    await ctx.send(f'Joined voice channel: {channel.name}')


@bot.command(name='leave')
async def leave_voice(ctx):
    voice_client = ctx.voice_client
    if voice_client is None:
        await ctx.send('I am not connected to a voice channel.')
        return
    await voice_client.disconnect()
    await ctx.send('Disconnected from voice channel.')


@bot.command(name='play')
async def play_music(ctx, *, url: str):
    if ctx.author.voice is None:
        await ctx.send('You need to be in a voice channel to play music.')
        return
    channel = ctx.author.voice.channel
    voice_client = ctx.voice_client
    if voice_client is None:
        voice_client = await channel.connect()
    elif voice_client.channel != channel:
        await voice_client.move_to(channel)

    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'noplaylist': True,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if 'entries' in info:
                info = info['entries'][0]
            audio_url = info['url']
            title = info.get('title', 'audio')
    except Exception as exc:
        logging.error(f'Error extracting audio from {url}: {exc}')
        await ctx.send(f'Error processing URL: {exc}')
        return

    source = FFmpegPCMAudio(audio_url, executable='ffmpeg')
    if voice_client.is_playing():
        voice_client.stop()
    voice_client.play(source)
    await ctx.send(f'Now playing: {title}')


@bot.tree.command(name='ping', description='Responds with Pong!')
async def ping_slash(interaction: discord.Interaction):
    await interaction.response.send_message('Pong!')


@bot.tree.command(name='echo', description='Echoes a message')
@app_commands.describe(message='Message to echo')
async def echo_slash(interaction: discord.Interaction, message: str):
    await interaction.response.send_message(message)


if __name__ == '__main__':
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        raise RuntimeError('Set DISCORD_TOKEN environment variable')
    bot.run(token)


from __future__ import annotations
from pathlib import Path

@bot.command(name='readfile')
async def read_file(ctx, *, filename: str):
    path = Path(filename)
    if not path.is_file():
        await ctx.send(f'File not found: {filename}')
        return
    try:
        content = path.read_text(encoding='utf-8')
        await ctx.send(f'Contents of {filename}:\n```\n{content}\n```')
    except Exception as exc:
        logging.error(f'Error reading file {filename}: {exc}')
        await ctx.send(f'Error reading file: {exc}')
    

@bot.command(name='random')
async def random_number(ctx, start: int = 0, end: int = 100
):
    number = random.randint(start, end)
    await ctx.send(f'Random number between {start} and {end}: {number}')


@bot.command(name='helpme')
async def help_me(ctx):
    help_text = (
        "Available commands:\n"
        "!ping - Responds with Pong!\n"
        "!echo <message> - Echoes the provided message\n"
        "!join - Joins your voice channel\n"
        "!leave - Leaves the voice channel\n"
        "!play <url> - Plays audio from a URL in voice chat\n"
        "!readfile <filename> - Reads and displays the contents of a file\n"
        "!random [start] [end] - Generates a random number between start and end (default 0-100)\n"
        "!helpme - Shows this help message\n"
    )
    await ctx.send(help_text)


