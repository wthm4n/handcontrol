import os
import logging

import discord
from discord.ext import commands

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


if __name__ == '__main__':
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        raise RuntimeError('Set DISCORD_TOKEN environment variable')
    bot.run(token)
