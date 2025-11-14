"""
Discord bot for FTD Top 200 reporting using Pycord.

This bot exposes a "/ftd_top_250" slash command that runs the local
FTD_Top_250.py script and posts the results (along with any generated
CSV and XLSX files) into the configured channel. The bot also
schedules a daily run at 3 AM Eastern. Make sure the bot is invited
with the applications.commands scope for slash commands to appear.
"""

import asyncio
import os
from datetime import datetime, timedelta, time as dt_time
from typing import Optional
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    ZoneInfo = None

import discord

async def run_ftd_script() -> str:
    """Run the FTD_Top_250.py script and return its combined stdout/stderr."""
    proc = await asyncio.create_subprocess_exec(
        os.getenv("PYTHON", "python3"),
        os.path.join(os.path.dirname(__file__), "FTD_Top_250.py"),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_bytes, stderr_bytes = await proc.communicate()
    output = stdout_bytes.decode().strip()
    errors = stderr_bytes.decode().strip()
    if errors:
        output = f"{output}\n\nErrors:\n{errors}"
    return output

async def send_report(bot: discord.Bot, channel_id: int) -> None:
    """Run the FTD script and post its output and any CSV/XLSX files."""
    channel = bot.get_channel(channel_id)
    if channel is None:
        raise RuntimeError(f"Could not find channel {channel_id}")
    report = await run_ftd_script()
    # Discord limits messages to 2000 characters; show the tail if long
    summary = report[-1800:] if len(report) > 1800 else report
    files = []
    try:
        import glob
        from pathlib import Path
        workdir = Path(os.path.dirname(__file__))
        csv_files = sorted(glob.glob(str(workdir / "FTD_Top_250_*.csv")), key=os.path.getmtime, reverse=True)
        xlsx_files = sorted(glob.glob(str(workdir / "FTD_Top_250_*.xlsx")), key=os.path.getmtime, reverse=True)
        for flist in (csv_files, xlsx_files):
            if flist:
                latest = flist[0]
                files.append(discord.File(latest, filename=os.path.basename(latest)))
    except Exception:
        pass
    if files:
        await channel.send(content=summary, files=files)
    else:
        await channel.send(content=summary)

def seconds_until_next_run(hour: int, minute: int, tz: Optional[ZoneInfo]) -> float:
    now = datetime.now(tz) if tz else datetime.now()
    target = datetime.combine(now.date(), dt_time(hour, minute), tzinfo=tz)
    if now >= target:
        target += timedelta(days=1)
    return (target - now).total_seconds()

async def scheduler(bot: discord.Bot, channel_id: int) -> None:
    tz = ZoneInfo("America/New_York") if ZoneInfo else None
    while True:
        await asyncio.sleep(seconds_until_next_run(3, 0, tz))
        try:
            await send_report(bot, channel_id)
        except Exception as exc:
            print(f"Error sending FTD_Top_250 report: {exc}")
        await asyncio.sleep(5)

class FTDTop200Client(discord.Bot):
    """Discord bot that schedules a daily report and exposes a slash command."""

    def __init__(self, channel_id: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.channel_id = channel_id

    async def on_ready(self) -> None:
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        self.loop.create_task(scheduler(self, self.channel_id))

    async def setup_hook(self) -> None:
        @self.slash_command(
            name="ftd_Top_250",
            description="Run FTD Top 200 script and post the results in this channel.",
        )
        async def ftd_Top_250(ctx: discord.ApplicationContext) -> None:
            await ctx.defer(ephemeral=True)
            try:
                await send_report(self, self.channel_id)
                await ctx.respond("FTD_Top_250 report posted.")
            except Exception as exc:
                await ctx.respond(f"Failed: {exc}", delete_after=30)
        # Register commands globally (note: may take up to an hour to appear)
        await self.sync()

def main() -> None:
    token = os.environ.get("DISCORD_BOT_TOKEN")
    channel_id = os.environ.get("DISCORD_CHANNEL_ID")
    if not token:
        raise RuntimeError("DISCORD_BOT_TOKEN must be set.")
    if not channel_id:
        raise RuntimeError("DISCORD_CHANNEL_ID must be set.")
    intents = discord.Intents.default()
    bot = FTDTop200Client(channel_id=int(channel_id), intents=intents)
    bot.run(token)

if __name__ == "__main__":
    main()