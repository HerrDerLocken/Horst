import discord
from discord import app_commands
import requests
import datetime
import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import urllib3
import os
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

TOKEN = os.getenv("TOKEN")
TOKEN = os.getenv("TOKEN")
ROLE_ID = 1439649825183371465  # Role to mention
CHANNEL_ID = 1439931429029941299# Channel to send messages

USERID = os.getenv("USERID")
HASH   = os.getenv("HASH")

intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)
scheduler = AsyncIOScheduler()

tz = pytz.timezone("Europe/Berlin")

def get_timetable():
    """Fetch all events from the API (week or full JSON)."""
    url = "https://selfservice.campus-dual.de/room/json"
    # We can request a wide range, then filter in Python
    now = datetime.datetime.now(tz)
    start_ts = int(now.timestamp())
    end_ts = int((now + datetime.timedelta(days=7)).timestamp())

    params = {
        "userid": USERID,
        "hash": HASH,
        "start": start_ts,
        "end": end_ts,
    }

    r = requests.get(url, params=params, verify=False)
    if r.status_code == 200:
        return r.json()
    else:
        return []

def filter_timetable_for_tomorrow(events):
    """Filter events between 01:00 and 23:00 tomorrow."""
    tomorrow = datetime.datetime.now(tz) + datetime.timedelta(days=1)
    start_dt = tz.localize(datetime.datetime(tomorrow.year, tomorrow.month, tomorrow.day, 1, 0, 0))
    end_dt = tz.localize(datetime.datetime(tomorrow.year, tomorrow.month, tomorrow.day, 23, 0, 0))
    
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    filtered = [
        e for e in events
        if start_ts <= e["start"] <= end_ts
    ]
    return filtered

async def send_long_message(channel, text):
    """Send long messages split into Discord limits."""
    chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
    for c in chunks:
        await channel.send(c)

async def send_timetable():
    """Send tomorrow's filtered events as a formatted table to Discord."""
    channel = client.get_channel(CHANNEL_ID)
    if not channel:
        print("Channel not found")
        return

    role_mention = f"<@&{ROLE_ID}>"
    all_events = get_timetable()
    events = filter_timetable_for_tomorrow(all_events)

    tomorrow = datetime.datetime.now(tz) + datetime.timedelta(days=1)
    if not events:
        await channel.send(f"{role_mention} Keine Veranstaltungen für morgen ({tomorrow.strftime('%d.%m.%Y')}).")
        return

    # Build table header
    table_lines = [
    f"```\nZeit       | Kurs                       | Raum   | Dozent",
        "-----------|----------------------------|--------|-------------------"
    ]

    # Add events
    for e in events:
        start = datetime.datetime.fromtimestamp(e["start"], tz).strftime("%H:%M")
        end = datetime.datetime.fromtimestamp(e["end"], tz).strftime("%H:%M")
        time_range = f"{start}-{end}"

        course = e.get("title", "Unbekannt")
        room = e.get("sroom", "?")
        instructor = e.get("instructor", "?")

        # Format each line with fixed-width columns
        table_lines.append(f"{time_range:<10}| {course:<28}| {room:<6}| {instructor}")

    table_lines.append("```")  # Close code block

    # Send table to Discord
    await send_long_message(channel, f"{role_mention} **Stundenplan für morgen ({tomorrow.strftime('%d.%m.%Y')})**\n" + "\n".join(table_lines))

async def send_weekly_schedule():
    """Send the timetable for the upcoming week (Monday-Sunday) every Sunday."""
    channel = client.get_channel(CHANNEL_ID)
    if not channel:
        print("Channel not found")
        return

    now = datetime.datetime.now(tz)
    # Find next Monday after today
    next_monday = now + datetime.timedelta(days=(7 - now.weekday()))  # Monday=0
    all_events = get_timetable()
    messages = []

    for i in range(7):
        day = next_monday + datetime.timedelta(days=i)
        events = filter_timetable_for_day(all_events, day)
        title = f"{day.strftime('%A %d.%m.%Y')}"
        messages.append(format_timetable_table(events, title))

    full_week_text = "\n".join(messages)
    await send_long_message(channel, f"<@&{ROLE_ID}> **Stundenplan für die kommende Woche**\n" + full_week_text)

def filter_timetable_for_day(events, day):
    """Filter events for a specific day between 01:00 and 23:00."""
    start_dt = tz.localize(datetime.datetime(day.year, day.month, day.day, 1, 0, 0))
    end_dt = tz.localize(datetime.datetime(day.year, day.month, day.day, 23, 0, 0))
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())
    return [e for e in events if start_ts <= e["start"] <= end_ts]


def format_timetable_table(events, title="Stundenplan"):
    """Format events as a clean Discord table."""
    if not events:
        return "Keine Veranstaltungen gefunden."

    table_lines = [
        f"```\nZeit       | Kurs                      | Raum   | Dozent",
        "-----------|----------------------------|--------|-------------------"
    ]
    for e in events:
        start = datetime.datetime.fromtimestamp(e["start"], tz).strftime("%H:%M")
        end = datetime.datetime.fromtimestamp(e["end"], tz).strftime("%H:%M")
        time_range = f"{start}-{end}"

        course = e.get("title", "Unbekannt")[:28]  # truncate long titles
        room = e.get("sroom", "?")
        instructor = e.get("instructor", "?")
        table_lines.append(f"{time_range:<10}| {course:<28}| {room:<6}| {instructor}")

    table_lines.append("```")
    return f"**{title}**\n" + "\n".join(table_lines)

# ---------------- Slash Commands ---------------- #

@tree.command(name="timetable_day", description="Zeige den Stundenplan für einen bestimmten Tag")
@app_commands.describe(tag="Tag des Monats (1-31)", monat="Monat (1-12)")
async def timetable_day(interaction: discord.Interaction, tag: int, monat: int):
    year = datetime.datetime.now(tz).year
    try:
        day = datetime.datetime(year, monat, tag)
    except ValueError:
        await interaction.response.send_message("Ungültiges Datum.", ephemeral=True)
        return

    all_events = get_timetable()
    events = filter_timetable_for_day(all_events, day)
    table_text = format_timetable_table(events, f"Stundenplan für {day.strftime('%d.%m.%Y')}")
    await interaction.response.send_message(table_text)


@tree.command(name="timetable_week", description="Zeige den Stundenplan für eine Woche")
@app_commands.describe(monday_tag="Tag des Montags", monday_monat="Monat des Montags")
async def timetable_week(interaction: discord.Interaction, monday_tag: int, monday_monat: int):
    year = datetime.datetime.now(tz).year
    try:
        monday = datetime.datetime(year, monday_monat, monday_tag)
    except ValueError:
        await interaction.response.send_message("Ungültiges Datum.", ephemeral=True)
        return

    all_events = get_timetable()
    messages = []

    for i in range(7):  # Monday to Sunday
        day = monday + datetime.timedelta(days=i)
        events = filter_timetable_for_day(all_events, day)
        title = f"{day.strftime('%A %d.%m.%Y')}"
        messages.append(format_timetable_table(events, title))

    # Combine all days into one message
    full_week_text = "\n".join(messages)
    await interaction.response.send_message(full_week_text)

@client.event
async def on_ready():
    print(f"Bot logged in as {client.user}")
    await tree.sync()
    # Schedule daily message at 20:00 Berlin time
    scheduler.add_job(
        send_timetable,
        trigger="cron",
        hour=20,
        minute=0,
        timezone="Europe/Berlin"
    )
    scheduler.add_job(
        send_weekly_schedule,
        trigger="cron",
        day_of_week="sun",
        hour=19,
        minute=45,
        timezone="Europe/Berlin"
    )
    scheduler.start()
client.run(TOKEN)
