import discord
from discord import app_commands
import requests
import datetime
import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import urllib3
import json
import asyncio
from urllib.parse import urlparse
import re
import os
from dotenv import load_dotenv
from openai import OpenAI
from building_graph import create_building

# ── Reine Logik aus bot_logic.py importieren ─────────────────────────────────
from bot_logic import (
    normalize_event, compare_timetables,
    filter_timetable_for_day, filter_timetable_for_today, filter_timetable_for_tomorrow,
    get_course_color, clean_dish_name, parse_json_from_text,
    load_praxisphasen, is_in_practical_phase,
    should_send_daily_message, should_send_weekly_schedule,
    check_debounce, reset_debounce, get_pending_state,
    save_timetable_state, load_timetable_state,
    generate_truth_table_image,
    DEBOUNCE_THRESHOLD, DEFAULT_COLOR,
)

load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Config ────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY")
TOKEN               = os.getenv("TOKEN")
ROLE_ID             = 1439649825183371465
CHANNEL_ID          = 1439649654055895080
ESSEN_CHANNEL_ID    = 1439977286882295909
CHANGES_ROLE_ID     = 1439649825183371465
CANTEEN_ROLE_ID     = 1439988749944487957
LOG_CHANNEL_ID      = 1439985824530829415
TEST_CHANNEL_ID     = 1439931429029941299   # Dedizierter Test-Channel
ADMIN_ROLE_ID       = 1439649601765511383   # Admin-Rolle für Test-Commands

USERID = os.getenv("USERID")
HASH   = os.getenv("HASH")

PRAXISPHASEN_FILE   = "praxisphasen.json"
TIMETABLE_STATE_FILE = "timetable_state.json"

# ── Intents & Client ──────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
client   = discord.Client(intents=intents)
tree     = app_commands.CommandTree(client)
scheduler = AsyncIOScheduler()

tz = pytz.timezone("Europe/Berlin")

previous_timetable = {"today": [], "tomorrow": []}
reaction_roles: dict = {}
REACTION_ROLES_FILE = "reaction_roles.json"

MENSA_COLORS = {"header": 3066993, "meal": 5793266}

CACHE_FILE = "nutrition_cache.json"

building = create_building()

openrouter_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)
MODEL_NAME = "gpt-oss-20b"

global last_update_day
last_update_day = datetime.datetime.now(tz).date()


# ═══════════════════════════════════════════════════════════════════════════════
#  Hilfsfunktionen (Discord-unabhängig, aber nicht in bot_logic weil IO/cache)
# ═══════════════════════════════════════════════════════════════════════════════

def is_valid_url(url):
    if not url or not isinstance(url, str):
        return False
    try:
        r = urlparse(url)
        return all([r.scheme, r.netloc]) and r.scheme in ['http', 'https']
    except Exception:
        return False


def load_cache():
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def clean_old_dates(cache):
    from datetime import timedelta
    today = datetime.datetime.now(tz).date()
    start_of_week = today - datetime.timedelta(days=today.weekday())
    end_of_next   = start_of_week + datetime.timedelta(days=13)
    new_cache = {}
    for key, value in cache.items():
        try:
            date_str  = key.split("::")[0]
            entry_date = datetime.datetime.fromisoformat(date_str).date()
            if start_of_week <= entry_date <= end_of_next:
                new_cache[key] = value
        except ValueError:
            continue
    return new_cache


def ask_model_for_nutrition(dish_name: str):
    prompt = f"""
Analyze the dish "{dish_name}" (German canteen food).
Estimate values for a standard portion (approx 350g).
Return a SINGLE JSON object with exactly these keys:
"kcal", "Eiweiss", "Kohlenhydrate", "Fette".
Values should be numbers (int or float).
Example: {{"kcal": 500, "Eiweiss": 20, "Kohlenhydrate": 50, "Fette": 15}}
DO NOT output Markdown. DO NOT output explanations. ONLY JSON.
"""
    try:
        resp = openrouter_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a precise nutrition estimation assistant."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.2,
        )
        return parse_json_from_text(resp.choices[0].message.content.strip())
    except Exception as e:
        print(f"  --> OpenRouter Exception for '{dish_name}': {e}")
        return None


def get_nutrition_for_day(dish_name: str, date, cache: dict):
    clean_name = clean_dish_name(dish_name)
    cache_key  = f"{date.isoformat()}::{clean_name}"
    if cache_key in cache:
        return cache[cache_key]
    nutrition = ask_model_for_nutrition(clean_name)
    if nutrition:
        cache[cache_key] = nutrition
    return nutrition


def get_canteen_meals(date):
    from datetime import timedelta
    date_str = date.strftime("%Y-%m-%d")
    url = f"https://api.studentenwerk-dresden.de/openmensa/v2/canteens/32/days/{date_str}/meals"
    today = datetime.datetime.today().date()
    start_of_week = today - datetime.timedelta(days=today.weekday())
    end_of_next   = start_of_week + datetime.timedelta(days=13)
    in_range = start_of_week <= date <= end_of_next
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            return []
        meals = response.json()
        cache = clean_old_dates(load_cache())
        cache_updated = False
        for meal in meals:
            nutrition = get_nutrition_for_day(meal.get("name", ""), date, cache) if in_range else None
            meal["nutrition"] = nutrition
            if nutrition:
                cache_updated = True
        if cache_updated:
            save_cache(cache)
        return meals
    except Exception as e:
        print(f"Error fetching canteen meals: {e}")
        return []


# ── API: Stundenplan ──────────────────────────────────────────────────────────

def get_timetable():
    """
    FIX: start_ts beginnt am Anfang des heutigen Tages (nicht bei 'now'),
    damit bereits gestartete Events weiterhin zurückgeliefert werden.
    Der bisherige Bug: start_ts=now → API ließ laufende Kurse weg → False Positive.
    """
    url  = "https://selfservice.campus-dual.de/room/json"
    now  = datetime.datetime.now(tz)
    # Beginn des heutigen Tages (00:00 Uhr)
    day_start  = tz.localize(datetime.datetime(now.year, now.month, now.day, 0, 0, 0))
    start_ts   = int(day_start.timestamp())
    end_ts     = int((now + datetime.timedelta(days=7)).timestamp())
    params = {"userid": USERID, "hash": HASH, "start": start_ts, "end": end_ts}
    r = requests.get(url, params=params, verify=False)
    return r.json() if r.status_code == 200 else []


# ═══════════════════════════════════════════════════════════════════════════════
#  Discord: Embed-Builder
# ═══════════════════════════════════════════════════════════════════════════════

def create_modern_embeds(events, date_str, course_group="IT-25APP"):
    if not events:
        return []
    embeds = [discord.Embed(
        color=5793266,
        description=f"**{course_group}** – {len(events)} Vorlesungen"
    )]
    embeds[0].set_author(name=f"Stundenplan – {date_str}")
    for e in events:
        start = datetime.datetime.fromtimestamp(e.get("start", 0), tz).strftime("%H:%M")
        end   = datetime.datetime.fromtimestamp(e.get("end",   0), tz).strftime("%H:%M")
        embed = discord.Embed(title=e.get("title", "Unbekannt"), color=get_course_color(e.get("title", "")))
        embed.add_field(name="Zeit",   value=f"`{start} – {end}`",       inline=True)
        embed.add_field(name="Raum",   value=f"`{e.get('sroom','?')}`",  inline=True)
        embed.add_field(name="Dozent", value=f"`{e.get('instructor','?')}`", inline=True)
        embeds.append(embed)
    return embeds


def create_change_embeds(added, removed, day_name):
    embeds = []
    if added or removed:
        embeds.append(discord.Embed(
            title="Stundenplanänderung erkannt!",
            description=f"Änderungen für **{day_name}**",
            color=15158332,
        ))
    for event in added:
        start = datetime.datetime.fromtimestamp(event.get("start", 0), tz).strftime("%H:%M")
        end   = datetime.datetime.fromtimestamp(event.get("end",   0), tz).strftime("%H:%M")
        embed = discord.Embed(title=f"NEU: {event.get('title','?')}", color=get_course_color(event.get("title","")))
        embed.add_field(name="Zeit",   value=f"`{start} – {end}`",         inline=True)
        embed.add_field(name="Raum",   value=f"`{event.get('sroom','?')}`", inline=True)
        embed.add_field(name="Dozent", value=f"`{event.get('instructor','?')}`", inline=True)
        embeds.append(embed)
    for event in removed:
        start = datetime.datetime.fromtimestamp(event.get("start", 0), tz).strftime("%H:%M")
        end   = datetime.datetime.fromtimestamp(event.get("end",   0), tz).strftime("%H:%M")
        embed = discord.Embed(title=f"ENTFERNT: {event.get('title','?')}", color=10038562)
        embed.add_field(name="Zeit",   value=f"`{start} – {end}`",         inline=True)
        embed.add_field(name="Raum",   value=f"`{event.get('sroom','?')}`", inline=True)
        embed.add_field(name="Dozent", value=f"`{event.get('instructor','?')}`", inline=True)
        embeds.append(embed)
    return embeds


def create_canteen_embeds(meals, date_str):
    if not meals:
        return []
    embeds = [discord.Embed(
        color=MENSA_COLORS["header"],
        description=f"**Mensa Johanna** – {len(meals)} Gerichte verfügbar"
    )]
    embeds[0].set_author(name=f"Mensa-Speiseplan – {date_str}")
    for meal in meals:
        name      = meal.get("name", "Unbekannt")
        prices    = meal.get("prices", {})
        notes     = meal.get("notes", [])
        nutrition = meal.get("nutrition")
        embed = discord.Embed(title=name, color=MENSA_COLORS["meal"])
        image_url = meal.get("image")
        if image_url:
            if image_url.startswith("//"):
                image_url = f"https:{image_url}"
            if is_valid_url(image_url):
                try: embed.set_thumbnail(url=image_url)
                except: pass
        cat = meal.get("category")
        if cat:
            embed.add_field(name="Kategorie", value=f"`{cat}`", inline=False)
        sp = prices.get("Studierende")
        if sp is not None:
            try:    embed.add_field(name="Preis", value=f"**{float(sp):.2f}€**", inline=True)
            except: embed.add_field(name="Preis", value=f"**{sp}**", inline=True)
        if nutrition and any(nutrition.values()):
            nt = ""
            if nutrition.get("kcal"):         nt += f"→ **{nutrition['kcal']} kcal**\n"
            if nutrition.get("Eiweiss"):      nt += f"→ Eiweiß: {nutrition['Eiweiss']}g\n"
            if nutrition.get("Kohlenhydrate"):nt += f"→ Kohlenhydrate: {nutrition['Kohlenhydrate']}g\n"
            if nutrition.get("Fette"):        nt += f"→ Fette: {nutrition['Fette']}g"
            if nt:
                embed.add_field(name="Nährwerte (ca. 350g, geschätzt)", value=nt, inline=True)
        else:
            embed.add_field(name="Nährwerte", value="*Keine Daten*", inline=True)
        if notes:
            nt = ", ".join(notes)
            embed.add_field(name="Hinweise", value=nt[:200] + ("..." if len(nt) > 200 else ""), inline=False)
        embeds.append(embed)
    return embeds


# ═══════════════════════════════════════════════════════════════════════════════
#  Logging
# ═══════════════════════════════════════════════════════════════════════════════

async def log_action(action: str):
    try:
        ch = client.get_channel(LOG_CHANNEL_ID)
        if not ch:
            print(f"Log channel not found. Action: {action}")
            return
        ts = datetime.datetime.now(tz).strftime("%d.%m.%Y %H:%M:%S")
        await ch.send(f"`[{ts}]` {action}")
    except Exception as e:
        print(f"Error sending log: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Scheduler-Jobs
# ═══════════════════════════════════════════════════════════════════════════════

async def send_timetable():
    tomorrow      = datetime.datetime.now(tz) + datetime.timedelta(days=1)
    should_send, reason = should_send_daily_message(tomorrow.date(), PRAXISPHASEN_FILE)
    if not should_send:
        await log_action(f"Stundenplan nicht gesendet: {reason}"); return
    ch = client.get_channel(CHANNEL_ID)
    if not ch:
        await log_action("ERROR: Stundenplan-Channel nicht gefunden"); return
    all_events = await asyncio.to_thread(get_timetable)
    events     = filter_timetable_for_tomorrow(all_events)
    date_str   = tomorrow.strftime("%A, %d. %B %Y")
    if not events:
        await ch.send(f"<@&{ROLE_ID}> Keine Veranstaltungen für morgen ({tomorrow.strftime('%d.%m.%Y')}).")
        await log_action(f"Stundenplan: Keine Veranstaltungen für morgen"); return
    embeds = create_modern_embeds(events, date_str)
    for i in range(0, len(embeds), 10):
        await ch.send(content=f"<@&{ROLE_ID}>" if i == 0 else None, embeds=embeds[i:i+10])
    await log_action(f"Stundenplan gesendet: {len(events)} Veranstaltungen für {tomorrow.strftime('%d.%m.%Y')}")


async def send_weekly_schedule():
    should_send, reason = should_send_weekly_schedule(tz, PRAXISPHASEN_FILE)
    if not should_send:
        await log_action(f"Wochenstundenplan nicht gesendet: {reason}"); return
    ch = client.get_channel(CHANNEL_ID)
    if not ch:
        await log_action("ERROR: Channel nicht gefunden"); return
    now         = datetime.datetime.now(tz)
    next_monday = now + datetime.timedelta(days=(7 - now.weekday()))
    all_events  = await asyncio.to_thread(get_timetable)
    await ch.send(f"<@&{ROLE_ID}> **Stundenplan für die kommende Woche**")
    total = 0
    for i in range(7):
        day    = next_monday + datetime.timedelta(days=i)
        events = filter_timetable_for_day(all_events, day)
        total += len(events)
        embeds = create_modern_embeds(events, day.strftime("%A, %d. %B %Y"))
        for j in range(0, len(embeds), 10):
            await ch.send(embeds=embeds[j:j+10])
    await log_action(f"Wochenstundenplan gesendet: {total} Veranstaltungen")


async def send_canteen_menu():
    today = datetime.datetime.now(tz).date()
    if today.weekday() in [5, 6]:
        await log_action(f"Mensa nicht gesendet: Wochenende"); return
    in_practical, phase = is_in_practical_phase(today, PRAXISPHASEN_FILE)
    if in_practical:
        await log_action(f"Mensa nicht gesendet: Praxisphase ({phase})"); return
    ch = client.get_channel(ESSEN_CHANNEL_ID)
    if not ch:
        await log_action("ERROR: Mensa-Channel nicht gefunden"); return
    meals    = await asyncio.to_thread(get_canteen_meals, today)
    date_str = datetime.datetime.now(tz).strftime("%A, %d. %B %Y")
    if not meals:
        await log_action(f"Mensa: Keine Gerichte für {today}"); return
    embeds = create_canteen_embeds(meals, date_str)
    for i in range(0, len(embeds), 10):
        await ch.send(content=f"<@&{CANTEEN_ROLE_ID}>" if i == 0 else None, embeds=embeds[i:i+10])
    await log_action(f"Mensa-Menü gesendet: {len(meals)} Gerichte für {today}")


async def check_timetable_changes():
    """
    Prüft alle 20 Minuten auf Änderungen.

    FIX 1: get_timetable() startet jetzt am Tagesbeginn → keine Events gehen verloren.
    FIX 2: Debounce – eine Änderung muss in DEBOUNCE_THRESHOLD aufeinander-
           folgenden Checks identisch sein, bevor ein Alert gesendet wird.
           Einmalige API-Aussetzer (bisheriger Bug) werden so unterdrückt.
    FIX 3: State wird in timetable_state.json persistiert → Bot-Neustart
           verursacht keine False Positives mehr.
    """
    global previous_timetable, last_update_day

    current_date = datetime.datetime.now(tz)

    # Tageswechsel → State neu laden, Debounce zurücksetzen
    if current_date.date() != last_update_day:
        all_events = await asyncio.to_thread(get_timetable)
        previous_timetable["today"]    = filter_timetable_for_today(all_events)
        previous_timetable["tomorrow"] = filter_timetable_for_tomorrow(all_events)
        save_timetable_state(previous_timetable, tz, TIMETABLE_STATE_FILE)
        reset_debounce()
        last_update_day = current_date.date()
        await log_action("Tageswechsel: Stundenplan-State aktualisiert, Debounce zurückgesetzt.")
        return   # Kein Change-Check direkt nach Reset

    ch = client.get_channel(CHANNEL_ID)
    if not ch:
        await log_action("ERROR: Änderungs-Check: Channel nicht gefunden"); return

    all_events     = await asyncio.to_thread(get_timetable)
    today_events   = filter_timetable_for_today(all_events)
    tomorrow_events = filter_timetable_for_tomorrow(all_events)

    today    = datetime.datetime.now(tz)
    tomorrow = today + datetime.timedelta(days=1)

    role_mention = f"<@&{CHANGES_ROLE_ID}>"

    # ── Heute ────────────────────────────────────────────────────────────────
    if previous_timetable["today"]:
        added_t, removed_t = compare_timetables(previous_timetable["today"], today_events)
        should_alert, data = check_debounce("today", added_t, removed_t)
        if should_alert:
            embeds = create_change_embeds(data["added"], data["removed"],
                                          f"HEUTE ({today.strftime('%A, %d. %B %Y')})")
            for i in range(0, len(embeds), 10):
                await ch.send(content=role_mention if i == 0 else None, embeds=embeds[i:i+10])
            await log_action(
                f"Stundenplanänderung HEUTE (bestätigt nach {DEBOUNCE_THRESHOLD} Checks): "
                f"+{len(data['added'])} neu, -{len(data['removed'])} entfernt"
            )
        elif added_t or removed_t:
            ps = get_pending_state("today")
            await log_action(
                f"Mögliche Änderung HEUTE erkannt (Debounce {ps['count']}/{DEBOUNCE_THRESHOLD}) – "
                f"noch nicht gemeldet."
            )

    # ── Morgen ───────────────────────────────────────────────────────────────
    if previous_timetable["tomorrow"]:
        added_tm, removed_tm = compare_timetables(previous_timetable["tomorrow"], tomorrow_events)
        should_alert, data = check_debounce("tomorrow", added_tm, removed_tm)
        if should_alert:
            embeds = create_change_embeds(data["added"], data["removed"],
                                          f"MORGEN ({tomorrow.strftime('%A, %d. %B %Y')})")
            for i in range(0, len(embeds), 10):
                await ch.send(content=role_mention if i == 0 else None, embeds=embeds[i:i+10])
            await log_action(
                f"Stundenplanänderung MORGEN (bestätigt nach {DEBOUNCE_THRESHOLD} Checks): "
                f"+{len(data['added'])} neu, -{len(data['removed'])} entfernt"
            )
        elif added_tm or removed_tm:
            ps = get_pending_state("tomorrow")
            await log_action(
                f"Mögliche Änderung MORGEN erkannt (Debounce {ps['count']}/{DEBOUNCE_THRESHOLD}) – "
                f"noch nicht gemeldet."
            )

    # State aktualisieren & persistieren
    previous_timetable["today"]    = today_events
    previous_timetable["tomorrow"] = tomorrow_events
    save_timetable_state(previous_timetable, tz, TIMETABLE_STATE_FILE)


# ═══════════════════════════════════════════════════════════════════════════════
#  Test-Suite (Discord)
# ═══════════════════════════════════════════════════════════════════════════════

def _user_has_admin(interaction: discord.Interaction) -> bool:
    if not interaction.guild:
        return False
    if interaction.user.guild_permissions.administrator:
        return True
    return any(r.id == ADMIN_ROLE_ID for r in interaction.user.roles)


def _test_embed(title: str, status: str, detail: str, ok: bool) -> discord.Embed:
    color  = 3066993 if ok else 15158332
    symbol = "✅" if ok else "❌"
    embed  = discord.Embed(title=f"{symbol} {title}", color=color)
    embed.add_field(name="Status", value=status, inline=False)
    if detail:
        embed.add_field(name="Detail", value=f"```{detail[:900]}```", inline=False)
    embed.set_footer(text=datetime.datetime.now(tz).strftime("%H:%M:%S"))
    return embed


async def _run_test_api(test_ch: discord.TextChannel) -> bool:
    """Testet die Campus-Dual- und Mensa-API-Verbindung."""
    results = []

    # Campus Dual
    try:
        events = await asyncio.to_thread(get_timetable)
        results.append((True, f"Campus Dual API: {len(events)} Events zurückgegeben"))
    except Exception as e:
        results.append((False, f"Campus Dual API Fehler: {e}"))

    # Mensa
    today = datetime.datetime.now(tz).date()
    try:
        meals = await asyncio.to_thread(get_canteen_meals, today)
        results.append((True, f"Mensa API: {len(meals)} Gerichte für heute"))
    except Exception as e:
        results.append((False, f"Mensa API Fehler: {e}"))

    all_ok = all(r[0] for r in results)
    detail = "\n".join(msg for _, msg in results)
    await test_ch.send(embed=_test_embed("API-Verbindungstest", "OK" if all_ok else "FEHLER", detail, all_ok))
    return all_ok


async def _run_test_stundenplan(test_ch: discord.TextChannel) -> bool:
    """Testet den Stundenplan-Abruf und sendet das Ergebnis in den Test-Channel."""
    try:
        all_events = await asyncio.to_thread(get_timetable)
        today_ev   = filter_timetable_for_today(all_events)
        tmrw_ev    = filter_timetable_for_tomorrow(all_events)
        now        = datetime.datetime.now(tz)
        tmrw       = now + datetime.timedelta(days=1)

        detail = f"Heute: {len(today_ev)} Events | Morgen: {len(tmrw_ev)} Events\n"
        if today_ev:
            detail += "\nHeutige Events:\n" + "\n".join(
                f"  • {e.get('title','?')} {datetime.datetime.fromtimestamp(e['start'],tz).strftime('%H:%M')}–"
                f"{datetime.datetime.fromtimestamp(e['end'],tz).strftime('%H:%M')} Raum {e.get('sroom','?')}"
                for e in today_ev
            )
        await test_ch.send(embed=_test_embed("Stundenplan-Test", "OK", detail, True))

        # Zeige Embeds wie sie im echten Channel aussehen würden
        if tmrw_ev:
            embeds = create_modern_embeds(tmrw_ev, tmrw.strftime("%A, %d. %B %Y"))
            await test_ch.send(content=" **Vorschau Morgen-Stundenplan:**", embeds=embeds[:10])
        return True
    except Exception as e:
        await test_ch.send(embed=_test_embed("Stundenplan-Test", "FEHLER", str(e), False))
        return False


async def _run_test_mensa(test_ch: discord.TextChannel) -> bool:
    """Testet den Mensa-Abruf und zeigt Embeds im Test-Channel."""
    today = datetime.datetime.now(tz).date()
    try:
        meals = await asyncio.to_thread(get_canteen_meals, today)
        if not meals:
            await test_ch.send(embed=_test_embed("Mensa-Test", "Keine Gerichte heute", "", True))
            return True
        detail = f"{len(meals)} Gerichte abgerufen"
        await test_ch.send(embed=_test_embed("Mensa-Test", "OK", detail, True))
        embeds = create_canteen_embeds(meals, today.strftime("%A, %d. %B %Y"))
        for i in range(0, min(len(embeds), 10), 10):
            await test_ch.send(content=" **Vorschau Mensa:**", embeds=embeds[i:i+10])
        return True
    except Exception as e:
        await test_ch.send(embed=_test_embed("Mensa-Test", "FEHLER", str(e), False))
        return False


async def _run_test_debounce(test_ch: discord.TextChannel) -> bool:
    """
    Testet die Debounce-Logik in-process.
    Simuliert den bekannten Bug (API-Aussetzer) und zeigt dass kein Alert ausgelöst wird.
    Simuliert dann eine echte Änderung (N Checks hintereinander) und prüft ob Alert ausgelöst wird.
    """
    from bot_logic import check_debounce, reset_debounce, DEBOUNCE_THRESHOLD
    ok = True
    lines = []

    # ── Szenario 1: API-Aussetzer (False Positive Bug) ────────────────────────
    reset_debounce("_test")
    fake_removed = [{"title": "DTDS", "start": 9*3600, "end": 11*3600, "sroom": "1.202", "instructor": "Test"}]

    alert1, _ = check_debounce("_test", [], fake_removed)  # 1. Detektion
    alert2, _ = check_debounce("_test", [], [])             # API wieder normal → Reset

    if alert1 or alert2:
        lines.append(f" Szenario 1 FEHLGESCHLAGEN: Einmaliger Aussetzer hätte Alert ausgelöst!")
        ok = False
    else:
        lines.append(f" Szenario 1: API-Aussetzer korrekt unterdrückt (kein False Positive)")

    # ── Szenario 2: Echte Änderung (N mal hintereinander) ────────────────────
    reset_debounce("_test")
    alerts = []
    for i in range(DEBOUNCE_THRESHOLD):
        should_alert, data = check_debounce("_test", [], fake_removed)
        alerts.append(should_alert)

    final_alert = alerts[-1]  # Letzter Check muss Alert auslösen
    if not final_alert:
        lines.append(f" Szenario 2 FEHLGESCHLAGEN: Echte Änderung nach {DEBOUNCE_THRESHOLD} Checks nicht gemeldet!")
        ok = False
    else:
        lines.append(f" Szenario 2: Echte Änderung nach {DEBOUNCE_THRESHOLD} Checks korrekt gemeldet")

    # ── Szenario 3: Unterschiedliche Änderungen → kein vorzeitiger Alert ─────
    reset_debounce("_test")
    fake2 = [{"title": "MATHE", "start": 8*3600, "end": 10*3600, "sroom": "2.003", "instructor": "X"}]
    check_debounce("_test", [], fake_removed)   # Änderung A, count=1
    check_debounce("_test", [], fake2)           # Änderung B → reset, count=1
    alert_early, _ = check_debounce("_test", [], fake_removed)  # Änderung A wieder → count=1
    if alert_early:
        lines.append(" Szenario 3 FEHLGESCHLAGEN: Vorzeitiger Alert bei wechselnden Änderungen!")
        ok = False
    else:
        lines.append(" Szenario 3: Wechselnde API-Daten lösen keinen Alert aus")

    reset_debounce("_test")  # Cleanup
    detail = "\n".join(lines) + f"\n\nDebounce-Schwellenwert: {DEBOUNCE_THRESHOLD} aufeinanderfolgende Checks"
    await test_ch.send(embed=_test_embed("Debounce-Test (False-Positive-Schutz)", "OK" if ok else "FEHLER", detail, ok))
    return ok


async def _run_test_notification(test_ch: discord.TextChannel) -> bool:
    """Sendet eine Mock-Änderungsbenachrichtigung NUR in den Test-Channel."""
    try:
        fake_added = [{
            "title": "TEST-Kurs (Simuliert)",
            "start": int((datetime.datetime.now(tz) + datetime.timedelta(hours=2)).timestamp()),
            "end":   int((datetime.datetime.now(tz) + datetime.timedelta(hours=4)).timestamp()),
            "sroom": "TEST.001",
            "instructor": "Test-Dozent"
        }]
        fake_removed = [{
            "title": "Ausgefallener Kurs (Simuliert)",
            "start": int((datetime.datetime.now(tz) + datetime.timedelta(hours=6)).timestamp()),
            "end":   int((datetime.datetime.now(tz) + datetime.timedelta(hours=8)).timestamp()),
            "sroom": "TEST.002",
            "instructor": "Test-Dozent 2"
        }]
        embeds = create_change_embeds(fake_added, fake_removed, "TEST-TAG (Simulation)")
        await test_ch.send(content=" **Vorschau: Änderungs-Benachrichtigung**", embeds=embeds)
        await test_ch.send(embed=_test_embed("Benachrichtigungs-Test", "Mock-Nachricht gesendet", "", True))
        return True
    except Exception as e:
        await test_ch.send(embed=_test_embed("Benachrichtigungs-Test", "FEHLER", str(e), False))
        return False


async def _run_test_state_persistence(test_ch: discord.TextChannel) -> bool:
    """Testet das Speichern und Laden des Timetable-State."""
    import tempfile
    try:
        test_state = {
            "today": [{"title":"TEST","start":1000,"end":2000,"sroom":"T","instructor":"X"}],
            "tomorrow": []
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        save_timetable_state(test_state, tz, tmp_path)
        loaded = load_timetable_state(tz, tmp_path)
        os.unlink(tmp_path)

        if loaded and len(loaded["today"]) == 1 and loaded["today"][0]["title"] == "TEST":
            detail = f"State erfolgreich gespeichert und geladen.\nGeladene Events Heute: {len(loaded['today'])}"
            await test_ch.send(embed=_test_embed("State-Persistenz-Test", "OK", detail, True))
            return True
        else:
            await test_ch.send(embed=_test_embed("State-Persistenz-Test", "FEHLER", "Geladener State stimmt nicht überein", False))
            return False
    except Exception as e:
        await test_ch.send(embed=_test_embed("State-Persistenz-Test", "FEHLER", str(e), False))
        return False


# ── Slash-Commands: Tests ─────────────────────────────────────────────────────

@tree.command(name="test_all", description="[ADMIN] Führt alle Bot-Tests durch (Ausgabe nur im Test-Channel)")
async def cmd_test_all(interaction: discord.Interaction):
    if not _user_has_admin(interaction):
        await interaction.response.send_message(" Nur für Admins.", ephemeral=True); return
    await interaction.response.send_message(" Starte Test-Suite… Ausgabe im Test-Channel.", ephemeral=True)

    test_ch = client.get_channel(TEST_CHANNEL_ID)
    if not test_ch:
        await interaction.followup.send(" Test-Channel nicht gefunden!", ephemeral=True); return

    await test_ch.send(
        embed=discord.Embed(
            title=" Test-Suite gestartet",
            description=f"Gestartet von **{interaction.user.display_name}**\n"
                        f"Zeitstempel: `{datetime.datetime.now(tz).strftime('%d.%m.%Y %H:%M:%S')}`",
            color=5793266
        )
    )

    tests = [
        ("API-Verbindung",         _run_test_api),
        ("Stundenplan-Abruf",      _run_test_stundenplan),
        ("Mensa-Abruf",            _run_test_mensa),
        ("Debounce / False-Positive", _run_test_debounce),
        ("Benachrichtigung",       _run_test_notification),
        ("State-Persistenz",       _run_test_state_persistence),
    ]

    results = []
    for name, fn in tests:
        try:
            ok = await fn(test_ch)
            results.append((name, ok))
        except Exception as e:
            await test_ch.send(embed=_test_embed(name, "UNERWARTETER FEHLER", str(e), False))
            results.append((name, False))

    # Zusammenfassung
    passed = sum(1 for _, ok in results if ok)
    total  = len(results)
    summary = "\n".join(f"{'✅' if ok else '❌'} {name}" for name, ok in results)
    color   = 3066993 if passed == total else (15105570 if passed > 0 else 15158332)
    await test_ch.send(embed=discord.Embed(
        title=f"🏁 Test-Suite abgeschlossen: {passed}/{total} bestanden",
        description=f"```\n{summary}\n```",
        color=color,
    ))
    await log_action(f"Test-Suite ausgeführt von {interaction.user}: {passed}/{total} Tests bestanden")


@tree.command(name="test_api", description="[ADMIN] Testet API-Verbindungen")
async def cmd_test_api(interaction: discord.Interaction):
    if not _user_has_admin(interaction):
        await interaction.response.send_message(" Nur für Admins.", ephemeral=True); return
    await interaction.response.send_message(" API-Test läuft…", ephemeral=True)
    test_ch = client.get_channel(TEST_CHANNEL_ID)
    if test_ch:
        await _run_test_api(test_ch)


@tree.command(name="test_stundenplan", description="[ADMIN] Testet Stundenplan-Abruf")
async def cmd_test_stundenplan(interaction: discord.Interaction):
    if not _user_has_admin(interaction):
        await interaction.response.send_message(" Nur für Admins.", ephemeral=True); return
    await interaction.response.send_message(" Stundenplan-Test läuft…", ephemeral=True)
    test_ch = client.get_channel(TEST_CHANNEL_ID)
    if test_ch:
        await _run_test_stundenplan(test_ch)


@tree.command(name="test_mensa", description="[ADMIN] Testet Mensa-Abruf")
async def cmd_test_mensa(interaction: discord.Interaction):
    if not _user_has_admin(interaction):
        await interaction.response.send_message(" Nur für Admins.", ephemeral=True); return
    await interaction.response.send_message(" Mensa-Test läuft…", ephemeral=True)
    test_ch = client.get_channel(TEST_CHANNEL_ID)
    if test_ch:
        await _run_test_mensa(test_ch)


@tree.command(name="test_debounce", description="[ADMIN] Testet False-Positive-Schutz")
async def cmd_test_debounce(interaction: discord.Interaction):
    if not _user_has_admin(interaction):
        await interaction.response.send_message(" Nur für Admins.", ephemeral=True); return
    await interaction.response.send_message(" Debounce-Test läuft…", ephemeral=True)
    test_ch = client.get_channel(TEST_CHANNEL_ID)
    if test_ch:
        await _run_test_debounce(test_ch)


@tree.command(name="test_notification", description="[ADMIN] Sendet Mock-Änderungsbenachrichtigung in Test-Channel")
async def cmd_test_notification(interaction: discord.Interaction):
    if not _user_has_admin(interaction):
        await interaction.response.send_message("Nur für Admins.", ephemeral=True); return
    await interaction.response.send_message(" Sende Test-Benachrichtigung…", ephemeral=True)
    test_ch = client.get_channel(TEST_CHANNEL_ID)
    if test_ch:
        await _run_test_notification(test_ch)


# ═══════════════════════════════════════════════════════════════════════════════
#  Bestehende Slash-Commands (unverändert)
# ═══════════════════════════════════════════════════════════════════════════════

@tree.command(name="weg", description="Berechnet den schnellsten Weg zwischen zwei Räumen")
@app_commands.describe(start="Start-Raum z.B. 3.005", ende="Ziel-Raum z.B. 1.202")
async def weg(interaction: discord.Interaction, start: str, ende: str):
    await interaction.response.defer()
    distance, path = building.dijkstra(start, ende)
    if not path:
        await interaction.followup.send(f"Kein Weg von **{start}** nach **{ende}** gefunden."); return
    await interaction.followup.send(
        f"**Start:** `{start}`\n**Ziel:** `{ende}`\n\n"
        f"**Weg:** `{'→'.join(path)}`\n**Distanz:** {distance}"
    )


@tree.command(name="truth", description="Erzeuge eine Wahrheitstabelle für einen Booleschen Ausdruck")
@app_commands.describe(ausdruck="Boolescher Ausdruck (z.B. A and !B or C)")
async def truth_command(interaction: discord.Interaction, ausdruck: str):
    await interaction.response.defer()
    try:
        path = generate_truth_table_image(ausdruck)
    except Exception as e:
        await interaction.followup.send(f"Fehler im Ausdruck:\n```\n{e}\n```"); return
    embed = discord.Embed(title="Wahrheitstabelle", description=f"Ausdruck: `{ausdruck}`", color=0x7289DA)
    embed.set_image(url="attachment://truth_table.png")
    await interaction.followup.send(embed=embed, file=discord.File(path, filename="truth_table.png"))


@tree.command(name="stundenplan_tag", description="Zeige den Stundenplan für einen bestimmten Tag")
@app_commands.describe(tag="Tag (1-31)", monat="Monat (1-12)")
async def stundenplan_tag(interaction: discord.Interaction, tag: int, monat: int):
    await interaction.response.defer()
    year = datetime.datetime.now(tz).year
    try:
        day = datetime.datetime(year, monat, tag, tzinfo=tz)
    except ValueError:
        await interaction.followup.send("Ungültiges Datum.", ephemeral=True); return
    all_events = await asyncio.to_thread(get_timetable)
    events     = filter_timetable_for_day(all_events, day)
    embeds     = create_modern_embeds(events, day.strftime("%A, %d. %B %Y"))
    if not embeds:
        await interaction.followup.send(f"Keine Veranstaltungen für {day.strftime('%d.%m.%Y')}."); return
    for i in range(0, len(embeds), 10):
        await interaction.followup.send(embeds=embeds[i:i+10])


@tree.command(name="stundenplan_woche", description="Zeige den Stundenplan für eine Woche")
@app_commands.describe(erster_tag_der_woche="Erster Tag (Montag)", monat="Monat des Montags")
async def timetable_week(interaction: discord.Interaction, erster_tag_der_woche: int, monat: int):
    await interaction.response.defer()
    year = datetime.datetime.now(tz).year
    try:
        monday = datetime.datetime(year, monat, erster_tag_der_woche)
    except ValueError:
        await interaction.followup.send("Ungültiges Datum.", ephemeral=True); return
    all_events = await asyncio.to_thread(get_timetable)
    await interaction.followup.send("**Stundenplan für die Woche**")
    total = 0
    for i in range(7):
        day    = monday + datetime.timedelta(days=i)
        events = filter_timetable_for_day(all_events, day)
        total += len(events)
        embeds = create_modern_embeds(events, day.strftime("%A, %d. %B %Y"))
        for j in range(0, len(embeds), 10):
            await interaction.followup.send(embeds=embeds[j:j+10])
    await log_action(f"/stundenplan_woche: {total} Events ab {monday.strftime('%d.%m.%Y')}")


@tree.command(name="speiseplan_tag", description="Zeige das Mensa-Angebot für einen bestimmten Tag")
@app_commands.describe(tag="Tag (1-31)", monat="Monat (1-12)")
async def canteen_day(interaction: discord.Interaction, tag: int, monat: int):
    try:
        await interaction.response.defer()
    except discord.errors.NotFound:
        return
    year = datetime.datetime.now(tz).year
    try:
        day = datetime.datetime(year, monat, tag).date()
    except ValueError:
        await interaction.followup.send("Ungültiges Datum."); return
    meals    = await asyncio.to_thread(get_canteen_meals, day)
    date_str = day.strftime("%A, %d. %B %Y")
    embeds   = create_canteen_embeds(meals, date_str)
    if not embeds:
        await interaction.followup.send(f"Keine Gerichte für {day.strftime('%d.%m.%Y')}."); return
    for i in range(0, len(embeds), 10):
        await interaction.followup.send(embeds=embeds[i:i+10])


# ── Reaction Roles (unverändert) ─────────────────────────────────────────────

def save_reaction_roles():
    try:
        out = {str(mid): {"mappings": d.get("mappings",{}), "raw": d.get("raw",{})}
               for mid, d in reaction_roles.items()}
        with open(REACTION_ROLES_FILE, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving reaction roles: {e}")


def load_reaction_roles():
    global reaction_roles
    if not os.path.isfile(REACTION_ROLES_FILE):
        reaction_roles = {}; return
    try:
        with open(REACTION_ROLES_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        reaction_roles = {}
        for mid_str, data in loaded.items():
            try:
                mid = int(mid_str)
            except:
                continue
            mappings = {k: int(v) for k, v in data.get("mappings", {}).items()}
            reaction_roles[mid] = {"mappings": mappings, "raw": data.get("raw", {})}
    except Exception as e:
        print(f"Error loading reaction roles: {e}"); reaction_roles = {}


def parse_emoji_input(emoji_input: str):
    if not emoji_input: return None, None
    emoji_input = emoji_input.strip()
    m = re.match(r'^<a?:([a-zA-Z0-9_]+):([0-9]+)>$', emoji_input)
    if m:
        return f"id:{m.group(2)}", ("partial", m.group(1), int(m.group(2)))
    m2 = re.match(r'^[0-9]{5,}$', emoji_input)
    if m2:
        return f"id:{m2.group(0)}", ("partial", None, int(m2.group(0)))
    return emoji_input, ("unicode", emoji_input)


def emoji_key_from_payload(payload_emoji):
    eid = getattr(payload_emoji, "id", None)
    if eid: return f"id:{eid}"
    name = getattr(payload_emoji, "name", None)
    return name if name is not None else str(payload_emoji)


async def apply_reaction_to_message(message, add_reaction_info):
    if not add_reaction_info: return False
    typ = add_reaction_info[0]
    if typ == "unicode":
        try: await message.add_reaction(add_reaction_info[1]); return True
        except: return False
    elif typ == "partial":
        _, name, eid = add_reaction_info
        try:
            partial = discord.PartialEmoji(name=name, id=eid)
            await message.add_reaction(partial); return True
        except:
            try: await message.add_reaction(f"<:{name}:{eid}>" if name else f"<:{eid}>"); return True
            except: return False
    return False


@client.event
async def on_raw_reaction_add(payload):
    if payload.user_id == client.user.id: return
    entry = reaction_roles.get(payload.message_id)
    if not entry: return
    emoji_key = emoji_key_from_payload(payload.emoji)
    role_id   = entry["mappings"].get(emoji_key)
    if not role_id: return
    guild = client.get_guild(payload.guild_id)
    if not guild: return
    try: member = await guild.fetch_member(payload.user_id)
    except: member = guild.get_member(payload.user_id)
    if not member: return
    role = guild.get_role(role_id)
    if not role: return
    try:
        await member.add_roles(role, reason="Reaction role add")
        await log_action(f"Rolle '{role.name}' zu {member} hinzugefügt (Reaction Role)")
    except Exception as e:
        await log_action(f"ERROR: Rolle hinzufügen fehlgeschlagen: {e}")


@client.event
async def on_raw_reaction_remove(payload):
    if payload.user_id == client.user.id: return
    entry = reaction_roles.get(payload.message_id)
    if not entry: return
    emoji_key = emoji_key_from_payload(payload.emoji)
    role_id   = entry["mappings"].get(emoji_key)
    if not role_id: return
    guild = client.get_guild(payload.guild_id)
    if not guild: return
    try: member = await guild.fetch_member(payload.user_id)
    except: member = guild.get_member(payload.user_id)
    if not member: return
    role = guild.get_role(role_id)
    if not role: return
    try:
        await member.remove_roles(role, reason="Reaction role remove")
        await log_action(f"Rolle '{role.name}' von {member} entfernt (Reaction Role)")
    except Exception as e:
        await log_action(f"ERROR: Rolle entfernen fehlgeschlagen: {e}")


@tree.command(name="setup_reaction_role", description="Richte Reaction Roles für eine Nachricht ein")
@app_commands.describe(message_id="Nachrichten-ID", emoji="Emoji", role="Rolle")
async def setup_reaction_role(interaction: discord.Interaction, message_id: str, emoji: str, role: discord.Role):
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message(" Admin-Rechte benötigt.", ephemeral=True); return
    await interaction.response.defer(ephemeral=True)
    try: msg_id = int(message_id)
    except: await interaction.followup.send(" Ungültige Nachrichten-ID.", ephemeral=True); return
    message = None
    for ch in interaction.guild.text_channels:
        try: message = await ch.fetch_message(msg_id)
        except: pass
        if message: break
    if not message:
        await interaction.followup.send(" Nachricht nicht gefunden.", ephemeral=True); return
    emoji_key, add_reaction_info = parse_emoji_input(emoji)
    if not emoji_key:
        await interaction.followup.send(" Ungültiges Emoji.", ephemeral=True); return
    if msg_id not in reaction_roles:
        reaction_roles[msg_id] = {"mappings": {}, "raw": {}}
    reaction_roles[msg_id]["mappings"][emoji_key] = role.id
    reaction_roles[msg_id]["raw"][emoji_key] = emoji
    save_reaction_roles()
    added = await apply_reaction_to_message(message, add_reaction_info)
    await interaction.followup.send(
        f"Reaction Role eingerichtet!\nNachricht: `{msg_id}`\nEmoji: {emoji}\nRolle: {role.mention}"
        + ("" if added else "\n\n Emoji konnte nicht automatisch hinzugefügt werden."),
        ephemeral=True
    )
    await log_action(f"Reaction Role eingerichtet: Nachricht {msg_id}, Emoji {emoji}, Rolle '{role.name}'")


@tree.command(name="remove_reaction_role", description="Entferne ein Reaction Role")
@app_commands.describe(message_id="Nachrichten-ID", emoji="Emoji")
async def remove_reaction_role(interaction: discord.Interaction, message_id: str, emoji: str):
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message(" Admin-Rechte benötigt.", ephemeral=True); return
    try: msg_id = int(message_id)
    except: await interaction.response.send_message(" Ungültige ID.", ephemeral=True); return
    emoji_key, _ = parse_emoji_input(emoji)
    if not emoji_key or msg_id not in reaction_roles or emoji_key not in reaction_roles[msg_id]["mappings"]:
        await interaction.response.send_message(" Kein Reaction Role gefunden.", ephemeral=True); return
    del reaction_roles[msg_id]["mappings"][emoji_key]
    reaction_roles[msg_id]["raw"].pop(emoji_key, None)
    if not reaction_roles[msg_id]["mappings"]:
        del reaction_roles[msg_id]
    save_reaction_roles()
    await interaction.response.send_message(f" Reaction Role entfernt!\nNachricht: {msg_id}\nEmoji: {emoji}", ephemeral=True)
    await log_action(f"Reaction Role entfernt: Nachricht {msg_id}, Emoji {emoji}")


@tree.command(name="list_reaction_roles", description="Zeige alle Reaction Roles")
async def list_reaction_roles(interaction: discord.Interaction):
    if not reaction_roles:
        await interaction.response.send_message("Keine Reaction Roles konfiguriert.", ephemeral=True); return
    embed = discord.Embed(title="Konfigurierte Reaction Roles", color=5793266)
    for msg_id, data in reaction_roles.items():
        role_list = []
        for ek, rid in data.get("mappings", {}).items():
            role = interaction.guild.get_role(rid)
            role_list.append(f"{data['raw'].get(ek, ek)} → {role.mention if role else f'<gelöscht: {rid}>'}")
        embed.add_field(name=f"Nachricht ID: {msg_id}", value="\n".join(role_list), inline=False)
    await interaction.response.send_message(embed=embed, ephemeral=True)


@tree.command(name="create_reaction_role_message", description="Erstelle eine neue Nachricht für Reaction Roles")
@app_commands.describe(title="Titel", description="Beschreibung")
async def create_reaction_role_message(interaction: discord.Interaction, title: str, description: str):
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message(" Admin-Rechte benötigt.", ephemeral=True); return
    embed = discord.Embed(title=title, description=description, color=5793266)
    embed.set_footer(text="Reagiere mit einem Emoji, um eine Rolle zu erhalten!")
    message = await interaction.channel.send(embed=embed)
    await interaction.response.send_message(
        f" Nachricht erstellt! ID: `{message.id}`\nNutze `/setup_reaction_role` um Roles hinzuzufügen.",
        ephemeral=True
    )
    await log_action(f"Reaction-Role-Nachricht erstellt: ID {message.id}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Startup
# ═══════════════════════════════════════════════════════════════════════════════

@client.event
async def on_ready():
    print(f"Bot logged in as {client.user}")
    if not hasattr(client, "synced"):
        try:
            guild = discord.Object(id=1439649601765511382)
            tree.copy_global_to(guild=guild)
            await tree.sync(guild=guild)  # register on guild (instant)
            tree.clear_commands(guild=None)
            await tree.sync()             # wipe global commands (removes duplicates)
            print(f"Commands synced")
            client.synced = True
        except Exception as e:
            print(f"Failed to sync commands: {e}")

    await log_action("Bot gestartet und bereit")
    load_reaction_roles()
    print(f"Loaded {len(reaction_roles)} reaction-role message(s)")

    # State aus JSON laden (verhindert False Positives nach Neustart)
    saved = load_timetable_state(tz, TIMETABLE_STATE_FILE)
    if saved:
        previous_timetable["today"]    = saved["today"]
        previous_timetable["tomorrow"] = saved["tomorrow"]
        print("Timetable state restored from disk")
        await log_action("Stundenplan-State aus timetable_state.json geladen")
    else:
        # Kein gespeicherter State → fresh von API laden
        all_events = await asyncio.to_thread(get_timetable)
        previous_timetable["today"]    = filter_timetable_for_today(all_events)
        previous_timetable["tomorrow"] = filter_timetable_for_tomorrow(all_events)
        save_timetable_state(previous_timetable, tz, TIMETABLE_STATE_FILE)
        print("Initial timetable loaded from API")
        await log_action("Initialer Stundenplan von API geladen und gespeichert")
        
    if not scheduler.running:
        scheduler.add_job(send_timetable,          "cron",     hour=20, minute=0,  timezone="Europe/Berlin")
        scheduler.add_job(send_weekly_schedule,    "cron",     day_of_week="sun", hour=19, minute=45, timezone="Europe/Berlin")
        scheduler.add_job(check_timetable_changes, "interval", minutes=30, timezone="Europe/Berlin")
        scheduler.add_job(send_canteen_menu,       "cron",     hour=11, minute=0,  timezone="Europe/Berlin")
        scheduler.start()
        print("Scheduler started")
        await log_action(f"Scheduler gestartet. Debounce-Schwellenwert: {DEBOUNCE_THRESHOLD} Checks (= {DEBOUNCE_THRESHOLD * 20} Min)")
    else:
        await log_action("Bot reconnected - Scheduler läuft breits, Jobs nicht neu registriert")


client.run(TOKEN)