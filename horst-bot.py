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
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---- CONFIG ----
# Replace these IDs with your actual Discord IDs
MODEL_NAME = "gemini-2.0-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TOKEN = os.getenv("TOKEN") # <-- replace with your NEW token
ROLE_ID=1439649825183371465
CHANNEL_ID=1439649654055895080
ESSEN_CHANNEL_ID=1439977286882295909
CHANGES_ROLE_ID=1439649825183371465
CANTEEN_ROLE_ID=1439988749944487957
LOG_CHANNEL_ID=1439985824530829415

USERID = os.getenv("USERID")  # <-- replace with your user ID for timetable API
HASH = os.getenv("HASH")  # <-- replace with your hash for timetable API

PRAXISPHASEN_FILE = "praxisphasen.json"
# ---- Intents & Client ----
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)
scheduler = AsyncIOScheduler()

tz = pytz.timezone("Europe/Berlin")

# Store previous timetable state
previous_timetable = {"today": [], "tomorrow": []}

# In-memory reaction roles structure:
# { message_id(int): { "mappings": {emoji_key: role_id}, "raw": {emoji_key: raw_input_string} } }
reaction_roles = {}

# JSON storage path
REACTION_ROLES_FILE = "reaction_roles.json"

# Farbzuordnung nach Schluesselwoertern
WORD_COLORS = {
    "MATHE": 3066993,  # Gruen
    "WISSA": 9442302,  # Lila
    "INGG": 15105570,  # Orange
    "TGI": 3447003,  # Blau
    "Imp. P.": 15548997,  # Rot
}

DEFAULT_COLOR = 7506394  # Grau fuer unbekannte Kurse

# Mensa color scheme
MENSA_COLORS = {
    "header": 3066993,  # Gruen
    "meal": 5793266,  # Blau
}

import json
import os
from datetime import datetime, timedelta

# Add this near the top with other constants
PRAXISPHASEN_FILE = "praxisphasen.json"

def load_praxisphasen():
    """Load practical phases from JSON file."""
    if not os.path.exists(PRAXISPHASEN_FILE):
        print(f"Warning: {PRAXISPHASEN_FILE} not found")
        return {}
    
    try:
        with open(PRAXISPHASEN_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading praxisphasen: {e}")
        return {}

def is_in_practical_phase(date):
    """Check if a given date falls within any practical phase."""
    praxisphasen = load_praxisphasen()
    
    for phase_name, phase_data in praxisphasen.items():
        try:
            start_date = datetime.strptime(phase_data["start"], "%Y-%m-%d").date()
            end_date = datetime.strptime(phase_data["end"], "%Y-%m-%d").date()
            
            if start_date <= date <= end_date:
                print(f"Date {date} is in practical phase: {phase_name}")
                return True, phase_name
        except (ValueError, KeyError) as e:
            print(f"Error parsing phase {phase_name}: {e}")
            continue
    
    return False, None

def should_send_daily_message(tomorrow):
    """
    Check if daily message should be sent.
    Returns (should_send: bool, reason: str)
    """
    # Check if tomorrow is weekend
    if tomorrow.weekday() in [5, 6]:  # Saturday = 5, Sunday = 6
        day_name = tomorrow.strftime("%A")
        return False, f"Morgen ist {day_name} (Wochenende)"
    
    # Check if tomorrow is in a practical phase
    in_practical, phase_name = is_in_practical_phase(tomorrow)
    if in_practical:
        return False, f"Praxisphase ({phase_name})"
    
    return True, "OK"

def should_send_weekly_schedule():
    """
    Check if weekly schedule should be sent.
    Returns (should_send: bool, reason: str)
    """
    now = datetime.now(tz)
    next_monday = now + timedelta(days=(7 - now.weekday()))
    
    # Check if the upcoming week overlaps with any practical phase
    # Check all 7 days of the week
    for i in range(7):
        day = (next_monday + timedelta(days=i)).date()
        in_practical, phase_name = is_in_practical_phase(day)
        if in_practical:
            return False, f"Kommende Woche in Praxisphase ({phase_name})"
    
    return True, "OK"

def generate_truth_table_image(expr: str):
    import re, itertools
    from PIL import Image, ImageDraw, ImageFont

    # ---------------- Tokenizer ----------------
    def tokenize(s):
        token_spec = [
            (r'\s+', None),
            (r'\(', '('),
            (r'\)', ')'),
            (r'\<\=\>', '<=>'),
            (r'==', '=='),
            (r'=', '='),
            (r'\&', '&'),
            (r'\|', '|'),
            (r'\!', '!'),
            (r'\bNOT\b', '!'),
            (r'\bnot\b', '!'),
            (r'\bAND\b', '&'),
            (r'\band\b', '&'),
            (r'\bOR\b', '|'),
            (r'\bor\b', '|'),
            (r'[A-Z]', 'VAR'),   # single-letter variables A-Z
        ]
        pos = 0
        tokens = []
        while pos < len(s):
            matched = False
            for pattern, typ in token_spec:
                m = re.match(pattern, s[pos:], flags=re.IGNORECASE)
                if not m:
                    continue
                matched = True
                txt = m.group(0)
                if typ is None:
                    pass
                elif typ == 'VAR':
                    tokens.append(('VAR', txt.upper()))
                else:
                    tokens.append((typ, txt))
                pos += len(txt)
                break
            if not matched:
                raise ValueError(f"Ungueltiges Symbol in Ausdruck bei Position {pos}: '{s[pos]}'")
        return tokens

    # -------------- Shunting-yard -> RPN --------------
    def shunting_yard(tokens):
        prec = {
            '!':  (4, 'right'),
            '&':  (3, 'left'),
            '|':  (2, 'left'),
            '<=>':(1, 'left'),
            '==': (1, 'left'),
            '=':  (1, 'left'),
        }
        output = []
        stack = []
        for ttype, tval in tokens:
            if ttype == 'VAR':
                output.append((ttype, tval))
            elif ttype == '(':
                stack.append((ttype, tval))
            elif ttype == ')':
                while stack and stack[-1][0] != '(':
                    output.append(stack.pop())
                if not stack:
                    raise ValueError("Fehlende oeffnende Klammer")
                stack.pop()
            else:
                # operator
                while stack and stack[-1][0] != '(':
                    top = stack[-1][0]
                    if top not in prec:
                        break
                    p_top, assoc_top = prec[top]
                    p_op, assoc_op = prec[ttype]
                    if (assoc_op == 'left' and p_op <= p_top) or (assoc_op == 'right' and p_op < p_top):
                        output.append(stack.pop())
                    else:
                        break
                stack.append((ttype, tval))
        while stack:
            if stack[-1][0] == '(':
                raise ValueError("Fehlende schliessende Klammer")
            output.append(stack.pop())
        return output

    # -------------- RPN -> AST --------------
    class Node:
        def __init__(self, kind, value=None, left=None, right=None):
            self.kind = kind  # 'VAR', 'NOT', 'AND', 'OR', 'EQ'
            self.value = value  # for VAR: name, for EQ maybe symbol
            self.left = left
            self.right = right

        def to_infix_label(self):
            if self.kind == 'VAR':
                return self.value
            if self.kind == 'NOT':
                return f"not {self.left.to_infix_label()}"
            if self.kind == 'AND':
                return f"{self.left.to_infix_label()} and {self.right.to_infix_label()}"
            if self.kind == 'OR':
                return f"{self.left.to_infix_label()} or {self.right.to_infix_label()}"
            if self.kind == 'EQ':
                return f"{self.left.to_infix_label()} = {self.right.to_infix_label()}"
            return "?"


        def to_python(self):
            # produce a fully-parenthesized python expression for this node
            if self.kind == 'VAR':
                return f"{self.value}"
            if self.kind == 'NOT':
                return f"(not ({self.left.to_python()}))"
            if self.kind == 'AND':
                return f"(({self.left.to_python()}) and ({self.right.to_python()}))"
            if self.kind == 'OR':
                return f"(({self.left.to_python()}) or ({self.right.to_python()}))"
            if self.kind == 'EQ':
                return f"(({self.left.to_python()}) == ({self.right.to_python()}))"
            return "False"

    def rpn_to_ast(rpn):
        stack = []
        for ttype, tval in rpn:
            if ttype == 'VAR':
                stack.append(Node('VAR', value=tval))
            elif ttype == '!':
                if not stack:
                    raise ValueError("Ungueltiger Ausdruck: '!' ohne Operand")
                a = stack.pop()
                stack.append(Node('NOT', left=a))
            else:
                # binary
                if len(stack) < 2:
                    raise ValueError("Ungueltiger Ausdruck: Binaerer Operator ohne zwei Operanden")
                b = stack.pop()
                a = stack.pop()
                if ttype == '&':
                    stack.append(Node('AND', left=a, right=b))
                elif ttype == '|':
                    stack.append(Node('OR', left=a, right=b))
                elif ttype in ('=', '==', '<=>'):
                    stack.append(Node('EQ', left=a, right=b))
                else:
                    raise ValueError(f"Unbekannter Operator: {ttype}")
        if len(stack) != 1:
            raise ValueError("Syntaxfehler im Ausdruck (ueberbleibsel im Stack)")
        return stack[0]

    # -------------- Collect "major" steps (Option C) --------------
    def collect_major_steps(ast_root):
        """
        We collect a list of nodes (in the order they are evaluated) that are:
         - any NOT (unary) node
         - any binary node (AND/OR/EQ)
        The order should reflect evaluation order -> post-order traversal.
        """
        steps = []
        seen = set()

        def postorder(node):
            if node is None:
                return
            if node.kind == 'VAR':
                return
            # visit children first
            if node.left:
                postorder(node.left)
            if node.right:
                postorder(node.right)
            # add this node if it's a major operator
            # avoid duplicates by object id
            nid = id(node)
            if nid not in seen:
                if node.kind in ('NOT', 'AND', 'OR', 'EQ'):
                    steps.append(node)
                    seen.add(nid)

        postorder(ast_root)
        return steps

    # ----------------- Build everything -----------------
    tokens = tokenize(expr)
    if not tokens:
        raise ValueError("Keine Tokens im Ausdruck.")
    variables = sorted({tval for ttype, tval in tokens if ttype == 'VAR'})
    rpn = shunting_yard(tokens)
    ast_root = rpn_to_ast(rpn)
    major_nodes = collect_major_steps(ast_root)

    # Prepare step labels and their python expressions
    step_labels = [n.to_infix_label() for n in major_nodes]
    step_py_exprs = [n.to_python() for n in major_nodes]
    # final expression is the last major node (should be whole expression) if exists,
    # otherwise ast_root (in case of single NOT or single VAR)
    if major_nodes:
        final_label = step_labels[-1]
        final_py = step_py_exprs[-1]
    else:
        final_label = ast_root.to_infix_label() if ast_root else expr
        final_py = ast_root.to_python() if ast_root else expr

    # Build full list of step labels including the final full expression column (ensure final is last)
    # If final is already included as last major step, keep it; otherwise add it.
    # For clarity, we will treat step_labels as all major steps and the last of them is final.
    header_steps = step_labels[:]  # will be displayed after variables; last is final

    # ----------------- Evaluate table -----------------
    all_step_exprs = step_py_exprs[:]  # python expressions for each step, in same order
    # If no major steps (e.g., single variable), ensure we still have the final expression
    if not all_step_exprs:
        all_step_exprs = [final_py]
        header_steps = [final_label]

    rows = []
    for comb in itertools.product([False, True], repeat=len(variables)):
        env = dict(zip(variables, comb))
        row_vars = [1 if v else 0 for v in comb]
        step_results = []
        for step_expr in all_step_exprs:
            try:
                val = eval(step_expr, {}, env)
                # after computing step, we DO NOT mutate env with intermediate names;
                # steps are purely expressions using variables or previously-composed sub-exprs.
                # This is fine since step_exprs are full python subexpressions.
                step_results.append(1 if bool(val) else 0)
            except Exception:
                step_results.append('?')
        rows.append(row_vars + step_results)

    # ----------------- Render image -----------------
    header = variables + header_steps

    # colors (discord-dark style)
    bg_color = (44, 47, 51)
    header_bg = (60, 63, 68)
    text_color = (220, 221, 222)
    line_color = (32, 34, 37)
    accent_color = (114, 137, 218)
    final_color = (67, 181, 129)  # green for final column
    step_color = (250, 166, 26)

    font_size = 22
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()

    cell_padding = 16
    col_widths = []

    # determine column widths based on header and rows
    for col_idx in range(len(header)):
        max_len = len(str(header[col_idx]))
        for row in rows:
            # row length should match header length
            if col_idx < len(row):
                max_len = max(max_len, len(str(row[col_idx])))
        # heuristic width
        col_w = max(max_len * (font_size // 2) + cell_padding, 70)
        col_widths.append(col_w)

    width = sum(col_widths) + 40
    height = (len(rows) + 2) * (font_size + 18) + 40

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # draw header row
    x = 20
    y = 20
    header_height = font_size + 12
    for i, col_name in enumerate(header):
        w = col_widths[i]
        draw.rectangle([x, y, x + w, y + header_height], fill=header_bg)
        # color variables in accent, steps in step_color, final in final_color
        if i < len(variables):
            color = accent_color
        elif i == len(header) - 1:
            color = final_color
        else:
            color = step_color
        # wrap long header text if needed (simple)
        text = str(col_name)
        draw.text((x + 8, y + 4), text, fill=color, font=font)
        x += w

    # draw rows
    y += header_height + 10
    for r in rows:
        x = 20
        for i, cell in enumerate(r):
            w = col_widths[i]
            # final column cell colored green text
            if i == len(header) - 1:
                cell_color = final_color
            else:
                cell_color = text_color
            draw.text((x + 8, y), str(cell), fill=cell_color, font=font)
            x += w
        y += font_size + 16

    # draw vertical lines
    x = 20
    for w in col_widths:
        draw.line((x, 20, x, height - 20), fill=line_color, width=2)
        x += w
    draw.line((x, 20, x, height - 20), fill=line_color, width=2)

    # draw outer horizontal lines
    draw.line((20, 20, width - 20, 20), fill=line_color, width=2)
    draw.line((20, 20 + header_height, width - 20, 20 + header_height), fill=line_color, width=3)

    output_path = "truth_table.png"
    img.save(output_path)
    return output_path

def is_valid_url(url):
    """Validate if a string is a proper URL."""
    if not url or not isinstance(url, str):
        return False
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
    except Exception:
        return False


def get_course_color(coursename):
    """Get color for a course based on keywords or exact match."""
    coursename_upper = coursename.upper()
    for word, color in WORD_COLORS.items():
        if word.upper() in coursename_upper:
            return color
    return DEFAULT_COLOR


async def log_action(action: str):
    """Send log message to log channel with timestamp."""
    try:
        log_channel = client.get_channel(LOG_CHANNEL_ID)
        if not log_channel:
            print(f"Log channel not found. Action: {action}")
            return

        now = datetime.now(tz)
        timestamp = now.strftime("%d.%m.%Y %H:%M:%S")
        log_message = f"`[{timestamp}]` {action}"

        await log_channel.send(log_message)
    except Exception as e:
        print(f"Error sending log: {e}")

CACHE_FILE = "nutrition_cache.json"

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
    """
    Cleans up the cache. 
    Keeps data only for the current week (starting Monday) and the next week.
    """
    # Use the timezone defined in your bot (Europe/Berlin)
    today = datetime.now(tz).date()
    
    # Calculate start of current week (Monday)
    start_of_current_week = today - timedelta(days=today.weekday())
    # Calculate end of next week (Sunday)
    end_of_next_week = start_of_current_week + timedelta(days=13)

    new_cache = {}
    kept_count = 0
    removed_count = 0

    for key, value in cache.items():
        try:
            date_str = key.split("::")[0]
            entry_date = datetime.fromisoformat(date_str).date()   # FIXED

            if start_of_current_week <= entry_date <= end_of_next_week:
                new_cache[key] = value
                kept_count += 1
            else:
                removed_count += 1
        except ValueError:
            continue

            
    if removed_count > 0:
        print(f"[Cache] Cleaned up: Kept {kept_count} entries, Removed {removed_count} old/far-future entries.")
    return new_cache


genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

def parse_json_from_text(text: str):
    """
    Try to extract and parse the first JSON object from arbitrary text.
    Handles common cases where model returns markdown code fences or extra text.
    Returns a Python dict or None.
    """
    if not text:
        return None

    original = text
    text = text.strip()

    # 1) Remove markdown code fences ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # 2) If the model wrapped it in triple backticks inline like `{"k":1}`, remove single backticks
    text = text.replace("`", "")

    # 3) Try to find the first {...} block (greedy to closing brace)
    m = re.search(r"\{(?:[^{}]|(?R))*\}", text, flags=re.DOTALL) if hasattr(re, "R") else None
    # If recursive pattern is not available, fall back to simple approach:
    if m is None:
        # find first '{' and last '}' after it
        try:
            first = text.index("{")
            last = text.rindex("}")
            candidate = text[first:last+1]
        except ValueError:
            candidate = text  # no braces => try whole text
    else:
        candidate = m.group(0)

    # strip whitespace
    candidate = candidate.strip()

    # 4) Try json.loads directly
    try:
        return json.loads(candidate)
    except Exception:
        pass

    # 5) Tolerant fixes: remove trailing commas before } or ]
    candidate2 = re.sub(r",\s*(?=[}\]])", "", candidate)

    # 6) Replace single quotes with double quotes if it looks like single-quoted JSON
    # (only do this if there are single quotes and not a quote inside already double-quoted JSON)
    if "'" in candidate2 and '"' not in candidate2[:50]:
        candidate2 = candidate2.replace("'", '"')

    # 7) Final attempt
    try:
        return json.loads(candidate2)
    except Exception:
        # last resort: try ast.literal_eval to handle python-like dicts (risky)
        try:
            import ast
            val = ast.literal_eval(candidate2)
            if isinstance(val, dict):
                return val
        except Exception:
            pass

    # nothing worked
    print("parse_json_from_text: failed to parse JSON. Original text was:\n", original)
    return None
# -------------------------------
# GEMINI QUERY
# -------------------------------
    
    
def ask_gemini_for_nutrition(dish_name: str):
    print(f"  --> Asking Gemini about: '{dish_name}'...")
    
    # Stricter prompt
    prompt = f"""
    Analyze the dish "{dish_name}" (German canteen food).
    Estimate values for a standard portion (approx 350g).
    Return a SINGLE JSON object with exactly these keys: "kcal", "Eiweiss", "Kohlenhydrate", "Fette".
    Values should be numbers (int or float).
    Example: {{"kcal": 500, "Eiweiss": 20, "Kohlenhydrate": 50, "Fette": 15}}
    DO NOT output Markdown. DO NOT output explanations. ONLY JSON.
    """

    try:
        response = model.generate_content(prompt)
        text = response.candidates[0].content.parts[0].text.strip()
        
        # Use your robust parser function
        data = parse_json_from_text(text)
        
        if data:
            return data
        else:
            print(f"  --> Gemini Error: Could not parse JSON. Raw text: {text[:50]}...")
            return None

    except Exception as e:
        print(f"  --> Gemini Exception for '{dish_name}': {e}")
        return None


def clean_dish_name(dish_name: str) -> str:
    """Remove allergen markers and other parentheses content from dish names."""
    # Remove anything in parentheses (including the parentheses)
    cleaned = re.sub(r'\([^)]*\)', '', dish_name)
    # Cut off everything from " mit " onwards (case insensitive)
    cleaned = re.split(r'\s+mit\s+', cleaned, flags=re.IGNORECASE)[0]
    # Replace dashes with spaces
    cleaned = cleaned.replace('-', ' ')
    # Remove extra whitespace
    cleaned = ' '.join(cleaned.split())
    return cleaned.strip()

def get_nutrition_for_day(dish_name: str, date: datetime.date, cache: dict):
    """
    Gets nutrition data. 
    NOTE: 'cache' must be passed in. This function updates the dict but DOES NOT save to file.
    """
    clean_name = clean_dish_name(dish_name)
    cache_key = f"{date.isoformat()}::{clean_name}"

    # 1. Check Cache
    if cache_key in cache:
        return cache[cache_key]

    # 2. Ask Gemini
    nutrition = ask_gemini_for_nutrition(clean_name)
    
    if nutrition is None:
        print(f"  --> FAILED to get data for '{clean_name}'.")
        return None

    # 3. Update the cache dictionary in memory (Caller will save it later)
    cache[cache_key] = nutrition
    print(f"  --> New data fetched for: {clean_name}")
    return nutrition


def get_canteen_meals(date):
    """Fetch meals and handle caching efficiently."""
    date_str = date.strftime("%Y-%m-%d")
    url = f"https://api.studentenwerk-dresden.de/openmensa/v2/canteens/32/days/{date_str}/meals"

    # --- Date Checks ---
    today = datetime.today().date() 
    start_of_current_week = today - timedelta(days=today.weekday())
    end_of_next_week = start_of_current_week + timedelta(days=13)

    is_within_valid_range = start_of_current_week <= date <= end_of_next_week

    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            print(f"Canteen API returned status code {response.status_code}")
            return []

        meals = response.json()
        print(f"[CANTEEN] API returned {len(meals)} meals for {date_str}")

        # --- BATCH CACHING START ---
        cache = load_cache()
        cache = clean_old_dates(cache)
        cache_updated = False

        if is_within_valid_range:
            print(f"[CANTEEN] Processing {len(meals)} meals for {date_str} ...")
        else:
            print(f"[CANTEEN] Date {date} is out of valid range, skipping Gemini calls.")

        for i, meal in enumerate(meals):
            dish_name = meal.get("name", "")
            print(f"  [{i+1}/{len(meals)}] Meal: {dish_name}")

            if is_within_valid_range:
                # Try to get nutrition (cache or Gemini)
                nutrition = get_nutrition_for_day(dish_name, date, cache)

                clean_name = clean_dish_name(dish_name)
                cache_key = f"{date.isoformat()}::{clean_name}"

                if cache_key in cache:
                    print(f"    Nutrition loaded (source: {'cache' if nutrition else 'none'})")
                else:
                    print(f"    No nutrition data returned")

                if nutrition:
                    print(f"        {nutrition}")

                # mark cache update
                if nutrition and cache.get(cache_key) == nutrition:
                    cache_updated = True
            else:
                nutrition = None

            meal["nutrition"] = nutrition

        # Save cache once
        if cache_updated:
            save_cache(cache)
            print("[CANTEEN] Cache saved to disk")

        # --- BATCH CACHING END ---
        return meals

    except Exception as e:
        print(f"Error fetching canteen meals: {e}")
        return []


def create_canteen_embeds(meals, date_str):
    """Create Discord embeds for canteen menu."""
    if not meals:
        return []

    embeds = []

    # Header Embed
    header_embed = discord.Embed(color=MENSA_COLORS["header"])
    header_embed.set_author(name=f"Mensa-Speiseplan - {date_str}")
    header_embed.description = f"**Mensa Johanna** - {len(meals)} Gerichte verfuegbar"
    embeds.append(header_embed)

    # Create an embed for each meal
    for meal in meals:
        name = meal.get("name", "Unbekannt")
        category = meal.get("category", "")
        prices = meal.get("prices", {})
        notes = meal.get("notes", [])
        nutrition = meal.get("nutrition")  # Already fetched in get_canteen_meals

        embed = discord.Embed(title=name, color=MENSA_COLORS["meal"])

        # Add image if available AND valid
        image_url = meal.get("image")
        if image_url:
            # Add https: protocol if missing (API returns URLs starting with //)
            if image_url.startswith("//"):
                image_url = f"https:{image_url}"

            if is_valid_url(image_url):
                try:
                    embed.set_thumbnail(url=image_url)
                except Exception as e:
                    print(f"Warning: Could not set thumbnail for {name}: {e}")

        if category:
            embed.add_field(name="Kategorie", value=f"`{category}`", inline=False)

        # Add student price prominently (API uses "Studierende" key)
        student_price = prices.get("Studierende")
        if student_price is not None:
            try:
                embed.add_field(name="Preis (Studierende)", value=f"**{float(student_price):.2f}€**", inline=True)
            except Exception:
                embed.add_field(name="Preis (Studierende)", value=f"**{student_price}**", inline=True)

        # Add nutrition information
        if nutrition and any(v for v in nutrition.values() if v is not None):
            nutrition_text = ""
            if nutrition.get("kcal"):
                nutrition_text += f"--> **{nutrition['kcal']} kcal**\n"
            if nutrition.get("Eiweiss"):
                nutrition_text += f"--> Eiweiss: {nutrition['Eiweiss']}g\n"
            if nutrition.get("Kohlenhydrate"):
                nutrition_text += f"--> Kohlenhydrate: {nutrition['Kohlenhydrate']}g\n"
            if nutrition.get("Fette"):
                nutrition_text += f"--> Fette: {nutrition['Fette']}g"

            if nutrition_text:
                embed.add_field(name="Naehrwerte (geschaetzt 350g) NICHT GENAU!", value=nutrition_text, inline=True)
        else:
            embed.add_field(name="Naehrwerte", value="*Keine Daten*", inline=True)

        # Add allergens/notes
        if notes:
            notes_text = ", ".join(notes)
            if len(notes_text) > 200:
                notes_text = notes_text[:197] + "..."
            embed.add_field(name="Hinweise", value=notes_text, inline=False)

        embeds.append(embed)

    return embeds


async def send_canteen_menu():
    """Send today's canteen menu to Discord."""
    tomorrow = datetime.now(tz) + timedelta(days=1)
    tomorrow_date = tomorrow.date()
    
    # Check if we should send (using tomorrow's date since we're posting for the next day)
    should_send, reason = should_send_daily_message(tomorrow_date)
    if not should_send:
        print(f"Skipping canteen menu send: {reason}")
        await log_action(f"Mensa-Menue nicht gesendet: {reason}")
        return
    
    channel = client.get_channel(ESSEN_CHANNEL_ID)
    if not channel:
        print("Channel not found for canteen menu")
        await log_action("ERROR: Mensa-Menue konnte nicht gesendet werden: Channel nicht gefunden")
        return

    role_mention = f"<@&{CANTEEN_ROLE_ID}>"
    today = datetime.now(tz).date()

    # Fetch meals in thread to avoid blocking
    meals = await asyncio.to_thread(get_canteen_meals, today)
    date_str = today.strftime("%A, %d. %B %Y")

    if not meals:
        await log_action(f"Mensa-Menue: Keine Gerichte verfuegbar fuer {today.strftime('%d.%m.%Y')}")
        return

    # Create embeds (meals already have nutrition data)
    embeds = create_canteen_embeds(meals, date_str)

    # Discord allows max 10 embeds per message
    if len(embeds) <= 10:
        await channel.send(content=role_mention, embeds=embeds)
    else:
        for i in range(0, len(embeds), 10):
            chunk = embeds[i:i+10]
            if i == 0:
                await channel.send(content=role_mention, embeds=chunk)
            else:
                await channel.send(embeds=chunk)

    await log_action(f"Mensa-Menue gesendet: {len(meals)} Gerichte fuer {today.strftime('%d.%m.%Y')}")


def get_timetable():
    """Fetch all events from the API (week or full JSON)."""
    url = "https://selfservice.campus-dual.de/room/json"
    now = datetime.now(tz)
    start_ts = int(now.timestamp())
    end_ts = int((now + timedelta(days=7)).timestamp())

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
    tomorrow = datetime.now(tz) + timedelta(days=1)
    start_dt = tz.localize(datetime(tomorrow.year, tomorrow.month, tomorrow.day, 1, 0, 0))
    end_dt = tz.localize(datetime(tomorrow.year, tomorrow.month, tomorrow.day, 23, 0, 0))

    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    filtered = [
        e for e in events
        if start_ts <= e.get("start", 0) <= end_ts
    ]
    return filtered


def filter_timetable_for_today(events):
    """Filter events between 01:00 and 23:00 today."""
    today = datetime.now(tz)
    start_dt = tz.localize(datetime(today.year, today.month, today.day, 1, 0, 0))
    end_dt = tz.localize(datetime(today.year, today.month, today.day, 23, 0, 0))

    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    filtered = [
        e for e in events
        if start_ts <= e.get("start", 0) <= end_ts
    ]
    return filtered


def filter_timetable_for_day(events, day):
    """Filter events for a specific day between 01:00 and 23:00."""
    start_dt = tz.localize(datetime(day.year, day.month, day.day, 1, 0, 0))
    end_dt = tz.localize(datetime(day.year, day.month, day.day, 23, 0, 0))
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())
    return [e for e in events if start_ts <= e.get("start", 0) <= end_ts]


def normalize_event(event):
    """Normalize event for comparison (remove dynamic fields)."""
    return {
        "title": event.get("title", ""),
        "start": event.get("start", 0),
        "end": event.get("end", 0),
        "sroom": event.get("sroom", ""),
        "instructor": event.get("instructor", "")
    }


def compare_timetables(old_events, new_events):
    """Compare two timetables and return changes."""
    old_normalized = [normalize_event(e) for e in old_events]
    new_normalized = [normalize_event(e) for e in new_events]

    # Convert to JSON strings for easy comparison
    old_set = set(json.dumps(e, sort_keys=True) for e in old_normalized)
    new_set = set(json.dumps(e, sort_keys=True) for e in new_normalized)

    added = [json.loads(e) for e in (new_set - old_set)]
    removed = [json.loads(e) for e in (old_set - new_set)]

    return added, removed


def create_modern_embeds(events, date_str, course_group="IT-25APP"):
    """Create modern Discord embeds for timetable."""
    if not events:
        return []

    embeds = []

    # Header Embed
    header_embed = discord.Embed(color=5793266)
    header_embed.set_author(name=f"Stundenplan - {date_str}")
    header_embed.description = f"**{course_group}** - {len(events)} Vorlesungen heute"
    embeds.append(header_embed)

    # Create an embed for each event
    for e in events:
        start = datetime.fromtimestamp(e.get("start", 0), tz).strftime("%H:%M")
        end = datetime.fromtimestamp(e.get("end", 0), tz).strftime("%H:%M")
        time_range = f"{start} - {end}"

        course = e.get("title", "Unbekannt")
        room = e.get("sroom", "?")
        instructor = e.get("instructor", "?")

        # Get color based on course
        color = get_course_color(course)

        embed = discord.Embed(title=course, color=color)
        embed.add_field(name="Zeit", value=f"`{time_range}`", inline=True)
        embed.add_field(name="Raum", value=f"`{room}`", inline=True)
        embed.add_field(name="Dozent", value=f"`{instructor}`", inline=True)

        embeds.append(embed)

    return embeds


def create_change_embeds(added, removed, day_name):
    """Create embeds for timetable changes."""
    embeds = []

    if added or removed:
        header_embed = discord.Embed(
            title="Stundenplanaenderung erkannt!",
            description=f"Ã„nderungen fuer **{day_name}**",
            color=15158332  # Red color for alerts
        )
        embeds.append(header_embed)

    # Added events
    for event in added:
        start = datetime.fromtimestamp(event.get("start", 0), tz).strftime("%H:%M")
        end = datetime.fromtimestamp(event.get("end", 0), tz).strftime("%H:%M")
        time_range = f"{start} - {end}"

        course = event.get("title", "Unbekannt")
        room = event.get("sroom", "?")
        instructor = event.get("instructor", "?")

        color = get_course_color(course)

        embed = discord.Embed(title=f"NEU: {course}", color=color)
        embed.add_field(name="Zeit", value=f"`{time_range}`", inline=True)
        embed.add_field(name="Raum", value=f"`{room}`", inline=True)
        embed.add_field(name="Dozent", value=f"`{instructor}`", inline=True)

        embeds.append(embed)

    # Removed events
    for event in removed:
        start = datetime.fromtimestamp(event.get("start", 0), tz).strftime("%H:%M")
        end = datetime.fromtimestamp(event.get("end", 0), tz).strftime("%H:%M")
        time_range = f"{start} - {end}"

        course = event.get("title", "Unbekannt")
        room = event.get("sroom", "?")
        instructor = event.get("instructor", "?")

        embed = discord.Embed(title=f"ERROR: ENTFERNT: {course}", color=10038562) 
        embed.add_field(name="Zeit", value=f"`{time_range}`", inline=True)
        embed.add_field(name="Raum", value=f"`{room}`", inline=True)
        embed.add_field(name="Dozent", value=f"`{instructor}`", inline=True)

        embeds.append(embed)

    return embeds
async def send_timetable():
    """Send tomorrow's filtered events as modern embeds to Discord."""
    tomorrow = datetime.now(tz) + timedelta(days=1)
    tomorrow_date = tomorrow.date()
    
    # Check if we should send
    should_send, reason = should_send_daily_message(tomorrow_date)
    if not should_send:
        print(f"Skipping timetable send: {reason}")
        await log_action(f"Stundenplan nicht gesendet: {reason}")
        return
    
    channel = client.get_channel(CHANNEL_ID)
    if not channel:
        print("Channel not found")
        await log_action("ERROR: Stundenplan konnte nicht gesendet werden: Channel nicht gefunden")
        return

    role_mention = f"<@&{ROLE_ID}>"
    all_events = await asyncio.to_thread(get_timetable)
    events = filter_timetable_for_tomorrow(all_events)

    date_str = tomorrow.strftime("%A, %d. %B %Y")
    
    if not events:
        await channel.send(f"{role_mention} Keine Veranstaltungen fuer morgen ({tomorrow.strftime('%d.%m.%Y')}).")
        await log_action(f"Stundenplan: Keine Veranstaltungen fuer morgen ({tomorrow.strftime('%d.%m.%Y')})")
        return

    embeds = create_modern_embeds(events, date_str)
    
    # Discord allows max 10 embeds per message
    if len(embeds) <= 10:
        await channel.send(content=role_mention, embeds=embeds)
    else:
        # Split into multiple messages if needed
        for i in range(0, len(embeds), 10):
            chunk = embeds[i:i+10]
            if i == 0:
                await channel.send(content=role_mention, embeds=chunk)
            else:
                await channel.send(embeds=chunk)
    
    await log_action(f"Stundenplan gesendet: {len(events)} Veranstaltungen fuer {tomorrow.strftime('%d.%m.%Y')}")


async def send_weekly_schedule():
    """Send the timetable for the upcoming week (Monday-Sunday) every Sunday."""
    # Check if we should send
    should_send, reason = should_send_weekly_schedule()
    if not should_send:
        print(f"Skipping weekly schedule send: {reason}")
        await log_action(f"Wochenstundenplan nicht gesendet: {reason}")
        return
    
    channel = client.get_channel(CHANNEL_ID)
    if not channel:
        print("Channel not found")
        await log_action("ERROR: Wochenstundenplan konnte nicht gesendet werden: Channel nicht gefunden")
        return

    now = datetime.now(tz)
    next_monday = now + timedelta(days=(7 - now.weekday()))
    all_events = await asyncio.to_thread(get_timetable)
    
    role_mention = f"<@&{ROLE_ID}>"
    await channel.send(f"{role_mention} **Stundenplan fuer die kommende Woche**")
    
    total_events = 0
    for i in range(7):
        day = next_monday + timedelta(days=i)
        events = filter_timetable_for_day(all_events, day)
        date_str = day.strftime("%A, %d. %B %Y")
        total_events += len(events)
        
        embeds = create_modern_embeds(events, date_str)
        
        if embeds:
            # Split if more than 10 embeds
            for j in range(0, len(embeds), 10):
                chunk = embeds[j:j+10]
                await channel.send(embeds=chunk)
    
    await log_action(f"Wochenstundenplan gesendet: {total_events} Veranstaltungen")

global last_update_day
last_update_day = datetime.now(tz).date()

async def check_timetable_changes():
    """Check for timetable changes every 10 minutes."""
    current_date = datetime.now(tz)
    date_old = current_date- timedelta(hours=23)
    global previous_timetable
    global last_update_day

    if current_date.date() != date_old.date() and current_date.date() != last_update_day:
        """Update previous timetable at day switch (midnight)."""
        all_events = await asyncio.to_thread(get_timetable)
        today_events = filter_timetable_for_today(all_events)
        tomorrow_events = filter_timetable_for_tomorrow(all_events)

        previous_timetable["today"] = today_events
        previous_timetable["tomorrow"] = tomorrow_events

        await log_action("Tageswechsel: Stundenplan-Daten aktualisiert.")
        last_update_day = current_date.date()


    channel = client.get_channel(CHANNEL_ID)
    if not channel:
        print("Channel not found for change detection")
        await log_action("ERROR: Stundenplan-Aenderungspruefung: Channel nicht gefunden")
        return

    all_events = await asyncio.to_thread(get_timetable)

    # Get today's and tomorrow's events
    today_events = filter_timetable_for_today(all_events)
    tomorrow_events = filter_timetable_for_tomorrow(all_events)

    today = datetime.now(tz)
    tomorrow = today + timedelta(days=1)

    today_str = today.strftime("%A, %d. %B %Y")
    tomorrow_str = tomorrow.strftime("%A, %d. %B %Y")

    role_mention = f"<@&{CHANGES_ROLE_ID}>"

    changes_detected = False

    # Check for changes in today's schedule
    if previous_timetable["today"]:
        added_today, removed_today = compare_timetables(previous_timetable["today"], today_events)

        if added_today or removed_today:
            changes_detected = True
            embeds = create_change_embeds(added_today, removed_today, f"HEUTE ({today_str})")

            if len(embeds) <= 10:
                await channel.send(content=role_mention, embeds=embeds)
            else:
                for i in range(0, len(embeds), 10):
                    chunk = embeds[i:i+10]
                    if i == 0:
                        await channel.send(content=role_mention, embeds=chunk)
                    else:
                        await channel.send(embeds=chunk)

            await log_action(f"Stundenplanaenderung HEUTE: +{len(added_today)} neu, -{len(removed_today)} entfernt")

    # Check for changes in tomorrow's schedule
    if previous_timetable["tomorrow"]:
        added_tomorrow, removed_tomorrow = compare_timetables(previous_timetable["tomorrow"], tomorrow_events)

        if added_tomorrow or removed_tomorrow:
            changes_detected = True
            embeds = create_change_embeds(added_tomorrow, removed_tomorrow, f"MORGEN ({tomorrow_str})")

            if len(embeds) <= 10:
                await channel.send(content=role_mention, embeds=embeds)
            else:
                for i in range(0, len(embeds), 10):
                    chunk = embeds[i:i+10]
                    if i == 0:
                        await channel.send(content=role_mention, embeds=chunk)
                    else:
                        await channel.send(embeds=chunk)

            await log_action(f"Stundenplanaenderung MORGEN: +{len(added_tomorrow)} neu, -{len(removed_tomorrow)} entfernt")

    # Update previous timetable
    previous_timetable["today"] = today_events
    previous_timetable["tomorrow"] = tomorrow_events


# ---------------- Reaction Roles ---------------- #

def save_reaction_roles():
    """
    Save reaction roles to a JSON file.
    Structure saved:
    {
      "<message_id>": {
          "mappings": { "<emoji_key>": role_id, ... },
          "raw": { "<emoji_key>": "<original_input>", ... }
      },
      ...
    }
    """
    try:
        out = {}
        for mid, data in reaction_roles.items():
            out[str(mid)] = {
                "mappings": data.get("mappings", {}),
                "raw": data.get("raw", {})
            }
        with open(REACTION_ROLES_FILE, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving reaction roles: {e}")


def load_reaction_roles():
    """Load reaction roles from a JSON file into memory (keys as ints)."""
    global reaction_roles
    try:
        if not os.path.isfile(REACTION_ROLES_FILE):
            reaction_roles = {}
            return

        with open(REACTION_ROLES_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            reaction_roles = {}
            for mid_str, data in loaded.items():
                try:
                    mid = int(mid_str)
                except:
                    continue
                mappings = data.get("mappings", {})
                raw = data.get("raw", {})
                # Cast role ids to int
                mappings = {k: int(v) for k, v in mappings.items()}
                reaction_roles[mid] = {"mappings": mappings, "raw": raw}
    except Exception as e:
        print(f"Error loading reaction roles: {e}")
        reaction_roles = {}


def parse_emoji_input(emoji_input: str):

    if not emoji_input:
        return None, None

    emoji_input = emoji_input.strip()

    m = re.match(r'^<a?:([a-zA-Z0-9_]+):([0-9]+)>$', emoji_input)
    if m:
        name = m.group(1)
        eid = m.group(2)
        key = f"id:{eid}"
        return key, ("partial", name, int(eid))

    m2 = re.match(r'^[0-9]{5,}$', emoji_input)
    if m2:
        eid = m2.group(0)
        key = f"id:{eid}"
        return key, ("partial", None, int(eid))
    return emoji_input, ("unicode", emoji_input)


def emoji_key_from_payload(payload_emoji):
    """Create emoji_key from a RawReactionActionEvent. Uses id if present, else the name (unicode)."""
    eid = getattr(payload_emoji, "id", None)
    if eid:
        return f"id:{eid}"
    
    name = getattr(payload_emoji, "name", None)
    return name if name is not None else str(payload_emoji)


async def apply_reaction_to_message(message, add_reaction_info):
    """
    Given a message and add_reaction_info returned from parse_emoji_input,
    attempt to add the reaction to the message object.
    """
    if not add_reaction_info:
        return False
    typ = add_reaction_info[0]
    if typ == "unicode":
        try:
            await message.add_reaction(add_reaction_info[1])
            return True
        except Exception:
            return False
    elif typ == "partial":
        _, name, eid = add_reaction_info
        try:
            if name:
                partial = discord.PartialEmoji(name=name, id=eid)
            else:
                partial = discord.PartialEmoji(name=None, id=eid)
            await message.add_reaction(partial)
            return True
        except Exception:
            try:
                raw = f"<:{name}:{eid}>" if name else f"<:{eid}>"
                await message.add_reaction(raw)
                return True
            except Exception:
                return False
    return False


# ---------------- Raw reaction handlers ---------------- #

@client.event
async def on_raw_reaction_add(payload):
    """Handle reaction add events for reaction roles."""
    # Ignore bot reactions
    if payload.user_id == client.user.id:
        return

    mid = payload.message_id
    entry = reaction_roles.get(mid)
    if not entry:
        return

    emoji_key = emoji_key_from_payload(payload.emoji)
    role_id = entry["mappings"].get(emoji_key)
    if not role_id:
        print(f"[RR] No mapping for emoji_key={emoji_key} on message {mid}. Stored keys: {list(entry['mappings'].keys())}")
        return

    guild = client.get_guild(payload.guild_id)
    if not guild:
        return

    try:
        member = await guild.fetch_member(payload.user_id)
    except Exception:
        member = guild.get_member(payload.user_id)

    if not member:
        return

    role = guild.get_role(role_id)
    if not role:
        print(f"Role {role_id} not found in guild {guild.id}")
        return

    try:
        await member.add_roles(role, reason="Reaction role add")
        print(f"Added role {role.name} to {member.display_name}")
        await log_action(f"Rolle '{role.name}' zu {member} hinzugefuegt (Reaction Role)")
    except Exception as e:
        print(f"Error adding role: {e}")
        await log_action(f"ERROR: Fehler beim Hinzufuegen der Rolle '{role.name}': {e}")


@client.event
async def on_raw_reaction_remove(payload):
    """Handle reaction remove events for reaction roles."""
    # Ignore bot reactions
    if payload.user_id == client.user.id:
        return

    mid = payload.message_id
    entry = reaction_roles.get(mid)
    if not entry:
        return

    emoji_key = emoji_key_from_payload(payload.emoji)
    role_id = entry["mappings"].get(emoji_key)
    if not role_id:
        print(f"[RR] No mapping for emoji_key={emoji_key} on message {mid}. Stored keys: {list(entry['mappings'].keys())}")
        return

    guild = client.get_guild(payload.guild_id)
    if not guild:
        return

    try:
        member = await guild.fetch_member(payload.user_id)
    except Exception:
        member = guild.get_member(payload.user_id)

    if not member:
        return

    role = guild.get_role(role_id)
    if not role:
        print(f"Role {role_id} not found in guild {guild.id}")
        return

    try:
        await member.remove_roles(role, reason="Reaction role remove")
        print(f"Removed role {role.name} from {member.display_name}")
        await log_action(f"Rolle '{role.name}' von {member} entfernt (Reaction Role)")
    except Exception as e:
        print(f"Error removing role: {e}")
        await log_action(f"ERROR: Fehler beim Entfernen der Rolle '{role.name}': {e}")


# ---------------- Slash Commands ---------------- #
@tree.command(name="truth", description="Erzeuge eine Wahrheitstabelle fuer einen Booleschen Ausdruck")
@app_commands.describe(ausdruck="Boolescher Ausdruck (z.B. A and !B or C)")
async def truth_command(interaction: discord.Interaction, ausdruck: str):

    await interaction.response.defer()

    try:
        path = generate_truth_table_image(ausdruck)
    except Exception as e:
        await interaction.followup.send(f"Fehler im Ausdruck:\n```\n{e}\n```")
        return

    file = discord.File(path, filename="truth_table.png")

    embed = discord.Embed(
        title="Wahrheitstabelle",
        description=f"Ausdruck: `{ausdruck}`",
        color=0x7289DA
    )
    embed.set_image(url="attachment://truth_table.png")

    await interaction.followup.send(embed=embed, file=file)

@tree.command(name="setup_reaction_role", description="Richte Reaction Roles fuer eine Nachricht ein")
@app_commands.describe(
    message_id="Die ID der Nachricht",
    emoji="Das Emoji fuer die Reaktion (Unicode oder <:name:id>)",
    role="Die Rolle, die vergeben werden soll"
)
async def setup_reaction_role(interaction: discord.Interaction, message_id: str, emoji: str, role: discord.Role):
    """Set up a reaction role on a message."""
    # Check if user has administrator permissions
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message("ERROR: Du benÃ¶tigst Administrator-Rechte fuer diesen Befehl.", ephemeral=True)
        return

    # Defer immediately to stop "Unknown interaction" errors
    await interaction.response.defer(ephemeral=True)
    try:
        msg_id = int(message_id)
    except ValueError:
        await interaction.followup.send("ERROR: Ungueltige Nachrichten-ID.", ephemeral=True)
        return

    # Try to find the message
    message = None
    guild = interaction.guild

    if not guild:
        await interaction.followup.send("ERROR: Dieser Befehl geht nur in einem Server.", ephemeral=True)
        return

    for ch in guild.text_channels:
        try:
            message = await ch.fetch_message(msg_id)
            if message:
                break
        except:
            pass

    if not message:
        await interaction.followup.send("ERROR: Nachricht nicht gefunden.", ephemeral=True)
        return

    # Parse emoji
    emoji_key, add_reaction_info = parse_emoji_input(emoji)
    if not emoji_key:
        await interaction.followup.send("ERROR: Ungueltiges Emoji.", ephemeral=True)
        return

    # Save reaction role entry
    if msg_id not in reaction_roles:
        reaction_roles[msg_id] = {"mappings": {}, "raw": {}}

    reaction_roles[msg_id]["mappings"][emoji_key] = role.id
    reaction_roles[msg_id]["raw"][emoji_key] = emoji
    save_reaction_roles()
    added = await apply_reaction_to_message(message, add_reaction_info)# Respond with final confirmation
    if added:
        await interaction.followup.send(
            f"Reaction Role eingerichtet!\n"
            f"Nachricht: `{msg_id}`\n"
            f"Emoji: {emoji}\n"
            f"Rolle: {role.mention}",
            ephemeral=True
        )
    else:
        await interaction.followup.send(
            f"Reaction Role gespeichert, aber konnte das Emoji nicht automatisch hinzufuegen.\n"
            f"Bitte fuege das Emoji manuell zur Nachricht hinzu.\n\n"
            f"Nachricht: `{msg_id}`\n"
            f"Emoji: {emoji}\n"
            f"Rolle: {role.mention}",
            ephemeral=True
        )


    # Try to add the reaction to the message (use the parsed info)
    try:
        added = await apply_reaction_to_message(message, add_reaction_info)
        if not added:
            await interaction.response.send(
                f"Reaction Role eingerichtet!\n"
                f"Nachricht: {msg_id}\n"
                f"Emoji: {emoji}\n"
                f"Rolle: {role.mention}\n\n"
                f"Hinweis: Konnte das Emoji nicht automatisch zur Nachricht hinzufuegen. Bitte fuege es manuell hinzu.",
                ephemeral=True
            )
            await log_action(f"Reaction Role eingerichtet fuer Nachricht {msg_id}, Emoji {emoji}, Rolle '{role.name}' (Emoji nicht automatisiert hinzugefuegt)")
            return
    except Exception as e:
        await interaction.response.send_message(f"Reaction Role eingerichtet, aber konnte Emoji nicht hinzufuegen: {e}", ephemeral=True)
        await log_action(f"Reaction Role eingerichtet fuer Nachricht {msg_id}, Emoji {emoji}, Rolle '{role.name}' (Fehler beim Hinzufuegen des Emoji: {e})")
        return

    await interaction.response.send_message(
        f"Reaction Role eingerichtet!\n"
        f"Nachricht: {msg_id}\n"
        f"Emoji: {emoji}\n"
        f"Rolle: {role.mention}",
        ephemeral=True
    )
    await log_action(f"Reaction Role eingerichtet: Nachricht {msg_id}, Emoji {emoji}, Rolle '{role.name}'")


@tree.command(name="remove_reaction_role", description="Entferne ein Reaction Role von einer Nachricht")
@app_commands.describe(
    message_id="Die ID der Nachricht",
    emoji="Das Emoji, das entfernt werden soll (Unicode oder <:name:id>)"
)
async def remove_reaction_role(interaction: discord.Interaction, message_id: str, emoji: str):
    """Remove a reaction role from a message."""
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message("ERROR: Du benÃ¶tigst Administrator-Rechte fuer diesen Befehl.", ephemeral=True)
        return

    try:
        msg_id = int(message_id)
    except ValueError:
        await interaction.response.send_message("ERROR: Ungueltige Nachrichten-ID.", ephemeral=True)
        return

    emoji_key, _ = parse_emoji_input(emoji)
    if not emoji_key:
        await interaction.response.send_message("ERROR: Ungueltiges Emoji.", ephemeral=True)
        return

    if msg_id not in reaction_roles or emoji_key not in reaction_roles[msg_id]["mappings"]:
        await interaction.response.send_message("ERROR: Kein Reaction Role fuer diese Kombination gefunden.", ephemeral=True)
        return

    # Remove mapping
    del reaction_roles[msg_id]["mappings"][emoji_key]
    if emoji_key in reaction_roles[msg_id].get("raw", {}):
        del reaction_roles[msg_id]["raw"][emoji_key]

    # If no more emojis for this message, remove the message entry
    if not reaction_roles[msg_id]["mappings"]:
        del reaction_roles[msg_id]

    save_reaction_roles()

    await interaction.response.send_message(
        f"Reaction Role entfernt!\n"
        f"Nachricht: {msg_id}\n"
        f"Emoji: {emoji}",
        ephemeral=True
    )
    await log_action(f"Reaction Role entfernt: Nachricht {msg_id}, Emoji {emoji}")


@tree.command(name="list_reaction_roles", description="Zeige alle konfigurierten Reaction Roles")
async def list_reaction_roles(interaction: discord.Interaction):
    """List all configured reaction roles."""
    if not reaction_roles:
        await interaction.response.send_message("Keine Reaction Roles konfiguriert.", ephemeral=True)
        return

    embed = discord.Embed(
        title="Konfigurierte Reaction Roles",
        color=5793266
    )

    for msg_id, data in reaction_roles.items():
        emoji_roles = data.get("mappings", {})
        raw_map = data.get("raw", {})
        role_list = []
        for emoji_key, role_id in emoji_roles.items():
            role = interaction.guild.get_role(role_id)
            role_name = role.mention if role else f"<gelÃ¶scht: {role_id}>"
            raw = raw_map.get(emoji_key, emoji_key)
            role_list.append(f"{raw} â†’ {role_name}")

        embed.add_field(
            name=f"Nachricht ID: {msg_id}",
            value="\n".join(role_list),
            inline=False
        )

    await interaction.response.send_message(embed=embed, ephemeral=True)


@tree.command(name="create_reaction_role_message", description="Erstelle eine neue Nachricht fuer Reaction Roles")
@app_commands.describe(
    title="Titel der Nachricht",
    description="Beschreibung der Nachricht"
)
async def create_reaction_role_message(interaction: discord.Interaction, title: str, description: str):
    """Create a new message for reaction roles."""
    # Check if user has administrator permissions
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message("ERROR: Du benÃ¶tigst Administrator-Rechte fuer diesen Befehl.", ephemeral=True)
        return

    embed = discord.Embed(
        title=title,
        description=description,
        color=5793266
    )
    embed.set_footer(text="Reagiere mit einem Emoji, um eine Rolle zu erhalten!")

    message = await interaction.channel.send(embed=embed)

    await interaction.response.send_message(
        f"Nachricht erstellt!\n"
        f"Nachrichten-ID: `{message.id}`\n\n"
        f"Nutze `/setup_reaction_role` um Reaction Roles hinzuzufuegen.",
        ephemeral=True
    )
    await log_action(f"Reaction Role Nachricht erstellt: ID {message.id}, Titel '{title}'")


@tree.command(name="stundenplan_tag", description="Zeige den Stundenplan fuer einen bestimmten Tag")
@app_commands.describe(tag="Tag des Monats (1-31)", monat="Monat (1-12)")
async def stundenplan_tag(interaction: discord.Interaction, tag: int, monat: int):
    await interaction.response.defer()

    year = datetime.now(tz).year
    try:
        day = datetime(year, monat, tag)
    except ValueError:
        await interaction.followup.send("Ungueltiges Datum.", ephemeral=True)
        await log_action(f"ERROR: /stundenplan_tag Befehl fehlgeschlagen: Ungueltiges Datum ({tag}.{monat})")
        return

    all_events = await asyncio.to_thread(get_timetable)
    events = filter_timetable_for_day(all_events, day)
    date_str = day.strftime("%A, %d. %B %Y")

    embeds = create_modern_embeds(events, date_str)

    if not embeds:
        await interaction.followup.send(f"Keine Veranstaltungen fuer {day.strftime('%d.%m.%Y')}.")
        await log_action(f" /stundenplan_tag: Keine Veranstaltungen fuer {day.strftime('%d.%m.%Y')}")
        return

    if len(embeds) <= 10:
        await interaction.followup.send(embeds=embeds)
    else:
        await interaction.followup(embeds=embeds[:10])
        # Send remaining embeds as follow-up because of 10 Embed limit
        for i in range(10, len(embeds), 10):
            chunk = embeds[i:i+10]
            await interaction.followup.send(embeds=chunk)

    await log_action(f"/stundenplan_tag ausgefuehrt: {len(events)} Veranstaltungen fuer {day.strftime('%d.%m.%Y')}")


@tree.command(name="stundenplan_woche", description="Zeige den Stundenplan fuer eine Woche")
@app_commands.describe(erster_tag_der_woche="Erster Tag der gewuenschten Woche (Montag)", monat="Monat des Montags")
async def timetable_week(interaction: discord.Interaction, erster_tag_der_woche: int, monat: int):
    # 1. Acknowledge the interaction immediately. This is your ONLY use of interaction.response.
    await interaction.response.defer()

    year = datetime.now(tz).year
    try:
        monday = datetime(year, monat, erster_tag_der_woche)
    except ValueError:

        await interaction.followup.send("Ungueltiges Datum.", ephemeral=True)
        await log_action(f"ERROR: /timetable_week Befehl fehlgeschlagen: Ungueltiges Datum ({erster_tag_der_woche}.{monat})")
        return

    all_events = await asyncio.to_thread(get_timetable)

    await interaction.followup.send("**Stundenplan fuer die Woche**")

    total_events = 0
    for i in range(7):
        day = monday + timedelta(days=i)
        events = filter_timetable_for_day(all_events, day)
        date_str = day.strftime("%A, %d. %B %Y")
        total_events += len(events)

        embeds = create_modern_embeds(events, date_str)

        if embeds:
            for j in range(0, len(embeds), 10):
                chunk = embeds[j:j+10]
                # 4. All other messages must also be followups.
                await interaction.followup.send(embeds=chunk)

    await log_action(f"/timetable_week ausgefuehrt: {total_events} Veranstaltungen ab {monday.strftime('%d.%m.%Y')}")

@tree.command(name="speiseplan_tag", description="Zeige das Mensa-Angebot fuer einen bestimmten Tag")
@app_commands.describe(tag="Tag des Monats (1-31)", monat="Monat (1-12)")
async def canteen_day(interaction: discord.Interaction, tag: int, monat: int):
    # Defer immediately to prevent timeout
    try:
        await interaction.response.defer()
    except discord.errors.NotFound:
        print("Interaction expired BEFORE defer() — likely double bot instance or slow system.")
        return
    
    year = datetime.now(tz).year
    try:
        day = datetime(year, monat, tag).date()
    except ValueError:
        await interaction.followup.send("Ungueltiges Datum.")
        await log_action(f"ERROR: /canteen_day Befehl fehlgeschlagen: Ungueltiges Datum ({tag}.{monat})")
        return

    # Fetch meals in thread
    meals = await asyncio.to_thread(get_canteen_meals, day)
    date_str = day.strftime("%A, %d. %B %Y")

    # Create embeds
    embeds = create_canteen_embeds(meals, date_str)

    if not embeds:
        await interaction.followup.send(f"Keine Mensa-Gerichte fuer {day.strftime('%d.%m.%Y')}.")
        await log_action(f"â„¹ /canteen_day: Keine Gerichte fuer {day.strftime('%d.%m.%Y')}")
        return

    if len(embeds) <= 10:
        await interaction.followup.send(embeds=embeds)
    else:
        await interaction.followup.send(embeds=embeds[:10])
        for i in range(10, len(embeds), 10):
            chunk = embeds[i:i+10]
            await interaction.followup.send(embeds=chunk)

    await log_action(f"/canteen_day ausgefuehrt: {len(meals)} Gerichte fuer {day.strftime('%d.%m.%Y')}")


# ---------------- Startup & Scheduler ---------------- #

@client.event
async def on_ready():
    print(f"Bot logged in as {client.user}")
    if not hasattr(client, "synced"):
        try:
            synced = await tree.sync()
            print(f"Synced {len(synced)} command(s)")
            client.synced = True
        except Exception as e:
            print(f"Failed to sync commands: {e}")

    await log_action("Bot gestartet und bereit")

    # Load reaction roles from disk
    load_reaction_roles()
    print(f"Loaded {len(reaction_roles)} reaction-role message(s) from disk")

    # Initialize previous timetable on startup
    all_events = await asyncio.to_thread(get_timetable)
    previous_timetable["today"] = filter_timetable_for_today(all_events)
    previous_timetable["tomorrow"] = filter_timetable_for_tomorrow(all_events)
    print("Initial timetable loaded for change detection")
    await log_action("Initialer Stundenplan geladen fuer uenderungserkennung")

    # Schedule daily message for Timetable of Tommorrow at 20:00 UTC +1 time
    scheduler.add_job(
        send_timetable,
        trigger="cron",
        hour=20,
        minute=0,
        timezone="Europe/Berlin"
    )

    # Schedule weekly overview on Sunday at 19:45 UTC +1 time
    scheduler.add_job(
        send_weekly_schedule,
        trigger="cron",
        day_of_week="sun",
        hour=19,
        minute=45,
        timezone="Europe/Berlin"
    )

    # Schedule change detection every 10 minutes
    scheduler.add_job(
        check_timetable_changes,
        trigger="interval",
        minutes=10,
        timezone="Europe/Berlin"
    )

    # Schedule daily canteen message at 10:30 UTC +1 time
    scheduler.add_job(
        send_canteen_menu,
        trigger="cron",
        hour=11,
        minute=0,
        timezone="Europe/Berlin"
    )

    scheduler.start()
    print("Scheduler started - checking for changes every 10 minutes")
    print("Canteen menu will be posted daily at 10:30")
    await log_action("Scheduler gestartet: Stundenplan um 20:00, Mensa um 10:30, Ãnderungen alle 10 Min")


client.run(TOKEN)
