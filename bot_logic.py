"""
bot_logic.py – Pure functions extracted from horst-bot.py.
No Discord, no API calls – fully unit-testable.
"""

import json
import re
import os
from datetime import datetime, timedelta
import pytz
import itertools

TZ = pytz.timezone("Europe/Berlin")

# ── Color config ─────────────────────────────────────────────────────────────
WORD_COLORS = {
    "MATHE":   3066993,   # Grün
    "WISSA":   9442302,   # Lila
    "INGG":   15105570,   # Orange
    "TGI":     3447003,   # Blau
    "Imp. P.":15548997,   # Rot
    "DTDS":    1752220,   # Teal
}
DEFAULT_COLOR = 7506394

# ── Debounce config ───────────────────────────────────────────────────────────
DEBOUNCE_THRESHOLD = 2   # wie viele aufeinanderfolgende Detektionen → Alert

_pending: dict = {
    "today":    {"hash": None, "count": 0, "data": None},
    "tomorrow": {"hash": None, "count": 0, "data": None},
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Event-Normalisierung & Vergleich
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_event(event: dict) -> dict:
    """Normalisiert ein Event für den Vergleich (Whitespace, fehlende Felder)."""
    return {
        "title":      event.get("title", "").strip(),
        "start":      event.get("start", 0),
        "end":        event.get("end", 0),
        "sroom":      event.get("sroom", "").strip(),
        "instructor": event.get("instructor", "").strip(),
    }


def compare_timetables(old_events: list, new_events: list) -> tuple[list, list]:
    """Gibt (added, removed) zurück."""
    old_set = {json.dumps(normalize_event(e), sort_keys=True) for e in old_events}
    new_set = {json.dumps(normalize_event(e), sort_keys=True) for e in new_events}
    added   = [json.loads(s) for s in (new_set - old_set)]
    removed = [json.loads(s) for s in (old_set - new_set)]
    return added, removed


# ═══════════════════════════════════════════════════════════════════════════════
#  Filterung
# ═══════════════════════════════════════════════════════════════════════════════

def filter_timetable_for_day(events: list, day: datetime, tz=TZ) -> list:
    """Filtert Events für einen bestimmten Tag (01:00–23:00)."""
    if day.tzinfo is None:
        day = tz.localize(day)
    start_dt = day.replace(hour=1,  minute=0, second=0, microsecond=0)
    end_dt   = day.replace(hour=23, minute=0, second=0, microsecond=0)
    start_ts = int(start_dt.timestamp())
    end_ts   = int(end_dt.timestamp())
    return [e for e in events if start_ts <= e.get("start", 0) <= end_ts]


def filter_timetable_for_today(events: list, tz=TZ) -> list:
    return filter_timetable_for_day(events, datetime.now(tz), tz)


def filter_timetable_for_tomorrow(events: list, tz=TZ) -> list:
    return filter_timetable_for_day(events, datetime.now(tz) + timedelta(days=1), tz)


# ═══════════════════════════════════════════════════════════════════════════════
#  Farben
# ═══════════════════════════════════════════════════════════════════════════════

def get_course_color(coursename: str) -> int:
    upper = coursename.upper()
    for word, color in WORD_COLORS.items():
        if word.upper() in upper:
            return color
    return DEFAULT_COLOR


# ═══════════════════════════════════════════════════════════════════════════════
#  Mensa-Hilfsfunktionen
# ═══════════════════════════════════════════════════════════════════════════════

def clean_dish_name(dish_name: str) -> str:
    """Entfernt Allergen-Kürzel und Beilagen aus Gerichtnamen."""
    cleaned = re.sub(r'\([^)]*\)', '', dish_name)
    cleaned = re.split(r'\s+mit\s+', cleaned, flags=re.IGNORECASE)[0]
    cleaned = cleaned.replace('-', ' ')
    return ' '.join(cleaned.split()).strip()


def parse_json_from_text(text: str):
    """Extrahiert JSON aus beliebigem Text (Markdown-Fences etc.)."""
    if not text:
        return None
    original = text
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.replace("`", "")
    try:
        first = text.index("{"); last = text.rindex("}")
        candidate = text[first:last + 1].strip()
    except ValueError:
        candidate = text
    try:
        return json.loads(candidate)
    except Exception:
        pass
    candidate2 = re.sub(r",\s*(?=[}\]])", "", candidate)
    try:
        return json.loads(candidate2)
    except Exception:
        pass
    try:
        import ast
        val = ast.literal_eval(candidate2)
        if isinstance(val, dict):
            return val
    except Exception:
        pass
    print("parse_json_from_text: failed. Original:\n", original)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Praxisphasen
# ═══════════════════════════════════════════════════════════════════════════════

def load_praxisphasen(filepath: str = "praxisphasen.json") -> dict:
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading praxisphasen: {e}")
        return {}


def is_in_practical_phase(date, filepath: str = "praxisphasen.json") -> tuple[bool, str | None]:
    if hasattr(date, 'date') and not isinstance(date, datetime):
        pass
    elif hasattr(date, 'date'):
        date = date.date()
    praxisphasen = load_praxisphasen(filepath)
    for phase_name, phase_data in praxisphasen.items():
        try:
            start = datetime.strptime(phase_data["start"], "%Y-%m-%d").date()
            end   = datetime.strptime(phase_data["end"],   "%Y-%m-%d").date()
            if start <= date <= end:
                return True, phase_name
        except (ValueError, KeyError):
            continue
    return False, None


def should_send_daily_message(tomorrow, filepath: str = "praxisphasen.json") -> tuple[bool, str]:
    day = tomorrow.date() if hasattr(tomorrow, 'date') else tomorrow
    if hasattr(day, 'weekday') and day.weekday() in [5, 6]:
        return False, f"Wochenende ({tomorrow.strftime('%A')})"
    in_practical, phase_name = is_in_practical_phase(day, filepath)
    if in_practical:
        return False, f"Praxisphase ({phase_name})"
    return True, "OK"


def should_send_weekly_schedule(tz=TZ, filepath: str = "praxisphasen.json") -> tuple[bool, str]:
    now = datetime.now(tz)
    next_monday = now + timedelta(days=(7 - now.weekday()))
    for i in range(7):
        day = (next_monday + timedelta(days=i)).date()
        in_practical, phase_name = is_in_practical_phase(day, filepath)
        if in_practical:
            return False, f"Kommende Woche in Praxisphase ({phase_name})"
    return True, "OK"


# ═══════════════════════════════════════════════════════════════════════════════
#  Debounce – verhindert False-Positive-Alerts
# ═══════════════════════════════════════════════════════════════════════════════

def _changes_hash(added: list, removed: list) -> str:
    return json.dumps({
        "added":   sorted(json.dumps(e, sort_keys=True) for e in added),
        "removed": sorted(json.dumps(e, sort_keys=True) for e in removed),
    }, sort_keys=True)


def check_debounce(
    key: str,
    added: list,
    removed: list,
    threshold: int = DEBOUNCE_THRESHOLD,
) -> tuple[bool, dict | None]:
    """
    Gibt (should_alert, data) zurück.

    Eine Änderung wird erst gemeldet, wenn sie in `threshold` aufeinander-
    folgenden Checks identisch erkannt wird. Einmalige API-Aussetzer (der
    bisherige Bug) werden so sicher unterdrückt.

    Aufruf mit leeren Listen → Counter für diesen Key zurücksetzen.
    """
    global _pending
    if key not in _pending:
        _pending[key] = {"hash": None, "count": 0, "data": None}

    if not added and not removed:
        _pending[key] = {"hash": None, "count": 0, "data": None}
        return False, None

    h = _changes_hash(added, removed)
    p = _pending[key]

    if h == p["hash"]:
        p["count"] += 1
    else:
        # Andere Änderung als vorher → Counter neu starten
        p["hash"]  = h
        p["count"] = 1
        p["data"]  = {"added": added, "removed": removed}

    if p["count"] >= threshold:
        data = p["data"]
        _pending[key] = {"hash": None, "count": 0, "data": None}
        return True, data

    return False, None


def reset_debounce(key: str | None = None) -> None:
    """Setzt den Debounce-State zurück. key=None → alle Keys."""
    global _pending
    if key is None:
        _pending = {
            "today":    {"hash": None, "count": 0, "data": None},
            "tomorrow": {"hash": None, "count": 0, "data": None},
        }
    else:
        _pending[key] = {"hash": None, "count": 0, "data": None}


def get_pending_state(key: str) -> dict:
    """Gibt den aktuellen Debounce-State für einen Key zurück (für Logging)."""
    return _pending.get(key, {"hash": None, "count": 0, "data": None}).copy()


# ═══════════════════════════════════════════════════════════════════════════════
#  State-Persistenz
# ═══════════════════════════════════════════════════════════════════════════════

def save_timetable_state(state: dict, tz=TZ, filepath: str = "timetable_state.json") -> None:
    try:
        out = {
            "date":     datetime.now(tz).date().isoformat(),
            "today":    state.get("today", []),
            "tomorrow": state.get("tomorrow", []),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving timetable state: {e}")


def load_timetable_state(tz=TZ, filepath: str = "timetable_state.json") -> dict | None:
    """Lädt den gespeicherten State. Gibt None zurück wenn veraltet oder nicht vorhanden."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
        if state.get("date") == datetime.now(tz).date().isoformat():
            return {
                "today":    state.get("today", []),
                "tomorrow": state.get("tomorrow", []),
            }
    except Exception as e:
        print(f"Error loading timetable state: {e}")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Wahrheitstabellen-Generator (PIL, kein Discord)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_truth_table_image(expr: str) -> str:
    from PIL import Image, ImageDraw, ImageFont

    # ── Tokenizer ────────────────────────────────────────────────────────────
    def tokenize(s):
        token_spec = [
            (r'\s+', None),
            (r'\(', '('), (r'\)', ')'),
            (r'\<\=\>', '<=>'), (r'==', '=='), (r'=', '='),
            (r'\&', '&'), (r'\|', '|'), (r'\!', '!'),
            (r'\bNOT\b', '!'), (r'\bnot\b', '!'),
            (r'\bAND\b', '&'), (r'\band\b', '&'),
            (r'\bOR\b',  '|'), (r'\bor\b',  '|'),
            (r'[A-Z]', 'VAR'),
        ]
        pos, tokens = 0, []
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
                raise ValueError(f"Ungültiges Symbol bei Position {pos}: '{s[pos]}'")
        return tokens

    # ── Shunting-Yard → RPN ───────────────────────────────────────────────────
    def shunting_yard(tokens):
        prec = {'!': (4,'right'), '&': (3,'left'), '|': (2,'left'),
                '<=>': (1,'left'), '==': (1,'left'), '=': (1,'left')}
        output, stack = [], []
        for ttype, tval in tokens:
            if ttype == 'VAR':
                output.append((ttype, tval))
            elif ttype == '(':
                stack.append((ttype, tval))
            elif ttype == ')':
                while stack and stack[-1][0] != '(':
                    output.append(stack.pop())
                if not stack:
                    raise ValueError("Fehlende öffnende Klammer")
                stack.pop()
            else:
                while stack and stack[-1][0] != '(':
                    top = stack[-1][0]
                    if top not in prec:
                        break
                    p_top, _ = prec[top]; p_op, assoc_op = prec[ttype]
                    if (assoc_op == 'left' and p_op <= p_top) or (assoc_op == 'right' and p_op < p_top):
                        output.append(stack.pop())
                    else:
                        break
                stack.append((ttype, tval))
        while stack:
            if stack[-1][0] == '(':
                raise ValueError("Fehlende schließende Klammer")
            output.append(stack.pop())
        return output

    class Node:
        def __init__(self, kind, value=None, left=None, right=None):
            self.kind = kind; self.value = value
            self.left = left; self.right = right
        def to_infix_label(self):
            if self.kind == 'VAR':  return self.value
            if self.kind == 'NOT':  return f"not {self.left.to_infix_label()}"
            if self.kind == 'AND':  return f"{self.left.to_infix_label()} and {self.right.to_infix_label()}"
            if self.kind == 'OR':   return f"{self.left.to_infix_label()} or {self.right.to_infix_label()}"
            if self.kind == 'EQ':   return f"{self.left.to_infix_label()} = {self.right.to_infix_label()}"
            return "?"
        def to_python(self):
            if self.kind == 'VAR':  return f"{self.value}"
            if self.kind == 'NOT':  return f"(not ({self.left.to_python()}))"
            if self.kind == 'AND':  return f"(({self.left.to_python()}) and ({self.right.to_python()}))"
            if self.kind == 'OR':   return f"(({self.left.to_python()}) or ({self.right.to_python()}))"
            if self.kind == 'EQ':   return f"(({self.left.to_python()}) == ({self.right.to_python()}))"
            return "False"

    def rpn_to_ast(rpn):
        stack = []
        for ttype, tval in rpn:
            if ttype == 'VAR':
                stack.append(Node('VAR', value=tval))
            elif ttype == '!':
                if not stack: raise ValueError("'!' ohne Operand")
                stack.append(Node('NOT', left=stack.pop()))
            else:
                if len(stack) < 2: raise ValueError("Binärer Operator ohne zwei Operanden")
                b, a = stack.pop(), stack.pop()
                kind = {'&': 'AND', '|': 'OR'}.get(ttype, 'EQ')
                stack.append(Node(kind, left=a, right=b))
        if len(stack) != 1:
            raise ValueError("Syntaxfehler im Ausdruck")
        return stack[0]

    def collect_major_steps(root):
        steps, seen = [], set()
        def post(node):
            if node is None or node.kind == 'VAR': return
            if node.left:  post(node.left)
            if node.right: post(node.right)
            nid = id(node)
            if nid not in seen and node.kind in ('NOT','AND','OR','EQ'):
                steps.append(node); seen.add(nid)
        post(root)
        return steps

    tokens   = tokenize(expr)
    if not tokens: raise ValueError("Leerer Ausdruck")
    variables = sorted({v for t, v in tokens if t == 'VAR'})
    ast_root  = rpn_to_ast(shunting_yard(tokens))
    major     = collect_major_steps(ast_root)
    step_labels  = [n.to_infix_label() for n in major] or [ast_root.to_infix_label()]
    step_pyexprs = [n.to_python() for n in major]      or [ast_root.to_python()]
    header = variables + step_labels

    rows = []
    for comb in itertools.product([False, True], repeat=len(variables)):
        env = dict(zip(variables, comb))
        row = [1 if v else 0 for v in comb]
        for se in step_pyexprs:
            try:    row.append(1 if bool(eval(se, {}, env)) else 0)
            except: row.append('?')
        rows.append(row)

    bg, hbg  = (44,47,51), (60,63,68)
    tc, lc   = (220,221,222), (32,34,37)
    acc, fin = (114,137,218), (67,181,129)
    step_c   = (250,166,26)
    font_size = 22
    try:    font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except: font = ImageFont.load_default()

    pad = 16
    col_widths = [max(max(len(str(h)), *(len(str(r[i])) for r in rows))
                      * (font_size // 2) + pad, 70)
                  for i, h in enumerate(header)]

    w = sum(col_widths) + 40
    h = (len(rows) + 2) * (font_size + 18) + 40
    img  = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)

    x, y, hh = 20, 20, font_size + 12
    for i, col in enumerate(header):
        cw = col_widths[i]
        draw.rectangle([x, y, x+cw, y+hh], fill=hbg)
        color = acc if i < len(variables) else (fin if i == len(header)-1 else step_c)
        draw.text((x+8, y+4), str(col), fill=color, font=font)
        x += cw
    y += hh + 10
    for row in rows:
        x = 20
        for i, cell in enumerate(row):
            color = fin if i == len(header)-1 else tc
            draw.text((x+8, y), str(cell), fill=color, font=font)
            x += col_widths[i]
        y += font_size + 16

    x = 20
    for cw in col_widths:
        draw.line((x,20,x,h-20), fill=lc, width=2); x += cw
    draw.line((x,20,x,h-20), fill=lc, width=2)
    draw.line((20,20,w-20,20), fill=lc, width=2)
    draw.line((20,20+hh,w-20,20+hh), fill=lc, width=3)

    path = "truth_table.png"
    img.save(path)
    return path