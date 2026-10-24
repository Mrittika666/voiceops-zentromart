import os, re, json, time
import pandas as pd
from groq import Groq
from tools.tools import lookup_order, create_ticket

MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
FALLBACK = "qwen/qwen3.8-27b"
_client = None

_W = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
      "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
      "shunya": "0", "ek": "1", "do": "2", "teen": "3", "char": "4", "chaar": "4",
      "paanch": "5", "panch": "5", "chhe": "6", "chhah": "6", "saat": "7",
      "aath": "8", "nau": "9"}


def words_to_digits(text):
    """Spoken digits (one two three...) ko numbers mein badlo, jab 3+ ek saath hon."""
    out, run = [], []

    def flush():
        nonlocal run
        if len(run) >= 3:
            out.append("".join(_W[w.lower().strip(",.")] for w in run))
        else:
            out.extend(run)
        run = []

    for t in text.split():
        if t.lower().strip(",.") in _W:
            run.append(t)
        else:
            flush()
            out.append(t)
    flush()
    return " ".join(out)


_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
         "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}


def spoken_numbers(text):
    """'four thousand five hundred thirty three' -> '4533' (jab hundred/thousand aaye)."""
    toks = text.split()
    out, i = [], 0
    while i < len(toks):
        w = toks[i].lower().strip(",.")
        if w in _ONES or w in _TENS:
            j, total, cur, mag = i, 0, 0, False
            while j < len(toks):
                x = toks[j].lower().strip(",.")
                if x == "and" and mag:
                    j += 1
                    continue
                if x in _ONES:
                    cur += _ONES[x]
                elif x in _TENS:
                    cur += _TENS[x]
                elif x == "hundred":
                    cur = max(cur, 1) * 100
                    mag = True
                elif x == "thousand":
                    total += max(cur, 1) * 1000
                    cur = 0
                    mag = True
                else:
                    break
                j += 1
            if mag:
                out.append(str(total + cur))
                i = j
                continue
        out.append(toks[i])
        i += 1
    return " ".join(out)


def build_system(data_dir):
    hol = pd.read_csv(os.path.join(data_dir, "data", "holidays.csv"), dtype=str)
    holidays = "; ".join(f"{r.date} ({r.name})" for r in hol.itertuples())
    return f"""You are ZentroMart's customer support assistant (Indian online store: groceries and electronics).
Today's date is 20 October 2026. Business days are Mon-Sat excluding these holidays: {holidays}. Return windows use calendar days.

RULES
- In your FIRST reply of the conversation, say you are an AI assistant.
- Answer ONLY from the POLICY CONTEXT and the VERIFIED ORDER DATA given below. Never invent a policy, date, amount or order detail. If not covered, say you don't have that information and offer a human agent.
- Remember what the customer already said (order, issue). Understand follow-ups without asking again.
- If a needed detail is missing, ask ONE clear question.
- If ORDER STATE says not verified, share NO order details; ask for the registered phone's last 4 digits OR the delivery pincode.
- Refuse politely anything the policy does not allow (for example extra compensation). If the policy says a human must decide, or the customer asks for something only a human can approve, set escalate=true.
- When escalate=true, give only a short polite explanation (for example why you cannot approve it). Do NOT mention a ticket or a handover; the system adds that sentence itself. NEVER claim a ticket was created.
- Keep replies short for the ear (1-3 sentences), plain text only: no markdown, no emojis, no bullet lists. Include the year in dates. Reply in the customer's language style (English, or Hinglish in Latin script if they speak Hinglish).
- For return, refund or delivery questions, give the exact number of days/hours for THIS order's payment method and category, and compute the exact date (return deadline = delivery date + window in calendar days; refund dates in business days, Mon-Sat excluding holidays). Always state dates with the year, and mention any fee (e.g. reverse pickup fee) and whether it applies to this customer.
- If the customer says they contacted before (repeated calls), is upset, or asks for a human, or the claim needs a human decision (check the escalation rules in the context, including value limits), set escalate=true. Never approve or promise something the policy reserves for a human.
- Do not say you do not know when the policy context contains the answer; read it carefully first.
- Before saying any claim, return or replacement is approved, check the claim rules and value limits in the policy context against the item price in the order data. If the limit is exceeded or a human must decide, do NOT approve: set escalate=true.
- If ORDER STATE says the order is verified, never ask for verification again.

Return ONLY a JSON object: {{"reply": "...", "escalate": true or false, "ticket_summary": "issue, checks made, why a human is needed (empty if not escalating)"}}"""


def chat_llm(messages, tries=6):
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    raw = None
    for t in range(tries):
        try:
            raw = _client.chat.completions.create(model=(MODEL if t == 0 else FALLBACK), messages=messages).choices[0].message.content
            break
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                print("LLM retry:", str(e)[:150])
                time.sleep(2 if t == 0 else 15 * t)
            else:
                print("LLM error:", str(e)[:300])
                break
    if raw:
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    if raw is None:
        return json.dumps({"reply": "Sorry, I am having trouble right now. Please try again in a moment.",
                           "escalate": False, "ticket_summary": ""})
    j = re.search(r"\{.*\}", raw, re.S)
    if j:
        try:
            json.loads(j.group(0))
            return raw
        except Exception:
            pass
    esc = bool(re.search(r'"escalate"\s*:\s*true', raw))
    m = re.search(r'"reply"\s*:\s*"(.*?)"\s*,\s*"escalate"', raw, re.S)
    reply = m.group(1) if m else re.split(r',?\s*"escalate"', raw)[0]
    reply = reply.strip().lstrip("{").strip()
    ts = re.search(r'"ticket_summary"\s*:\s*"(.*?)"\s*\}', raw, re.S)
    return json.dumps({"reply": reply, "escalate": esc, "ticket_summary": ts.group(1) if ts else ""})


class Agent:
    def __init__(self, kb, system):
        self.kb, self.system = kb, system
        self.history, self.user_msgs = [], []
        self.order_id, self.order = None, None
        self.verified, self.fails = False, 0
        self.ticket_id, self.tool_log, self.last_chunks = None, [], []

    def respond(self, msg):
        msg = words_to_digits(spoken_numbers(msg))
        self.tool_log = []
        self.user_msgs.append(msg)

        # 1) order ID (8 digits, beech mein space ho sakta hai)
        rest = msg
        m = re.search(r"(?<!\d)(\d{4})[\s-]?(\d{4})(?!\d)", msg)
        if m:
            new_id = m.group(1) + m.group(2)
            if new_id != self.order_id:
                self.order_id, self.order, self.verified, self.fails = new_id, None, False, 0
            rest = msg[:m.start()] + " " + msg[m.end():]

        # 2) verification: phone last 4 ya 6-digit pincode
        if self.order_id and not self.verified and self.fails < 2:
            p = re.search(r"(?<!\d)(\d{6}|\d{4})(?!\d)", rest)
            if p and p.group(1) not in ("2025", "2026", "2027"):
                res = lookup_order(self.order_id, p.group(1))
                self.tool_log.append(f"lookup_order({self.order_id})")
                if res.get("verified"):
                    self.verified, self.order = True, res
                else:
                    self.fails += 1

        # 3) order state
        if not self.order_id:
            st = "No order ID given yet. Ask ONLY for the order ID (one short question); do not ask for verification yet."
        elif self.verified:
            o = {k: v for k, v in self.order.items() if v is not None}
            st = "Order verified. VERIFIED ORDER DATA: " + json.dumps(o, ensure_ascii=False, default=str)[:3500]
        elif self.fails >= 2:
            st = f"Order {self.order_id}: verification FAILED twice. Share nothing; set escalate=true for human verification."
        elif self.fails == 1:
            st = f"Order {self.order_id}: verification failed once. NOT verified. Ask once more for last 4 phone digits OR delivery pincode. Share no order details."
        else:
            st = f"Order {self.order_id} given but NOT verified yet. Ask for registered phone last 4 digits OR delivery pincode. Share no order details."

        # 4) RAG
        ctx = self.kb.retrieve(" ".join(self.user_msgs[-3:]), k=5)
        self.last_chunks = [c["section_id"] for c in ctx]
        context = "\n\n".join(f'[{c["section_id"]}] {c["text"]}' for c in ctx)
        turn = "FIRST reply: introduce yourself as an AI assistant." if not self.history else "follow-up: do NOT introduce yourself again."
        block = f"POLICY CONTEXT:\n{context}\n\nORDER STATE: {st}\n\nTURN: {turn}\n\nCustomer: {msg}"
        messages = [{"role": "system", "content": self.system}] + self.history + [{"role": "user", "content": block}]
        raw = chat_llm(messages)

        reply, esc, summ = raw, False, ""
        j = re.search(r"\{.*\}", raw, re.S)
        if j:
            try:
                d = json.loads(j.group(0))
                reply = d.get("reply", raw)
                esc = bool(d.get("escalate"))
                summ = d.get("ticket_summary") or ""
            except Exception:
                pass

        # 5) handover: ticket code banata hai, claim tabhi jab created=True
        if esc and not self.ticket_id:
            if len(summ.strip()) < 20:
                summ = f"Customer needs human help. Last message: {msg}. Order: {self.order_id or 'n/a'}. Verified: {self.verified}."
            t = create_ticket(summ, order_id=self.order_id, reason="human decision required", intent="escalation")
            self.tool_log.append("create_ticket()")
            if t.get("created"):
                self.ticket_id = t["ticket_id"]
                reply += f" I have created a ticket ending {self.ticket_id[-6:]} and handed this over to a human colleague."
            else:
                reply += " I could not create a handover ticket right now. Please try again shortly."

        self.history += [{"role": "user", "content": msg}, {"role": "assistant", "content": reply}]
        return reply
