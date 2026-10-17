"""Synthetic agentic traces, short-chat data, and per-token role labels.

Every token of a rendered chat gets a role label so calibration can mask by role:
  template     <|im_start|>role\\n headers and <|im_end|>
  system       system prompt incl. the JSON tool schemas (the long, repeated part)
  user         user turns
  tool_result  <tool_response> blocks (rendered as user turns by Qwen's template)
  assistant    assistant turns (tool calls and final answers)
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field

ROLES = ("template", "system", "user", "tool_result", "assistant")
ROLE_ID = {r: i for i, r in enumerate(ROLES)}

CITIES = ["Toronto", "Ottawa", "Montreal", "Vancouver", "Calgary", "Halifax", "Paris", "Berlin",
          "Tokyo", "Lagos", "Lima", "Oslo", "Madrid", "Seoul", "Nairobi", "Dublin", "Austin", "Quebec City"]
NAMES = ["Priya", "Marc", "Aisha", "Tomás", "Wei", "Olga", "Kofi", "Sara", "Liam", "Noor", "Hiro", "Emma"]
PRODUCTS = ["laptop stand", "USB-C hub", "desk lamp", "noise-cancelling headphones", "standing desk",
            "mechanical keyboard", "webcam", "monitor arm", "ergonomic chair", "power bank"]
CURRENCIES = ["CAD", "USD", "EUR", "JPY", "GBP", "MXN"]
LANGS = ["French", "Spanish", "German", "Japanese", "Portuguese", "Korean"]


def _p(typ, desc, enum=None):
    d = {"type": typ, "description": desc}
    if enum:
        d["enum"] = enum
    return d


def _date(r):
    return f"2026-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}"


# name -> (description, properties, required, request templates, arg sampler)
def _tool_specs():
    return {
        "get_weather": ("Get the current weather for a city.",
                        {"city": _p("string", "City name"), "unit": _p("string", "Temperature unit", ["celsius", "fahrenheit"])},
                        ["city", "unit"],
                        ["What's the weather in {city} right now, in {unit}?", "Tell me the current temperature in {city} ({unit}).",
                         "Is it cold in {city}? Give it to me in {unit}."],
                        lambda r: {"city": r.choice(CITIES), "unit": r.choice(["celsius", "fahrenheit"])}),
        "convert_currency": ("Convert an amount between currencies.",
                             {"amount": _p("number", "Amount to convert"), "from_currency": _p("string", "ISO code"),
                              "to_currency": _p("string", "ISO code")},
                             ["amount", "from_currency", "to_currency"],
                             ["How much is {amount} {from_currency} in {to_currency}?", "Convert {amount} {from_currency} to {to_currency} please.",
                              "I have {amount} {from_currency}; what is that in {to_currency}?"],
                             lambda r: dict(zip(["amount", "from_currency", "to_currency"],
                                                [r.choice([25, 40, 99, 120, 250, 500, 1200]), *r.sample(CURRENCIES, 2)]))),
        "create_calendar_event": ("Create a calendar event.",
                                  {"title": _p("string", "Event title"), "date": _p("string", "Date, YYYY-MM-DD"),
                                   "duration_minutes": _p("integer", "Length in minutes")},
                                  ["title", "date", "duration_minutes"],
                                  ["Book a {duration_minutes}-minute '{title}' on {date}.", "Put '{title}' in my calendar for {date}, {duration_minutes} minutes.",
                                   "Schedule {title} on {date} lasting {duration_minutes} minutes."],
                                  lambda r: {"title": r.choice(["design review", "1:1 with " + r.choice(NAMES), "dentist", "team standup", "budget sync"]),
                                             "date": _date(r), "duration_minutes": r.choice([15, 30, 45, 60, 90])}),
        "send_email": ("Send an email to a contact.",
                       {"to": _p("string", "Recipient name"), "subject": _p("string", "Subject line")},
                       ["to", "subject"],
                       ["Email {to} with the subject '{subject}'.", "Send {to} a note titled '{subject}'.",
                        "Can you write to {to}? Subject: {subject}."],
                       lambda r: {"to": r.choice(NAMES), "subject": r.choice(["Q3 numbers", "lunch tomorrow", "contract draft",
                                                                             "travel plans", "invoice 4471"])}),
        "search_products": ("Search the product catalog.",
                            {"query": _p("string", "Search text"), "max_price": _p("number", "Maximum price in CAD")},
                            ["query", "max_price"],
                            ["Find me a {query} under {max_price} dollars.", "Search for a {query} that costs at most {max_price}.",
                             "I need a {query}, budget {max_price}."],
                            lambda r: {"query": r.choice(PRODUCTS), "max_price": r.choice([30, 50, 80, 150, 300])}),
        "get_order_status": ("Look up the status of an order.",
                             {"order_id": _p("string", "Order identifier")}, ["order_id"],
                             ["Where is order {order_id}?", "What's the status of my order {order_id}?", "Track order {order_id} for me."],
                             lambda r: {"order_id": f"ORD-{r.randint(10000, 99999)}"}),
        "translate_text": ("Translate text into a target language.",
                           {"text": _p("string", "Text to translate"), "target_language": _p("string", "Language name")},
                           ["text", "target_language"],
                           ["Translate '{text}' into {target_language}.", "How do you say '{text}' in {target_language}?",
                            "Put '{text}' in {target_language}."],
                           lambda r: {"text": r.choice(["good morning", "where is the station", "thank you very much", "the meeting is cancelled"]),
                                      "target_language": r.choice(LANGS)}),
        "book_flight": ("Search flights between two cities on a date.",
                        {"origin": _p("string", "Departure city"), "destination": _p("string", "Arrival city"),
                         "date": _p("string", "Date, YYYY-MM-DD")},
                        ["origin", "destination", "date"],
                        ["Find flights from {origin} to {destination} on {date}.", "I want to fly {origin} to {destination} on {date}.",
                         "Any flights {origin} -> {destination} for {date}?"],
                        lambda r: dict(zip(["origin", "destination"], r.sample(CITIES, 2)), date=_date(r))),
        "set_reminder": ("Set a reminder.",
                         {"message": _p("string", "Reminder text"), "minutes_from_now": _p("integer", "Delay in minutes")},
                         ["message", "minutes_from_now"],
                         ["Remind me to {message} in {minutes_from_now} minutes.", "In {minutes_from_now} minutes, remind me: {message}.",
                          "Set a reminder to {message} {minutes_from_now} minutes from now."],
                         lambda r: {"message": r.choice(["call mom", "take the pizza out", "stretch", "join the call", "water the plants"]),
                                    "minutes_from_now": r.choice([5, 10, 20, 30, 45, 90])}),
        "query_database": ("Run a read-only SQL query on the analytics database.",
                           {"table": _p("string", "Table name", ["orders", "users", "events", "invoices"]),
                            "limit": _p("integer", "Maximum rows")},
                           ["table", "limit"],
                           ["Show me the first {limit} rows of the {table} table.", "Pull {limit} records from {table}.",
                            "Query {table}, limit {limit}."],
                           lambda r: {"table": r.choice(["orders", "users", "events", "invoices"]), "limit": r.choice([5, 10, 20, 50, 100])}),
        "get_stock_price": ("Get the latest stock price for a ticker.",
                            {"ticker": _p("string", "Ticker symbol")}, ["ticker"],
                            ["What's {ticker} trading at?", "Get me the price of {ticker}.", "How is {ticker} doing today?"],
                            lambda r: {"ticker": r.choice(["SHOP", "RY", "TD", "ENB", "CNR", "BCE", "AAPL", "NVDA"])}),
        "lookup_contact": ("Find a contact's phone number and email.",
                           {"name": _p("string", "Contact name")}, ["name"],
                           ["What's {name}'s phone number?", "Look up {name} in my contacts.", "How do I reach {name}?"],
                           lambda r: {"name": r.choice(NAMES)}),
    }


TOOLS = _tool_specs()


def tool_schema(name):
    desc, props, req, *_ = TOOLS[name]
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": req}}}


def _fake_result(name, args, r):
    if name == "get_weather":
        return {"city": args["city"], "temperature": r.randint(-15, 33), "unit": args["unit"], "conditions": r.choice(["sunny", "light rain", "overcast", "snow"])}
    if name == "convert_currency":
        rate = round(r.uniform(0.5, 1.6), 4)
        return {"rate": rate, "converted_amount": round(args["amount"] * rate, 2), "currency": args["to_currency"]}
    if name == "search_products":
        return {"results": [{"name": f"{args['query']} {m}", "price": round(r.uniform(10, args['max_price']), 2)} for m in r.sample(["Pro", "Lite", "Max", "Air"], 2)]}
    if name == "get_order_status":
        return {"order_id": args["order_id"], "status": r.choice(["shipped", "processing", "delivered"]), "eta_days": r.randint(1, 9)}
    if name == "get_stock_price":
        return {"ticker": args["ticker"], "price": round(r.uniform(20, 400), 2), "change_pct": round(r.uniform(-4, 4), 2)}
    if name == "book_flight":
        return {"flights": [{"flight": f"AC{r.randint(100, 999)}", "depart": f"{r.randint(6, 21):02d}:{r.choice(['00', '15', '30', '45'])}", "price": r.randint(180, 1400)}]}
    return {"ok": True, "id": f"{name[:3].upper()}-{r.randint(1000, 9999)}", **{k: v for k, v in args.items() if k in ("date", "to", "name")}}


def _final_answer(name, res):
    if name == "get_weather":
        return f"It's currently {res['temperature']} degrees {res['unit']} and {res['conditions']} in {res['city']}."
    if name == "convert_currency":
        return f"That comes to {res['converted_amount']} {res['currency']} at a rate of {res['rate']}."
    if name == "search_products":
        return "I found: " + "; ".join(f"{x['name']} at ${x['price']}" for x in res["results"]) + "."
    if name == "get_order_status":
        return f"Order {res['order_id']} is {res['status']}, with an estimated {res['eta_days']} days to go."
    if name == "get_stock_price":
        return f"{res['ticker']} is at ${res['price']}, {res['change_pct']}% on the day."
    if name == "book_flight":
        f = res["flights"][0]
        return f"The best option is {f['flight']} departing at {f['depart']} for ${f['price']}."
    return f"Done — reference {res['id']}."


@dataclass
class Episode:
    tools: list
    turns: list = field(default_factory=list)   # (request, tool, args, result, answer)


def sample_episode(r: random.Random, n_tools=(6, 12), n_rounds=(1, 3)) -> Episode:
    offered = r.sample(list(TOOLS), r.randint(*n_tools))
    ep = Episode(tools=offered)
    for _ in range(r.randint(*n_rounds)):
        name = r.choice(offered)
        _, _, _, templates, sampler = TOOLS[name]
        args = sampler(r)
        req = r.choice(templates).format(**args)
        res = _fake_result(name, args, r)
        ep.turns.append((req, name, args, res, _final_answer(name, res)))
    return ep


def episode_messages(ep: Episode):
    msgs = []
    for req, name, args, res, ans in ep.turns:
        msgs.append({"role": "user", "content": req})
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}]})
        msgs.append({"role": "tool", "content": json.dumps(res)})
        msgs.append({"role": "assistant", "content": ans})
    return msgs


_BLOCK = re.compile(r"<\|im_start\|>(\w+)\n(.*?)(<\|im_end\|>)", re.S)


def char_roles(text: str) -> list[int]:
    """Role id for every character of a rendered Qwen chat."""
    roles = [ROLE_ID["template"]] * len(text)
    for m in _BLOCK.finditer(text):
        role, body = m.group(1), m.group(2)
        if role == "user" and body.lstrip().startswith("<tool_response>"):
            role = "tool_result"
        rid = ROLE_ID.get(role, ROLE_ID["user"])
        for c in range(m.start(2), m.end(2)):
            roles[c] = rid
    return roles


def tokenize_with_roles(tok, text: str, max_len: int | None = None):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    cr = char_roles(text)
    ids, roles = [], []
    for t, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]):
        ids.append(t)
        roles.append(cr[a] if b > a else ROLE_ID["template"])
    if max_len:
        ids, roles = ids[:max_len], roles[:max_len]
    return ids, roles


def render(tok, msgs, tools=None, add_generation_prompt=False):
    kw = {"tools": [tool_schema(t) for t in tools]} if tools else {}
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt, **kw)
