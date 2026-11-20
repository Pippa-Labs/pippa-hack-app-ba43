"""DIVERT — what getting across London actually costs, right now.

A Pippa agent app. One call in, one result out.

Given two station names it returns every way of making the journey, each with
its real fare in pence, flags the ones whose line is disrupted, and warns when
the route involves leaving one station and entering another in a way Transport
for London charges as two journeys instead of one.

Four things worth knowing about the design:

**The second-fare warning is the reason this exists.** Transport for London
designates certain station pairs as a single journey; every other pair is
charged twice. Each designated pair has its own time limit, differing by
direction. Their list is no longer published — the PDF 404s and nothing is
archived — so it is rebuilt and bundled in `tapdata.json` beside this file.
No journey planner shows this, including theirs.

**Transport for London does not price everything it offers.** A journey
request without `mode=tube` comes back with `fare: null` on every result, and
even with it the extra multi-modal results are unpriced. Rather than drop
those options, rail journeys are priced from the two stations' zones using a
table read from the live API. Walking and cycling are not priced at all,
because Transport for London does not charge for them.

**Options are ranked by what they cost the person, not by fare alone.**
Sorting on price alone recommends a 29-minute walk over a 2-minute train to
save £3.00, which is not advice anybody wants. Minutes are converted to money
at the National Living Wage and added. The rate is a named constant and the
app reports which option is genuinely cheapest as well as which is best, so
the trade-off stays visible rather than buried in a sort key.

**No language model is called from here.** Pippa routes the question, extracts
the two inputs and writes the reply. This file does three HTTP requests —
five when cycling is on the table, none at all for `interchange.check` — and
all of the arithmetic, then returns eight short fields. The expensive thing in
an assistant is a conversation that takes four turns to reach an answer, so the
design goal is to need exactly one.
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import time as _time
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from zoneinfo import ZoneInfo

from pippa_agentapp_sdk import (
    AgentAppError,
    Call,
    Manifest,
    Result,
    load_manifest_file,
    validate_manifest,
)

MANIFEST_PATH = Path(__file__).with_name("manifest.yaml")

#: Bundled lookups. MUST sit beside app.py — `docs/` and `scripts/` are not
#: deployed, so a data file in either passes every local check and is missing
#: in production with no error to find.
DATA_PATH = Path(__file__).with_name("tapdata.json")

#: Transport for London's API returns 403 with body `error code: 1010` to
#: Python's default urllib User-Agent. It is a Cloudflare bot-signature rule,
#: not an IP block, and any identifying User-Agent is accepted. Verified
#: 16 Nov 2026: `Python-urllib/3.12` -> 403, this string -> 200, same IP, same
#: second. Pippa's own docs/IDEAS.md records this API as permanently blocked
#: ("nothing you write fixes that"); it is not, and this line is the fix.
USER_AGENT = "pippa-agent-app-divert/0.1 (https://github.com/Pippa-Labs)"

API = "https://api.tfl.gov.uk"

#: Optional. Without a key Transport for London allow roughly 50 requests a
#: minute per IP address — ample for one person, and nowhere near enough for a
#: room on one venue wifi, because the limit is per source address rather than
#: per app. A free key from api-portal.tfl.gov.uk raises it to about 500.
#:
#: The value never appears in this repository. It is declared in manifest.yaml
#: under `app_secrets`, read from the environment here, and set in Pippa under
#: Apps > Created by me > your app > Keys — which means it can be added, or
#: changed, without a push. Locally it comes from `.env`, which git ignores.
#:
#: Declared `required: false`: the app works unkeyed and only wants one if the
#: network it is calling from is being throttled.
def app_key() -> str:
    """This app's Transport for London key, read at the moment it is used.

    Not at import. Pippa puts an app's keys into the environment for the
    duration of a call and takes them out again, so a module-level read runs
    once, before any call has started, and binds the empty string for the
    life of the process. The key would then be set correctly in Pippa, look
    right in every screenshot, and never once be sent — the app quietly
    running unkeyed at fifty requests a minute instead of five hundred, which
    is the failure it was obtained to prevent.

    Their own checker names this: "Pippa puts the key in os.environ for each
    call only, so read it inside handle() (or a method it calls), never at
    the top of a file."
    """
    return os.environ.get("TFL_APP_KEY", "")

#: Per request. urlopen has no default timeout: without one a single
#: unresponsive host holds the app until it is killed with nothing to show.
TIMEOUT = 4.0

#: The runtime kills the app at 20 seconds, and up to five requests at four
#: seconds each would spend 20 on its own — so the app would be killed rather
#: than return the answer it already had. A deadline keeps the whole call
#: inside the budget: every request is given at most the time left, and the
#: optional ones at the end are skipped rather than started too late. Measured
#: against the real API this never bites — a call is about half a second —
#: but a venue network is exactly where it would.
DEADLINE_SECONDS = 13.0

#: Below this much remaining, an optional extra — the wider journey search,
#: the hire docks — is not worth starting.
OPTIONAL_MIN_SECONDS = 3.0

#: What one request may take when it is the only thing standing between the
#: person and an answer. The ordinary ceiling is TIMEOUT, which is right for a
#: request that widens a list we already have and wrong for the one that is
#: the list: Heathrow Terminal 5 to Bond Street has no tube route, so
#: `mode=tube` returns 404 and the multimodal query is the whole answer — and
#: it takes about seven seconds, so it timed out at four every single time
#: while eleven seconds of the call's budget went unused. Bounded by what is
#: actually left, so the deadline still governs.
LAST_CHANCE_SECONDS = 9.0

#: At most this many options. The output is read aloud by an assistant, so a
#: handful of numbers a person can follow beats completeness.
MAX_OPTIONS = 5

#: Value of a minute, in pence, for ranking time against money. The National
#: Living Wage is £12.71 an hour from 1 April 2026 (gov.uk), which is 21.18p a
#: minute. Rounded to 21. This is a judgement, not a fact, which is why it is
#: named, is reported in the narrative, and does not hide the cheapest option.
PENCE_PER_MINUTE = 21

#: Modes Transport for London charges a fare for.
FARE_MODES = frozenset({"tube", "dlr", "elizabeth-line", "overground",
                        "national-rail", "tram", "bus", "river-bus", "cable-car"})
#: Modes it does not.
FREE_MODES = frozenset({"walking", "cycle"})
#: Modes whose `lineIdentifier` is a railway line rather than a bus route
#: number. Without this filter a journey on the 148 renders as "148 then 35".
RAIL_MODES = frozenset({"tube", "dlr", "elizabeth-line", "overground", "national-rail"})

#: How far counts as "I'd walk that if the line were down". Most designated
#: interchanges in Transport for London's own list are under 800 m and almost
#: all are under a kilometre, so this is the distance at which walking out
#: stops being hypothetical.
WALKABLE_METRES = 750
#: A brisk but ordinary walking pace, for turning metres into minutes.
WALK_METRES_PER_MINUTE = 80
#: Two station records closer together than this are the same building under
#: two names — "Euston" and "London Euston" are seventeen metres apart.
COINCIDENT_METRES = 80

#: Suffixes Transport for London attach to a station's name. They describe the
#: mode, not a different place: "Ampere Way" and "Ampere Way Tram Stop" are
#: the same stop under two naptans, nought metres apart.
NAME_SUFFIX = re.compile(
    r"\s+(Tram Stop|Rail Station|Underground Station|DLR Station)$", re.I)

#: Croydon tram naptan prefixes. The second-fare rule is about rail gatelines:
#: you tap out of one and into another and are charged twice. Trams are
#: tap-in-only at a flat fare, and the Hopper gives an hour of unlimited tram
#: and bus travel, so a walk between two tram stops is very often free — the
#: opposite of what this app would otherwise say, since no tram stop appears
#: in the interchange list and every pair would read as a second fare.
TRAM_PREFIXES = ("940GZZCR", "9400ZZCR")

#: How far to look for a hire dock. Most stations in zone 1 have one within
#: a couple of minutes; beyond this it stops being the same journey.
BIKE_RADIUS = 450

#: Transport for London peak: 06:30–09:30 and 16:00–19:00, Monday to Friday.
#: Only used for the zone-table fallback — where the API prices a journey it
#: also returns `chargeLevel`, which is authoritative and preferred.
PEAK_WINDOWS = ((_time(6, 30), _time(9, 30)), (_time(16, 0), _time(19, 0)))

#: What Londoners type that is not what Transport for London call it.
#: Consulted only after the ordinary lookup has already failed, so nothing
#: that resolves today can start resolving differently. "bakerloo" is
#: deliberately absent: it is a line, not a station, and answering it with a
#: station would be a confident guess at something the person did not ask.
SPOKEN_ALIASES = {
    "kings x": "kings cross",
    "kings cross x": "kings cross",
    "the angel": "angel",
    "lpool st": "liverpool street",
    "lpool street": "liverpool street",
    # Terminals 2 and 3 share one station and Terminal 1 closed in 2015,
    # so all three go to the same place. An alias pointing at a station that
    # does not exist is worse than none: it refuses by a name the person
    # never typed.
    "heathrow t1": "heathrow terminals 2 and 3",
    "heathrow t2": "heathrow terminals 2 and 3",
    "heathrow t3": "heathrow terminals 2 and 3",
    "heathrow terminal 1": "heathrow terminals 2 and 3",
    "heathrow terminal 2": "heathrow terminals 2 and 3",
    "heathrow terminal 3": "heathrow terminals 2 and 3",
    "heathrow t4": "heathrow terminal 4",
    "heathrow t5": "heathrow terminal 5",
    "t5": "heathrow terminal 5",
    "tcr": "tottenham court road",
    "kx": "kings cross",
}

#: Waiting for off-peak is only advice if the wait is short. Asked at 16:05,
#: off-peak is nearly three hours away and "wait" is not a useful answer; asked
#: at 09:10 it is twenty minutes and worth real money.
PEAK_TIP_MAX_WAIT = 45

#: ...and only if there is something in it. Eight of the twenty-five zone pairs
#: differ by 10p or 20p between peak and off-peak, and "wait twenty-two minutes
#: to save ten pence" is worse than silence: this app already values a minute
#: at PENCE_PER_MINUTE when it ranks, so suggesting a wait worth £4.62 of
#: somebody's time to save £0.10 contradicts its own arithmetic out loud.
PEAK_TIP_MIN_SAVING = 100

#: Those windows are London local time, and `call.context.now` need not be.
#: Read in UTC through the summer the whole thing sits an hour out: 05:45 UTC
#: is 06:45 in London and squarely peak, and an app reading the clock as it
#: arrives would quote the off-peak fare. In November the two coincide, which
#: is exactly why it would have gone unnoticed.
LONDON = ZoneInfo("Europe/London")


# ---------------------------------------------------------------------------
# Errors. `user_message` is a CLASS attribute on AgentAppError — passing it as
# a keyword raises TypeError. Subclass for the default, then set it on the
# instance to name the specific thing that went wrong.
# ---------------------------------------------------------------------------

class NeedTwoStations(AgentAppError):
    user_message = "I need both a starting station and a destination."
    retryable = False


class StationUnknown(AgentAppError):
    user_message = "I don't recognise one of those station names."
    retryable = False


class SameStation(AgentAppError):
    user_message = "Those are the same station."
    retryable = False


class NoStepFreeRoute(AgentAppError):
    user_message = "There is no step-free route between those two stations."
    retryable = False


class NoRouteFound(AgentAppError):
    user_message = "Transport for London isn't returning a route between those two right now."
    retryable = True


class ServiceBusy(AgentAppError):
    user_message = "Transport for London is rate limiting me — try again shortly."
    retryable = True


class ServiceUnavailable(AgentAppError):
    user_message = "I couldn't reach Transport for London just now."
    retryable = True


# ---------------------------------------------------------------------------
# Name handling
# ---------------------------------------------------------------------------

#: Words people abbreviate, longest form mapped to the short one, so that
#: "bond street" and "bond st" land on the same key. Built identically in
#: scripts/local/build_data.py — the two must not drift.
_ABBREV = {
    "street": "st", "saint": "st", "road": "rd", "square": "sq",
    "park": "pk", "international": "", "junction": "jn",
    "gardens": "gdns", "bridge": "br", "court": "ct", "green": "grn",
}

_NOISE = re.compile(r"\b(underground|station|rail|tube|lu|nr|dlr|el)\b")


def norm(name: str) -> str:
    """Canonical key for a station name, as a person would type it."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = s.lower().replace("&", " and ")
    s = s.replace("st.", "st").replace("'", "").replace(".", "")
    s = _NOISE.sub(" ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(w for w in (_ABBREV.get(w, w) for w in s.split()) if w)


def echo(text: str, limit: int = 48) -> str:
    """User text quoted back inside a message, bounded and flattened.

    Nothing limits how long an extracted string can be — the length is
    whatever the model produced — and these messages are read aloud. An
    earlier version quoted a 5,000-character input back in full, giving a
    5,036-character sentence. Newlines and tabs collapse for the same
    reason: a message is one spoken line.
    """
    clean = " ".join(str(text).split())
    return clean if len(clean) <= limit else clean[:limit - 1] + "\u2026"


#: Strings an extractor may send for "no". `bool("no")` is True in Python, so
#: a model that answers a boolean field with a word rather than a boolean
#: would otherwise turn the flag ON when the person said the opposite.
NEGATIVE_WORDS = frozenset(
    ("", "0", "f", "false", "n", "no", "none", "nope", "not", "null", "off")
)


def truthy(value: object) -> bool:
    """Whether an optional input means yes.

    `step_free` is declared a boolean and a well-behaved extractor sends one.
    But models routinely answer a boolean field with the STRING "false" or
    "no", and `bool("no")` is True — so the plain cast turns step-free on for
    somebody who just said they do not need it.

    That is the worst answer this app can give. Oxford Circus to Warren Street
    is one minute by tube, and on the step-free network there is no tube route
    at all: the flag set backwards replaces a one-minute train with a
    twenty-two minute walk, and the person reading it has no way to tell.
    """
    if isinstance(value, str):
        return value.strip().lower() not in NEGATIVE_WORDS
    return bool(value)


def pounds(pence: int) -> str:
    """Pence as a readable amount. Money is integers everywhere else."""
    return f"£{pence // 100}.{pence % 100:02d}"


class App:
    """One journey question in, one costed answer out."""

    def __init__(self) -> None:
        self._manifest = validate_manifest(load_manifest_file(MANIFEST_PATH))
        with DATA_PATH.open(encoding="utf-8") as handle:
            data = json.load(handle)
        self._osi: dict = data["osi_pairs"]
        self._gateline: dict = data["gateline_osi"]
        self._stations: dict = data["stations"]
        # Each record carries the key it is filed under. A few stations are
        # filed under the short form the interchange list uses — "west
        # hampstead nll" — while their full name normalises to something else
        # entirely ("west hampstead north london line"). Looking the pair up
        # by name then misses a designated interchange and reports a free walk
        # as a second fare.
        for key, record in self._stations.items():
            record["key"] = key
        # Ten stations are filed under the short key the interchange list uses
        # while their display name normalises to something else. The app emits
        # display names — in a warning, in an aside — so every name it can say
        # has to be a name it can be asked about. Without this, Divert tells
        # you about "West Hampstead (North London Line)" and then does not
        # recognise it when you ask.
        for record in list(self._stations.values()):
            spoken = norm(record["name"])
            if spoken and spoken not in self._stations:
                self._stations[spoken] = record

        # One station can be reachable under several keys: the one the line
        # data filed it under, the short one the interchange list uses, and
        # its display name. They are genuinely different strings —
        # "hammersmith h and c line", "hammersmith cir h and c",
        # "hammersmith h and c line" — and the interchange list is keyed on
        # its own. Collect every alias per station so a pair can be looked up
        # under any combination of them.
        # Grouped by naptan, not by object identity: an alias is a copy of
        # the record it points at, so `id()` puts the two in separate buckets
        # and the interchange key is never tried. The naptan is what actually
        # says "this is the same station".
        self._aliases: dict[str, list[str]] = {}
        for key, record in self._stations.items():
            naptan = record.get("naptan")
            if naptan:
                self._aliases.setdefault(naptan, []).append(key)
        self._zone_fares: dict = data["zone_fares"]

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    # -- network ----------------------------------------------------------

    def _remaining(self) -> float:
        """Seconds left of this call's budget."""
        started = getattr(self, "_started", None)
        if started is None:
            return TIMEOUT
        return DEADLINE_SECONDS - (time.monotonic() - started)

    def _fetch(self, url: str, timeout: float = TIMEOUT) -> bytes:
        """Every outbound request goes through here, so one stub covers them
        all in tests. The network is blocked during the build's test run."""
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    def _get(self, path: str, ceiling: float = TIMEOUT) -> object:
        """Fetch and decode, turning every failure into something actionable.

        HTTPError must be caught before URLError: it is a subclass, and the
        wrong order reports a 404 as "unreachable, try again" — a wrong
        message, marked retryable, about a request that will never succeed.
        """
        key = app_key()
        if key:
            path += ("&" if "?" in path else "?") + f"app_key={key}"
        # Never ask for longer than the call has left.
        budget = max(0.5, min(ceiling, self._remaining()))
        try:
            raw = self._fetch(f"{API}{path}", timeout=budget)
        except urllib.error.HTTPError as exc:
            if exc.code == 300:
                # Not a failure. Transport for London answers an ambiguous
                # place name with 300 and a DisambiguationResult, which urllib
                # raises as an error. Names are resolved from the bundled list
                # before we get here, so this means a naptan id they reject.
                # Both names were resolved from the bundled list before
                # this call, so a 300 means they rejected a naptan we
                # hold — their records and ours have diverged. Saying
                # "I do not recognise that station" blames the wrong
                # side and sends somebody off to retype a name that was
                # already right.
                error = StationUnknown(f"disambiguation for {path!r}")
                error.user_message = (
                    "Transport for London did not recognise one of those "
                    "stations, though I do. Their records may have changed.")
                raise error from exc
            if exc.code == 404:
                raise NoRouteFound(f"404 for {path!r}") from exc
            if exc.code == 429:
                raise ServiceBusy(f"rate limited on {path!r}") from exc
            raise ServiceUnavailable(f"tfl returned {exc.code} for {path!r}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ServiceUnavailable(f"could not reach tfl: {exc}") from exc
        try:
            return json.loads(raw)
        except ValueError as exc:
            # A 200 carrying something that is not JSON is nearly always a
            # throttle or block page rather than a broken API — worth saying,
            # because the two have different answers.
            head = raw[:200].decode("utf-8", "replace").strip()
            if "error code" in head.lower() or head.startswith("<"):
                error = ServiceBusy(f"tfl returned a non-JSON page: {head[:80]!r}")
                error.user_message = (
                    "Transport for London is throttling requests from here — "
                    "try again in a moment.")
                raise error from exc
            raise ServiceUnavailable(
                f"tfl returned something that is not json: {head[:80]!r}") from exc

    # -- inputs -----------------------------------------------------------

    def _resolve(self, raw: object, role: str) -> dict:
        """A typed-in station name to a bundled station record.

        An optional string input arrives as either None or "" — absent and
        deliberately-blank are different and both reach us.
        """
        text = (raw or "").strip() if isinstance(raw, str) else ""
        if not text:
            raise NeedTwoStations(f"no {role} station in the message")

        key = norm(text)
        if not key:
            # Everything the person typed was punctuation, an emoji or a
            # non-Latin script, and normalising left nothing. An empty key is
            # a prefix of every station name, so without this it would "match"
            # all of them.
            error = StationUnknown(f"nothing usable in {text!r}")
            error.user_message = f"I can't read {echo(text, 40)} as a station name."
            raise error

        hit = self._stations.get(key)
        if hit is not None:
            return hit

        # One forgiving pass: a prefix match, so "liverpool" finds Liverpool
        # Street. Several distinct matches is not a failure to recognise the
        # name — "Hammersmith" is two separate stations 56 m apart, and so is
        # Edgware Road — so say which, rather than claiming not to know it.
        matches = {v["name"]: v for k, v in self._stations.items() if k.startswith(key)}
        if len(matches) == 1:
            return next(iter(matches.values()))
        if len(matches) > 1:
            names = sorted(matches)
            error = StationUnknown(f"{text!r} matches {len(names)}: {names}")
            shown = ", ".join(names[:3])
            error.user_message = (
                f"There is more than one {text} — did you mean {shown}?"
                if len(names) <= 3 else
                f"There are {len(names)} stations starting with {text}, "
                f"including {shown}. Which did you mean?")
            raise error

        # Last resort: what people actually type. Only reached once the
        # ordinary lookup has failed, so it can add matches but never change
        # one. Recursion is safe because every value here resolves directly.
        spoken = SPOKEN_ALIASES.get(key)
        if spoken and spoken != key:
            return self._resolve(spoken, role)

        error = StationUnknown(f"no station matching {text[:120]!r}")
        error.user_message = f"I don't recognise a station called {echo(text)}."
        raise error

    def _is_peak(self, call: Call) -> bool:
        """Use call.context.now, never datetime.now() — it is what makes this
        testable, and it is the user's clock rather than the server's."""
        moment = call.context.now
        if moment.tzinfo is not None:
            moment = moment.astimezone(LONDON)
        if moment.weekday() >= 5:
            return False
        clock = moment.time()
        return any(start <= clock < end for start, end in PEAK_WINDOWS)

    def _peak_tip(self, a: dict, b: dict, call: Call) -> str | None:
        """What waiting for off-peak would save, when the wait is short.

        Both figures come from the same bundled zone table, so the difference
        is a real Transport for London price gap rather than a comparison
        across two sources — the live fare is authoritative for what you pay
        now, but it cannot tell you what the same journey costs later, and
        subtracting one source from another would invent a number.

        Nothing else surfaces this. Zone 1-4 is £4.80 peak and £3.60 off-peak,
        and at 09:10 that is twenty minutes for £1.20.
        """
        moment = call.context.now
        if moment.tzinfo is not None:
            moment = moment.astimezone(LONDON)
        if moment.weekday() >= 5:
            return None
        clock = moment.time()
        window = next((w for w in PEAK_WINDOWS if w[0] <= clock < w[1]), None)
        if window is None:
            return None

        dear = self._zone_fare(a, b, True)
        cheap = self._zone_fare(a, b, False)
        if dear is None or cheap is None:
            return None
        if dear - cheap < PEAK_TIP_MIN_SAVING:
            return None

        ends = moment.replace(hour=window[1].hour, minute=window[1].minute,
                              second=0, microsecond=0)
        wait = int((ends - moment).total_seconds() // 60)
        if wait <= 0 or wait > PEAK_TIP_MAX_WAIT:
            return None

        minutes = "a minute" if wait == 1 else f"{wait} minutes"
        return (f"Off-peak starts at {window[1].strftime('%H:%M')}. "
                f"Waiting {minutes} makes these zones "
                f"{pounds(dear - cheap)} cheaper.")

    def _zone_fare(self, a: dict, b: dict, peak: bool) -> int | None:
        """Fallback fare from the two stations' zones, or None if unknown.

        Returning None rather than a guess matters: an option shown without a
        price is honest, an option shown with an invented one is not.
        """
        try:
            za, zb = int(a.get("zone") or 0), int(b.get("zone") or 0)
        except (TypeError, ValueError):
            return None
        if not za or not zb:
            return None
        row = self._zone_fares.get(f"{min(za, zb)}-{max(za, zb)}")
        return None if row is None else (row["peak"] if peak else row["off_peak"])

    # -- journey shaping --------------------------------------------------

    @staticmethod
    def _modes(journey: dict) -> list[str]:
        return [(leg.get("mode") or {}).get("name", "") for leg in journey.get("legs") or []]

    @staticmethod
    def _rail_lines(journey: dict) -> list[str]:
        """Railway line names a journey uses, in order, without repeats.

        Only legs on a rail mode are considered. Transport for London puts bus
        route numbers in the same `lineIdentifier` field, so without the filter
        a journey on the 148 and the 35 renders as "148 then 35".

        Their `name` is used in preference to the id: it is already written for
        a human — "Elizabeth line", "Hammersmith & City" — whereas the id is
        `elizabeth` and `hammersmith-city`.
        """
        found: list[str] = []
        for leg in journey.get("legs") or []:
            if (leg.get("mode") or {}).get("name") not in RAIL_MODES:
                continue
            for option in leg.get("routeOptions") or []:
                identifier = option.get("lineIdentifier") or {}
                name = identifier.get("name") or identifier.get("id")
                if name and name not in found:
                    found.append(str(name))
        return found

    @staticmethod
    def _bus_routes(journey: dict) -> list[str]:
        found: list[str] = []
        for leg in journey.get("legs") or []:
            if (leg.get("mode") or {}).get("name") != "bus":
                continue
            for option in leg.get("routeOptions") or []:
                identifier = option.get("lineIdentifier") or {}
                name = identifier.get("name") or identifier.get("id")
                if name and name not in found:
                    found.append(str(name))
        return found

    @staticmethod
    def _out_of_station_walk(journey: dict) -> tuple[str, str] | None:
        """A walking leg between two *different* stations — the thing
        Transport for London may charge as two journeys.

        A walk from the last station to the destination doorstep is not one of
        these, so both ends must be station stop points.
        """
        for leg in journey.get("legs") or []:
            if (leg.get("mode") or {}).get("name") != "walking":
                continue
            start, end = leg.get("departurePoint") or {}, leg.get("arrivalPoint") or {}
            a, b = start.get("naptanId"), end.get("naptanId")
            if not a or not b or a == b:
                continue
            if a.startswith("940") and b.startswith("940"):
                return start.get("commonName", ""), end.get("commonName", "")
        return None

    @staticmethod
    def _base_name(name: str) -> str:
        """A station name without its qualifier.

        Transport for London name separate entrances to one complex as
        "Paddington (H&C Line)-Underground" or "Hammersmith (Dist&Picc Line)".
        The interchange list names the complex. Without this, an entrance 254 m
        from its own main entrance is reported as a station you would be
        charged twice to walk to, which is a false alarm in the one feature
        that has to be right.

        Only bracketed qualifiers and trailing dashes are stripped. Nothing
        here touches "Euston Square", which really is a different station from
        Euston and really is a designated interchange with it.
        """
        # Split on the bracket only. An earlier version also split on a plain
        # hyphen, which turned Bromley-by-Bow into Bromley and
        # Harrow-on-the-Hill into Harrow — neither of which is the same place,
        # and both of which would have merged wrongly the moment a station
        # called Bromley or Harrow existed in the list. A trailing
        # "-Underground" always follows the bracket, so taking the head is
        # enough: "Paddington (H&C Line)-Underground" -> "Paddington".
        head = NAME_SUFFIX.sub("", name.split("(", 1)[0]).strip()
        return head or name

    def _gateline_for(self, station: dict) -> dict | None:
        """The gateline record for a station, under any name it is filed
        beneath.

        Looking this up by the station's display name alone stranded St
        Pancras. The record is keyed `st pancras`; the station it belongs to
        is "King's Cross & St Pancras International", which normalises to
        `kings cross and st pancras` and matched nothing — so one of only
        twenty gateline facts, at one of the busiest stations in London,
        could never be told. The aliases already held the right key.
        """
        for key in (norm(station["name"]), station.get("key", ""),
                    *(self._aliases.get(station.get("naptan")) or [])):
            record = self._gateline.get(key) if key else None
            if record:
                return record
        return None

    def _pair_strict(self, leaving: dict, entering: dict) -> dict | None:
        """The interchange record for two stations, under any key either is
        filed beneath — and only those keys.

        No base-name fallback here, deliberately. The list contains
        "paddington|paddington", meaning Paddington Underground to Paddington
        main line: two operators in one building. Reaching it by stripping
        qualifiers would make the app read that as "the main entrance and the
        Hammersmith & City entrance are two different stations", which is the
        opposite of true.
        """
        for a in self._aliases.get(leaving.get("naptan"), [leaving.get("key", "")]):
            for b in self._aliases.get(entering.get("naptan"), [entering.get("key", "")]):
                hit = self._osi.get(f"{a}|{b}")
                if hit is not None:
                    return hit
        return None

    def _pair(self, leaving: dict, entering: dict) -> dict | None:
        """As above, then falling back to the names for callers who have a
        pair the keys do not cover."""
        return (self._pair_strict(leaving, entering)
                or self._interchange(leaving["name"], entering["name"]))

    def _interchange(self, exit_name: str, entry_name: str) -> dict | None:
        hit = self._osi.get(f"{norm(exit_name)}|{norm(entry_name)}")
        if hit is not None:
            return hit
        a, b = self._base_name(exit_name), self._base_name(entry_name)
        if a == exit_name and b == entry_name:
            return None
        return self._osi.get(f"{norm(a)}|{norm(b)}")

    @staticmethod
    def _metres(a: dict, b: dict) -> float | None:
        """Great-circle distance. Straight-line, so it understates a walk —
        which is the safe direction: it never makes a station look further
        away than it is, and the designated list carries the real distances."""
        try:
            lat1, lon1 = radians(float(a["lat"])), radians(float(a["lon"]))
            lat2, lon2 = radians(float(b["lat"])), radians(float(b["lon"]))
        except (TypeError, ValueError, KeyError):
            return None
        dlat, dlon = lat2 - lat1, lon2 - lon1
        h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return 2 * 6_371_000 * asin(sqrt(h))

    def _walkable(self, station: dict) -> list[dict]:
        """Stations within a short walk, and whether that walk is free.

        This is the proactive half of the app, and the reason it is here:
        Transport for London's planner routes you through interchanges inside
        the gateline, so a walk between two stations almost never appears in
        a suggested journey. But it is exactly what a passenger does when a
        line fails — walk out, walk round, tap in again — and it is the moment
        the second fare lands. Knowing which neighbours are designated is only
        useful *before* you do it.
        """
        near: list[dict] = []
        seen_naptans = {station["naptan"]}
        for other in self._stations.values():
            if other["naptan"] in seen_naptans:
                continue
            if str(other.get("naptan", "")).startswith(TRAM_PREFIXES):
                continue
            straight = self._metres(station, other)
            # A generous straight-line gate, because the published distance
            # can be shorter than the straight one: Oxford Circus to Bond
            # Street is 280 m published and 551 m between centroids, since
            # both have entrances strung along Oxford Street. Gating on the
            # straight line alone would drop pairs that are genuinely a short
            # walk.
            if straight is None or straight > WALKABLE_METRES * 1.6:
                continue
            record = self._pair(station, other)
            # Transport for London's own figure where there is one — measured
            # entrance to entrance along roads. It is what `interchange.check`
            # reports, and the two outcomes must not give different distances
            # for the same pair.
            metres = (record["metres"] if record and record.get("metres")
                      else straight)
            if metres > WALKABLE_METRES:
                continue
            seen_naptans.add(other["naptan"])
            near.append({
                "name": other["name"],
                "metres": int(round(metres)),
                "minutes": max(1, round(metres / WALK_METRES_PER_MINUTE)),
                "designated": record is not None,
                "allowed": record["minutes"] if record else None,
            })
        near.sort(key=lambda n: n["metres"])

        # Collapse entries that are the same place under two names. "Euston"
        # and "London Euston" are seventeen metres apart and share a gateline;
        # offering both as separate walks, and possibly warning about one
        # while the other is designated, would be nonsense. Anything within
        # COINCIDENT_METRES of a nearer neighbour is the same building.
        distinct: list[dict] = []
        for candidate in near:
            def same_place(other: dict) -> bool:
                """Two names for one building, rather than two stations."""
                # The published list decides first, and overrides any
                # distance. Where it names a pair as an interchange it is
                # saying they are two stations — West Hampstead and West
                # Hampstead (North London Line) are 220 m apart with 20
                # minutes allowed between them. That matters here because five
                # same-name records are aliased onto a neighbour's
                # coordinates, so their distance apart reads as zero and would
                # otherwise merge them and delete the only useful fact about
                # the pair.
                a = self._stations.get(norm(other["name"]))
                b = self._stations.get(norm(candidate["name"]))
                if a and b and self._pair_strict(a, b):
                    return False
                # Two entrances to one complex: a shared base name
                # ("Paddington" and "Paddington (H&C Line)-Underground") ...
                if self._base_name(other["name"]) == self._base_name(candidate["name"]):
                    return True
                # ... or close enough to be one building under two names
                # ("Euston" and "London Euston", seventeen metres apart).
                a = self._stations.get(norm(other["name"]))
                b = self._stations.get(norm(candidate["name"]))
                if not a or not b:
                    return False
                gap = self._metres(a, b)
                return gap is not None and gap < COINCIDENT_METRES

            twin = next((k for k in distinct if same_place(k)), None)
            if twin is None:
                distinct.append(candidate)
            elif candidate["designated"] and not twin["designated"]:
                # Prefer the name the interchange list knows: if either of the
                # two is designated, walking there is designated.
                twin.update(candidate)
        return distinct

    # -- the call ---------------------------------------------------------

    def handle(self, call: Call) -> Result:
        if call.outcome_id == "interchange.check":
            return self._check_interchange(call)
        if call.outcome_id != "journey.cost":
            # Only reachable if the manifest declares an outcome the code has
            # no branch for. The base class default is "Something went wrong",
            # which tells nobody anything; naming it says where to look.
            error = AgentAppError(f"no branch for outcome {call.outcome_id!r}")
            error.user_message = (
                f"I can't do {echo(str(call.outcome_id), 40)} — this app "
                f"handles journeys and interchanges.")
            raise error
        return self._cost_journey(call)

    def _check_interchange(self, call: Call) -> Result:
        """One fare or two, for a walk between two named stations.

        The question the whole app exists to answer, asked directly: somebody
        standing at a gateline deciding whether to walk. It needs no network —
        the answer is a lookup in the bundled list and one fare from the zone
        table — so it is the one thing here that still works when Transport
        for London is unreachable, and it answers in milliseconds.
        """
        leaving = self._resolve(call.inputs.get("exit_station"), "starting")
        entering = self._resolve(call.inputs.get("entry_station"), "destination")
        if leaving["naptan"] == entering["naptan"]:
            error = SameStation(f"identical: {leaving['name']!r}")
            error.user_message = (f"That is {leaving['name']} on both sides — "
                                  f"you would not be leaving at all.")
            raise error

        peak = self._is_peak(call)
        # By the key each station is filed under first — the same reason as in
        # `_walkable`. "Hammersmith (H&C Line)" normalises to
        # "hammersmith h and c line" while the interchange list files it as
        # "hammersmith cir h and c", so a lookup by name misses it and reports
        # a free walk as a second fare.
        record = self._pair(leaving, entering)

        # Prefer the published distance. It is measured entrance to entrance
        # along roads, and the two differ more than you would expect: Oxford
        # Circus to Bond Street is 280 m published and 551 m straight between
        # the station centroids, because both have entrances strung along
        # Oxford Street. Where there is no published figure — every
        # undesignated pair — a straight line is all there is, and it
        # understates the walk.
        metres = record["metres"] if record and record.get("metres") else None
        if not metres:
            straight = self._metres(leaving, entering)
            metres = straight if straight is not None else 0
        second = 0 if record else (self._zone_fare(entering, entering, peak) or 0)

        data = {
            "designated": record is not None,
            "minutes_allowed": record["minutes"] if record else 0,
            "walk_metres": int(round(metres)),
            "journeys_charged": 1 if record else 2,
            "second_fare_pence": second,
            "exit_station_name": leaving["name"],
            "entry_station_name": entering["name"],
        }

        if record:
            narrative = (
                f"{leaving['name']} to {entering['name']} is a designated "
                f"interchange — one journey, so long as you are back through "
                f"the gates within {record['minutes']} minutes.")
            if metres:
                narrative += f" It is about {int(round(metres))} metres."
        else:
            narrative = (
                f"{leaving['name']} to {entering['name']} is not a designated "
                f"interchange, so tapping out and back in is charged as two "
                f"journeys")
            narrative += (f" — a second fare from {pounds(second)}."
                          if second else ".")
            # Where there IS a free alternative nearby, that is the useful half.
            safe = next((n for n in self._walkable(leaving) if n["designated"]), None)
            if safe:
                narrative += (f" {safe['name']} is designated, if that works "
                              f"for you — {safe['allowed']} minutes.")

        aside = self._did_you_know(leaving, entering, narrative)
        if aside:
            data["did_you_know"] = aside
        return Result(data=data, narrative=narrative)

    def _cost_journey(self, call: Call) -> Result:
        self._started = time.monotonic()
        origin = self._resolve(call.inputs.get("from_station"), "starting")
        destination = self._resolve(call.inputs.get("to_station"), "destination")
        if origin["naptan"] == destination["naptan"]:
            error = SameStation(f"identical stations: {origin['name']!r}")
            error.user_message = f"{origin['name']} is both ends of that journey."
            raise error

        # A non-string optional input is left out of `call.inputs` entirely
        # when the person did not mention it — never sent as a falsy value —
        # so `.get` returning None is the ordinary "they did not say" path.
        step_free = truthy(call.inputs.get("step_free"))

        peak = self._is_peak(call)
        disrupted = self._disrupted_lines()
        journeys = self._journeys(origin, destination, step_free)
        if not journeys:
            raise NoRouteFound(f"no journeys {origin['name']} -> {destination['name']}")

        options = self._shape(journeys, origin, destination, peak, disrupted)
        if not options:
            raise NoRouteFound("journeys returned but none could be shaped")

        # Ranked by what each costs the person: the fare, plus the time valued
        # at PENCE_PER_MINUTE. An unpriced option sorts last — "no fare quoted"
        # is not the same as "free".
        def cost(option: dict) -> tuple[int, int]:
            fare = option["pence"] if option["pence"] is not None else 10 ** 5
            return (fare + option["minutes"] * PENCE_PER_MINUTE, option["minutes"])

        # Only now, and only if cycling is genuinely on the table, is it
        # worth two more requests to find out whether there is a bike.
        cycle = next((o for o in options if o["label"] == "Cycle"), None)
        if cycle is not None and self._remaining() >= OPTIONAL_MIN_SECONDS:
            here = self._bikes_near(origin)
            # The limiting factor is getting a bike. Without that, how many
            # empty docks wait at the far end is not advice — an earlier
            # version said "183 spaces at the other end" on its own when the
            # first lookup failed, which tells somebody nothing they can act
            # on. If this end is unknown, say nothing about either.
            if here:
                notes = [f"{here['bikes']} hire bikes at {here['nearest_with_bikes']}"
                         if here["bikes"] and here["nearest_with_bikes"]
                         else "no hire bikes free at this end"]
                there = self._bikes_near(destination) if here["bikes"] else None
                if there:
                    notes.append(f"{there['spaces']} spaces at the other end"
                                 if there["spaces"]
                                 else "no empty docks at the other end")
                cycle["text"] += " · " + ", ".join(notes)

        options.sort(key=cost)
        # Cycling is offered but never recommended, unless it is the only
        # thing on offer. Transport for London's planner assumes you have a
        # bicycle with you; this app has no way to know whether you do, and
        # recommending one you do not have is worse than useless. Walking
        # needs no such guard — everyone arrives with their legs, and the
        # arithmetic above already declines to recommend a half-hour walk
        # over a two-minute train.
        recommendable = [o for o in options if o["label"] != "Cycle"] or options
        best = recommendable[0]

        on_route = sorted({line for o in options for line in o["disrupted"]})
        # A walk that is genuinely in the suggested route is exact, so it wins;
        # otherwise fall back to what walking out of the origin would cost,
        # which is the case that actually comes up.
        warning = (next((o["warning"] for o in options if o["warning"]), None)
                   or self._walk_out_warning(origin, peak, bool(on_route)))

        # The recommendation leads, then the rest in cost order. A list whose
        # first entry is not the one being recommended reads as a mistake.
        ordered = [best] + [o for o in options if o is not best]
        shown = ordered[:MAX_OPTIONS]

        # Every figure in the payload has to describe one of the options in
        # the payload. Taking the cheapest across all of them quoted a fare
        # belonging to a walk that ranked outside the five returned — true,
        # and impossible for Pippa to explain, because it was not given the
        # option it referred to. Cycling is excluded for the same reason it
        # cannot be recommended: a fare you avoid on a bicycle you may not
        # have is not a fare you can actually avoid.
        # A walk and a cycle do not have a cheap fare, they have no fare, and
        # this field promises "the lowest fare among the options that carry
        # one". Excluding the bicycle by name but not the walk let a free
        # 81-minute walk set the figure: the cheapest way from New Cross to
        # London Bridge came back as £0.00 with "Southeastern · 5 min · £3.20"
        # recommended beside it. Nothing priced is ever zero — the Hopper is
        # said in the sentence, never subtracted from a fare — so a falsy
        # fare means no fare.
        priced = [o for o in shown if o["pence"]]

        data = {
            "options": [o["text"] for o in shown],
            "best": best["text"],
            "best_pence": best["pence"] or 0,
            "cheapest_pence": min((o["pence"] for o in priced), default=0),
            # From `shown`, not from every option, for exactly the reason
            # `cheapest_pence` above is: a figure in the payload has to
            # describe an option in the payload. Six options where the
            # quickest is also the dearest put "fastest 2 minutes" beside a
            # list whose shortest was 20 — true, and impossible for Pippa to
            # explain, because it was never given the journey it meant.
            "fastest_minutes": min(o["minutes"] for o in shown),
            "disrupted_lines": on_route,
        }
        # Fields declared `required: false` are omitted entirely when there is
        # nothing to report, rather than returned empty.
        if warning:
            data["fare_warning"] = warning
        if step_free:
            data["step_free"] = True
        aside = self._did_you_know(origin, destination, warning)
        if aside:
            data["did_you_know"] = aside
        tip = self._peak_tip(origin, destination, call)
        if tip:
            data["peak_tip"] = tip

        return Result(data=data, narrative=self._narrative(
            origin, destination, best, options, on_route, warning, step_free))

    # -- the pieces of handle() -------------------------------------------

    def _disrupted_lines(self) -> list[str]:
        """Line names reporting anything other than a good service.

        `statusSeverity` is NOT ordered by badness — 10 is Good Service, 20 is
        Service Closed, 9 is Minor Delays — so it is never compared, only
        matched. `lineStatuses` is an array and repeats, so this de-duplicates.
        """
        try:
            payload = self._get("/Line/Mode/tube/Status")
        except AgentAppError:
            # Which lines are delayed is an enhancement, not the answer. If
            # the status call fails the routes and the fares are still worth
            # having, and losing them because a second endpoint was slow
            # would be a worse outcome than quietly leaving out the flags.
            return []
        if not isinstance(payload, list):
            return []
        bad: list[str] = []
        for line in payload:
            statuses = line.get("lineStatuses") or []
            # A MISSING description must not count. `"" != "good service"` is
            # true, so the obvious test turns silence into a positive claim
            # that the line is delayed — the one kind of error this app cannot
            # afford, because every figure it gives is meant to be checkable
            # against Transport for London a minute later.
            if any(told and told != "good service"
                   for s in statuses
                   for told in [(s.get("statusSeverityDescription") or "").strip().lower()]):
                name = line.get("name") or line.get("id")
                if name and name not in bad:
                    bad.append(str(name))
        return bad

    def _planner_naptan(self, record: dict) -> str:
        """The code the journey planner will accept for this station.

        Hub codes describe a place, not a platform, and the planner answers a
        hub with a 300 asking which station was meant. Resolved once per call
        against the bundled list: the nearest concrete station to the hub,
        which for King's Cross is the Underground station and for New Cross
        the Overground one.
        """
        naptan = record.get("naptan") or ""
        if not naptan.startswith("HUB"):
            return naptan
        lat, lon = record.get("lat"), record.get("lon")
        if lat is None or lon is None:
            return naptan
        best, best_gap = naptan, None
        for other in self._stations.values():
            code = other.get("naptan") or ""
            if code.startswith("HUB") or other.get("lat") is None:
                continue
            gap = abs(other["lat"] - lat) + abs(other["lon"] - lon)
            if gap < 0.01 and (best_gap is None or gap < best_gap):
                best, best_gap = code, gap
        return best

    def _journeys(self, origin: dict, destination: dict,
                  step_free: bool = False) -> list[dict]:
        """Journeys from Transport for London.

        `mode=tube` is what makes them price a journey: the same request
        without it returns `fare: null` on every result. The second request
        widens the set to walking, cycling and bus alternatives, which are
        often the right answer during disruption and which the first omits.
        """
        # Transport for London answer HTTP 300 — a disambiguation, not a
        # journey — when either end is a HUB naptan, so "Kings Cross" and
        # "St Pancras" (both HUBKGX) and "New Cross" (HUBNWX) failed outright
        # with "they did not recognise one of those stations". They are among
        # the most asked-for names in London. The planner takes the concrete
        # station at the same place perfectly well, so send that instead.
        start = self._planner_naptan(origin)
        finish = self._planner_naptan(destination)
        route = f"/Journey/JourneyResults/{start}/to/{finish}"
        # Verified 19 Nov 2026: Oxford Circus to Warren Street is one minute by
        # tube, and with this preference there is no tube journey at all — the
        # only result is a 22-minute walk. Warren Street to Barbican goes from
        # 15 minutes to 58. The constraint is not a filter, it is a different
        # network.
        access = "&accessibilityPreference=stepFreeToPlatform" if step_free else ""
        found: list[dict] = []
        queries = (f"{route}?mode=tube{access}",
                   f"{route}?useMultiModalCall=true{access}")
        for index, query in enumerate(queries):
            # The first carries the priced options and is worth waiting for.
            # The second only widens the alternatives, so it is skipped rather
            # than started with too little time left to finish it — unless
            # the first found nothing, in which case it is not an extra, it
            # is the answer, and skipping it reports "no route" for a journey
            # Transport for London will happily describe.
            if index and found and self._remaining() < OPTIONAL_MIN_SECONDS:
                break
            # Nothing found yet and no query behind this one: give it the time
            # that is left rather than a slice sized for an optional extra.
            last_chance = index == len(queries) - 1 and not found
            try:
                payload = self._get(
                    query,
                    ceiling=LAST_CHANCE_SECONDS if last_chance else TIMEOUT)
            except NoRouteFound:
                # A 404 here means Transport for London have no route matching
                # the constraint, not that they are unreachable. Said as
                # "try again shortly" it would be both wrong and dismissive.
                #
                # It must not end the call either. `mode=tube` 404s for any
                # pair no tube line serves — New Cross is Overground only —
                # while the multimodal query right behind it returns six
                # perfectly good journeys. Raising on the first query threw
                # those away and told the person there was no way to get
                # there at all. So carry on, and decide once both have run.
                if index == 0:
                    continue
                # Whether an empty result is "no step-free route" or "no route
                # at all" is decided once, below, where both queries are in.
                break
            except (ServiceUnavailable, ServiceBusy):
                # The first query is the important one. A failure on the
                # second means fewer alternatives, not a failed call.
                if not found:
                    raise
                break
            if isinstance(payload, dict):
                found.extend(payload.get("journeys") or [])

        if not found:
            # Nothing from either query. Which failure this is depends on what
            # was asked for: a step-free request that comes back empty means
            # no step-free route exists, which is a different and more useful
            # thing to say than "try again shortly".
            if step_free:
                error = NoStepFreeRoute(f"no step-free route {route}")
                error.user_message = (
                    f"Transport for London have no step-free route from "
                    f"{origin['name']} to {destination['name']}.")
                raise error
            raise NoRouteFound(f"no journeys for {route}")
        return found

    def _bikes_near(self, station: dict) -> dict | None:
        """Hire bikes and empty docks within a short walk of a station.

        Transport for London's planner will happily offer a cycle journey
        whether or not you own a bicycle. For most people the question is
        really about Santander hire, and that turns on two live facts it never
        mentions: whether there is a bike at this end, and whether there is
        anywhere to put it at the other. A dock with no bikes makes the
        suggestion useless; a dock with no spaces makes it worse, because you
        find out at the end.

        Returns None rather than guessing when the lookup fails — a cycle
        option with no dock figures is still a cycle option.
        """
        lat, lon = station.get("lat"), station.get("lon")
        if lat is None or lon is None:
            return None
        try:
            payload = self._get(f"/BikePoint?lat={lat}&lon={lon}&radius={BIKE_RADIUS}")
        except AgentAppError:
            return None

        # With a radius this returns {"places": [...]}; without one it returns
        # a bare list. Two different shapes from one path.
        places = payload.get("places") if isinstance(payload, dict) else payload
        if not isinstance(places, list) or not places:
            return None

        def count(dock: dict, key: str) -> int:
            for prop in dock.get("additionalProperties") or []:
                if prop.get("key") == key:
                    try:
                        return int(prop.get("value") or 0)
                    except (TypeError, ValueError):
                        return 0
            return 0

        usable = [d for d in places
                  if not any(p.get("key") == "Locked" and
                             str(p.get("value")).lower() == "true"
                             for p in d.get("additionalProperties") or [])]
        if not usable:
            return None

        best = max(usable, key=lambda d: count(d, "NbBikes"))
        return {
            "bikes": sum(count(d, "NbBikes") for d in usable),
            "spaces": sum(count(d, "NbEmptyDocks") for d in usable),
            "docks": len(usable),
            # Dock names are "Drummond Street , Euston" — area after the
            # comma, and an inconsistent space before it.
            "nearest_with_bikes": (best.get("commonName") or "").split(",")[0].strip()
            if count(best, "NbBikes") else None,
        }

    def _shape(self, journeys: list[dict], origin: dict, destination: dict,
               peak: bool, disrupted: list[str]) -> list[dict]:
        """Each journey to one readable line, de-duplicated."""
        shaped: list[dict] = []
        seen: dict[tuple, dict] = {}

        # Quickest first, so that where several results collapse to one entry
        # the one kept is the quickest rather than whichever came back first.
        journeys = sorted(journeys, key=lambda j: j.get("duration") or 10 ** 6)

        for journey in journeys:
            minutes = journey.get("duration")
            if not isinstance(minutes, int) or minutes < 0:
                continue

            modes = [m for m in self._modes(journey) if m]
            if not modes:
                continue
            lines = self._rail_lines(journey)
            paying = [m for m in modes if m in FARE_MODES]

            pence = (journey.get("fare") or {}).get("totalCost")
            source = "tfl"
            if not isinstance(pence, int):
                if not paying:
                    # Transport for London does not charge you to walk or to
                    # ride your own bicycle. A zone fare here would be wrong.
                    pence, source = 0, "none"
                else:
                    pence = self._zone_fare(origin, destination, peak)
                    source = "zone" if pence is not None else "unknown"

            label = self._label(modes, lines, journey)
            # Collapse near-identical results: Transport for London returns
            # the same route several times a few minutes apart, and the two
            # requests overlap. Walking and cycling collapse to one entry each
            # — two cycle routes four minutes apart is noise, not a choice.
            # The same route priced by them beats the same route priced from
            # the zone table, so a real fare replaces an estimated one.
            # Keyed on the label alone: two results reading "Victoria · 3 min"
            # and "Victoria · 4 min" are the same advice twice. Journeys were
            # sorted quickest-first above, so the one kept is the quickest.
            key = (label,)
            if key in seen:
                if source == "tfl" and seen[key]["source"] != "tfl":
                    seen[key].update(pence=pence, source=source,
                                     text=self._option_text(label, seen[key]["minutes"],
                                                            pence, source,
                                                            seen[key]["disrupted"]))
                continue

            hit = [line for line in lines if line in disrupted]
            warning = self._warning(journey, destination, peak, pence)

            entry = {
                "minutes": minutes,
                "pence": pence,
                "source": source,
                "lines": lines,
                "label": label,
                "disrupted": hit,
                "warning": warning,
                "text": self._option_text(label, minutes, pence, source, hit),
            }
            seen[key] = entry
            shaped.append(entry)
        return shaped

    def _warning(self, journey: dict, destination: dict, peak: bool,
                 pence: int | None) -> str | None:
        """The second-fare warning for a walk that is actually in the route.

        Rare — see `_walkable` — but exact when it happens, so it takes
        precedence over the proactive version.
        """
        walk = self._out_of_station_walk(journey)
        if not walk:
            return None
        record = self._interchange(*walk)
        if record is not None:
            return (f"This route walks from {walk[0]} to {walk[1]}, which is a "
                    f"designated interchange: one journey, but only if you are "
                    f"back through the gates within {record['minutes']} minutes.")
        second = self._zone_fare(destination, destination, peak)
        amount = f" — a second fare from {pounds(second)}" if second else ""
        return (f"This route walks from {walk[0]} to {walk[1]}, which is not a "
                f"designated interchange, so tapping out and back in is charged "
                f"as two journeys{amount}.")

    def _walk_out_warning(self, origin: dict, peak: bool,
                          disrupted: bool) -> str | None:
        """What walking out of the origin station would cost.

        Fires when there is a station close enough to walk to that is **not**
        designated, because that is the trap: the obvious thing to do when a
        line fails is also the thing that charges you twice. Where the station
        does have a designated neighbour, it is named too — the pair of facts
        together is the whole answer, and neither half is any use alone.
        """
        near = self._walkable(origin)
        if not near:
            return None
        traps = [n for n in near if not n["designated"]]
        safe = [n for n in near if n["designated"]]
        if not traps:
            if not safe:
                return None
            first = safe[0]
            return (f"Walking out of {origin['name']} to {first['name']} is a "
                    f"designated interchange — one journey, if you are back "
                    f"through the gates within {first['allowed']} minutes.")

        trap = traps[0]
        second = self._zone_fare(origin, origin, peak)
        amount = f", a second fare from {pounds(second)}" if second else ""
        opening = ("The line is disrupted, so if you walk out of "
                   if disrupted else "If you walk out of ")
        text = (f"{opening}{origin['name']}: {trap['name']} is about "
                f"{trap['minutes']} minutes away but is not a designated "
                f"interchange, so tapping out and back in there is charged as "
                f"two journeys{amount}.")
        if safe:
            first = safe[0]
            text += (f" {first['name']} is designated — one journey if you are "
                     f"back through the gates within {first['allowed']} minutes.")
        return text

    @staticmethod
    def _label(modes: list[str], lines: list[str], journey: dict) -> str:
        unique = set(modes)
        if unique <= {"walking"}:
            return "Walk"
        if "cycle" in unique:
            return "Cycle"
        if lines:
            label = " then ".join(lines)
            if "bus" in unique:
                label += " and bus"
            return label
        routes = App._bus_routes(journey)
        if routes:
            return "Bus " + "/".join(routes[:2])
        # Last resort: the mode names themselves. They arrive lower case
        # ("tube", "national-rail") and go straight into an answer a person
        # reads, so they are tidied rather than passed through raw.
        pretty = {"tube": "Tube", "dlr": "DLR", "national-rail": "National Rail",
                  "overground": "Overground", "elizabeth-line": "Elizabeth line",
                  "tram": "Tram", "bus": "Bus", "river-bus": "River bus",
                  "cable-car": "Cable car", "coach": "Coach"}
        parts = [pretty.get(m, m.replace("-", " ").capitalize())
                 for m in dict.fromkeys(modes) if m != "walking"]
        return " then ".join(parts) or "Travel"

    @staticmethod
    def _option_text(label: str, minutes: int, pence: int | None,
                     source: str, disrupted: list[str]) -> str:
        """One option as a line a person can read aloud."""
        if pence is None:
            money = "fare not quoted"
        elif pence == 0:
            money = "no fare"
        else:
            money = pounds(pence) + (" (zone estimate)" if source == "zone" else "")
        text = f"{label} · {minutes} min · {money}"
        if disrupted:
            text += f" · {', '.join(disrupted)} disrupted"
        return text

    # -- the aside ---------------------------------------------------------

    def _did_you_know(self, origin: dict, destination: dict,
                      warning: str | None) -> str | None:
        """One true, surprising thing about the stations in this journey.

        The fare rules are full of small absurdities, and they are the most
        likeable part of the subject: a designated interchange between two
        stations two metres apart, a gateline you can walk out of by the wrong
        door and start a clock you did not know existed, forty minutes allowed
        one way and twenty coming back.

        Everything here is read from the bundled list. Nothing is generated,
        nothing is embellished, and when there is no fact to tell, the field
        is left out rather than padded — an invented curiosity about somebody's
        fare is worse than none.
        """
        ends = [s for s in (origin, destination) if s]
        said = (warning or "").lower()

        def fresh(text: str, *names: str) -> str | None:
            """Skip anything the warning has already covered."""
            return None if any(n.lower() in said for n in names) else text

        # 1. A gateline inside one of these stations. Almost nobody knows that
        #    leaving by the wrong door can start a clock.
        for end in ends:
            record = self._gateline_for(end)
            if not record:
                continue
            if record.get("step_free_forced"):
                return (f"At {record['station']} the step-free route leaves and "
                        f"re-enters the paid area, so getting out by the only "
                        f"accessible way starts a {record['minutes']}-minute clock.")
            # The published text is a noun phrase with proper nouns in it —
            # "Argyll Street exit and main entrance" — so it goes in as
            # written. Lower-casing the first letter turned Argyll into argyll.
            return (f"{record['station']} has an interchange inside the station, "
                    f"between {record['between']}, and you have "
                    f"{record['minutes']} minutes to cross it.")

        # 2. A designated walk from one of these stations that is absurd in
        #    one direction or the other.
        best_short: tuple | None = None
        best_long: tuple | None = None
        for end in ends:
            prefix = norm(end["name"]) + "|"
            for key, pair in self._osi.items():
                if not key.startswith(prefix):
                    continue
                metres = pair["metres"]
                if 0 < metres <= 80 and (best_short is None or metres < best_short[0]):
                    best_short = (metres, pair)
                if metres >= 900 and (best_long is None or metres > best_long[0]):
                    best_long = (metres, pair)
        if best_short:
            metres, pair = best_short
            text = fresh(
                f"{pair['exit']} and {pair['entry']} are a designated "
                f"interchange — and they are {metres} metres apart, with "
                f"{pair['minutes']} minutes allowed to cross.",
                pair["entry"])
            if text:
                return text
        if best_long:
            metres, pair = best_long
            text = fresh(
                f"You can walk the {metres / 1000:.1f} km from {pair['exit']} to "
                f"{pair['entry']} and still be charged one journey, so long as "
                f"it takes under {pair['minutes']} minutes.",
                pair["entry"])
            if text:
                return text

        # 3. An allowance that differs by direction.
        #
        #    This used to end "the longer side is the one where you wait on a
        #    concourse rather than a platform" — a tidy explanation, and an
        #    invented one: the bundled list carries entry, exit, metres,
        #    minutes and variants, and nothing about concourses anywhere. The
        #    field promises one TRUE thing read from the list, so a plausible
        #    causal story is exactly the wrong thing to put in it.
        #
        #    What the data does say is stronger. Every one of the eighteen
        #    asymmetric pairs gives the longer allowance to the direction
        #    ending at the National Rail station — 36 of 36 directed entries,
        #    no exceptions. That is checkable, and more interesting than the
        #    guess it replaces.
        for end in ends:
            prefix = norm(end["name"]) + "|"
            for key, out in self._osi.items():
                if not key.startswith(prefix):
                    continue
                a, b = key.split("|")
                back = self._osi.get(f"{b}|{a}")
                if not back or back["minutes"] == out["minutes"]:
                    continue
                longer = out if out["minutes"] > back["minutes"] else back
                to_rail = any("->NR" in v for v in longer.get("variants") or [])
                tail = (f" The extra time always goes to the direction ending "
                        f"at the National Rail station — every asymmetric pair "
                        f"on the list does the same."
                        if to_rail else "")
                text = fresh(
                    f"{out['exit']} to {out['entry']} allows {out['minutes']} "
                    f"minutes; coming back allows {back['minutes']}.{tail}",
                    out["entry"])
                if text:
                    return text

        # 4. A station that is unusually well connected for free walks.
        for end in ends:
            prefix = norm(end["name"]) + "|"
            out = {self._osi[k]["entry"] for k in self._osi if k.startswith(prefix)}
            if len(out) >= 3:
                text = fresh(
                    f"{end['name']} is one of the better-connected stations for "
                    f"this: {len(out)} places you can walk to and still be "
                    f"charged a single journey.", *out)
                if text:
                    return text

        # 5. Nothing about this journey, so something about the network. Said
        #    as a general fact, because that is what it is — the alternative is
        #    dressing a network-wide quirk up as local, which is a small lie.
        general = self._network_facts()
        if not general:
            return None
        # Deterministic, so the same journey always says the same thing and a
        # test can assert it, while different journeys vary. Both ends go into
        # the hash: keyed on the origin alone, every journey out of Euston
        # told the same fact however far apart the destinations were.
        seed = origin["naptan"] + "|" + destination["naptan"]
        pick = sum(ord(c) for c in seed) % len(general)
        return general[pick]

    def _network_facts(self) -> list[str]:
        """Superlatives read from the bundled list, not written by hand.

        This pool was three facts, and three facts over five hundred stations
        meant eighty-nine per cent of answers carried one of the same three
        lines — somebody trying three journeys saw a repeat. Everything here
        is still computed from the data rather than typed out, so the figures
        cannot drift away from the file they describe.
        """
        pairs = [p for p in self._osi.values() if p["metres"] > 0]
        if not pairs or not self._gateline:
            return []
        shortest = min(pairs, key=lambda p: p["metres"])
        longest = max(pairs, key=lambda p: p["metres"])
        tightest = min(self._gateline.values(), key=lambda g: g["minutes"])
        roomiest = max(self._gateline.values(), key=lambda g: g["minutes"])
        quickest = min(self._osi.values(), key=lambda p: p["minutes"])

        # How many places you can walk to free, from each station.
        reach: dict[str, int] = {}
        for key in self._osi:
            reach[key.split("|")[0]] = reach.get(key.split("|")[0], 0) + 1
        busiest_key = max(reach, key=lambda k: reach[k])
        busiest = next(v for k, v in self._osi.items()
                       if k.startswith(busiest_key + "|"))

        # Pairs whose two directions disagree, and the widest disagreement.
        gaps = []
        for key, out in self._osi.items():
            a, b = key.split("|")
            back = self._osi.get(f"{b}|{a}")
            if back and back["minutes"] != out["minutes"]:
                gaps.append((out, back))
        widest = max(gaps, key=lambda g: abs(g[0]["minutes"] - g[1]["minutes"])) \
            if gaps else None
        # Only claim the National Rail pattern while the data still shows it.
        rail_rule = bool(gaps) and all(
            any("->NR" in v for v in
                (out if out["minutes"] > back["minutes"] else back)
                .get("variants") or [])
            for out, back in gaps)

        same_name = sorted({p["exit"] for p in self._osi.values()
                            if p["exit"] == p["entry"]})
        spread: dict[int, int] = {}
        for pair in self._osi.values():
            spread[pair["minutes"]] = spread.get(pair["minutes"], 0) + 1
        usual = max(spread.items(), key=lambda kv: kv[1])
        listed = len({k.split("|")[0] for k in self._osi})
        total = len({s["naptan"] for s in self._stations.values()})

        facts = [
            (f"{shortest['exit']} and {shortest['entry']} are two different "
             f"stations with the same name, and a designated interchange "
             f"between them — {shortest['metres']} metres apart, with "
             f"{shortest['minutes']} minutes allowed to cross."
             if shortest['exit'] == shortest['entry'] else
             f"Somewhere on the network, {shortest['exit']} and "
             f"{shortest['entry']} are a designated interchange — "
             f"{shortest['metres']} metres apart, with {shortest['minutes']} "
             f"minutes allowed to cross."),
            f"The longest walk you can make and still be charged one journey is "
            f"{longest['exit']} to {longest['entry']}: {longest['metres'] / 1000:.1f} "
            f"km, inside {longest['minutes']} minutes.",
            f"At {tightest['station']} there is an interchange between "
            f"{tightest['between']} — and only {tightest['minutes']} minutes "
            f"to cross it.",
            f"{roomiest['station']} is the most forgiving interchange inside a "
            f"station: {roomiest['minutes']} minutes to cross it.",
            f"{busiest['exit']} is the best-connected station for this: "
            f"{reach[busiest_key]} places you can walk to and still be charged "
            f"one journey.",
            f"The tightest designated walk on the list is {quickest['exit']} to "
            f"{quickest['entry']} — {quickest['metres']} metres, and "
            f"{quickest['minutes']} minutes to do it in.",
            f"There are {len(self._osi)} designated interchanges on the "
            f"network, and no journey planner shows you a single one.",
            f"The usual allowance is {usual[0]} minutes — {usual[1]} of the "
            f"{len(self._osi)} designated pairs give you exactly that.",
            f"Only {listed} of {total} stations appear on the interchange list "
            f"at all, which is why the ones that do are worth knowing.",
            f"There are {len(same_name)} station names that turn up on both "
            f"sides of a designated pair — {same_name[0]} to {same_name[0]}, "
            f"{same_name[-1]} to {same_name[-1]}.",
        ]
        if widest:
            out, back = widest
            facts.append(
                f"{out['exit']} to {out['entry']} allows {out['minutes']} "
                f"minutes and the way back allows {back['minutes']} — the "
                f"widest disagreement on the list.")
        if rail_rule:
            facts.append(
                f"There are {len(gaps) // 2} pairs that allow more time one "
                f"way than the other, and every one of them gives the extra "
                f"time to the direction ending at a National Rail station.")
        return facts

    def _narrative(self, origin: dict, destination: dict, best: dict,
                   options: list[dict], disrupted: list[str],
                   warning: str | None, step_free: bool = False) -> str:
        """One sentence Pippa may use. Names the subject rather than only the
        number, because a bare figure gives a follow-up nothing to lean on."""
        lead = "Step-free, " if step_free else ""
        parts = [f"{lead}{origin['name']} to {destination['name']}: {best['text']}."]

        # Cycling is excluded here for the same reason it cannot be `best`:
        # suggesting someone save £3.00 on a bicycle they may not have is not
        # a saving, it is a non-sequitur.
        priced = [o for o in options
                  if o["pence"] is not None and o["label"] != "Cycle"]
        cheaper = min(priced, key=lambda o: o["pence"], default=None)
        if (cheaper and cheaper is not best and best["pence"] is not None
                and cheaper["pence"] < best["pence"]):
            saved = best["pence"] - cheaper["pence"]
            longer = cheaper["minutes"] - best["minutes"]
            # Only worth saying when it is a close call. Saving £3.00 by
            # walking for an extra 45 minutes is 4p a minute, far under what
            # the ranking values a minute at — so the ranking has already
            # rejected it, and raising it here would contradict the
            # recommendation in the same breath as making it.
            worth_it = longer > 0 and saved >= longer * PENCE_PER_MINUTE * 0.8
            if worth_it:
                parts.append(f"{pounds(saved)} cheaper to {cheaper['text'].split(' · ')[0]} "
                             f"instead, for {longer} more minutes.")
        # The Hopper. Transport for London give you unlimited bus and tram
        # journeys within an hour of your first bus tap, which means a bus leg
        # is often genuinely free rather than merely cheap — the second London
        # money rule this app knows, after the interchange list.
        if "Bus" in best["label"] or "bus" in best["label"]:
            parts.append("That bus fare is free if you have tapped a bus or "
                         "tram in the last hour.")
        if disrupted:
            parts.append(f"{', '.join(disrupted)} reporting delays.")
        if warning:
            parts.append(warning)
        return " ".join(parts)
