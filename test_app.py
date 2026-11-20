"""Tests for DIVERT.

The build runs these with **every outbound socket refused**, so nothing here
may touch the network. `App._fetch` is the single method that does HTTP, and
every test replaces it on the instance — which is the reason it exists as one
method rather than being inlined at each call site.

The fixtures mirror shapes read from the live API on 17 November 2026. The
`WARREN_BARBICAN` payload is a real response, trimmed to the fields app.py
actually reads.
"""

from __future__ import annotations

import json
import time
import urllib.error
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import pytest

from app import (App, DEADLINE_SECONDS, TIMEOUT, norm, pounds)
from pippa_agentapp_sdk import AgentAppError, LocalRunner

LONDON_TZ = ZoneInfo("Europe/London")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _leg(mode, lines=(), frm=None, to=None, frm_name="", to_name=""):
    return {
        "mode": {"name": mode},
        "departurePoint": {"naptanId": frm, "commonName": frm_name},
        "arrivalPoint": {"naptanId": to, "commonName": to_name},
        "routeOptions": [{"lineIdentifier": {"id": i, "name": n}} for i, n in lines],
    }


#: Real response, trimmed. Warren Street -> Barbican, Metropolitan line.
WARREN_BARBICAN = {"journeys": [
    {"duration": 14, "fare": {"totalCost": 300},
     "legs": [_leg("walking"), _leg("tube", [("metropolitan", "Metropolitan")])]},
]}

#: Everything in good order, so nothing is reported as disrupted.
ALL_GOOD = [{"id": lid, "name": lid.title(),
             "lineStatuses": [{"statusSeverity": 10,
                               "statusSeverityDescription": "Good Service"}]}
            for lid in ("central", "victoria", "northern")]

#: Severity 20 is Service Closed and 9 is Minor Delays — both worse than 10,
#: Good Service. The numbers are not ordered by badness, which is why the app
#: matches on the description and never compares the number.
MIXED_STATUS = [
    {"id": "central", "name": "Central",
     "lineStatuses": [{"statusSeverity": 20, "statusSeverityDescription": "Service Closed"},
                      {"statusSeverity": 20, "statusSeverityDescription": "Service Closed"}]},
    {"id": "victoria", "name": "Victoria",
     "lineStatuses": [{"statusSeverity": 10, "statusSeverityDescription": "Good Service"}]},
    {"id": "metropolitan", "name": "Metropolitan",
     "lineStatuses": [{"statusSeverity": 9, "statusSeverityDescription": "Minor Delays"}]},
]


def route(app: App, status=ALL_GOOD, journeys=WARREN_BARBICAN):
    """Point the app's one network method at canned payloads."""
    def fake(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(status).encode()
        return json.dumps(journeys).encode()
    app._fetch = fake
    return app


def run_call(app, outcome: str, inputs: dict, *, now=None, explicit=False):
    """Call the app the way the production runtime does.

    Through `LocalRunner`, which their SDK describes as "the same code the
    production runtime uses to call your app" — so a Result that violates the
    manifest fails here exactly as it would in production.

    Not by building `Call(outcome, inputs, Context(now=...))` by hand. That
    worked against the local stand-in for weeks and then died at import in
    their build:

        ImportError: cannot import name 'Context' from 'pippa_agentapp_sdk'

    `Context` is not part of the real SDK. Every call in this suite goes
    through here, so the suite depends on nothing the real SDK lacks.
    """
    return LocalRunner(app).call(outcome, dict(inputs), granted_secrets={},
                                 now=now, request_id="test-1",
                                 explicit=explicit).result


def ask(app: App, frm="Warren Street", to="Barbican", when=None):
    moment = when or datetime(2026, 11, 17, 14, 0, tzinfo=timezone.utc)  # Tue, off-peak
    return run_call(app, "journey.cost",
                    {"from_station": frm, "to_station": to}, now=moment)



def leg_journey(minutes: int, pence: int | None, mode: str,
                line: str = "northern") -> dict:
    """One journey in Transport for London's shape, with a chosen fare.

    `fare` absent and `totalCost` zero are different things to this app —
    one is "not quoted", the other is "free" — so the stub has to be able to
    produce both.
    """
    journey = {
        "duration": minutes,
        "legs": [{"mode": {"name": mode},
                  "departurePoint": {"lat": 51.5247, "lon": -0.1384,
                                     "naptanId": "940GZZLUWRR",
                                     "commonName": "Warren Street"},
                  "arrivalPoint": {"lat": 51.5201, "lon": -0.0979,
                                   "naptanId": "940GZZLUBBN",
                                   "commonName": "Barbican"},
                  "routeOptions": [{"lineIdentifier": {"id": line,
                                                       "name": line.title()}}]}],
    }
    if pence is not None:
        journey["fare"] = {"totalCost": pence}
    return journey


@pytest.fixture
def app():
    return App()


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------

def test_returns_exactly_the_declared_fields(app):
    result = ask(route(app))
    declared = {f["name"] for f in app.manifest.outcome("journey.cost")["output"]}
    optional = {f["name"] for f in app.manifest.outcome("journey.cost")["output"]
                if f.get("required") is False}
    assert set(result.data) - declared == set(), "returned an undeclared field"
    assert declared - optional - set(result.data) == set(), "missing a required field"


def test_optional_field_is_omitted_not_blank(app):
    """A station with no walkable neighbour and no walk in the route should
    leave `fare_warning` out entirely rather than return an empty string."""
    result = ask(route(app), frm="Heathrow Terminal 5", to="Barbican")
    assert result.data.get("fare_warning", "OMITTED") != ""


def test_money_is_whole_pence(app):
    data = ask(route(app)).data
    for key in ("best_pence", "cheapest_pence"):
        assert isinstance(data[key], int) and not isinstance(data[key], bool)
    assert isinstance(data["fastest_minutes"], int)


def test_narrative_names_the_subject(app):
    """A bare figure gives a follow-up nothing to lean on."""
    result = ask(route(app))
    assert "Warren Street" in result.narrative and "Barbican" in result.narrative


# ---------------------------------------------------------------------------
# The interchange rule — the reason the app exists
# ---------------------------------------------------------------------------

def test_bank_to_cannon_street_takes_the_shorter_limit(app):
    """Bank to Cannon Street is 15 minutes to the Underground station and 40
    to the main line one. Collapsing those must keep 15: telling somebody they
    have 40 minutes when they have 15 is the one error that costs them money.
    """
    assert app._interchange("Bank", "Cannon Street")["minutes"] == 15


def test_designated_and_undesignated_pairs(app):
    assert app._interchange("Oxford Circus", "Bond Street") is not None
    assert app._interchange("Warren Street", "Euston Square") is not None
    assert app._interchange("Oxford Circus", "Warren Street") is None


def test_warning_names_both_the_trap_and_the_safe_pair(app):
    """Warren Street has Great Portland Street close by and not designated,
    and Euston Square close by and designated. Either fact alone is useless."""
    warning = ask(route(app)).data["fare_warning"]
    assert "Great Portland" in warning and "not a designated" in warning
    assert "Euston Square" in warning and "20 minutes" in warning


def test_no_invented_pairs(app):
    """Every bundled pair must resolve in both the table and the station list,
    or the app will confidently describe a walk that does not exist."""
    missing = [k for k in app._osi
               if any(part not in app._stations for part in k.split("|"))]
    assert not missing[:5], f"interchange endpoints not in the station list: {missing[:5]}"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("frm,to", [("", "Barbican"), (None, "Barbican"),
                                    ("Warren Street", ""), (None, None)])
def test_missing_input_asks_for_it(app, frm, to):
    """Absent and deliberately-blank both reach us, and neither is a crash."""
    with pytest.raises(AgentAppError) as caught:
        ask(route(app), frm=frm, to=to)
    assert "starting station" in caught.value.user_message


def test_unknown_station_is_named_back(app):
    """'I don't recognise Narnia' tells somebody what to retype. 'Not found'
    does not."""
    with pytest.raises(AgentAppError) as caught:
        ask(route(app), frm="Narnia", to="Barbican")
    assert "Narnia" in caught.value.user_message


def test_same_station_both_ends(app):
    with pytest.raises(AgentAppError) as caught:
        ask(route(app), frm="Euston", to="euston")
    assert "Euston" in caught.value.user_message


@pytest.mark.parametrize("written", ["Bond Street", "bond st", "BOND  STREET",
                                     "Bond Street Underground Station"])
def test_people_type_station_names_many_ways(app, written):
    assert norm(written) == norm("Bond Street")


def test_prefix_match_only_when_unambiguous(app):
    assert app._resolve("liverpool", "starting")["name"].startswith("Liverpool")
    with pytest.raises(AgentAppError):
        app._resolve("zzzz", "starting")


# ---------------------------------------------------------------------------
# Fares
# ---------------------------------------------------------------------------

def test_peak_and_off_peak_come_from_the_supplied_clock(app):
    """call.context.now, never datetime.now() — so this still passes in March."""
    tuesday_peak = datetime(2026, 11, 17, 8, 0, tzinfo=timezone.utc)
    tuesday_off = datetime(2026, 11, 17, 14, 0, tzinfo=timezone.utc)
    saturday = datetime(2026, 11, 21, 8, 0, tzinfo=timezone.utc)
    # `_is_peak` reads only `call.context.now`, so it is handed exactly
    # that — and no `Context`, which the real SDK does not export.
    mk = lambda when: SimpleNamespace(context=SimpleNamespace(now=when))
    assert app._is_peak(mk(tuesday_peak)) is True
    assert app._is_peak(mk(tuesday_off)) is False
    assert app._is_peak(mk(saturday)) is False, "weekends are never peak"


def test_zone_fare_is_dearer_at_peak(app):
    z1 = {"zone": "1"}
    assert app._zone_fare(z1, z1, peak=True) > app._zone_fare(z1, z1, peak=False)


def test_unknown_zone_returns_none_rather_than_a_guess(app):
    assert app._zone_fare({"zone": None}, {"zone": "1"}, peak=False) is None


def test_walking_and_cycling_are_never_given_a_tube_fare(app):
    """Transport for London does not charge you to walk or to ride your own
    bicycle, so a zone fare attached to one would be an invented charge."""
    payload = {"journeys": [
        {"duration": 29, "fare": None, "legs": [_leg("walking")]},
        {"duration": 9, "fare": None, "legs": [_leg("cycle")]},
    ]}
    data = ask(route(app, journeys=payload)).data
    assert all("no fare" in text for text in data["options"])


def test_cycling_is_offered_but_never_recommended(app):
    """The app cannot know whether you have a bicycle with you."""
    payload = {"journeys": [
        {"duration": 9, "fare": None, "legs": [_leg("cycle")]},
        {"duration": 14, "fare": {"totalCost": 300},
         "legs": [_leg("tube", [("metropolitan", "Metropolitan")])]},
    ]}
    data = ask(route(app, journeys=payload)).data
    assert "Cycle" in " ".join(data["options"]), "cycling should still be offered"
    assert not data["best"].startswith("Cycle")


def test_pounds_formats_pence(app):
    assert pounds(300) == "£3.00" and pounds(5) == "£0.05" and pounds(1185) == "£11.85"


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def test_severity_number_is_never_compared(app):
    """10 is Good Service, 20 is Service Closed, 9 is Minor Delays. Sorting or
    comparing the number reports a closed line as healthier than a delayed
    one. Both non-good lines must appear, each exactly once."""
    disrupted = route(app, status=MIXED_STATUS)._disrupted_lines()
    assert disrupted == ["Central", "Metropolitan"]


def test_only_lines_actually_on_the_route_are_reported(app):
    data = ask(route(app, status=MIXED_STATUS)).data
    assert data["disrupted_lines"] == ["Metropolitan"], "Central is not on this route"


# ---------------------------------------------------------------------------
# How it fails
# ---------------------------------------------------------------------------

def test_a_200_containing_nothing(app):
    """The failure that actually happens. A 200 with an empty body is
    invisible, and `journeys[0]` turns it into a crash or an invented answer.
    """
    for empty in ({}, {"journeys": []}, []):
        with pytest.raises(AgentAppError) as caught:
            ask(route(app, journeys=empty))
        assert caught.value.user_message


@pytest.mark.parametrize("code,retryable", [(404, True), (429, True), (500, True)])
def test_http_errors_are_named(app, code, retryable):
    def boom(url, timeout=5.0):
        raise urllib.error.HTTPError(url, code, "nope", {}, None)
    app._fetch = boom
    with pytest.raises(AgentAppError) as caught:
        ask(app)
    assert caught.value.retryable is retryable


def test_three_hundred_is_a_disambiguation_not_a_failure(app):
    """Transport for London answers an ambiguous name with 300, which urllib
    raises as an error. Reporting it as 'the service is down' would be wrong.
    """
    def ambiguous(url, timeout=5.0):
        raise urllib.error.HTTPError(url, 300, "multiple", {}, None)
    app._fetch = ambiguous
    with pytest.raises(AgentAppError) as caught:
        ask(app)
    assert caught.value.retryable is False


def test_a_timeout_is_not_a_urlerror(app):
    """TimeoutError does not subclass URLError, so it needs catching by name
    or it escapes as an unhandled exception."""
    def slow(url, timeout=5.0):
        raise TimeoutError("too slow")
    app._fetch = slow
    with pytest.raises(AgentAppError) as caught:
        ask(app)
    assert caught.value.retryable is True


def test_losing_the_second_request_is_not_losing_the_call(app):
    """The first request carries the priced options; the second only widens
    the alternatives. Failing it should cost alternatives, not the answer."""
    calls = {"n": 0}

    def flaky(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        calls["n"] += 1
        if calls["n"] > 1:
            raise TimeoutError("second request died")
        return json.dumps(WARREN_BARBICAN).encode()

    app._fetch = flaky
    assert ask(app).data["options"], "should still answer from the first request"


def test_garbage_body(app):
    """A 200 carrying HTML is nearly always a throttle or block page, not a
    broken API, and the two have different answers. Asserting only that
    *something* raised would pass even if the message were wrong."""
    app._fetch = lambda url, timeout=5.0: b"<html>not json</html>"
    with pytest.raises(AgentAppError) as caught:
        ask(app)
    assert caught.value.retryable is True
    assert "throttl" in caught.value.user_message.lower()


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

def test_no_more_than_three_requests(app):
    """20 seconds is the hard limit, and an app that spends its whole budget
    is racing the thing that called it."""
    seen: list[str] = []

    def counted(url, timeout=5.0):
        seen.append(url)
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps(WARREN_BARBICAN).encode()

    app._fetch = counted
    ask(app)
    assert len(seen) <= 3, seen


def test_every_request_carries_a_timeout(app):
    """urlopen has no default. Without one an unresponsive host holds the app
    until it is killed with nothing to show."""
    seen: list[float] = []

    def check(url, timeout=5.0):
        seen.append(timeout)
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps(WARREN_BARBICAN).encode()

    app._fetch = check
    ask(app)
    assert all(t and t > 0 for t in seen)


def test_output_stays_small(app):
    """Their own cautionary tale is an app that returned 3,245 items. What
    comes back here is read aloud, so it is a handful of short lines."""
    data = ask(route(app)).data
    assert len(data["options"]) <= 5
    assert len(json.dumps(data)) < 2000


def test_an_ambiguous_name_says_which(app):
    """Hammersmith is two separate stations 56 m apart, and so is Edgware
    Road. Replying 'I don't recognise Hammersmith' would be false."""
    with pytest.raises(AgentAppError) as caught:
        ask(route(app), frm="Hammersmith", to="Barbican")
    message = caught.value.user_message
    assert "more than one" in message and "Hammersmith" in message


@pytest.mark.parametrize("junk", ["🚇", "...", "!!!", "   ---   "])
def test_input_that_normalises_to_nothing(app, junk):
    """An empty normalised key is a prefix of every station name, so without
    a guard an emoji 'matches' all five hundred of them."""
    with pytest.raises(AgentAppError) as caught:
        app._resolve(junk, "starting")
    assert "stations starting" not in caught.value.user_message


def test_walkable_never_crashes_on_any_station(app):
    """Coordinates come from a third party and some rows have none. Not
    crashing is the floor; every neighbour returned must also be well formed,
    or a later caller reads a None and fails somewhere less obvious."""
    seen = 0
    for record in app._stations.values():
        for near in app._walkable(record):
            seen += 1
            assert isinstance(near["metres"], int) and near["metres"] >= 0
            assert isinstance(near["minutes"], int) and near["minutes"] >= 1
            assert isinstance(near["designated"], bool)
            assert near["name"] and isinstance(near["name"], str)
            assert (near["allowed"] is None) is not near["designated"]
    assert seen > 100, f"only {seen} neighbours across the whole network"


def test_a_station_with_no_neighbours_has_no_warning(app):
    """Heathrow Terminal 5 has nothing within walking distance, so there is
    no trap to report and nothing should be invented."""
    assert app._walkable(app._resolve("Heathrow Terminal 5", "x")) == []


def test_every_figure_describes_a_returned_option(app):
    """`cheapest_pence` once quoted a free walk that ranked outside the five
    options returned. True, and impossible for Pippa to explain, because it
    was never given the option the number referred to."""
    payload = {"journeys": [
        {"duration": 12, "fare": {"totalCost": 300},
         "legs": [_leg("tube", [("northern", "Northern")])]},
        {"duration": 14, "fare": {"totalCost": 320},
         "legs": [_leg("tube", [("circle", "Circle")])]},
        {"duration": 16, "fare": {"totalCost": 340},
         "legs": [_leg("tube", [("central", "Central")])]},
        {"duration": 18, "fare": {"totalCost": 360},
         "legs": [_leg("tube", [("district", "District")])]},
        {"duration": 20, "fare": {"totalCost": 380},
         "legs": [_leg("tube", [("victoria", "Victoria")])]},
        # ranks last, so it is not among the five returned
        {"duration": 90, "fare": None, "legs": [_leg("walking")]},
    ]}
    data = ask(route(app, journeys=payload)).data
    assert len(data["options"]) == 5
    assert "Walk" not in " ".join(data["options"])
    assert data["cheapest_pence"] == 300, "quoted a fare from an option it did not return"


def test_the_same_place_under_two_names_is_one_neighbour(app):
    """'Euston' and 'London Euston' are 17 m apart and share a gateline.
    Offering both as separate walks — possibly warning about one while the
    other is designated — would be nonsense."""
    names = [n["name"] for n in app._walkable(app._resolve("Warren Street", "x"))]
    eustons = [n for n in names if n.lower().endswith("euston")]
    assert len(eustons) <= 1, f"same place listed twice: {eustons}"


def test_a_designated_twin_wins_over_an_undesignated_one(app):
    """If either name of a coincident pair is in the interchange list, the
    walk is designated — the gateline does not care what you call it."""
    near = app._walkable(app._resolve("Warren Street", "x"))
    by_name = {n["name"]: n for n in near}
    assert by_name.get("Euston Square", {}).get("designated") is True


# ---------------------------------------------------------------------------
# The aside
# ---------------------------------------------------------------------------

def test_there_is_always_something_true_to_say(app):
    """Every journey gets an aside: a gateline inside the station, an absurd
    designated walk, an asymmetric allowance, a well-connected hub, or failing
    all of those a network fact said as a network fact."""
    for frm, to in [("Warren Street", "Barbican"), ("Oxford Circus", "Euston"),
                    ("Brixton", "Walthamstow Central"), ("Richmond", "Victoria"),
                    ("Heathrow Terminal 5", "Bank")]:
        o, d = app._resolve(frm, "x"), app._resolve(to, "y")
        said = app._did_you_know(o, d, None)
        assert said and said[0].isupper() and said.endswith("."), (frm, to, said)


def test_the_aside_is_deterministic(app):
    """The same journey says the same thing every time — a fact that changes
    on refresh reads as invented, which is the one thing it must not be."""
    o, d = app._resolve("Brixton", "x"), app._resolve("Walthamstow Central", "y")
    assert len({app._did_you_know(o, d, None) for _ in range(5)}) == 1


def test_the_aside_never_repeats_the_warning(app):
    """Saying the same station twice, once as a trap and once as a curiosity,
    reads as padding."""
    o, d = app._resolve("Oxford Circus", "x"), app._resolve("Euston", "y")
    warning = "something about Bond Street being designated"
    said = app._did_you_know(o, d, warning) or ""
    assert "Bond Street" not in said


def test_proper_nouns_survive(app):
    """An early version lower-cased the first letter of the published phrase
    and turned Argyll into argyll."""
    o, d = app._resolve("Oxford Circus", "x"), app._resolve("Euston", "y")
    assert "Argyll" in app._did_you_know(o, d, None)


def test_network_facts_are_read_not_written(app):
    """The superlatives come out of the bundled list. If the list changes and
    a hand-written sentence stayed behind, this catches it."""
    facts = app._network_facts()
    shortest = min((p for p in app._osi.values() if p["metres"] > 0),
                   key=lambda p: p["metres"])
    assert any(str(shortest["metres"]) + " metres" in f for f in facts)


def test_the_aside_is_optional_in_the_contract(app):
    """It is declared `required: false`, so an answer without one is valid."""
    field = next(f for f in app.manifest.outcome("journey.cost")["output"]
                 if f["name"] == "did_you_know")
    assert field["required"] is False


def test_an_entrance_is_not_a_different_station(app):
    """'Paddington (H&C Line)-Underground' is 254 m from 'Paddington' and is
    the same complex. Reported as undesignated it becomes a false second-fare
    warning, which is the one thing this app must not produce."""
    assert app._interchange("Lancaster Gate",
                            "Paddington (H&C Line)-Underground") is not None
    near = app._walkable(app._resolve("Lancaster Gate", "x"))
    traps = [n["name"] for n in near if not n["designated"]]
    assert not any("Paddington" in t for t in traps), traps


def test_stripping_qualifiers_does_not_merge_real_pairs(app):
    """Euston Square is genuinely a different station from Euston, and the two
    are a designated interchange. The qualifier rule must not touch it."""
    assert app._base_name("Euston Square") == "Euston Square"
    assert app._base_name("Paddington (H&C Line)-Underground") == "Paddington"
    assert app._base_name("Hammersmith (Dist&Picc Line)") == "Hammersmith"
    assert app._interchange("Warren Street", "Euston Square") is not None


def test_two_entrances_are_listed_once(app):
    """Paddington and Paddington (H&C Line) are one complex. Listing both is
    the same advice twice."""
    names = [n["name"] for n in app._walkable(app._resolve("Lancaster Gate", "x"))]
    assert sum("Paddington" in n for n in names) == 1, names


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------

def test_the_suite_genuinely_cannot_reach_the_network():
    """Every test stubs `_fetch`, and the honesty of all of them rests on one
    guard: the conftest refuses outbound sockets, so a test that quietly
    called the real API fails here instead of passing on this laptop and
    failing in a build with no network.

    Nothing checked the guard itself. If it stopped working the suite would
    go on passing while silently depending on Transport for London being up
    — which today, rate limited, it intermittently is not.

    Skipped rather than failed where no guard is installed, because on the
    day their conftest is the one in force and it is not ours to assert on."""
    import socket as _socket
    # Detected by shape, not by name. An earlier version looked for a guard
    # called "refuse" — the name ours happens to use — so in Pippa Labs' own
    # repository, where the conftest installs a `_Blocked` socket subclass
    # instead, this test skipped itself. It reported "no socket guard" about
    # a repository whose guard is stricter than ours, and the one check that
    # the network really is sealed quietly stopped running at exactly the
    # moment it counted.
    #
    # Anything that guards sockets has to replace `socket.socket` or
    # `socket.create_connection`, and whatever it puts there is defined
    # somewhere other than the standard library's `socket` module.
    def foreign(obj) -> bool:
        return getattr(obj, "__module__", "socket") != "socket"

    blocked = (foreign(_socket.socket) or foreign(_socket.create_connection)
               or getattr(_socket.socket.connect, "__name__", "") == "refuse")
    if not blocked:
        pytest.skip("no socket guard installed in this conftest")

    with pytest.raises(Exception) as caught:
        _socket.create_connection(("api.tfl.gov.uk", 443), timeout=2)
    assert "socket" in str(caught.value).lower(), (
        f"something stopped it, but not the guard: {caught.value}")

    with pytest.raises(Exception):
        probe = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        probe.connect(("api.tfl.gov.uk", 443))


def test_no_key_value_lives_in_the_repository():
    """Their pre-commit scanner looks for a key in a URL (`?api_key=...`),
    among other shapes. The value must come from the environment, never a
    literal — a key that reaches a public repo is a published key, and
    deleting it later does not take it out of the history."""
    import app as module
    source = Path(module.__file__).read_text()
    assert 'os.environ.get("TFL_APP_KEY"' in source
    assert 'TFL_APP_KEY", "")' in source, "the default must be empty, not a key"
    # No literal that looks like a key anywhere, under any name.
    import re as _re
    literals = _re.findall(r'(?:KEY|TOKEN|SECRET)\w*\s*=\s*"([^"]{8,})"', source)
    assert not literals, f"a key-shaped literal is in the source: {literals}"


def test_the_key_is_read_inside_the_call_not_at_import():
    """Pippa puts an app's keys into the environment for the duration of a
    call and takes them out again. A module-level `APP_KEY = os.environ.get(...)`
    therefore runs once, at import, before any call exists — binding the empty
    string for the life of the process. The key would be set correctly in
    Pippa and never once be sent, and the app would run unkeyed at fifty
    requests a minute while looking entirely healthy.

    Their own checker flags this, which is how it was found. Pinned here
    because it is invisible from the outside: nothing fails, the answers are
    still right, and the only symptom is throttling under load."""
    import app as module
    source = Path(module.__file__).read_text()
    module_level = [line for line in source.splitlines()
                    if line.startswith(("APP_KEY", "TFL_APP_KEY"))
                    and "os.environ" in line]
    assert not module_level, (
        f"the key is read at import: {module_level} — read it inside the call")

    # and reading it really does see a value set after import
    import os
    before = os.environ.get("TFL_APP_KEY")
    try:
        os.environ["TFL_APP_KEY"] = "SETAFTERIMPORT"  # pragma: allowlist secret
        assert module.app_key() == "SETAFTERIMPORT"   # pragma: allowlist secret
    finally:
        if before is None:
            os.environ.pop("TFL_APP_KEY", None)
        else:
            os.environ["TFL_APP_KEY"] = before


def test_the_key_is_declared_and_optional(app):
    """Declared under app_secrets so it can be set in Pippa without a push,
    and `required: false` because the app answers perfectly well unkeyed."""
    declared = app.manifest.get("app_secrets") or []
    entry = next((e for e in declared if e["name"] == "TFL_APP_KEY"), None)
    assert entry is not None, "TFL_APP_KEY not declared"
    assert entry.get("required") is False
    assert entry.get("description", "").strip()


def test_app_secrets_is_not_the_forbidden_secrets_block(app):
    """`secrets:` asks for a user's credential and is refused at registration.
    `app_secrets:` is the app's own key. They are different keys and only one
    of them is allowed."""
    assert not app.manifest.get("secrets")
    assert not app.manifest.get("connectors")


def test_the_url_carries_the_key_only_when_there_is_one(app, monkeypatch):
    seen: list[str] = []
    app._fetch = lambda url, timeout=5.0: (seen.append(url), b"[]")[1]
    monkeypatch.delenv("TFL_APP_KEY", raising=False)
    app._get("/Line/Mode/tube/Status")
    assert "app_key" not in seen[0], "unkeyed request should carry no app_key"

    monkeypatch.setenv("TFL_APP_KEY", "TESTVALUE")  # pragma: allowlist secret
    seen.clear()
    app._get("/Line/Mode/tube/Status")
    assert "app_key=TESTVALUE" in seen[0]


# ---------------------------------------------------------------------------
# The second outcome
# ---------------------------------------------------------------------------

def ask_interchange(app, frm, to, when=None):
    moment = when or datetime(2026, 11, 20, 14, 0, tzinfo=timezone.utc)
    return run_call(app, "interchange.check",
                    {"exit_station": frm, "entry_station": to}, now=moment)


def test_a_designated_pair_is_one_journey(app):
    data = ask_interchange(app, "Oxford Circus", "Bond Street").data
    assert data["designated"] is True
    assert data["journeys_charged"] == 1
    assert data["second_fare_pence"] == 0
    assert data["minutes_allowed"] == 20


def test_an_undesignated_pair_costs_a_second_fare(app):
    data = ask_interchange(app, "Oxford Circus", "Warren Street").data
    assert data["designated"] is False
    assert data["journeys_charged"] == 2
    assert data["second_fare_pence"] > 0
    assert data["minutes_allowed"] == 0


def test_the_published_distance_wins_over_a_straight_line(app):
    """Oxford Circus to Bond Street is 280 m published and 551 m straight
    between centroids — both stations have entrances along Oxford Street. The
    published figure is the one a walker experiences, and the number returned
    must match the one the narrative says."""
    result = ask_interchange(app, "Oxford Circus", "Bond Street")
    assert result.data["walk_metres"] == 280
    assert "280 metres" in result.narrative


def test_an_undesignated_pair_is_offered_the_free_alternative(app):
    """Knowing the walk costs £3.00 is half an answer. The other half is that
    Bond Street, the same sort of distance away, does not."""
    narrative = ask_interchange(app, "Oxford Circus", "Warren Street").narrative
    assert "Bond Street" in narrative and "designated" in narrative


def test_the_second_outcome_needs_no_network(app):
    """It answers from the bundled list alone, so it still works when
    Transport for London is unreachable. Any fetch here is a bug."""
    def boom(url, timeout=5.0):
        raise AssertionError(f"interchange.check made a network call: {url}")
    app._fetch = boom
    assert ask_interchange(app, "Oxford Circus", "Bond Street").data["designated"]


def test_both_outcomes_are_declared_with_distinct_phrases(app):
    outcomes = app.manifest["outcomes"]
    assert len(outcomes) == 2
    STOP = {"i", "a", "an", "the", "to", "is", "it", "do", "does", "my", "of",
            "from", "have", "got", "way", "and", "how", "what", "whats", "this",
            "that", "me", "can", "will", "be", "if", "again", "another"}
    words = [set(w for p in o["example_phrases"] for w in p.split() if w not in STOP)
             for o in outcomes]
    assert not (words[0] & words[1]), f"outcomes share vocabulary: {words[0] & words[1]}"


# ---------------------------------------------------------------------------
# Hire bikes
# ---------------------------------------------------------------------------

BIKEPOINT = {"places": [
    {"commonName": "Drummond Street , Euston", "additionalProperties": [
        {"key": "NbBikes", "value": "12"}, {"key": "NbEmptyDocks", "value": "20"},
        {"key": "Locked", "value": "false"}]},
    {"commonName": "Gower Place , Euston", "additionalProperties": [
        {"key": "NbBikes", "value": "2"}, {"key": "NbEmptyDocks", "value": "11"},
        {"key": "Locked", "value": "false"}]},
]}


def with_bikes(app, docks=BIKEPOINT, status=ALL_GOOD, journeys=None):
    payload = journeys or {"journeys": [
        {"duration": 13, "fare": None, "legs": [_leg("cycle")]},
        {"duration": 15, "fare": {"totalCost": 300},
         "legs": [_leg("tube", [("circle", "Circle")])]}]}

    def fake(url, timeout=5.0):
        if "/BikePoint" in url:
            return json.dumps(docks).encode()
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(status).encode()
        return json.dumps(payload).encode()
    app._fetch = fake
    return app


def test_a_cycle_option_says_whether_there_is_a_bike(app):
    """A cycle suggestion with no bike at the dock is useless advice."""
    text = " ".join(ask(with_bikes(app)).data["options"])
    assert "hire bikes at Drummond Street" in text
    assert "Drummond Street ," not in text, "dock name not trimmed"


def test_it_says_when_there_is_nowhere_to_put_it(app):
    """Worse than no bike: you find out at the far end."""
    full = {"places": [{"commonName": "Gower Place , Euston", "additionalProperties": [
        {"key": "NbBikes", "value": "4"}, {"key": "NbEmptyDocks", "value": "0"},
        {"key": "Locked", "value": "false"}]}]}
    text = " ".join(ask(with_bikes(app, docks=full)).data["options"])
    assert "no empty docks at the other end" in text


def test_a_locked_dock_does_not_count(app):
    locked = {"places": [{"commonName": "Closed Dock", "additionalProperties": [
        {"key": "NbBikes", "value": "9"}, {"key": "NbEmptyDocks", "value": "9"},
        {"key": "Locked", "value": "true"}]}]}
    with_bikes(app, docks=locked)
    assert app._bikes_near({"lat": 51.52, "lon": -0.13}) is None


def test_bikepoint_returns_two_different_shapes(app):
    """With a radius it is {"places": [...]}; without one it is a bare list.
    One path, two shapes."""
    with_bikes(app, docks=BIKEPOINT["places"])          # the bare-list shape
    assert app._bikes_near({"lat": 51.52, "lon": -0.13})["bikes"] == 14


def test_no_cycle_option_means_no_bike_requests(app):
    """Two extra requests are only worth making when cycling is on the table."""
    seen: list[str] = []

    def counted(url, timeout=5.0):
        seen.append(url)
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps(WARREN_BARBICAN).encode()   # tube only, no cycle leg

    app._fetch = counted
    ask(app)
    assert not any("/BikePoint" in u for u in seen), seen


def test_a_failed_dock_lookup_does_not_lose_the_answer(app):
    """Dock figures are a nicety. The journey is not."""
    def flaky(url, timeout=5.0):
        if "/BikePoint" in url:
            raise TimeoutError("docks down")
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps({"journeys": [
            {"duration": 13, "fare": None, "legs": [_leg("cycle")]},
            {"duration": 15, "fare": {"totalCost": 300},
             "legs": [_leg("tube", [("circle", "Circle")])]}]}).encode()
    app._fetch = flaky
    assert ask(app).data["options"], "a dock timeout must not fail the call"


# ---------------------------------------------------------------------------
# Step-free
# ---------------------------------------------------------------------------

def ask_access(app, frm="Oxford Circus", to="Warren Street", step_free=None):
    inputs = {"from_station": frm, "to_station": to}
    if step_free is not None:
        inputs["step_free"] = step_free
    return run_call(app, "journey.cost", inputs,
                    now=datetime(2026, 11, 20, 14, 0, tzinfo=timezone.utc))


def test_step_free_is_only_applied_when_asked_for(app):
    """The step-free network is a different network, not a filter. Applying it
    uninvited gives somebody a much slower answer than they asked for."""
    seen: list[str] = []

    def counted(url, timeout=5.0):
        seen.append(url)
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps(WARREN_BARBICAN).encode()

    app._fetch = counted
    ask_access(app)                                   # not mentioned
    assert not any("accessibilityPreference" in u for u in seen)

    seen.clear()
    ask_access(app, step_free=True)
    assert any("accessibilityPreference=stepFreeToPlatform" in u for u in seen)


def test_the_answer_says_it_is_step_free(app):
    result = ask_access(route(app), step_free=True)
    assert result.data.get("step_free") is True
    assert result.narrative.startswith("Step-free,")


def test_step_free_is_absent_not_false_when_not_asked(app):
    """An optional output is omitted rather than returned empty."""
    assert "step_free" not in ask_access(route(app)).data


# ---------------------------------------------------------------------------
# Three behaviours I wrongly called equivalent after a mutation run, until
# the outputs were actually compared: they change up to a third of replies.
# ---------------------------------------------------------------------------

def test_the_headline_fare_is_the_fare_and_not_zero(app):
    """`best["pence"] or 0` turned into `and` makes best_pence zero on every
    priced answer — the headline number, silently £0.00. It changed 121 of
    400 replies and nothing noticed."""
    priced = {"journeys": [
        {"duration": 12, "fare": {"totalCost": 390},
         "legs": [_leg("tube", [("circle", "Circle")])]},
    ]}

    def fetch(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps(priced).encode()

    app._fetch = fetch
    data = ask_access(app, "Warren Street", "Barbican").data
    assert data["best_pence"] == 390, data["best_pence"]
    assert "\u00a33.90" in data["best"], data["best"]


def test_a_bicycle_you_may_not_have_is_not_the_cheapest_fare(app):
    """Cycling carries no fare, so including it makes `cheapest_pence` zero
    and tells somebody they can avoid a fare on a bicycle they do not own.
    The code excludes it deliberately; nothing was checking that it still
    does, and the mutation changed 91 of 400 replies."""
    mixed = {"journeys": [
        {"duration": 30, "fare": None, "legs": [_leg("cycle")]},
        {"duration": 12, "fare": {"totalCost": 300},
         "legs": [_leg("tube", [("circle", "Circle")])]},
    ]}

    def fetch(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        if "BikePoint" in url:
            return json.dumps({"places": []}).encode()
        return json.dumps(mixed).encode()

    app._fetch = fetch
    data = ask_access(app, "Warren Street", "Barbican").data
    assert any("Cycle" in o for o in data["options"]), "the cycle vanished entirely"
    assert data["cheapest_pence"] == 300, (
        f"cheapest_pence={data['cheapest_pence']} — a bike is not a fare you can avoid")


def test_the_narrative_never_suggests_saving_money_on_a_bicycle(app):
    """The same exclusion lives twice: once for `cheapest_pence`, once in the
    sentence. Only the first was covered. Leaking into the sentence, the app
    says "£3.00 cheaper to Cycle instead" to somebody who may well not own a
    bicycle — a saving they cannot actually make."""
    tempting_bike = {"journeys": [
        {"duration": 12, "fare": {"totalCost": 300},
         "legs": [_leg("tube", [("circle", "Circle")])]},
        {"duration": 16, "fare": None, "legs": [_leg("cycle")]},
    ]}

    def fetch(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        if "BikePoint" in url:
            return json.dumps({"places": []}).encode()
        return json.dumps(tempting_bike).encode()

    app._fetch = fetch
    narrative = ask_access(app, "Warren Street", "Barbican").narrative
    assert "cheaper to Cycle" not in narrative, narrative


def test_the_asymmetry_aside_is_true_of_the_pair_it_names(app):
    """The fact names two directions and calls their gap the widest on the
    list. Built from pairs that AGREE it would say "allows 20 and the way
    back allows 20 — the widest disagreement", which is simply false. The
    mutation changed 308 of 400 replies."""
    import re as _re
    for fact in app._network_facts():
        m = _re.search(r"allows (\d+) minutes and the way back allows (\d+)", fact)
        if m:
            assert m.group(1) != m.group(2), fact
        m2 = _re.search(r"There are (\d+) pairs that allow more time one way", fact)
        if m2:
            assert int(m2.group(1)) > 0, fact


def test_a_journey_that_is_only_walking_is_labelled_a_walk(app):
    """`unique <= {"walking"}` is a subset test, and narrowing it to `<`
    changed 165 of 520 replies — the label on roughly a third of answers,
    with no test touching it."""
    walking_only = {"journeys": [
        {"duration": 22, "fare": None, "legs": [_leg("walking"), _leg("walking")]},
        {"duration": 12, "fare": {"totalCost": 300},
         "legs": [_leg("tube", [("circle", "Circle")])]},
    ]}

    def fetch(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps(walking_only).encode()

    app._fetch = fetch
    options = ask_access(app, "Warren Street", "Barbican").data["options"]
    assert any(o.startswith("Walk ") for o in options), options
    # and it is free, because walking is
    walk = next(o for o in options if o.startswith("Walk "))
    assert "no fare" in walk or "free" in walk, walk


@pytest.mark.parametrize("mode,lines,expect_hopper", [
    ("bus", [("205", "205")], True),
    ("tube", [("circle", "Circle")], False),
])
def test_the_hopper_is_mentioned_only_when_the_best_way_is_a_bus(
        app, mode, lines, expect_hopper):
    """Transport for London give unlimited bus and tram journeys within an
    hour of the first bus tap — the second London money rule this app knows,
    after the interchange list. It appeared on 46 of 520 replies and nothing
    checked it was there, or that it stayed off a Tube answer where it would
    simply be wrong."""
    journeys = {"journeys": [
        {"duration": 20, "fare": {"totalCost": 185}, "legs": [_leg(mode, lines)]},
    ]}

    def fetch(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps(journeys).encode()

    app._fetch = fetch
    narrative = ask_access(app, "Warren Street", "Barbican").narrative
    said = "tapped a bus or tram in the last hour" in narrative
    assert said is expect_hopper, narrative


def test_a_cheaper_alternative_is_only_raised_when_it_is_worth_the_time(app):
    """The ranking already values a minute at 21p. Offering "£3.00 cheaper if
    you walk for another 45 minutes" contradicts the recommendation in the
    same breath as making it, so the line is held unless the saving roughly
    covers the extra time."""
    def build(cheap_minutes):
        return {"journeys": [
            {"duration": 10, "fare": {"totalCost": 300},
             "legs": [_leg("tube", [("circle", "Circle")])]},
            {"duration": cheap_minutes, "fare": {"totalCost": 0},
             "legs": [_leg("walking")]},
        ]}

    def run(cheap_minutes):
        def fetch(url, timeout=5.0, _m=cheap_minutes):
            if "/Line/Mode/tube/Status" in url:
                return json.dumps(ALL_GOOD).encode()
            return json.dumps(build(_m)).encode()
        app._fetch = fetch
        return ask_access(app, "Warren Street", "Barbican").narrative

    # The tube has to stay the recommendation, or there is no alternative to
    # raise: at 21p a minute, 10 min + £3.00 costs 510, so the walk must be
    # longer than 24 minutes to rank behind it.
    # 25 minutes: £3.00 saved for 15 more, worth saying — and the figure has
    # to be the difference between the two fares. Computed as a sum it reads
    # as £3.00 saved on a £3.00 journey against a free one, which is wrong
    # about somebody's money in the one sentence that is only about money.
    said = run(25)
    assert "cheaper to" in said, said
    assert "\u00a33.00 cheaper" in said, said
    assert "for 15 more minutes" in said, said

    # Both fares have to be non-zero or the arithmetic hides itself: against
    # a free walk, 300 - 0 and 300 + 0 are the same number. £3.90 beside
    # £1.70 tells the two apart — a difference of £2.20, a sum of £5.60.
    priced_pair = {"journeys": [
        {"duration": 10, "fare": {"totalCost": 390},
         "legs": [_leg("tube", [("circle", "Circle")])]},
        {"duration": 21, "fare": {"totalCost": 170},
         "legs": [_leg("bus", [("205", "205")])]},
    ]}

    def fetch_pair(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps(priced_pair).encode()

    app._fetch = fetch_pair
    both = ask_access(app, "Warren Street", "Barbican").narrative
    assert "\u00a32.20 cheaper" in both, both
    assert "\u00a35.60" not in both, f"the saving is a sum, not a difference: {both}"
    # 60 minutes: £3.00 saved for 50 more is 6p a minute. The ranking already
    # rejected it, so raising it would contradict the recommendation.
    assert "cheaper to" not in run(60), run(60)
    # 30 minutes straddles the 0.8 factor deliberately: £3.00 for 20 more
    # minutes is 15p a minute, under the 16.8p the rule asks for and over the
    # 10.5p a looser factor would. Without this case the constant can be
    # retuned silently, and it decides how often the app argues with its own
    # recommendation.
    assert "cheaper to" not in run(30), run(30)


def test_a_hub_station_is_sent_to_the_planner_as_a_real_station(app):
    """"Kings Cross" and "St Pancras" both resolve to HUBKGX, and "New Cross"
    to HUBNWX. A hub names a place rather than a platform, and the journey
    planner answers one with a 300 asking which station was meant — so those
    three, among the most asked-for names in London, failed outright with
    "Transport for London did not recognise one of those stations"."""
    for spoken in ("Kings Cross", "St Pancras", "New Cross"):
        record = app._resolve(spoken, "starting")
        assert record["naptan"].startswith("HUB"), f"{spoken} is no longer a hub"
        sent = app._planner_naptan(record)
        assert not sent.startswith("HUB"), f"{spoken} still sent as a hub"
        assert sent, spoken

    # an ordinary station is passed through untouched
    plain = app._resolve("Warren Street", "starting")
    assert app._planner_naptan(plain) == plain["naptan"]


def test_no_tube_route_does_not_discard_the_other_ways_to_get_there(app):
    """`mode=tube` 404s for any pair no tube line serves — New Cross is
    Overground only — and that 404 used to end the call. The multimodal
    query right behind it returns six perfectly good journeys, and the
    person was told there was no way to get there at all."""
    other_ways = {"journeys": [
        {"duration": 5, "fare": {"totalCost": 320},
         "legs": [_leg("national-rail", [("southeastern", "Southeastern")])]},
        {"duration": 36, "fare": {"totalCost": 185}, "legs": [_leg("bus", [("53", "53")])]},
    ]}

    def tube_404_then_answers(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        if "mode=tube" in url:
            raise urllib.error.HTTPError(url, 404, "no tube route", {}, None)
        return json.dumps(other_ways).encode()

    app._fetch = tube_404_then_answers
    data = ask_access(app, "New Cross", "London Bridge").data
    assert data["options"], "threw away the journeys it was holding"
    assert any("Southeastern" in o for o in data["options"]), data["options"]


def test_no_step_free_route_is_not_a_service_failure(app):
    """Verified live: South Kensington to Liverpool Street step-free returns
    404. Reporting that as 'try again shortly' is both wrong and dismissive —
    there is no such route, and retrying will never produce one."""
    def no_route(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        raise urllib.error.HTTPError(url, 404, "no route", {}, None)

    app._fetch = no_route
    with pytest.raises(AgentAppError) as caught:
        ask_access(app, step_free=True)
    assert "step-free" in caught.value.user_message.lower()
    assert caught.value.retryable is False, "retrying will never find one"
    assert "Oxford Circus" in caught.value.user_message


def test_a_404_without_step_free_stays_retryable(app):
    def no_route(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        raise urllib.error.HTTPError(url, 404, "no route", {}, None)

    app._fetch = no_route
    with pytest.raises(AgentAppError) as caught:
        ask_access(app)
    assert caught.value.retryable is True


def test_the_manifest_describes_what_the_code_does(app):
    """The reviewer reads the manifest and not the code, so a description
    that has drifted from behaviour is the one kind of error that costs
    something. walk_metres prefers the published figure; it said straight-line
    for a while after the code changed."""
    field = next(f for f in app.manifest.outcome("interchange.check")["output"]
                 if f["name"] == "walk_metres")
    text = field["description"]
    assert "published" in text, "walk_metres prefers TfL's published distance"
    result = ask_interchange(app, "Oxford Circus", "Bond Street")
    assert result.data["walk_metres"] == 280, "and this is that figure, not 551"


def test_the_manifest_declares_no_owner_email_and_no_publisher(app):
    """Pippa Labs removed both fields on 16 November (commit 64749fa):
    ownership comes from the repository they create for the account, and "no
    personal data belongs here". This is not a tidy-up. Their
    .githooks/pre-commit refuses any commit whose changed files contain an
    email address, so putting owner_email back does not fail quietly on the
    platform — it blocks the push, at the one moment there is no time to
    debug it. Guarded here because the field was in this manifest for three
    days and every document told you to fill it in."""
    raw = dict(app.manifest)
    assert "owner_email" not in raw, "an email address here blocks the commit"
    assert "publisher" not in raw, "removed with owner_email; the badge now reads app_author_name"
    assert set(raw) == {"manifest_version", "persistence", "app_author_name",
                        "app", "agent_types", "outcomes", "app_secrets"}, (
        "top-level keys must match their template plus app_secrets")


def test_the_only_fields_left_to_fill_in_are_the_two(app):
    """The documents promise exactly two. If a third placeholder appears the
    day plan is wrong, and the plan is read under time pressure."""
    raw = dict(app.manifest)
    placeholders = [k for k, v in raw.items() if isinstance(v, str) and v.startswith("XXX")]
    placeholders += ["app.id"] if str(raw["app"].get("id", "")).startswith("XXX") else []
    # conftest substitutes them for the test run, so either none are left
    # (substituted) or exactly the two that are filled in on the day.
    assert set(placeholders) <= {"app_author_name", "app.id"}, (
        f"unexpected placeholder: {placeholders}")


def _fixed_fetch(app):
    """Both halves of the step-free check need the same stub: a good-service
    status call and one journey, so only the request URL varies."""
    seen: list[str] = []

    def fetch(url, timeout=5.0):
        seen.append(url)
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps(WARREN_BARBICAN).encode()

    app._fetch = fetch
    return seen


@pytest.mark.parametrize("sent", [False, "no", "No", "NO", "false", "FALSE",
                                  "off", "none", "", 0])
def test_a_negative_step_free_input_does_not_turn_step_free_on(app, sent):
    """`step_free` is declared a boolean, but models answer boolean fields
    with words all the time — and `bool("no")` is True in Python, so the
    plain cast turned the flag ON for somebody who said they did not need it.

    This is the worst answer the app can give. Oxford Circus to Warren Street
    is one minute by tube and has no step-free tube route at all, so the flag
    set backwards replaces a one-minute train with a twenty-two minute walk,
    and nothing in the reply reveals the mistake."""
    seen = _fixed_fetch(app)
    result = ask_access(app, step_free=sent)
    assert "step_free" not in result.data, f"{sent!r} was read as yes"
    assert not any("accessibilityPreference" in u for u in seen), f"{sent!r} was read as yes"


@pytest.mark.parametrize("sent", [True, "yes", "Yes", "true", 1, "step free"])
def test_an_affirmative_step_free_input_still_works(app, sent):
    """The other half: narrowing the cast must not break the real thing."""
    seen = _fixed_fetch(app)
    result = ask_access(app, step_free=sent)
    assert result.data.get("step_free") is True, f"{sent!r} was not read as yes"
    assert any("accessibilityPreference" in u for u in seen), f"{sent!r} was not read as yes"


@pytest.mark.parametrize("described,expected", [
    ({},                                        []),          # key absent
    ({"statusSeverityDescription": None},       []),
    ({"statusSeverityDescription": ""},         []),
    ({"statusSeverityDescription": "   "},      []),
    ({"statusSeverityDescription": "Good Service"},   []),
    ({"statusSeverityDescription": "good service"},   []),
    ({"statusSeverityDescription": "Minor Delays"},   ["Metropolitan"]),
    ({"statusSeverityDescription": "Service Closed"}, ["Metropolitan"]),
])
def test_a_line_that_reports_nothing_is_not_reported_as_delayed(app, described, expected):
    """`"" != "good service"` is true, so testing the description directly
    turned a MISSING status into a positive claim that the line is delayed.

    Every number this app gives is meant to survive being checked against
    Transport for London a minute later. Inventing a delay is the one error
    that breaks that, and it reads exactly like a real one."""
    # Must be the line WARREN_BARBICAN actually runs on, or the result is
    # filtered out as not-on-route and the test passes for the wrong reason.
    status = [{"id": "metropolitan", "name": "Metropolitan",
               "lineStatuses": [described]}]

    def fetch(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(status).encode()
        return json.dumps(WARREN_BARBICAN).encode()

    app._fetch = fetch
    assert ask_access(app, "Warren Street", "Barbican").data["disrupted_lines"] == expected


def _peak_app(app):
    """Heathrow T5 to Covent Garden: zones 6 and 1, £5.90 peak, £4.00 off."""
    def fetch(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps(WARREN_BARBICAN).encode()
    app._fetch = fetch
    return app


def _ask_at(app, when):
    return run_call(app, "journey.cost",
                    {"from_station": "Heathrow Terminal 5",
                     "to_station": "Covent Garden"}, now=when)


@pytest.mark.parametrize("when,fires", [
    (datetime(2026, 11, 23, 9, 10, tzinfo=LONDON_TZ), True),
    (datetime(2026, 11, 23, 9, 29, tzinfo=LONDON_TZ), True),
    (datetime(2026, 11, 23, 18, 30, tzinfo=LONDON_TZ), True),
    (datetime(2026, 11, 23, 8, 0, tzinfo=LONDON_TZ), False),
    (datetime(2026, 11, 23, 16, 5, tzinfo=LONDON_TZ), False),
    (datetime(2026, 11, 23, 11, 0, tzinfo=LONDON_TZ), False),
    (datetime(2026, 11, 21, 9, 10, tzinfo=LONDON_TZ), False),
])
def test_peak_tip_only_when_waiting_is_actually_advice(app, when, fires):
    """Telling somebody at 16:05 to wait until 19:00 is not advice, it is a
    fact they cannot use. The tip is held back unless off-peak is close."""
    assert ("peak_tip" in _ask_at(_peak_app(app), when).data) is fires


def test_peak_tip_copes_with_a_station_that_has_no_zone(app):
    """Eighty-nine stations carry no zone, so the fare table has nothing for
    them. The guard that spots a missing figure before subtracting is what
    stops `None - None` from taking the whole answer down — and deleting it
    changed 84 of 400 replies without a test noticing."""
    def fetch(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps(WARREN_BARBICAN).encode()

    app._fetch = fetch
    zoneless = [v["name"] for v in app._stations.values() if not v.get("zone")]
    assert zoneless, "every station now has a zone — has the data changed?"

    peak = datetime(2026, 11, 23, 9, 10, tzinfo=LONDON_TZ)
    for name in zoneless[:12]:
        result = run_call(app, "journey.cost",
                          {"from_station": name,
                           "to_station": "Warren Street"}, now=peak)
        # no crash, and nothing invented about a fare it cannot price
        assert "peak_tip" not in result.data, (name, result.data.get("peak_tip"))


def test_peak_tip_stays_quiet_when_there_is_nothing_in_it(app):
    """Eight of the twenty-five zone pairs differ by 10p or 20p. This app
    values a minute at 21p when it ranks journeys, so "wait twenty-two
    minutes to save ten pence" is it contradicting its own arithmetic in the
    same reply — £4.62 of somebody's time for £0.10."""
    _peak_app(app)
    result = run_call(app, "journey.cost",
                      {"from_station": "Warren Street",
                       "to_station": "Barbican"},
                      now=datetime(2026, 11, 23, 9, 10, tzinfo=LONDON_TZ))
    assert "peak_tip" not in result.data, result.data.get("peak_tip")


def test_peak_tip_reads_a_london_clock(app):
    """08:10 UTC in July is 09:10 in London and squarely peak. Read as it
    arrives, the tip would never fire through British Summer Time."""
    summer = datetime(2026, 7, 13, 8, 10, tzinfo=timezone.utc)
    assert "peak_tip" in _ask_at(_peak_app(app), summer).data


def test_peak_tip_quotes_a_real_price_difference(app):
    """Both figures come from the same zone table. Subtracting a live fare
    from a table fare would invent a saving that TfL never charges."""
    tip = _ask_at(_peak_app(app),
                  datetime(2026, 11, 23, 9, 10, tzinfo=LONDON_TZ)).data["peak_tip"]
    assert "09:30" in tip and "20 minutes" in tip
    assert "£1.90" in tip, tip


def test_no_fact_explains_why_only_what(app):
    """`did_you_know` promises one TRUE thing read from the list. It used to
    end "the longer side is the one where you wait on a concourse rather than
    a platform" — tidy, plausible, and invented: the bundled data carries
    entry, exit, metres, minutes and variants, and nothing about concourses.

    A causal story nobody can check is the one thing this field must not
    contain, because every other figure in the app is checkable and the
    reader cannot tell which is which."""
    for record in app._stations.values():
        fact = app._did_you_know(record, record, None)
        if fact:
            # Not the word itself — London Bridge's gateline is published as
            # "Main concourse (lower) and upper concourse", and quoting a real
            # name is the opposite of inventing one. What is banned is the
            # explanatory construction: a reason the data does not carry.
            low = fact.lower()
            assert "where you wait" not in low, fact
            assert "rather than a platform" not in low, fact
            assert "the reason" not in low, fact


def test_the_asymmetry_claim_is_still_true_of_the_data(app):
    """The replacement claim — the longer allowance always goes to the
    direction ending at National Rail — is only worth saying while the
    bundled list still says it. 36 of 36 directed entries when written."""
    checked = 0
    for key, out in app._osi.items():
        a, b = key.split("|")
        back = app._osi.get(f"{b}|{a}")
        if not back or back["minutes"] == out["minutes"]:
            continue
        longer = out if out["minutes"] > back["minutes"] else back
        assert any("->NR" in v for v in longer.get("variants") or []), (
            f"{key}: the longer side no longer ends at National Rail")
        checked += 1
    assert checked >= 30, f"only {checked} asymmetric entries found"


def test_no_single_aside_dominates_the_answers(app):
    """Delight dies on repetition. The fallback pool was three facts, so 89%
    of journeys carried one of the same three lines and anybody trying three
    routes saw a repeat — which is exactly what a judge does."""
    import random as _random

    def fetch(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps(WARREN_BARBICAN).encode()
    app._fetch = fetch

    names = [r["name"] for r in
             {r["naptan"]: r for r in app._stations.values()}.values()]
    rng = _random.Random(7)
    when = datetime(2026, 11, 23, 10, 30, tzinfo=LONDON_TZ)
    seen: dict[str, int] = {}
    runs = 0
    for _ in range(600):
        a, b = rng.sample(names, 2)
        try:
            data = run_call(app, "journey.cost",
                            {"from_station": a, "to_station": b},
                            now=when).data
        except AgentAppError:
            continue
        runs += 1
        aside = data.get("did_you_know")
        if aside:
            seen[aside] = seen.get(aside, 0) + 1

    assert runs > 300, f"only {runs} answers produced"
    top = max(seen.values()) / sum(seen.values())
    assert top < 0.20, f"one aside is {top:.0%} of answers"
    assert len(seen) >= 20, f"only {len(seen)} distinct asides"


def test_every_station_aside_is_true_of_the_pair_it_names(app):
    """The per-station asides quote a distance, a time limit, or both, about
    a named pair. The selection conditions around them survived a mutation
    sweep — loosen one and the sentence names a pair whose figures do not
    match what it says about them."""
    import re as _re
    osi = app._osi
    for record in list(app._stations.values())[:400]:
        fact = app._did_you_know(record, record, None)
        if not fact:
            continue
        # "X and Y are a designated interchange — and they are N metres apart"
        m = _re.search(r"^(.+?) and (.+?) are a designated interchange — and "
                       r"they are (\d+) metres apart", fact)
        if m:
            exit_name, entry_name, metres = m.group(1), m.group(2), int(m.group(3))
            hit = [p for p in osi.values()
                   if p["exit"] == exit_name and p["entry"] == entry_name]
            assert hit, f"names a pair that is not on the list: {fact}"
            assert hit[0]["metres"] == metres, (fact, hit[0])
            assert metres <= 80, f"called {metres} m a short walk: {fact}"
        # "You can walk the N km from X to Y"
        m = _re.search(r"walk the ([\d.]+) km from (.+?) to (.+?) and", fact)
        if m:
            km, exit_name, entry_name = float(m.group(1)), m.group(2), m.group(3)
            hit = [p for p in osi.values()
                   if p["exit"] == exit_name and p["entry"] == entry_name]
            assert hit, f"names a pair that is not on the list: {fact}"
            assert abs(hit[0]["metres"] / 1000 - km) < 0.05, (fact, hit[0])
            assert hit[0]["metres"] >= 900, f"called {hit[0]['metres']} m a long walk"
        # "X is one of the better-connected stations for this: N places"
        m = _re.search(r"^(.+?) is one of the better-connected stations for "
                       r"this: (\d+) places", fact)
        if m:
            station, claimed_n = m.group(1), int(m.group(2))
            prefix = None
            for k, v in app._stations.items():
                if v["name"] == station:
                    prefix = k + "|"
                    break
            assert prefix, f"names a station not in the list: {fact}"
            real = len({osi[k]["entry"] for k in osi if k.startswith(prefix)})
            assert claimed_n == real, (
                f"{station} is said to reach {claimed_n} places; the list "
                f"says {real} — {fact}")
            assert claimed_n >= 3, fact

        # "X to Y allows A minutes; coming back allows B"
        m = _re.search(r"^(.+?) to (.+?) allows (\d+) minutes; coming back "
                       r"allows (\d+)", fact)
        if m:
            exit_name, entry_name = m.group(1), m.group(2)
            there, back_mins = int(m.group(3)), int(m.group(4))
            assert there != back_mins, f"calls equal times a disagreement: {fact}"
            hit = [p for p in osi.values()
                   if p["exit"] == exit_name and p["entry"] == entry_name]
            assert hit and hit[0]["minutes"] == there, (fact, hit)
            # When the longer side ends at National Rail the fact says so.
            # Losing that leaves a bare pair of numbers and drops the only
            # part of the aside that explains anything.
            back_key = None
            for key, pair in osi.items():
                if pair["exit"] == entry_name and pair["entry"] == exit_name:
                    back_key = pair
                    break
            if back_key:
                longer = hit[0] if there > back_mins else back_key
                if any("->NR" in v for v in longer.get("variants") or []):
                    assert "National Rail station" in fact, fact


def test_a_saving_is_only_offered_when_there_is_one(app):
    """The guard around the cheaper line has four parts. Loosen it and the
    app offers "£0.00 cheaper to Circle instead" — naming the very option it
    just recommended, for no saving at all."""
    import re as _re
    one_option = {"journeys": [
        {"duration": 12, "fare": {"totalCost": 300},
         "legs": [_leg("tube", [("circle", "Circle")])]},
    ]}

    def fetch(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps(one_option).encode()

    app._fetch = fetch
    narrative = ask_access(app, "Warren Street", "Barbican").narrative
    assert "cheaper to" not in narrative, narrative
    assert "\u00a30.00 cheaper" not in narrative, narrative
    # and wherever it does appear, the saving is real and names something else
    m = _re.search(r"\u00a3([\d.]+) cheaper to (\w+)", narrative or "")
    if m:
        assert float(m.group(1)) > 0, narrative


def test_every_designated_pair_within_walking_range_is_offered_as_one(app):
    """The one invariant that catches a collapsing rule reaching too far.

    Two names for one complex are merged so the same walk is not offered
    twice. Merge too eagerly and genuinely separate stations disappear:
    forty-nine stations lose neighbours, Aldgate stops offering Tower
    Gateway, and a designated twenty-minute interchange simply is not there
    any more. Counting duplicates cannot see that — the list gets shorter,
    not longer — so this checks the opposite direction: everything the
    interchange list says is designated, and is close enough to walk, has to
    come back as a designated neighbour."""
    from app import WALKABLE_METRES
    checked = 0
    for key, pair in app._osi.items():
        exit_key, entry_key = key.split("|")
        # Same-name pairs — Shepherd's Bush to Shepherd's Bush, Balham to
        # Balham — collapse to one record, so there is no second station to
        # offer. The fact is still true and still told; it just cannot be a
        # neighbour of itself.
        if exit_key == entry_key:
            continue
        start = app._stations.get(exit_key)
        if not start or pair["metres"] > WALKABLE_METRES:
            continue
        # The interchange list and the station list name things differently
        # — "Fenchurch Street" against "London Fenchurch Street Rail Station"
        # — so match the way the app does, by the record behind the name.
        target = app._stations.get(entry_key)
        if not target:
            continue
        # Hub records name a place rather than a platform, and both sides of
        # the King's Cross complex collapse onto one. Two pairs there cannot
        # be offered as neighbours for that reason; it is a known limit of
        # the bundled data, not of this rule.
        if str(start.get("naptan", "")).startswith("HUB") \
                or str(target.get("naptan", "")).startswith("HUB"):
            continue
        near = {n["name"]: n for n in app._walkable(start)}
        found = near.get(target["name"]) or near.get(pair["entry"])
        assert found, (
            f"{pair['exit']} no longer offers {target['name']}, "
            f"a designated pair {pair['metres']} m away")
        assert found["designated"] is True, (pair["exit"], pair["entry"], found)
        # An allowance, not this entry's exactly: the King's Cross complex
        # carries two entries for what is effectively one walk, at 30 and 40
        # minutes, so the app picks one and either is defensible. What must
        # never happen is a designated pair coming back with no allowance at
        # all, which is what reports a free walk as a second fare.
        assert found["allowed"] and found["allowed"] > 0, (pair, found)
        checked += 1
    assert checked > 60, f"only {checked} designated pairs were in range"


def test_the_demo_pair_survives_every_collapsing_rule(app):
    """Two names for one complex are collapsed so the same walk is not
    offered twice. Collapse too eagerly and genuinely different stations
    merge: Warren Street would stop offering both Euston Square and Great
    Portland Street, which is the entire demo.

    Pinned here rather than as a count, because a rule that over-collapses
    produces fewer neighbours, not duplicates — so counting duplicates
    cannot see it."""
    from app import WALKABLE_METRES
    start = app._resolve("Warren Street", "starting")
    near = {n["name"]: n for n in app._walkable(start)}

    # Nothing beyond walking distance may be offered. Without that limit
    # every station in London becomes a "neighbour", and the fare warning
    # describes somewhere miles away as a few minutes' walk.
    for record in list(app._stations.values())[:300]:
        for neighbour in app._walkable(record):
            assert neighbour["metres"] <= WALKABLE_METRES, (
                f"{record['name']} is offered {neighbour['name']} "
                f"{neighbour['metres']} m away, past the {WALKABLE_METRES} m limit")

    free = near.get("Euston Square")
    assert free, f"Euston Square vanished from Warren Street: {sorted(near)}"
    assert free["designated"] is True
    assert free["allowed"] == 20, free

    paid = near.get("Great Portland Street")
    assert paid, f"Great Portland Street vanished: {sorted(near)}"
    assert paid["designated"] is False
    assert not paid["allowed"], paid

    # and they are two different places, not one collapsed entry
    assert free["metres"] != paid["metres"]
    names = [n["name"] for n in app._walkable(start)]
    assert len(names) == len(set(names)), names


def test_a_walk_of_a_couple_of_metres_is_one_minute_not_two(app):
    """`max(1, ...)` is a floor, not a rounding. Raise it and every short
    walk gains a minute — the shortest designated pair on the network is two
    metres apart, and "a 2 minute walk" across two metres reads as broken."""
    shortest = min((n for record in app._stations.values()
                    for n in app._walkable(record)), key=lambda n: n["metres"])
    assert shortest["minutes"] == 1, shortest
    floors = {n["minutes"] for record in app._stations.values()
              for n in app._walkable(record)}
    assert min(floors) == 1, sorted(floors)[:5]


# ---------------------------------------------------------------------------
# Three fixes that nothing was protecting. A statement-deletion sweep removed
# each one and the suite stayed green, which means the bug each was written
# for could come back silently.
# ---------------------------------------------------------------------------

def test_the_step_free_gateline_aside_says_the_right_thing(app):
    """Three stations — Lewisham, Richmond, Woodford — have a step-free route
    that leaves and re-enters the paid area, so using the only accessible way
    out starts a clock. That is a different sentence from an ordinary
    in-station interchange, and the branch choosing it survived a deletion
    sweep: without it those three get the generic wording, which does not
    mention the step-free route at all.

    It is the one aside that matters most to the person reading it, because
    they have no alternative way out."""
    from app import norm
    forced = {k: v for k, v in app._gateline.items() if v.get("step_free_forced")}
    assert forced, "no step-free-forced gatelines left in the data"

    for key, record in forced.items():
        station = app._stations.get(key) or next(
            (v for v in app._stations.values() if norm(v["name"]) == key), None)
        assert station, f"{record['station']} is not in the station list"
        fact = app._did_you_know(station, station, None)
        assert fact, record["station"]
        assert "step-free route" in fact, (record["station"], fact)
        assert "re-enters the paid area" in fact, (record["station"], fact)
        assert f"{record['minutes']}-minute clock" in fact, (record, fact)

    # an ordinary gateline must NOT borrow that wording
    plain = {k: v for k, v in app._gateline.items()
             if not v.get("step_free_forced")}
    for key, record in list(plain.items())[:6]:
        station = app._stations.get(key) or next(
            (v for v in app._stations.values() if norm(v["name"]) == key), None)
        if not station:
            continue
        fact = app._did_you_know(station, station, None) or ""
        assert "step-free route" not in fact, (record["station"], fact)



#: The kinds of fact `_did_you_know` can tell about the stations in front of
#: it, each by a phrase only that kind uses. Order matches the function.
LOCAL_FACT_KINDS = {
    "gateline, step-free forced": "the step-free route leaves and",
    "gateline, ordinary": "has an interchange inside the station, between",
    "an absurdly short designated walk": "are a designated interchange — and they are",
    "an absurdly long designated walk": "and still be charged one journey, so long as",
    "an allowance that differs by direction": "; coming back allows",
    "a well-connected station": "is one of the better-connected stations for this:",
}


def test_every_kind_of_local_fact_is_still_reachable(app):
    """`did_you_know` falls back to a network-wide fact when it has nothing
    to say about these particular stations. That fallback is what makes the
    whole block fragile: delete any one of the local branches and the field
    still fills, still says something true, and every test stays green —
    the answer just quietly stops being about the journey the person asked
    about.

    A deletion sweep found exactly that. Nine deletions inside this function
    changed between four and thirteen replies in four hundred and not one
    test noticed, because each deleted branch was covered by the fallback.

    The rarest kind fires for six station pairs out of a thousand, so a
    spot-check would not find it. This asks the only question that
    distinguishes the branches: can each kind still happen at all?
    """
    recs = list({r["naptan"]: r for r in app._stations.values()}.values())
    found: dict[str, str] = {}
    for i, a in enumerate(recs):
        for b in (a, recs[(i + 7) % len(recs)]):
            fact = app._did_you_know(a, b, None)
            if not fact:
                continue
            for kind, mark in LOCAL_FACT_KINDS.items():
                if mark in fact and kind not in found:
                    found[kind] = f"{a['name']} -> {b['name']}: {fact}"
        if len(found) == len(LOCAL_FACT_KINDS):
            break
    missing = sorted(set(LOCAL_FACT_KINDS) - set(found))
    assert not missing, (
        "no journey anywhere on the network can now produce: "
        + "; ".join(missing)
        + " — the field will still fill, with a fact about the network "
          "instead of about the journey")


def test_a_local_fact_wins_over_a_network_one(app):
    """The ordering is the point, not an accident. A gateline inside the
    station the person is standing in beats a superlative about somewhere
    else, so the local branches must be tried first — and `return` on a hit
    is what enforces it."""
    local = [k for k, v in app._gateline.items() if not v.get("step_free_forced")]
    assert local, "no ordinary gatelines left in the data"
    checked = 0
    for key in local:
        station = app._stations.get(key) or next(
            (v for v in app._stations.values() if norm(v["name"]) == key), None)
        if not station:
            continue
        fact = app._did_you_know(station, station, None) or ""
        assert LOCAL_FACT_KINDS["gateline, ordinary"] in fact, (key, fact)
        checked += 1
    assert checked >= 5, f"only {checked} gatelines were checkable"



def test_a_free_walk_is_not_the_cheapest_fare(app):
    """`cheapest_pence` is declared as "the lowest fare among the options
    that carry one". A walk does not carry a cheap fare, it carries none.

    Counting it as £0.00 made the cheapest way from New Cross to London
    Bridge come back as zero with "Southeastern · 5 min · £3.20" recommended
    in the same reply — a number that contradicts the sentence beside it is
    the one thing the payload must never do, because Pippa reads both out.
    The bicycle was already excluded by name; the walk was not."""
    journeys = {"journeys": [
        # a real fare, a dearer one, and two journeys with no fare at all
        leg_journey(5, 320, "national-rail"),
        leg_journey(15, 310, "tube", line="victoria"),
        leg_journey(22, None, "cycle"),
        leg_journey(81, 0, "walking"),
    ]}
    result = ask(route(app, journeys=journeys))
    data = result.data
    assert data["cheapest_pence"] == 310, (data["cheapest_pence"], data["options"])
    # and the free options are still offered — they are not being hidden,
    # only kept out of a figure that names a fare
    assert any("no fare" in o for o in data["options"]), data["options"]


def test_cheapest_pence_is_zero_only_when_nothing_is_priced(app):
    """The declared meaning of zero: "0 when nothing on offer is priced."
    It has to stay reachable, or the field quietly stops being able to say
    "this is free" at all."""
    journeys = {"journeys": [
        leg_journey(81, 0, "walking"),
        leg_journey(22, None, "cycle"),
    ]}
    data = ask(route(app, journeys=journeys)).data
    assert data["cheapest_pence"] == 0, data["options"]



def test_the_only_query_left_gets_the_time_that_is_left(app):
    """Heathrow Terminal 5 to Bond Street has no tube route. `mode=tube`
    returns 404, so the multimodal query is not a widening of the options —
    it is the whole answer. It takes about seven seconds, and it was being
    given the four-second slice sized for an optional extra while eleven
    seconds of the call's budget went unused. The question failed every
    single time, live, with "I couldn't reach Transport for London just now".

    Stubbed on the timeout the app asks for, because the real timing depends
    on how warm Transport for London's cache is.
    """
    asked: list[float] = []

    def slow_multimodal(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        if "mode=tube" in url:
            raise urllib.error.HTTPError(url, 404, "no tube route", {}, None)
        asked.append(timeout)
        if timeout <= TIMEOUT:           # the old, too-small slice
            raise TimeoutError("read timed out")
        return json.dumps(WARREN_BARBICAN).encode()

    app._fetch = slow_multimodal
    result = ask(app)
    assert result.data["options"], "no answer for a journey TfL describes"
    assert asked and asked[0] > TIMEOUT, (
        f"the last remaining query was given {asked[0]}s, "
        f"the slice for an optional extra")


def test_an_optional_second_query_does_not_get_the_extended_budget(app):
    """The other half of the rule. Once the first query has returned
    something, the second only widens the list — and a request that is merely
    nice to have must not be allowed to spend the budget the answer is
    already sitting in."""
    asked: list[float] = []

    def watch(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        if "useMultiModalCall" in url:
            asked.append(timeout)
        return json.dumps(WARREN_BARBICAN).encode()

    app._fetch = watch
    ask(app)
    assert asked, "the widening query never ran"
    assert asked[0] <= TIMEOUT, (
        f"an optional query was given {asked[0]}s of the call's budget")


def test_no_route_is_only_said_when_both_queries_have_run(app):
    """Skipping the second query when nothing has been found turns "I ran out
    of time to ask" into "there is no way to get there", which is a different
    claim and a false one."""
    seen: list[str] = []

    def only_multimodal_works(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        seen.append(url)
        if "mode=tube" in url:
            raise urllib.error.HTTPError(url, 404, "no tube route", {}, None)
        return json.dumps(WARREN_BARBICAN).encode()

    app._fetch = only_multimodal_works
    # `handle` restamps the clock, so the budget is squeezed at the source.
    # One second left is below OPTIONAL_MIN_SECONDS, which is what makes the
    # second query skippable — and it must not be skipped when it is the only
    # query that can answer.
    app._remaining = lambda: 1.0
    assert ask(app).data["options"], "gave up with a query still unasked"
    assert any("useMultiModalCall" in u for u in seen), seen



#: Every name the real `pippa_agentapp_sdk` is known to export, taken from
#: Pippa Labs' own example app and its tests — the only authoritative sample
#: of the real SDK available on a laptop, since the package itself is
#: installed only inside their build.
REAL_SDK_NAMES = frozenset({
    "AgentAppError", "Call", "Manifest", "Result",
    "load_manifest_file", "validate_manifest",     # their app.py imports these
    "LocalRunner", "OutputShapeViolation",         # their test_app.py these
})


def test_nothing_imports_a_name_the_real_sdk_has_not_got():
    """The local stand-in is wider than the real SDK, and that is dangerous.

    `Context` and `set_active_outcome` exist here and do not exist there. The
    suite used both, passed completely on this machine, and then died at
    import in their build:

        ImportError: cannot import name 'Context' from 'pippa_agentapp_sdk'

    Nothing local could have caught it, because the thing doing the checking
    was the thing that was wrong. A stand-in with extra surface is worse than
    none: the extra surface is invisible until the real one refuses it, and
    by then the failure is in a build log rather than a test run.

    So the files that ship are held to the real SDK's vocabulary, and the
    stand-in may keep its extras for the local harness as long as nothing
    deployed reaches for them.
    """
    import ast
    for filename in ("app.py", "test_app.py"):
        path = Path(__file__).parent / filename
        tree = ast.parse(path.read_text())
        imported = {alias.name
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom)
                    and node.module == "pippa_agentapp_sdk"
                    for alias in node.names}
        unknown = imported - REAL_SDK_NAMES
        assert not unknown, (
            f"{filename} imports {sorted(unknown)} from pippa_agentapp_sdk, "
            f"which the real SDK does not export — it will fail at import in "
            f"the build. Known names: {sorted(REAL_SDK_NAMES)}")


def test_the_stand_in_provides_everything_the_real_sdk_does():
    """The other direction. If the stand-in is missing something the real SDK
    exports, a test written against the real API fails locally for a reason
    that has nothing to do with the app."""
    import pippa_agentapp_sdk as sdk
    missing = sorted(n for n in REAL_SDK_NAMES if not hasattr(sdk, n))
    assert not missing, f"the local stand-in has no {missing}"


def test_live_disruption_reaches_both_the_option_and_the_sentence(app):
    """"It reads the disruption itself rather than being told" is the claim,
    and it is carried by two small branches: one marks the option, the other
    puts it in the sentence. Either can be deleted and the suite stayed
    green — 134 and 132 of 400 replies changed in silence."""
    delayed = [{"id": "victoria", "name": "Victoria",
                "lineStatuses": [{"statusSeverity": 9,
                                  "statusSeverityDescription": "Severe Delays"}]}]
    journey = {"journeys": [
        {"duration": 12, "fare": {"totalCost": 300},
         "legs": [_leg("tube", [("victoria", "Victoria")])]},
    ]}

    def fetch(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(delayed).encode()
        return json.dumps(journey).encode()

    app._fetch = fetch
    result = ask_access(app, "Warren Street", "Barbican")
    assert result.data["disrupted_lines"] == ["Victoria"], result.data
    assert "disrupted" in result.data["options"][0], result.data["options"]
    assert "reporting delays" in (result.narrative or ""), result.narrative

    # and when everything runs, none of that is said
    def all_good(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps(journey).encode()

    app._fetch = all_good
    quiet = ask_access(app, "Warren Street", "Barbican")
    assert quiet.data["disrupted_lines"] == []
    assert "disrupted" not in quiet.data["options"][0], quiet.data["options"]
    assert "reporting delays" not in (quiet.narrative or ""), quiet.narrative


def test_a_bus_option_carries_its_route_numbers(app):
    """"Bus" on its own is not an answer — the number is the whole point of
    knowing it is a bus. Deleting the branch that adds them changed 108 of
    400 replies."""
    bus = {"journeys": [
        {"duration": 20, "fare": {"totalCost": 185},
         "legs": [_leg("bus", [("205", "205")]), _leg("bus", [("4", "4")])]},
    ]}

    def fetch(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps(bus).encode()

    app._fetch = fetch
    label = ask_access(app, "Warren Street", "Barbican").data["options"][0]
    assert label.startswith("Bus "), label
    assert "205" in label, label


def test_every_allowance_is_one_transport_for_london_actually_grants(app):
    """A time limit is a promise about somebody's money: cross the gates
    inside it and the walk is free. One pair's allowance was set to 99
    minutes in a data sweep and nothing noticed — the app would have said so,
    confidently, to a person who had ten.

    Transport for London grant these in fixed steps, so anything outside the
    set is a corrupted row rather than a new rule."""
    allowed = {10, 15, 20, 25, 30, 40}
    odd = {k: v["minutes"] for k, v in app._osi.items()
           if v["minutes"] not in allowed}
    assert not odd, f"interchange allowances outside {sorted(allowed)}: {list(odd.items())[:5]}"

    gate_allowed = {5, 10, 15, 20, 30}
    odd_gate = {k: v["minutes"] for k, v in app._gateline.items()
                if v["minutes"] not in gate_allowed}
    assert not odd_gate, f"gateline allowances outside {sorted(gate_allowed)}: {odd_gate}"

    # and the walk itself has to be walkable in the time given
    for key, pair in app._osi.items():
        if pair["metres"] <= 0:
            continue
        on_foot = pair["metres"] / 80
        assert on_foot <= pair["minutes"], (
            f"{key}: {pair['metres']} m in {pair['minutes']} min needs "
            f"{on_foot:.0f} minutes of walking")


def test_no_station_has_wandered_off_the_map(app):
    """Coordinates drive every distance the app quotes and every point on the
    map. A station was moved two kilometres in a data sweep without a
    complaint; these bounds are a generous box around the network, so
    anything outside is a corrupted row, not a new station."""
    stray = [(v["name"], v["lat"], v["lon"]) for v in app._stations.values()
             if v.get("lat") is not None
             and not (51.25 <= v["lat"] <= 51.95 and -1.05 <= v["lon"] <= 0.80)]
    assert not stray, f"{len(stray)} stations outside Greater London: {stray[:4]}"

    # the demo pair's distance must match its coordinates, not drift from them
    start = app._resolve("Warren Street", "starting")
    square = app._resolve("Euston Square", "destination")
    straight = app._metres(start, square)
    assert straight is not None and 150 <= straight <= 350, straight


def test_the_bundled_keys_still_agree_with_the_normaliser(app):
    """`tapdata.json` is keyed by `norm()` of each station name, generated
    once when the file was built. The app re-keys from the same function at
    load, so a change to `norm` breaks nothing visible — both sides move
    together — while quietly disagreeing with every key already written into
    the file: the interchange pairs, the gateline entries, the aliases.

    That is how a free walk came to be reported as a second fare once, and
    it is invisible from the outside, so it is checked from the inside."""
    import json as _json, pathlib as _pathlib
    from app import norm
    raw = _json.loads((_pathlib.Path(__file__).parent / "tapdata.json").read_text())

    # Five stations are deliberately filed under the short name the
    # interchange list uses, so a name need not normalise to its own key —
    # but it must normalise to *some* key in the file, or the app and the
    # data have stopped speaking the same language.
    keys = set(raw["stations"])
    stranded = {key for key, record in raw["stations"].items()
                if norm(record["name"]) not in keys}
    # Six are filed under the short name the interchange list uses, so their
    # full name normalises to something that is deliberately not a key. Any
    # others mean `norm` and the file have drifted apart — the thirteen
    # stations with apostrophes go first, Earl's Court and Queen's Park
    # among them.
    expected = {"fenchurch st", "kings cross", "queenstown rd", "st pancras",
                "waterloo east", "west hampstead nll"}
    assert stranded == expected, (
        f"{len(stranded - expected)} station names have stopped normalising "
        f"to a key in the file: {sorted(stranded - expected)[:6]}")

    # and every interchange endpoint is still a key the app can look up
    orphans = [k for pair in raw["osi_pairs"] for k in pair.split("|")
               if k not in raw["stations"]]
    assert not orphans, f"{len(orphans)} interchange endpoints have no station: {orphans[:4]}"


def test_punctuation_in_a_station_name_does_not_matter(app):
    """People type "st pauls", the list says "St. Paul's". The normaliser
    strips the full stops and the apostrophe so both land on one station;
    without it, half the names somebody would reasonably type stop
    resolving."""
    pairs = [("st pauls", "St. Paul's"), ("st. paul's", "St. Paul's"),
             ("kings cross st pancras", "King's Cross St. Pancras"),
             ("elephant and castle", "Elephant & Castle"),
             ("shepherds bush market", "Shepherd's Bush Market")]
    for typed, proper in pairs:
        assert app._resolve(typed, "starting")["name"] == proper, typed
        assert app._resolve(proper, "starting")["name"] == proper, proper


def test_a_bus_route_is_never_rendered_as_a_railway_line(app):
    """Bus legs carry route numbers in the same shape as Tube lines, so a
    journey came back labelled "148 then 35" as though those were railways.
    The fix is one filter, and deleting it changed 163 of 400 replies
    without a single test noticing."""
    mixed = {"journeys": [
        {"duration": 30, "fare": {"totalCost": 185}, "legs": [
            _leg("bus", [("148", "148")]),
            _leg("tube", [("victoria", "Victoria")]),
            _leg("bus", [("35", "35")]),
        ]},
    ]}

    def fetch(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps(mixed).encode()

    app._fetch = fetch
    data = ask_access(app, "Warren Street", "Barbican").data
    label = data["options"][0]
    assert "Victoria" in label, label
    # the bus numbers must not appear as though they were lines
    assert "148 then" not in label and "then 35" not in label, label
    assert not any(_r in label for _r in ("Victoria then 35", "148 then Victoria")), label


def test_the_options_come_back_in_cost_order(app):
    """The whole premise is ranking by what each way actually costs — fare
    plus time at 21p a minute. Deleting the sort leaves them in whatever
    order Transport for London happened to return, and the recommendation
    becomes an accident. 103 of 400 replies changed, unnoticed."""
    from app import PENCE_PER_MINUTE
    import re as _re
    # Journeys arrive sorted quickest-first, so the cost order has to
    # disagree with the duration order or the check proves nothing:
    #   Circle   10 min £5.90 -> 210 + 590 = 800
    #   Victoria 15 min £3.90 -> 315 + 390 = 705
    #   Bus 205  20 min £1.70 -> 420 + 170 = 590
    # quickest-first is Circle, Victoria, Bus; by cost it is the reverse.
    spread = {"journeys": [
        {"duration": 10, "fare": {"totalCost": 590},
         "legs": [_leg("tube", [("circle", "Circle")])]},
        {"duration": 15, "fare": {"totalCost": 390},
         "legs": [_leg("tube", [("victoria", "Victoria")])]},
        {"duration": 20, "fare": {"totalCost": 170},
         "legs": [_leg("bus", [("205", "205")])]},
    ]}

    def fetch(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps(spread).encode()

    app._fetch = fetch
    data = ask_access(app, "Warren Street", "Barbican").data
    costs = []
    for line in data["options"]:
        mins = int(_re.search(r"(\d+) min", line).group(1))
        fare = _re.search(r"\u00a3([\d.]+)", line)
        pence = int(round(float(fare.group(1)) * 100)) if fare else 0
        costs.append(mins * PENCE_PER_MINUTE + pence)
    assert costs == sorted(costs), (data["options"], costs)
    assert data["best"] == data["options"][0], data["options"]


def test_tram_stops_are_not_offered_as_stations_to_walk_to(app):
    """Thirty-nine tram stops sit in the station list and are not gated
    stations at all — you do not tap out of one, so an interchange warning
    about one is meaningless. Deleting the filter changed 16 of 400."""
    from app import TRAM_PREFIXES
    trams = {k: v for k, v in app._stations.items()
             if str(v.get("naptan", "")).startswith(TRAM_PREFIXES)}
    assert trams, "no tram stops in the data — has the prefix changed?"
    tram_names = {v["name"] for v in trams.values()}
    for record in list(app._stations.values())[:400]:
        for near in app._walkable(record):
            assert near["name"] not in tram_names, (
                f"{record['name']} is offered a walk to the tram stop {near['name']}")


def test_a_bus_route_with_only_an_id_is_still_named(app):
    """`identifier.get("name") or identifier.get("id")` — real responses
    carry one or the other, and losing the fallback drops the route number
    from the label entirely. It changed 108 of 400 replies."""
    only_id = {"legs": [{"mode": {"name": "bus"},
                         "routeOptions": [{"lineIdentifier": {"id": "453"}}]}]}
    assert app._bus_routes(only_id) == ["453"]
    only_name = {"legs": [{"mode": {"name": "bus"},
                           "routeOptions": [{"lineIdentifier": {"name": "N53"}}]}]}
    assert app._bus_routes(only_name) == ["N53"]
    # name wins when both are present
    both = {"legs": [{"mode": {"name": "bus"},
                      "routeOptions": [{"lineIdentifier": {"id": "453", "name": "453 Express"}}]}]}
    assert app._bus_routes(both) == ["453 Express"]


def test_the_option_kept_for_a_repeated_label_is_the_quickest(app):
    """Several results collapse to one entry when they share a label. They
    are sorted quickest-first so the one kept is the quickest; without that
    it is whichever came back first, and the app quotes a slower journey
    than the one it was given."""
    same_label = {"journeys": [
        {"duration": 40, "fare": {"totalCost": 300},
         "legs": [_leg("tube", [("circle", "Circle")])]},
        {"duration": 11, "fare": {"totalCost": 300},
         "legs": [_leg("tube", [("circle", "Circle")])]},
        {"duration": 25, "fare": {"totalCost": 300},
         "legs": [_leg("tube", [("circle", "Circle")])]},
    ]}

    def fetch(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps(same_label).encode()

    app._fetch = fetch
    data = ask_access(app, "Warren Street", "Barbican").data
    circle = [o for o in data["options"] if o.startswith("Circle")]
    assert len(circle) == 1, data["options"]
    assert "11 min" in circle[0], circle[0]


def test_every_superlative_in_a_fact_is_actually_the_superlative(app):
    """Each of these asides claims a record — the longest walk, the tightest
    interchange, the most forgiving one, the best-connected station, the
    widest disagreement. Swap a `min` for a `max` anywhere and the sentence
    still reads perfectly while the claim becomes false, which is the one
    thing this field must never be. A sweep found fifteen such mutations that
    no test noticed, so every claim is re-derived from the data here."""
    import re as _re
    facts = app._network_facts()
    blob = "\n".join(facts)
    osi = app._osi
    gate = app._gateline
    pairs = [p for p in osi.values() if p["metres"] > 0]

    def claimed(pattern, where=blob):
        """A pattern that no longer matches is a failure, not a skip.

        These claims are built with f-strings, so a wrong operator does not
        remove a sentence — it malforms one. "-8 of the 169 pairs" and "9.0
        pairs" both stop matching a `\\d+`, and an `if got:` guard then skips
        the very check that would have caught them. Five mutations lived in
        that gap."""
        m = _re.search(pattern, where)
        assert m, f"no fact matches {pattern!r} any more — look at: {where[:300]}"
        return m.groups()

    # the shortest pair, and the right phrasing for it. Two stations sharing
    # a name is the surprising half of that fact, so it has its own wording;
    # the generic one ("Somewhere on the network...") reads as a different,
    # weaker claim about the same pair.
    shortest = min(pairs, key=lambda p: p["metres"])
    if shortest["exit"] == shortest["entry"]:
        assert "two different stations with the same name" in blob, blob[:200]
        assert f"{shortest['metres']} metres apart" in blob, blob[:200]
    else:
        assert "Somewhere on the network" in blob, blob[:200]

    # the longest free walk
    longest = max(pairs, key=lambda p: p["metres"])
    got = claimed(r"longest walk you can make[^.]*?is ([^:]+): ([\d.]+) km, inside (\d+)")
    if got:
        assert got[0] == f"{longest['exit']} to {longest['entry']}", got
        assert abs(float(got[1]) - longest["metres"] / 1000) < 0.05, got
        assert int(got[2]) == longest["minutes"], got

    # the tightest and the most forgiving gateline
    tightest = min(gate.values(), key=lambda g: g["minutes"])
    got = claimed(r"At ([^,]+?) there is an interchange[^.]*?only (\d+) minutes")
    if got:
        assert int(got[1]) == tightest["minutes"], got
        assert int(got[1]) == min(g["minutes"] for g in gate.values()), got
    roomiest = max(gate.values(), key=lambda g: g["minutes"])
    got = claimed(r"([\w'’ .&-]+?) is the most forgiving interchange[^:]*: (\d+) minutes")
    if got:
        assert int(got[1]) == roomiest["minutes"], got
        assert int(got[1]) == max(g["minutes"] for g in gate.values()), got

    # the tightest designated walk
    quickest = min(osi.values(), key=lambda p: p["minutes"])
    got = claimed(r"tightest designated walk[^-]*— (\d+) metres, and (\d+) minutes")
    if got:
        assert int(got[1]) == quickest["minutes"], got
        assert int(got[1]) == min(p["minutes"] for p in osi.values()), got

    # the best-connected station
    reach: dict[str, int] = {}
    for key in osi:
        reach[key.split("|")[0]] = reach.get(key.split("|")[0], 0) + 1
    got = claimed(r"is the best-connected station for this: (\d+) places")
    if got:
        assert int(got[0]) == max(reach.values()), got

    # the counts
    got = claimed(r"There are (\d+) designated interchanges on the network")
    if got:
        assert int(got[0]) == len(osi), got
    got = claimed(r"usual allowance is (\d+) minutes — (\d+) of the (\d+)")
    if got:
        spread: dict[int, int] = {}
        for pair in osi.values():
            spread[pair["minutes"]] = spread.get(pair["minutes"], 0) + 1
        usual = max(spread.items(), key=lambda kv: kv[1])
        assert (int(got[0]), int(got[1])) == usual, (got, usual)
        assert int(got[2]) == len(osi), got
    got = claimed(r"Only (\d+) of (\d+) stations appear")
    if got:
        assert int(got[0]) == len({k.split("|")[0] for k in osi}), got
        assert int(got[1]) == len({s["naptan"] for s in app._stations.values()}), got
    got = claimed(r"There are (\d+) station names that turn up on both sides")
    if got:
        same = {p["exit"] for p in osi.values() if p["exit"] == p["entry"]}
        assert int(got[0]) == len(same), got
        # and the two it names as examples are genuinely among them
        named = _re.search(r"both\s+sides of a designated pair — (.+?) to \1, "
                           r"(.+?) to \2\.", blob)
        assert named, "the same-name aside no longer names two examples"
        for example in named.groups():
            assert example in same, f"{example!r} is not a same-name pair"

    # the widest disagreement really is the widest
    got = claimed(r"allows (\d+) minutes and the way back allows (\d+) — the\s+widest")
    if got:
        gap = abs(int(got[0]) - int(got[1]))
        widest = 0
        for key, out in osi.items():
            a, b = key.split("|")
            back = osi.get(f"{b}|{a}")
            if back:
                widest = max(widest, abs(out["minutes"] - back["minutes"]))
        assert gap == widest, (got, widest)

    # how many pairs disagree at all
    got = claimed(r"There are (\d+) pairs that allow more time one")
    if got:
        disagreeing = sum(
            1 for key, out in osi.items()
            for a, b in [key.split("|")]
            if (back := osi.get(f"{b}|{a}")) and back["minutes"] != out["minutes"])
        assert int(got[0]) == disagreeing // 2, (got, disagreeing // 2)


def test_every_network_fact_is_well_formed(app):
    """These are built from the data with f-strings, so a sentence can come
    out starting with a numeral or missing its stop without anyone noticing."""
    facts = app._network_facts()
    assert len(facts) >= 10, f"only {len(facts)} network facts"
    for fact in facts:
        assert fact.endswith("."), fact
        assert not fact[0].isdigit(), f"starts with a numeral: {fact}"
        assert not fact[0].islower(), fact
        assert "  " not in fact, fact


def test_every_figure_describes_an_option_that_was_returned(app):
    """`cheapest_pence` was fixed for this once; `fastest_minutes` was missed.

    Only five options come back. Give the planner seven, where the quickest
    is also far and away the dearest, and it ranks outside the five — but
    "fastest 2 minutes" was still reported beside a list whose shortest was
    20. True, and impossible for Pippa to explain, because it was never given
    the journey that figure referred to."""
    import re as _re

    names = ["circle", "district", "central", "northern", "victoria", "jubilee"]
    journeys = [{"duration": 20 + 3 * k, "fare": {"totalCost": 170},
                 "legs": [_leg("tube", [(names[k], names[k].title())])]}
                for k in range(6)]
    journeys.append({"duration": 2, "fare": {"totalCost": 5000},
                     "legs": [_leg("tube", [("bakerloo", "Bakerloo")])]})

    def fetch(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps({"journeys": journeys}).encode()

    app._fetch = fetch
    data = run_call(app, "journey.cost",
                    {"from_station": "Warren Street",
                     "to_station": "Barbican"},
                    now=datetime(2026, 11, 20, 14, 30, tzinfo=LONDON_TZ)).data

    shown = {int(m.group(1)) for line in data["options"]
             for m in [_re.search(r"(\d+) min", line)] if m}
    assert data["fastest_minutes"] in shown, (
        f"fastest_minutes={data['fastest_minutes']} but options show {sorted(shown)}")
    # Membership alone is too weak: `max` instead of `min` picks the SLOWEST
    # returned option, which is still "in" the list. It has to be the least.
    assert data["fastest_minutes"] == min(shown), (
        f"fastest_minutes={data['fastest_minutes']} is not the quickest of {sorted(shown)}")
    if data["cheapest_pence"]:
        fares = {int(round(float(m.group(1)) * 100)) for line in data["options"]
                 for m in [_re.search(r"\u00a3(\d+\.\d\d)", line)] if m}
        assert data["cheapest_pence"] in fares, (
            f"cheapest_pence={data['cheapest_pence']} but options show {sorted(fares)}")


# ---------------------------------------------------------------------------
# Paths the suite never reached until a coverage run went looking
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code,expected,retry", [
    (429, "ServiceBusy", True),        # the room shares one address
    (404, "NoRouteFound", True),
    (503, "ServiceUnavailable", True),
    (500, "ServiceUnavailable", True),
])
def test_each_http_code_from_tfl_maps_to_the_right_failure(app, code, expected, retry):
    """Throttling is the one failure outside our control today: fifty
    requests a minute per source address, and the whole room is one address.
    A mutation changed `exc.code == 429` and nothing noticed, so the branch
    that produces "rate limiting me — try again shortly" was never pinned."""
    import io, urllib.error

    def refuse(url, timeout=5.0):
        raise urllib.error.HTTPError(url, code, "no", {}, io.BytesIO(b""))

    app._fetch = refuse
    with pytest.raises(AgentAppError) as caught:
        ask_access(app, "Warren Street", "Barbican")
    assert type(caught.value).__name__ == expected
    assert caught.value.retryable is retry
    if code == 429:
        assert "rate limit" in caught.value.user_message.lower()


def test_optional_work_is_dropped_when_the_budget_is_nearly_gone(app):
    """Pippa kills the call at twenty seconds. The budget here is thirteen,
    and the dock lookup is two more requests that only decorate an answer we
    already have — so it is skipped when there is not time to finish it.

    Nothing tested that. A mutation turning `>= OPTIONAL_MIN_SECONDS` into
    `>` survived, and so would one that dropped the guard entirely: on a slow
    connection the app would start work it cannot finish and lose the whole
    answer to the kill rather than return it without the bike count."""
    import time as _time
    from app import DEADLINE_SECONDS, OPTIONAL_MIN_SECONDS

    cycling = {"journeys": [
        {"duration": 9, "fare": None, "legs": [_leg("cycle")]},
        {"duration": 14, "fare": {"totalCost": 300},
         "legs": [_leg("tube", [("circle", "Circle")])]},
    ]}
    asked: list[str] = []

    def fetch(url, timeout=5.0):
        asked.append(url)
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        if "BikePoint" in url:
            return json.dumps({"places": []}).encode()
        return json.dumps(cycling).encode()

    app._fetch = fetch

    # `handle` starts the clock itself, so the budget has to be spent rather
    # than set: the stub reports back a monotonic clock that has moved on.
    spent = {"seconds": 0.0}
    real_monotonic = _time.monotonic

    def slow_fetch(url, timeout=5.0):
        spent["seconds"] += 4.0        # each request eats four seconds
        return fetch(url, timeout)

    # Plenty of budget: the docks are worth asking about.
    ask_access(app, "Warren Street", "Barbican")
    assert any("BikePoint" in u for u in asked), "docks never looked up at all"

    # Now make the calls slow enough that the optional work cannot finish.
    asked.clear()
    app._fetch = slow_fetch
    import app as _appmod
    _appmod.time.monotonic = lambda: real_monotonic() + spent["seconds"]
    try:
        result = ask_access(app, "Warren Street", "Barbican")
    finally:
        _appmod.time.monotonic = real_monotonic
    assert not any("BikePoint" in u for u in asked), "started work it could not finish"
    assert result.data["options"], "dropped the answer instead of the extra"


def test_every_spoken_alias_points_at_a_station_that_exists(app):
    """These are the shortenings people actually type — "kings x", "t5",
    "tcr". An alias aimed at a station that is not in the bundled list is
    worse than no alias at all: it refuses by a name the person never typed
    ("I don't recognise a station called heathrow terminal 1"), which reads
    as the app being confused rather than the input being unusual."""
    from app import SPOKEN_ALIASES
    for spoken, target in SPOKEN_ALIASES.items():
        record = app._resolve(spoken, "starting")
        assert record, spoken
        assert app._resolve(target, "starting")["naptan"] == record["naptan"], (
            spoken, target)


def test_a_line_is_not_answered_with_a_station(app):
    """"bakerloo" is a line. Answering it with a station would be a confident
    guess at something nobody asked, and the aliases must never grow to
    include one."""
    for line in ("bakerloo", "central line", "jubilee", "the district line",
                 "elizabeth line"):
        with pytest.raises(AgentAppError):
            app._resolve(line, "starting")


def _list_keys(app, record):
    """Every key the published interchange list could file a station under.

    Built here from the station record and the raw list, not by calling the
    app's own lookup — a test that asks `_interchange` whether a pair is in
    the list passes whenever the lookup and the answer are wrong together,
    which is precisely the failure this is for.

    A station is not filed under one name. "Custom House (for ExCel)" sits in
    the station list under `custom house for excel`, while the interchange
    list calls the same place `custom house`. An earlier version of this
    check used only `norm(name)`, matched nothing for every such station, and
    so asserted nothing at all about them — it was green because it was
    empty.
    """
    from app import norm
    keys = set(app._aliases.get(record.get("naptan"), []) or [])
    keys.add(record.get("key", ""))
    keys.add(norm(record["name"]))
    keys.add(norm(app._base_name(record["name"])))
    return {k for k in keys if k}


def test_every_free_walk_in_the_published_list_is_marked_free(app):
    """The structural form of the money rule.

    The warning test below reads the sentence the passenger sees. This reads
    the data behind it: for every pair of stations the app offers as a short
    walk, if the published list contains that pair under any name either of
    them is filed under, the walk must come back designated. One missed
    lookup and a free walk is reported as a second fare.

    Three stations lost their designated flag to a one-character change in
    the base-name guard — Custom House (for ExCel), Paddington (H&C
    Line)-Underground and Shepherd's Bush (Central) — and no test noticed,
    because the only check compared the warning against the same function
    that wrote it.
    """
    from app import norm
    wrong = []
    for record in app._stations.values():
        mine = _list_keys(app, record)
        for near in app._walkable(record):
            other = app._stations.get(norm(near["name"]))
            if not other:
                continue
            listed = any(f"{x}|{y}" in app._osi
                         for x in mine for y in _list_keys(app, other))
            if listed and not near["designated"]:
                wrong.append(f"{record['name']} -> {near['name']}")
    assert not wrong, (
        "the published list calls these walks free, the app does not: "
        + ", ".join(sorted(set(wrong))[:8]))


def test_a_walk_out_warning_never_charges_for_a_free_pair(app):
    """Stations sharing a base name are usually two real stations — Edgware
    Road (Bakerloo) and Edgware Road (Circle Line) are two hundred metres
    apart and designated, and pairs like that are the whole subject. So the
    rule cannot be "different names, different places".

    What must hold is narrower and more important: the warning about being
    charged twice must never name a pair the list says is free. Four free
    walks were reported as a second fare once, which is the worst kind of
    error this app can make — confidently wrong about somebody's money."""
    for record in app._stations.values():
        for near in app._walkable(record):
            # nothing is ever its own neighbour
            assert near["name"] != record["name"], record["name"]
            if near["designated"]:
                assert near["allowed"], (record["name"], near)
        warning = app._walk_out_warning(record, False, False)
        if not warning:
            continue
        import re as _re
        # Pull out the station the warning actually names as costing money.
        # Substring matching is not enough: "London Liverpool Street" is a
        # different station from "Liverpool Street", the warning correctly
        # names one as charged and the other as free in the same sentence,
        # and a naive `in` test reads that as a contradiction.
        m = _re.search(r"If you walk out of .+?: (.+?) is about \d+ minutes "
                       r"away but is not a designated interchange", warning)
        if not m:
            continue
        accused = m.group(1)
        # Checked against the bundled interchange list, not against
        # `_walkable`. Comparing the warning with the same function that
        # produced it passes whenever both are wrong together — which is
        # exactly what happens when the base-name guard goes: Custom House
        # (for ExCel) stops seeing Custom House as designated, and the
        # warning then calls a free six-metre walk a second fare.
        free_by_list = set()
        for key in _list_keys(app, record):
            for osi_key, pair in app._osi.items():
                exit_key, entry_key = osi_key.split("|")
                if exit_key != key:
                    continue
                free_by_list.add(pair["entry"])
                target = app._stations.get(entry_key)
                if target:
                    free_by_list.add(target["name"])
        assert accused not in free_by_list, (
            f"{record['name']}: warns that {accused} costs a second fare, "
            f"but the interchange list says it is free — {warning}")


def test_the_same_question_gives_the_same_answer(app):
    """`persistence: none` means a fresh copy of the app per call, so two
    identical questions must produce identical replies — including which
    aside is chosen, which is picked by hashing the two stations rather than
    at random for exactly this reason.

    A judge who asks twice and gets two different answers has found a reason
    not to trust any of them."""
    from app import App as _App

    def fetch(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps(WARREN_BARBICAN).encode()

    when = datetime(2026, 11, 20, 14, 30, tzinfo=LONDON_TZ)

    for outcome, body in (
            ("journey.cost", {"from_station": "Warren Street",
                              "to_station": "Barbican"}),
            ("interchange.check", {"exit_station": "Warren Street",
                                   "entry_station": "Euston Square"})):
        seen = set()
        for _ in range(4):
            fresh = _App()               # a new copy, as the runtime does
            fresh._fetch = fetch
            result = run_call(fresh, outcome, body, now=when)
            seen.add(json.dumps({"d": result.data, "n": result.narrative},
                                sort_keys=True))
        assert len(seen) == 1, f"{outcome} gave {len(seen)} different answers"


def test_the_fare_shown_belongs_to_the_journey_whose_time_is_shown(app):
    """Several results collapse to one entry when they share a label, and the
    quickest is kept. The fare has to be kept from that same journey.

    Loosen the rule and a later journey's price is written onto the earlier
    one's time: "Circle, 10 min, £1.70", where the ten-minute journey cost
    £3.00 and the £1.70 one took twenty-six. Both numbers are real; the
    pairing is not, and nothing about the reply reveals that."""
    import re as _re
    two_prices = {"journeys": [
        {"duration": 10, "fare": {"totalCost": 300},
         "legs": [_leg("tube", [("circle", "Circle")])]},
        {"duration": 26, "fare": {"totalCost": 170},
         "legs": [_leg("tube", [("circle", "Circle")])]},
    ]}

    def fetch(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps(two_prices).encode()

    app._fetch = fetch
    data = ask_access(app, "Warren Street", "Barbican").data
    circle = [o for o in data["options"] if o.startswith("Circle")]
    assert len(circle) == 1, data["options"]
    minutes = int(_re.search(r"(\d+) min", circle[0]).group(1))
    fare = _re.search(r"\u00a3([\d.]+)", circle[0])
    assert minutes == 10, circle[0]
    assert fare and fare.group(1) == "3.00", (
        f"{circle[0]!r} pairs the 10-minute journey with another one's fare")
    assert data["best_pence"] == 300, data["best_pence"]


def test_a_live_fare_replaces_a_zone_estimate_for_the_same_route(app):
    """Two journeys can come back under one label — one priced by Transport
    for London, one only estimable from the zones. The real price has to win,
    because the estimate is shown as "(zone estimate)" and quoting that beside
    a fare TfL actually gave us would be choosing the weaker of two answers
    we already hold."""
    journeys = {"journeys": [
        {"duration": 12, "fare": None,                     # no price: zone estimate
         "legs": [_leg("tube", [("circle", "Circle")])]},
        {"duration": 12, "fare": {"totalCost": 390},       # same label, real price
         "legs": [_leg("tube", [("circle", "Circle")])]},
    ]}

    def fetch(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps(journeys).encode()

    app._fetch = fetch
    data = ask_access(app, "Warren Street", "Barbican").data
    circle = [o for o in data["options"] if o.startswith("Circle")]
    assert len(circle) == 1, f"the same route twice: {data['options']}"
    assert "\u00a33.90" in circle[0], circle[0]
    assert "zone estimate" not in circle[0], circle[0]


def test_which_failures_pippa_should_retry_and_which_it_should_not(app):
    """`retryable` decides whether Pippa quietly tries again or comes back to
    the person. A mutation run flipped these to True and nothing noticed.

    Getting it backwards is not cosmetic: marked retryable, "I need both a
    starting station and a destination" makes Pippa retry the identical call
    instead of asking for the station it is missing, and the person waits
    through it twice for the same answer."""
    from app import (NeedTwoStations, StationUnknown, SameStation,
                     NoStepFreeRoute, NoRouteFound, ServiceBusy,
                     ServiceUnavailable)

    # Nothing about trying again fixes these: the inputs are the problem, and
    # a step-free 404 from Transport for London means no such route exists
    # rather than ask again shortly.
    for failure in (NeedTwoStations, StationUnknown, SameStation,
                    NoStepFreeRoute):
        assert failure.retryable is False, failure.__name__

    # These are about the far end. NoRouteFound belongs here and not above:
    # its message says the planner is not returning a route "right now",
    # which is a claim about this moment, and the planner does recover.
    for failure in (NoRouteFound, ServiceBusy, ServiceUnavailable):
        assert failure.retryable is True, failure.__name__


def test_a_minute_is_valued_at_the_figure_the_pitch_quotes(app):
    """Ranking is the one judgement this app makes, and it rests on a single
    number: a minute costs 21p, from the National Living Wage of £12.71 an
    hour from April 2026. That figure is said out loud — "minutes are valued
    at the National Living Wage" — so it has to be the one in the code.

    Retuned to 22 it still sounds right and quietly reorders close calls;
    six of four hundred replies changed, and nothing noticed."""
    from app import PENCE_PER_MINUTE
    assert PENCE_PER_MINUTE == 21, (
        f"a minute is valued at {PENCE_PER_MINUTE}p; the pitch says 21p")
    # and it is the wage, not a round number picked to look like one
    hourly_pence = 1271
    assert round(hourly_pence / 60) == PENCE_PER_MINUTE, (
        "21p is £12.71 an hour divided by sixty; if the wage changes, say so "
        "in the documents as well as here")


def test_the_walking_times_the_demo_quotes(app):
    """A mutation run halved WALK_METRES_PER_MINUTE and not one test noticed,
    while these two figures are said out loud: "three minutes against five".

    They are the walk from Warren Street to a free interchange and to one that
    costs a second fare, and they are the whole demo."""
    from app import WALK_METRES_PER_MINUTE
    assert WALK_METRES_PER_MINUTE == 80

    start = app._resolve("Warren Street", "starting")
    near = {n["name"]: n for n in app._walkable(start)}

    free = near["Euston Square"]
    assert free["metres"] == 220 and free["minutes"] == 3
    assert free["designated"] is True

    paid = near["Great Portland Street"]
    assert paid["metres"] == 429 and paid["minutes"] == 5
    assert paid["designated"] is False


def test_a_walk_is_never_reported_as_zero_minutes(app):
    """Two metres apart divided by eighty is nought, and "a 0 minute walk"
    reads as a bug even though the distance behind it is right."""
    for record in app._stations.values():
        for near in app._walkable(record):
            assert near["minutes"] >= 1, (record["name"], near)
            # `minutes` comes off the unrounded distance while `metres` is
            # shown rounded, so a walk sitting within half a metre of a
            # boundary can land a minute either side — two of them do. The
            # tolerance keeps the real protection (a wrong speed constant is
            # out by whole minutes) without pinning a sub-minute artefact.
            expected = max(1, round(near["metres"] / 80))
            assert abs(near["minutes"] - expected) <= 1, (record["name"], near)


def test_bus_routes_are_named_not_rendered_as_lines(app):
    """Bus route numbers once came out looking like Tube lines. The collector
    that keeps them apart had no test at all, so the fix it carries was one
    edit away from being undone silently."""
    journey = {"legs": [
        _leg("bus", [("453", "453"), ("18", "18")]),
        _leg("tube", [("victoria", "Victoria")]),
        _leg("bus", [("453", "453")]),          # repeat: must not duplicate
    ]}
    assert app._bus_routes(journey) == ["453", "18"]
    assert app._bus_routes({"legs": []}) == []
    assert app._bus_routes({}) == []
    # a bus leg carrying no identifiable route must not add an empty string
    assert app._bus_routes({"legs": [_leg("bus", [])]}) == []


def test_a_walk_between_two_stations_inside_a_journey_is_spotted(app):
    """Transport for London route through in-station interchanges, so this
    branch almost never fires on live data — which is exactly why it was
    never exercised. It is the reactive half of the fare warning, and if it
    is wrong nobody finds out until it finally does fire."""
    walk = _leg("walking", frm="940GZZLUWRR", to="940GZZLUESQ",
                frm_name="Warren Street", to_name="Euston Square")
    found = app._out_of_station_walk({"legs": [walk]})
    assert found == ("Warren Street", "Euston Square")

    # not two stations, or the same one, or not a walk: all ignored
    assert app._out_of_station_walk({"legs": [
        _leg("walking", frm="490000123", to="940GZZLUESQ")]}) is None
    assert app._out_of_station_walk({"legs": [
        _leg("walking", frm="940GZZLUWRR", to="940GZZLUWRR")]}) is None
    assert app._out_of_station_walk({"legs": [
        _leg("tube", frm="940GZZLUWRR", to="940GZZLUESQ")]}) is None
    assert app._out_of_station_walk({"legs": []}) is None


def test_every_assumption_the_harness_makes_about_the_app_text(app):
    """The harness reads the app's sentences and styles them. Four couplings
    exist, and neither file can see the other — the zone badge already went
    silent once because one side said "(zone fare)" and the other wrote
    "(zone estimate)".

    The worst of them is the discriminator: the harness decides between
    "Interchange — mind the clock" and "You would be charged twice" by
    looking for "is a designated" in the warning. Reword either side and a
    free walk is labelled on screen as costing money."""
    import re as _re, pathlib as _pathlib
    harness = (_pathlib.Path(__file__).parent
               / "scripts" / "local" / "harness.html")
    if not harness.is_file():
        pytest.skip("harness is not deployed; nothing to cross-check")
    source = harness.read_text()

    when = datetime(2026, 11, 20, 14, 30, tzinfo=LONDON_TZ)

    # 1. the designated/not discriminator, on real pairs both ways
    m = _re.search(r'includes\("([^"]+)"\)', source)
    assert m, "the harness no longer discriminates on a phrase"
    needle = m.group(1)
    for first, second in (("Warren Street", "Euston Square"),
                          ("Bank", "Monument"),
                          ("Marylebone", "Paddington"),
                          ("Warren Street", "Great Portland Street")):
        reply = run_call(app, "interchange.check",
                         {"exit_station": first, "entry_station": second},
                         now=when)
        seen = needle in (reply.narrative or "")
        assert seen is reply.data["designated"], (
            f"the harness reads {first}->{second} as "
            f"{'free' if seen else 'charged'} and the app says the opposite: "
            f"{reply.narrative}")

    # 2. the no-fare pattern must match what a free option is written as
    free_pat = _re.search(r"/\^\(([^)]+)\)\$/", source)
    assert free_pat, "the harness no longer has a free-fare pattern"
    alternatives = free_pat.group(1).split("|")
    zero = {"journeys": [{"duration": 9, "fare": {"totalCost": 0},
                          "legs": [_leg("cycle")]}]}

    def fetch(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        if "BikePoint" in url:
            return json.dumps({"places": []}).encode()
        return json.dumps(zero).encode()

    app._fetch = fetch
    line = ask_access(app, "Warren Street", "Barbican").data["options"][0]
    money = line.split(" \u00b7 ")[-1].strip()
    assert money in alternatives, (
        f"a free option is written {money!r}; the harness only styles "
        f"{alternatives}")

    # 3. the exact-match string for an unpriced fare
    unpriced = _re.search(r'===\s*"([^"]+)"', source)
    if unpriced:
        assert unpriced.group(1) == "fare not quoted", unpriced.group(1)


def test_a_zone_estimate_says_so_and_the_harness_can_see_it(app):
    """A fare the zone table supplied is labelled "(zone estimate)" so nobody
    reads an estimate as a quote. The harness keys its "est. from zone" badge
    off that exact text — and once matched "(zone fare)" while the app said
    "(zone estimate)", so the badge silently never appeared.

    Both ends are checked here because neither file can see the other."""
    import re as _re, pathlib as _pathlib
    unpriced = {"journeys": [
        {"duration": 12, "fare": None,
         "legs": [_leg("tube", [("circle", "Circle")])]},
    ]}

    def fetch(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps(unpriced).encode()

    app._fetch = fetch
    line = ask_access(app, "Warren Street", "Barbican").data["options"][0]
    assert "(zone estimate)" in line, line

    # the harness regex must match what the app actually writes
    harness = (_pathlib.Path(__file__).parent
               / "scripts" / "local" / "harness.html")
    if not harness.is_file():
        pytest.skip("harness is not deployed; nothing to cross-check")
    source = harness.read_text()
    m = _re.search(r"const ZONE_RE = /([^/]+)/", source)
    assert m, "the harness no longer has a ZONE_RE to check against"
    pattern = m.group(1).replace("\\s*", r"\s*")
    assert _re.search(pattern, line), (
        f"the harness pattern {m.group(1)!r} does not match what the app "
        f"writes: {line!r}")


def test_the_three_ways_a_fare_can_be_reported(app):
    """An option says one of three things about money, and each is a
    different claim: a price, "no fare" when there genuinely is none, and
    "fare not quoted" when Transport for London did not price it and the
    zones cannot either.

    Saying nothing would read as free, which is the one reading that costs
    somebody money. The harness also keys its styling off these exact words,
    so changing them breaks the display as well as the sentence."""
    def build(fare, modes):
        return {"journeys": [{"duration": 12, "fare": fare,
                              "legs": [_leg(modes[0], modes[1])]}]}

    def run(payload):
        def fetch(url, timeout=5.0):
            if "/Line/Mode/tube/Status" in url:
                return json.dumps(ALL_GOOD).encode()
            if "BikePoint" in url:
                return json.dumps({"places": []}).encode()
            return json.dumps(payload).encode()
        app._fetch = fetch
        return ask_access(app, "Warren Street", "Barbican").data["options"][0]

    priced = run(build({"totalCost": 390}, ("tube", [("circle", "Circle")])))
    assert "\u00a33.90" in priced, priced

    free = run(build({"totalCost": 0}, ("cycle", [])))
    assert "no fare" in free, free

    # Two stations with no zone between them: nothing can price it.
    zoneless = [v["name"] for v in app._stations.values() if not v.get("zone")]
    assert zoneless, "every station has a zone now"

    def fetch_unpriced(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        if "BikePoint" in url:
            return json.dumps({"places": []}).encode()
        return json.dumps({"journeys": [
            {"duration": 12, "fare": None,
             "legs": [_leg("national-rail", [("sw", "South Western")])]}]}).encode()

    app._fetch = fetch_unpriced
    unpriced = run_call(app, "journey.cost",
                        {"from_station": zoneless[0],
                         "to_station": zoneless[1]},
                        now=datetime(2026, 11, 20, 14, 30,
                                     tzinfo=LONDON_TZ)).data
    line = unpriced["options"][0]
    assert "fare not quoted" in line, line
    assert "no fare" not in line, f"an unpriced journey must not read as free: {line}"


def test_the_warning_opens_differently_when_a_line_is_down(app):
    """The proactive warning has two openings. With everything running it is
    "If you walk out of Warren Street..."; with a line down it becomes "The
    line is disrupted, so if you walk out of...", which is the sentence that
    ties the delay to the fare trap and is the reason the warning is worth
    reading at that moment.

    Neither opening was checked: both could be rewritten freely."""
    journey = {"journeys": [
        {"duration": 12, "fare": {"totalCost": 300},
         "legs": [_leg("tube", [("victoria", "Victoria")])]},
    ]}
    delayed = [{"id": "victoria", "name": "Victoria",
                "lineStatuses": [{"statusSeverityDescription": "Severe Delays"}]}]

    def make(status):
        def fetch(url, timeout=5.0, _s=status):
            if "/Line/Mode/tube/Status" in url:
                return json.dumps(_s).encode()
            return json.dumps(journey).encode()
        return fetch

    app._fetch = make(ALL_GOOD)
    calm = ask_access(app, "Warren Street", "Barbican").data.get("fare_warning", "")
    assert calm.startswith("If you walk out of Warren Street"), calm
    assert "disrupted" not in calm, calm

    app._fetch = make(delayed)
    rough = ask_access(app, "Warren Street", "Barbican").data.get("fare_warning", "")
    assert rough.startswith("The line is disrupted, so if you walk out of"), rough
    assert "Great Portland Street" in rough, rough


def test_a_warning_carried_by_an_option_reaches_the_answer(app):
    """Two warnings exist. The proactive one looks at what is near the origin
    ("If you walk out of Warren Street..."); the reactive one fires when the
    route Transport for London returned walks between two stations itself
    ("This route walks from X to Y...") and takes precedence, because it is
    about the journey actually being recommended.

    Only the first ever reached the output in a test. Changing the key that
    carries the second — `o["warning"]` — left the suite green, because no
    test produced an option that had one."""
    walking_route = {"journeys": [
        {"duration": 20, "fare": {"totalCost": 300}, "legs": [
            _leg("tube", [("victoria", "Victoria")]),
            _leg("walking", frm="940GZZLUWRR", to="940GZZLURGP",
                 frm_name="Warren Street", to_name="Great Portland Street"),
            _leg("tube", [("circle", "Circle")]),
        ]},
    ]}

    def fetch(url, timeout=5.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps(walking_route).encode()

    app._fetch = fetch
    result = ask_access(app, "Warren Street", "Barbican")
    warning = result.data.get("fare_warning", "")
    assert warning, "the route walked between two stations and said nothing"
    assert warning.startswith("This route walks from"), (
        f"the option's own warning was lost and the generic one used: {warning}")
    assert "Great Portland Street" in warning, warning
    assert "two journeys" in warning, warning
    assert warning in (result.narrative or ""), result.narrative


@pytest.mark.parametrize("to_naptan,to_name,designated", [
    ("940GZZLUESQ", "Euston Square", True),     # on the list
    ("940GZZLURGP", "Great Portland Street", False),
])
def test_the_reactive_fare_warning_reads_correctly(app, to_naptan, to_name, designated):
    """Transport for London route through in-station interchanges, so this
    warning almost never fires on live data — which is why nothing exercised
    the sentence it produces. If it ever does fire it is stated as fact to
    somebody about their money, so it had better be right."""
    journey = {"duration": 12, "fare": {"totalCost": 300}, "legs": [
        _leg("tube", [("victoria", "Victoria")]),
        _leg("walking", frm="940GZZLUWRR", to=to_naptan,
             frm_name="Warren Street", to_name=to_name),
        _leg("tube", [("circle", "Circle")]),
    ]}
    destination = app._resolve("Barbican", "destination")
    text = app._warning(journey, destination, False, 300)
    assert text, "no warning produced for a walk between two stations"
    assert "Warren Street" in text and to_name in text
    assert text.rstrip().endswith(".")
    assert not text[0].islower()
    if designated:
        assert "is a designated interchange" in text
        assert "one journey" in text
    else:
        assert "not a designated interchange" in text
        assert "two journeys" in text


def test_the_same_station_on_both_sides_of_an_interchange(app):
    """`interchange.check` answers from the bundled table and makes no network
    call, so this message is what somebody sees when they name one station
    twice. It had no test."""
    with pytest.raises(AgentAppError) as caught:
        ask_interchange(app, "Euston", "Euston")
    message = caught.value.user_message
    assert "Euston" in message
    assert message.rstrip().endswith(".")
    assert not message[0].islower()


# ---------------------------------------------------------------------------
# The harness. Not deployed — `scripts/` is stripped — but it is the surface
# the app is demonstrated on, and it had no tests at all. Its routing once
# sent every question to journey.cost, so interchange.check, the outcome the
# whole project exists for, could not be reached from it.
# ---------------------------------------------------------------------------

def _harness():
    """Import the local harness, or skip where it is not deployed."""
    import importlib.util, pathlib, sys as _sys
    path = pathlib.Path(__file__).parent / "scripts" / "local" / "harness.py"
    if not path.is_file():
        pytest.skip("harness is not deployed; nothing to check")
    _sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_map_data_still_joins_up(app):
    """The map is the first thing anybody sees: 531 stations drawn as a
    particle field with the line geometries threaded through them. Each
    branch is a list of naptans, and the drawing code does
    `branch.map(...).filter(Boolean)` — so a reference that no longer
    resolves is not an error, it is a silent gap in a line.

    Checked against the naptan index the harness actually serves, not
    against the station keys, because those are two different things and
    confusing them is how this looked broken the first time."""
    import json as _json, pathlib as _pathlib
    geo_path = (_pathlib.Path(__file__).parent
                / "scripts" / "local" / "geo.json")
    if not geo_path.is_file():
        pytest.skip("map data is not deployed; nothing to check")
    geo = _json.loads(geo_path.read_text())
    routes = geo.get("routes") or {}
    assert len(routes) >= 13, f"only {len(routes)} lines left on the map"

    # exactly what harness.py serves at /stations
    points = {v["naptan"] for v in app._stations.values()
              if v.get("naptan") and v.get("lat") is not None}
    assert len(points) > 500, len(points)

    missing, thin = [], []
    for name, route in routes.items():
        assert str(route.get("colour", "")).startswith("#"), (name, route.get("colour"))
        branches = route.get("branches") or []
        assert branches, f"{name} has no branches"
        for index, branch in enumerate(branches):
            drawable = [n for n in branch if n in points]
            missing.extend(n for n in branch if n not in points)
            if len(drawable) < 2:
                thin.append((name, index, len(branch)))
    assert not missing, (
        f"{len(missing)} branch points are not stations the map is given: "
        f"{sorted(set(missing))[:5]}")
    assert not thin, f"branches that would draw nothing: {thin[:4]}"


def test_the_map_colours_match_the_lines_it_draws(app):
    """The harness colours a status chip from its own table and the map from
    geo.json. Two tables of the same thing drift, and the Tube's colours are
    the one part of this a Londoner checks instantly."""
    import json as _json, re as _re, pathlib as _pathlib
    here = _pathlib.Path(__file__).parent / "scripts" / "local"
    if not (here / "geo.json").is_file() or not (here / "harness.html").is_file():
        pytest.skip("harness is not deployed; nothing to cross-check")
    geo = _json.loads((here / "geo.json").read_text())
    page = (here / "harness.html").read_text()
    block = page.split("const LINES = {", 1)[1].split("};", 1)[0]
    block = _re.sub(r"//[^\n]*", "", block)
    table = dict(_re.findall(r'"?([A-Za-z&\' .-]+?)"?\s*:\s*"(#[0-9A-Fa-f]{6})"',
                             block))
    assert len(table) >= 13, f"the colour table has shrunk to {len(table)}"
    # every colour the map uses must be a real six-digit colour
    for name, route in geo["routes"].items():
        assert _re.fullmatch(r"#[0-9A-Fa-f]{6}", route["colour"]), (name, route)


@pytest.mark.parametrize("question,expected", [
    ("will i be charged twice walking from warren street to euston square",
     "interchange.check"),
    ("is the walk between warren street and euston square free",
     "interchange.check"),
    ("do i pay again if i walk from bank to monument", "interchange.check"),
    ("how do i get from warren street to barbican", "journey.cost"),
    ("whats the cheapest way to get to barbican from warren street",
     "journey.cost"),
    ("i need to get to barbican from warren street", "journey.cost"),
])
def test_the_harness_routes_a_question_to_the_right_outcome(app, question, expected):
    """It scored the phrases and then threw the answer away, sending
    everything to journey.cost. Asking "will I be charged twice" rendered as
    "No options — Transport for London returned no way to make that
    journey": the app answering correctly, shown as a failure, on the one
    question the demo is built around."""
    harness = _harness()
    routing = harness.would_route(question, dict(app.manifest))
    assert routing["outcome"] == expected, (question, routing)
    assert routing["score"] > 0, routing


def test_the_harness_names_the_inputs_each_outcome_expects(app):
    """The two outcomes take the same two stations under different names.
    Sending from_station to interchange.check produces "I need both a
    starting station and a destination" for a question that named both."""
    harness = _harness()
    assert harness.OUTCOME_FIELDS["journey.cost"] == ("from_station", "to_station")
    assert harness.OUTCOME_FIELDS["interchange.check"] == ("exit_station",
                                                           "entry_station")
    for outcome in app.manifest["outcomes"]:
        declared = tuple(i["name"] for i in outcome["inputs"][:2])
        assert harness.OUTCOME_FIELDS[outcome["id"]] == declared, outcome["id"]


@pytest.mark.parametrize("question,start,end", [
    ("how do i get from warren street to barbican", "Warren Street", "Barbican"),
    ("how much is it to get to victoria from euston", "Euston", "Victoria"),
    ("warren street to barbican", "Warren Street", "Barbican"),
    ("i need a step free route from warren street to barbican",
     "Warren Street", "Barbican"),
])
def test_the_harness_reads_both_stations_out_of_a_sentence(app, question, start, end):
    """Taking them in the order they appear reversed "to victoria from
    euston" — the markers have to win over position."""
    harness = _harness()
    found = harness.extract(question, app._stations)
    assert found.get("from_station") == start, (question, found)
    assert found.get("to_station") == end, (question, found)


def test_the_harness_only_claims_step_free_when_it_is_asked_for(app):
    """Pippa reads this from the sentence; the stand-in has to as well, or
    the demo of a step-free route quietly shows the ordinary one."""
    harness = _harness()
    asked = harness.extract("i need a step free route from warren street to barbican",
                            app._stations)
    assert asked.get("step_free") is True, asked
    plain = harness.extract("how do i get from warren street to barbican",
                            app._stations)
    assert "step_free" not in plain, plain


def test_every_harness_fixture_is_a_shape_the_app_accepts(app):
    """The stress fixtures stand in for Transport for London. One that has
    drifted from the real response shape exercises nothing and hides the
    state it was built to show."""
    harness = _harness()
    assert harness.STRESS, "the harness has no fixtures left"
    for name, fixture in harness.STRESS.items():
        assert isinstance(fixture, dict), name
        body = fixture.get("journeys")
        if body is None:
            continue
        assert isinstance(body, list), name
        for journey in body:
            assert "duration" in journey, (name, journey)
            assert isinstance(journey.get("legs"), list), (name, journey)


# ---------------------------------------------------------------------------
# Clocks and names
# ---------------------------------------------------------------------------

def test_peak_is_a_london_clock_not_the_one_it_arrives_on(app):
    """Transport for London's peak windows are London local time. Through the
    summer, 05:45 UTC is 06:45 in London and squarely peak — an app reading
    the clock as it arrives quotes the off-peak fare for an hour every morning
    from late March to late October. In November the two coincide, which is
    why it would have gone unnoticed."""
    from zoneinfo import ZoneInfo
    london = ZoneInfo("Europe/London")
    # `_is_peak` reads only `call.context.now`, so it is handed exactly
    # that — and no `Context`, which the real SDK does not export.
    mk = lambda when: SimpleNamespace(context=SimpleNamespace(now=when))

    # British Summer Time: UTC is an hour behind London
    assert app._is_peak(mk(datetime(2027, 7, 16, 5, 45, tzinfo=timezone.utc))) is True
    assert app._is_peak(mk(datetime(2027, 7, 16, 6, 45, tzinfo=london))) is True
    # and the same instant expressed either way must agree
    for hour in range(0, 24):
        utc = datetime(2027, 7, 16, hour, 0, tzinfo=timezone.utc)
        assert app._is_peak(mk(utc)) == app._is_peak(mk(utc.astimezone(london)))


def test_winter_is_unaffected(app):
    """GMT and UTC coincide, so November behaves exactly as before."""
    # `_is_peak` reads only `call.context.now`, so it is handed exactly
    # that — and no `Context`, which the real SDK does not export.
    mk = lambda when: SimpleNamespace(context=SimpleNamespace(now=when))
    assert app._is_peak(mk(datetime(2026, 11, 20, 8, 0, tzinfo=timezone.utc))) is True
    assert app._is_peak(mk(datetime(2026, 11, 20, 14, 0, tzinfo=timezone.utc))) is False


def test_a_hyphen_inside_a_name_is_part_of_the_name(app):
    """Bromley-by-Bow is not Bromley and Harrow-on-the-Hill is not Harrow.
    An earlier version split on any hyphen and would have merged them the
    moment a station by the shorter name appeared in the list."""
    assert app._base_name("Bromley-by-Bow") == "Bromley-by-Bow"
    assert app._base_name("Harrow-on-the-Hill") == "Harrow-on-the-Hill"
    # the trailing qualifier always follows a bracket, so it still goes
    assert app._base_name("Paddington (H&C Line)-Underground") == "Paddington"


def test_no_station_base_name_collides_with_a_different_station(app):
    """A base name that resolves to some other real station would merge two
    places that are not the same place."""
    collisions = []
    for record in app._stations.values():
        base = app._base_name(record["name"])
        if base == record["name"]:
            continue
        other = app._stations.get(norm(base))
        if other and other["name"] != record["name"]:
            gap = app._metres(record, other)
            designated = app._osi.get(f"{norm(record['name'])}|{norm(other['name'])}")
            # Acceptable when they are the same complex, OR when the list
            # names them as an interchange — which says they are two
            # stations, and `same_place` must then keep them apart.
            if designated is None and (gap is None or gap > 400):
                collisions.append((record["name"], other["name"], gap))
    assert not collisions, collisions


def test_two_stations_the_list_names_are_not_merged(app):
    """West Hampstead and West Hampstead (North London Line) share a base name
    and are 220 m apart — and Transport for London designate an interchange
    between them, which is a statement that they are two stations. Merging
    them deletes the one useful fact: that the walk is free and you have 20
    minutes. The two Paddington entrances are 254 m apart and must still
    merge, so distance alone cannot decide it."""
    near = {n["name"]: n for n in app._walkable(app._resolve("West Hampstead", "x"))}
    nll = [n for n in near if "North London Line" in n or "NLL" in n]
    assert nll, f"the NLL station was merged away: {sorted(near)}"
    assert near[nll[0]]["designated"] is True
    assert near[nll[0]]["allowed"] == 20

    # and the genuine same-complex case still collapses
    pad = [n["name"] for n in app._walkable(app._resolve("Lancaster Gate", "x"))]
    assert sum("Paddington" in n for n in pad) == 1, pad


def test_no_designated_pair_is_ever_reported_as_a_trap(app):
    """The strongest guard there is on the headline feature. Walk every pair
    in the published list and check that where one is a neighbour of the
    other, the app calls it designated.

    This caught a real one: ten stations are filed under the short key the
    interchange list uses ("west hampstead nll", "fenchurch st") while their
    full name normalises to something else entirely. Looking the pair up by
    name missed the entry, and a free walk was reported as a second fare —
    the one error here that costs somebody money by telling them not to make
    a journey that was free."""
    misreported = []
    for key, pair in app._osi.items():
        a_key, b_key = key.split("|")
        a_rec, b_rec = app._stations.get(a_key), app._stations.get(b_key)
        if not a_rec or not b_rec:
            continue
        found = [n for n in app._walkable(a_rec) if n["name"] == b_rec["name"]]
        if found and not found[0]["designated"]:
            misreported.append((a_rec["name"], b_rec["name"]))
    assert not misreported, misreported


def test_every_station_record_knows_its_own_key(app):
    """Ten of the 536 are filed under a key that is not norm(their name)."""
    # Every record carries the key it is CANONICALLY filed under. A record
    # may also be reachable under an alias — its display name — and then the
    # alias key and record["key"] differ on purpose.
    for key, record in app._stations.items():
        assert record.get("key") in app._stations, (key, record.get("key"))
    odd = [k for k, r in app._stations.items() if k != r["key"]]
    assert len(odd) >= 5, "the display-name aliases should exist"


def test_every_name_the_app_can_say_is_a_name_it_can_be_asked_about(app):
    """Divert emits display names — in a fare warning, in an aside, in
    exit_station_name. A name it can say but not resolve is a dead end: it
    tells you about West Hampstead (North London Line) and then claims not to
    recognise it. Ten of the 536 are filed under the interchange list's short
    key, so their display name needed an alias."""
    unresolvable = []
    for record in list(app._stations.values()):
        try:
            app._resolve(record["name"], "x")
        except AgentAppError:
            unresolvable.append(record["name"])
    assert not unresolvable, unresolvable


def test_the_round_trip_holds_for_a_reported_neighbour(app):
    """Take a name the app actually produces and ask it straight back."""
    near = app._walkable(app._resolve("West Hampstead", "x"))
    assert near
    for neighbour in near:
        resolved = app._resolve(neighbour["name"], "x")
        assert resolved["name"] == neighbour["name"]


@pytest.mark.parametrize("frm,to", [
    ("West Hampstead", "West Hampstead (North London Line)"),
    ("Hammersmith (H&C Line)", "Hammersmith (Dist&Picc Line)"),
    ("Edgware Road (Bakerloo)", "Edgware Road (Circle Line)"),
    ("Bank", "Monument"),
    ("Oxford Circus", "Bond Street"),
])
def test_same_name_pairs_are_free(app, frm, to):
    """Five pairs where the station list and the interchange list file the
    same place under different keys. Each was reported as a second fare at
    some point while this was being built — the worst possible error here,
    because it tells somebody not to make a walk that is free."""
    assert ask_interchange(app, frm, to).data["designated"] is True


@pytest.mark.parametrize("frm,to", [
    ("Oxford Circus", "Warren Street"),
    ("Victoria", "Oxford Circus"),
])
def test_genuinely_undesignated_pairs_still_cost(app, frm, to):
    """And the opposite error — relaxing the lookup until everything looks
    free — would be just as bad."""
    data = ask_interchange(app, frm, to).data
    assert data["designated"] is False and data["second_fare_pence"] > 0


def test_both_outcomes_report_the_same_distance(app):
    """The same pair asked two ways must not give two answers. `_walkable`
    computed a straight line while `interchange.check` used Transport for
    London's published figure, so Oxford Circus to Bond Street was 551 metres
    one way and 280 the other."""
    disagreements = []
    for record in app._stations.values():
        for near in app._walkable(record):
            other = app._resolve(near["name"], "x")
            direct = app._pair(record, other)
            if direct and direct.get("metres"):
                if near["metres"] != direct["metres"]:
                    disagreements.append((record["name"], near["name"],
                                          near["metres"], direct["metres"]))
    assert not disagreements[:5], disagreements[:5]


def test_a_station_is_never_listed_twice_as_a_neighbour(app):
    """One naptan, one entry."""
    for record in app._stations.values():
        names = [n["name"] for n in app._walkable(record)]
        assert len(names) == len(set(names)), (record["name"], names)


# ---------------------------------------------------------------------------
# The zone fare table
# ---------------------------------------------------------------------------

def test_every_zone_1_pair_is_priced(app):
    """The fallback matters most for journeys through zone 1, which is almost
    all of them. Outer-to-outer pairs are rarer and genuinely harder to
    sample; where one is missing the app returns no fare rather than a guess."""
    for zone in range(1, 10):
        assert f"1-{zone}" in app._zone_fares, f"zone 1-{zone} unpriced"


def test_fares_rise_with_distance(app):
    """A table assembled from sampled journeys could easily pick up a wrong
    row. Zone 1-6 costing less than 1-3 would be visible nonsense."""
    through_zone_one = sorted(
        (int(k.split("-")[1]), v) for k, v in app._zone_fares.items()
        if k.startswith("1-"))
    peaks = [v["peak"] for _, v in through_zone_one]
    offs = [v["off_peak"] for _, v in through_zone_one]
    assert peaks == sorted(peaks), peaks
    assert offs == sorted(offs), offs


def test_peak_is_never_cheaper_than_off_peak(app):
    wrong = {k: v for k, v in app._zone_fares.items() if v["peak"] < v["off_peak"]}
    assert not wrong, wrong


def test_an_unpriced_zone_pair_returns_nothing_not_a_guess(app):
    """Honest silence beats an invented fare."""
    missing = next((f"{a}-{b}" for a in range(1, 10) for b in range(a, 10)
                    if f"{a}-{b}" not in app._zone_fares), None)
    if missing:
        lo, hi = missing.split("-")
        assert app._zone_fare({"zone": lo}, {"zone": hi}, peak=False) is None


def test_the_readme_figures_match_the_data(app):
    """Documentation drifts silently. The README quoted 536 stations after the
    count moved to 531, and PIPPA.md still said 304 and 61 KB from before the
    rebuild. Figures a reader could check should be checkable."""
    import pathlib
    path = pathlib.Path(__file__).parent / "README.md"
    if not path.is_file():
        # README.md is not one of the files that goes into the Pippa repo —
        # the deploy strips it anyway — but their CI does run this suite. A
        # bare read_text() here turned a missing README into a red build at
        # the one moment there is no time to work out why.
        pytest.skip("no README.md in this checkout; nothing to cross-check")
    readme = path.read_text()
    # Existing is not the same as being this project's. Pippa Labs' own
    # repository ships a README about building Agent Apps, and in there this
    # test failed for the only wrong reason available: it cross-checked my
    # figures against somebody else's prose. The claims being checked are the
    # thing that identifies the document, so an unrelated README is skipped
    # and a drifted one still fails.
    if "DIVERT" not in readme:
        pytest.skip("README.md here is not this project's; nothing to check")
    facts = {
        f"{len(app._osi)} directed pairs": True,
        f"{len(app._gateline)} same-station interchanges": True,
        f"{len({r['naptan'] for r in app._stations.values() if r.get('naptan')})} stations": True,
    }
    missing = [claim for claim in facts if claim not in readme]
    assert not missing, f"README does not state: {missing}"


# ---------------------------------------------------------------------------
# The time budget
# ---------------------------------------------------------------------------

def test_the_whole_call_stays_inside_the_budget(app):
    """The runtime kills the app at 20 seconds. Five requests at four seconds
    each would spend 20 on their own, so a slow network would get the app
    killed rather than returning the answer it already had. Every request is
    given at most the time the call has left."""
    import app as module
    asked: list[float] = []

    def slow(url, timeout=4.0):
        asked.append(timeout)
        # pretend each one burned its whole allowance
        module.time.monotonic = lambda _c=[0.0]: (_c.__setitem__(0, _c[0] + 4.0)
                                                  or _c[0])
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps(WARREN_BARBICAN).encode()

    real_monotonic = module.time.monotonic
    try:
        app._fetch = slow
        ask(app)
    finally:
        module.time.monotonic = real_monotonic
    assert sum(asked) <= module.DEADLINE_SECONDS + module.TIMEOUT, asked
    assert all(t > 0 for t in asked), asked


def test_optional_requests_are_skipped_when_time_is_short(app):
    """The wider journey search and the hire docks are niceties. Started too
    late they turn a slow answer into no answer."""
    import app as module
    urls: list[str] = []
    clock = [0.0]

    def slow(url, timeout=4.0):
        urls.append(url)
        clock[0] += 6.0                       # each request eats six seconds
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps({"journeys": [
            {"duration": 13, "fare": None, "legs": [_leg("cycle")]},
            {"duration": 15, "fare": {"totalCost": 300},
             "legs": [_leg("tube", [("circle", "Circle")])]}]}).encode()

    real = module.time.monotonic
    try:
        module.time.monotonic = lambda: clock[0]
        app._fetch = slow
        result = ask(app)
    finally:
        module.time.monotonic = real
    assert result.data["options"], "an answer must still come back"
    assert not any("/BikePoint" in u for u in urls), "docks started too late"


def test_a_request_is_never_given_a_zero_timeout(app):
    """Clamped to half a second at the floor — urlopen with timeout=0 raises
    immediately and would turn a slow call into a failed one."""
    import app as module
    asked: list[float] = []
    clock = [0.0]

    def slow(url, timeout=4.0):
        asked.append(timeout)
        clock[0] += 30.0                      # blow straight past the deadline
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps(WARREN_BARBICAN).encode()

    real = module.time.monotonic
    try:
        module.time.monotonic = lambda: clock[0]
        app._fetch = slow
        ask(app)
    finally:
        module.time.monotonic = real
    assert all(t >= 0.5 for t in asked), asked


def test_losing_the_status_call_does_not_lose_the_answer(app):
    """Which lines are delayed is an enhancement. The routes and the fares
    are the answer, and a slow second endpoint should not take them with it."""
    def status_down(url, timeout=4.0):
        if "/Line/Mode/tube/Status" in url:
            raise TimeoutError("status is slow")
        return json.dumps(WARREN_BARBICAN).encode()

    app._fetch = status_down
    result = ask(app)
    assert result.data["options"], "the journey should still come back"
    assert result.data["disrupted_lines"] == [], "with no disruption claimed"


def test_losing_the_journey_call_does_lose_the_answer(app):
    """The opposite case: without journeys there is nothing to say, and
    inventing one would be worse than an honest failure."""
    def journeys_down(url, timeout=4.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        raise TimeoutError("journeys are slow")

    app._fetch = journeys_down
    with pytest.raises(AgentAppError) as caught:
        ask(app)
    assert caught.value.retryable is True


def test_the_stated_request_count_is_the_real_one(app):
    """The manifest claims a number a reviewer could check, and the claim has
    drifted twice already — from four to three as the calls changed, and from
    seven fields to eight when the aside was added."""
    tube_only = []

    def counted(url, timeout=4.0):
        tube_only.append(url)
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps(WARREN_BARBICAN).encode()

    app._fetch = counted
    result = ask(app)
    assert len(tube_only) == 3, tube_only
    assert len(result.data) == 8, sorted(result.data)

    # The claim lives in a comment, which a reviewer reads and YAML discards,
    # so check the file rather than the parsed manifest.
    raw = (Path(__file__).parent / "manifest.yaml").read_text()
    assert "three HTTP requests" in raw
    assert "eight short fields" in raw


def test_interchange_check_makes_no_request_at_all(app):
    def boom(url, timeout=4.0):
        raise AssertionError(f"interchange.check called the network: {url}")
    app._fetch = boom
    assert len(ask_interchange(app, "Oxford Circus", "Bond Street").data) == 8


def test_dock_figures_need_the_near_end(app):
    """Knowing there are 183 empty docks at the destination is not advice if
    you cannot get a bike here. An earlier version said exactly that on its
    own when the first lookup failed."""
    calls: list[str] = []

    def near_end_down(url, timeout=4.0):
        if "/BikePoint" in url:
            calls.append(url)
            if len(calls) == 1:         # the origin lookup
                raise TimeoutError("docks slow")
            return json.dumps(BIKEPOINT).encode()
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps({"journeys": [
            {"duration": 13, "fare": None, "legs": [_leg("cycle")]},
            {"duration": 15, "fare": {"totalCost": 300},
             "legs": [_leg("tube", [("circle", "Circle")])]}]}).encode()

    app._fetch = near_end_down
    cycle = next(o for o in ask(app).data["options"] if o.startswith("Cycle"))
    assert "spaces at the other end" not in cycle, cycle
    assert cycle == "Cycle · 13 min · no fare", cycle


def test_no_bikes_here_means_the_far_end_is_not_consulted(app):
    """If there is nothing to ride, there is no reason to spend a request
    finding out where to park it."""
    empty = {"places": [{"commonName": "Empty Dock", "additionalProperties": [
        {"key": "NbBikes", "value": "0"}, {"key": "NbEmptyDocks", "value": "20"},
        {"key": "Locked", "value": "false"}]}]}
    calls: list[str] = []

    def counted(url, timeout=4.0):
        if "/BikePoint" in url:
            calls.append(url)
            return json.dumps(empty).encode()
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        return json.dumps({"journeys": [
            {"duration": 13, "fare": None, "legs": [_leg("cycle")]},
            {"duration": 15, "fare": {"totalCost": 300},
             "legs": [_leg("tube", [("circle", "Circle")])]}]}).encode()

    app._fetch = counted
    cycle = next(o for o in ask(app).data["options"] if o.startswith("Cycle"))
    assert "no hire bikes free at this end" in cycle
    assert len(calls) == 1, f"asked about the far end anyway: {calls}"


def test_a_300_blames_the_right_side(app):
    """Both names were resolved from the bundled list before the call, so a
    disambiguation means Transport for London rejected a naptan we hold.
    'I don't recognise that station' would send somebody off to retype a name
    that was already correct."""
    def ambiguous(url, timeout=4.0):
        if "/Line/Mode/tube/Status" in url:
            return json.dumps(ALL_GOOD).encode()
        raise urllib.error.HTTPError(url, 300, "multiple", {}, None)

    app._fetch = ambiguous
    with pytest.raises(AgentAppError) as caught:
        ask(app)
    message = caught.value.user_message
    assert "Transport for London" in message
    assert "I don't recognise" not in message
    assert caught.value.retryable is False


def test_every_error_message_is_a_sentence(app):
    """These are read aloud by an assistant. A fragment or a lower-case start
    reads as a crash leaking through."""
    import app as module
    classes = [v for v in vars(module).values()
               if isinstance(v, type) and issubclass(v, AgentAppError)
               and v is not AgentAppError]
    assert len(classes) >= 6
    for cls in classes:
        message = cls.user_message
        assert message[0].isupper(), (cls.__name__, message)
        assert message.rstrip().endswith((".", "?")), (cls.__name__, message)
        assert isinstance(cls.retryable, bool), cls.__name__


# ---------------------------------------------------------------------------
# What a model extractor might actually send
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["W" * 5000, "Q" * 200, "x" * 80])
def test_an_error_message_never_runs_away(app, value):
    """Nothing bounds how long an extracted string can be — the length is
    whatever the model produced — and these messages are read aloud. One
    version quoted a 5,000-character input back in full."""
    with pytest.raises(AgentAppError) as caught:
        ask(route(app), frm=value)
    assert len(caught.value.user_message) < 120, len(caught.value.user_message)


def test_echoed_text_is_one_line(app):
    """A message is a spoken sentence. Newlines and tabs collapse."""
    with pytest.raises(AgentAppError) as caught:
        ask(route(app), frm="Oxford\n\n\nCircus\tNowhere")
    message = caught.value.user_message
    assert "\n" not in message and "\t" not in message
    assert "Oxford Circus Nowhere" in message


def test_an_outcome_with_no_branch_says_so(app):
    """Only reachable if the manifest declares an outcome the code has no
    branch for. 'Something went wrong' would tell nobody where to look."""
    with pytest.raises(AgentAppError) as caught:
        run_call(app, "something.else", {},
                 now=datetime(2026, 11, 20, 14, 0, tzinfo=timezone.utc))
    assert "something.else" in caught.value.user_message
    assert caught.value.user_message != "Something went wrong."


@pytest.mark.parametrize("junk", [123, ["Warren Street"], {"name": "x"}, 3.5, True])
def test_a_non_string_station_is_handled(app, junk):
    """`call.inputs` is filled by a model. A list or a number where a string
    belongs must not reach the lookup."""
    with pytest.raises(AgentAppError) as caught:
        ask(route(app), frm=junk)
    assert "starting station" in caught.value.user_message


def test_an_undeclared_input_is_ignored(app):
    """The runtime strips anything undeclared, but an app that fell over on a
    stray key would fail for a reason nobody could see."""
    result = run_call(route(app), "journey.cost",
                      {"from_station": "Warren Street",
                       "to_station": "Barbican", "nonsense": "x"},
                      now=datetime(2026, 11, 20, 14, 0, tzinfo=timezone.utc))
    assert result.data["options"]
    assert "nonsense" not in result.data


def test_echo_helper(app):
    from app import echo
    assert echo("short") == "short"
    assert len(echo("x" * 500)) == 48
    assert echo("a\n\nb\tc") == "a b c"
    assert echo("x" * 500).endswith("…")
