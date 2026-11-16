"""Roblox Live Analytics — a Streamlit dashboard for live Roblox game statistics.

The app reads Roblox's public, unauthenticated web API. It needs no Roblox
cookie, no Roblox Studio changes and no paid service.

History is recorded only while a browser session is open and is kept in that
session's memory; nothing is written to disk or to a database.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone, tzinfo
from typing import Any, Final, Literal, TypedDict
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

APP_TITLE: Final = "Roblox Live Analytics"

GAMES_URL: Final = "https://games.roblox.com/v1/games"
PLACE_TO_UNIVERSE_URL: Final = "https://apis.roblox.com/universes/v1/places/{place_id}/universe"
GAME_PAGE_URL: Final = "https://www.roblox.com/games/{place_id}"

REFRESH_SECONDS: Final = 10  # automatic refresh interval
TIMER_TOLERANCE_SECONDS: Final = 2  # the refresh timer may fire slightly early
STATS_CACHE_SECONDS: Final = 5  # shortest gap between two real API calls
MAX_BACKOFF_SECONDS: Final = 300  # longest pause after repeated failures
STALE_AFTER_SECONDS: Final = 45  # data older than this is flagged as stale
REQUEST_TIMEOUT: Final = (5, 10)  # (connect, read) seconds
MAX_GAMES: Final = 20  # games monitored per session
MAX_SNAPSHOTS: Final = 8640  # per game; about 24 hours at one per 10 s
MAX_ROBLOX_ID: Final = 2**63 - 1

SERIES_COLORS: Final = {"visits": "#3987e5", "playing": "#d95926"}
FALLBACK_SURFACE_COLOR: Final = "#0f1116"

ID_TYPE_AUTO: Final = "Auto-detect"
ID_TYPE_PLACE: Final = "Place ID"
ID_TYPE_UNIVERSE: Final = "Universe ID"
ID_TYPES: Final = (ID_TYPE_AUTO, ID_TYPE_PLACE, ID_TYPE_UNIVERSE)

ALL_RANGE: Final = "All (session)"
TIME_RANGES: Final[tuple[tuple[str, int], ...]] = (
    ("5 min", 5 * 60),
    ("15 min", 15 * 60),
    ("1 hour", 60 * 60),
    ("6 hours", 6 * 60 * 60),
)

# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #

ErrorKind = Literal["rate_limited", "timeout", "network", "http", "not_found", "bad_response"]
InputKind = Literal["place", "universe", "unknown"]


class ParsedInput(TypedDict):
    kind: InputKind
    id: int


class GameStats(TypedDict):
    universe_id: int
    root_place_id: int | None
    name: str
    creator_name: str
    creator_type: str
    visits: int
    playing: int
    favorites: int | None
    max_players: int | None


class FetchResult(TypedDict):
    fetched_at: float  # Unix time (UTC) of the API response
    games: dict[int, GameStats]


class Snapshot(TypedDict):
    ts: float  # Unix time (UTC)
    visits: int
    playing: int
    favorites: int | None


class TrackedGame(TypedDict):
    stats: GameStats  # latest successful API values
    first: Snapshot  # baseline: first snapshot of this monitoring run
    history: list[Snapshot]  # most recent MAX_SNAPSHOTS snapshots
    peak_playing: int
    peak_ts: float
    error: str | None  # problem specific to this game, if any


class PollState(TypedDict):
    last_attempt: float
    last_success: float | None
    failures: int
    blocked_until: float
    error: str | None
    error_kind: ErrorKind | None


class RobloxAPIError(Exception):
    """A request to the Roblox API failed in a way the user should hear about."""

    def __init__(self, message: str, *, kind: ErrorKind, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.kind: ErrorKind = kind
        self.retry_after = retry_after


class InputError(ValueError):
    """The user's URL or ID could not be turned into a Roblox game."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:])")


def escape_md(text: str) -> str:
    """Escape Markdown so game and creator names are always shown literally."""
    return _MD_SPECIAL.sub(r"\\\1", text)


def to_int(value: Any) -> int | None:
    """Convert an API value to int, or None if it is not a whole number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def user_timezone() -> tuple[tzinfo, str]:
    """Return the viewer's browser time zone, falling back to UTC."""
    try:
        name = st.context.timezone
        if name:
            return ZoneInfo(name), name
    except Exception:  # unknown zone name or no browser context
        pass
    return timezone.utc, "UTC"


def format_clock(ts: float, tz: tzinfo) -> str:
    """Format a Unix time as a local clock time, adding the date if not today."""
    moment = datetime.fromtimestamp(ts, tz)
    clock = moment.strftime("%I:%M:%S %p").lstrip("0")
    zone = moment.tzname() or ""
    if moment.date() != datetime.now(tz).date():
        clock = f"{moment.strftime('%b')} {moment.day}, {clock}"
    return f"{clock} {zone}".strip()


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds} s"
    minutes, _ = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def format_signed(value: int) -> str:
    return f"{value:+,}" if value else "0"


def format_rate(rate: float | None) -> str:
    if rate is None:
        return "—"
    return f"{rate:,.1f}" if abs(rate) >= 10 else f"{rate:,.2f}"


def percent_change(change: int, base: int) -> str | None:
    """Signed percentage text for a change relative to its starting value."""
    if base <= 0:
        return None
    pct = change / base * 100
    if pct == 0:
        return "0%"
    if abs(pct) < 0.01:
        return "<0.01%"  # the card's arrow and colour show the direction
    return f"{pct:+.1f}%" if abs(pct) >= 1 else f"{pct:+.2f}%"


# --------------------------------------------------------------------------- #
# Input parsing
# --------------------------------------------------------------------------- #


def _validated_id(text: str) -> int:
    number = int(text)
    if not 0 < number <= MAX_ROBLOX_ID:
        raise InputError("That ID is out of range. Roblox IDs are positive whole numbers.")
    return number


def parse_game_input(raw: str) -> ParsedInput:
    """Turn a pasted Roblox URL or a bare number into a Place ID or Universe ID.

    Recognised forms include:
      https://www.roblox.com/games/920587237/Adopt-Me            (Place ID)
      https://www.roblox.com/games/start?placeId=920587237       (Place ID)
      https://create.roblox.com/dashboard/creations/experiences/383310974/overview
                                                                 (Universe ID)
      920587237                                                  (bare number)
    """
    text = (raw or "").strip()
    if not text:
        raise InputError("Paste a Roblox game URL or enter a Place ID or Universe ID.")

    compact = re.sub(r"[\s,_]", "", text)
    if compact.isdigit():
        return {"kind": "unknown", "id": _validated_id(compact)}

    if text.lower().startswith("roblox:"):  # app deep link, e.g. roblox://placeId=123
        match = re.search(r"placeid=(\d+)", text, re.IGNORECASE)
        if match:
            return {"kind": "place", "id": _validated_id(match.group(1))}
        raise InputError(
            "That Roblox app link has no placeId. Paste the game's web address instead."
        )

    parsed = urlparse(text if "://" in text else f"https://{text}")
    host = (parsed.hostname or "").lower()
    if not (host == "roblox.com" or host.endswith(".roblox.com")):
        if host == "blox.com" or host.endswith(".blox.com"):  # ro.blox.com short links
            raise InputError(
                "Short share links can't be resolved without signing in. Open the link, "
                "then copy the address of the game page (roblox.com/games/…) and paste that."
            )
        raise InputError(
            "That doesn't look like a Roblox link. Paste a roblox.com game URL, "
            "or enter the game's Place ID or Universe ID."
        )

    query = {key.lower(): values[0] for key, values in parse_qs(parsed.query).items() if values}
    path = parsed.path or ""

    if query.get("universeid", "").isdigit():
        return {"kind": "universe", "id": _validated_id(query["universeid"])}
    if query.get("placeid", "").isdigit():
        return {"kind": "place", "id": _validated_id(query["placeid"])}

    if host == "create.roblox.com":  # Creator Hub links carry the Universe ID
        match = re.search(r"/experiences/(\d+)", path)
        if match:
            return {"kind": "universe", "id": _validated_id(match.group(1))}
        match = re.search(r"/places/(\d+)", path)
        if match:
            return {"kind": "place", "id": _validated_id(match.group(1))}

    match = re.search(r"/games/(\d+)", path)
    if match:
        return {"kind": "place", "id": _validated_id(match.group(1))}

    if path.rstrip("/").lower().endswith("/share") or "code" in query:
        raise InputError(
            "Share links can't be resolved without signing in. Open the link, then copy "
            "the address of the game page (roblox.com/games/…) and paste that."
        )
    raise InputError(
        "No game ID was found in that link. Use the address of the game's page, "
        "for example https://www.roblox.com/games/920587237/Adopt-Me."
    )


# --------------------------------------------------------------------------- #
# Roblox API client
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def http_session() -> requests.Session:
    """One shared HTTP session so connections to Roblox are reused."""
    session = requests.Session()
    session.headers.update(
        {"Accept": "application/json", "User-Agent": "roblox-live-analytics-streamlit/1.0"}
    )
    return session


def _retry_after_seconds(response: requests.Response) -> float | None:
    value = response.headers.get("Retry-After", "").strip()
    return float(value) if value.isdigit() else None


def _error_detail(response: requests.Response) -> str:
    """Pull Roblox's own error message out of an error response, if present."""
    try:
        errors = response.json().get("errors")
        message = errors[0].get("message") if errors else None
        if isinstance(message, str) and message.strip():
            return f" Roblox said: “{message.strip()[:200]}”."
    except (ValueError, AttributeError, IndexError, KeyError, TypeError):
        pass
    return ""


def get_json(url: str) -> dict[str, Any]:
    """GET a Roblox API URL and return its JSON object, or raise RobloxAPIError."""
    try:
        response = http_session().get(url, timeout=REQUEST_TIMEOUT)
    except requests.Timeout as exc:
        raise RobloxAPIError("Roblox took too long to respond (timeout).", kind="timeout") from exc
    except requests.ConnectionError as exc:
        raise RobloxAPIError(
            "Couldn't reach Roblox (network connection problem).", kind="network"
        ) from exc
    except requests.RequestException as exc:
        raise RobloxAPIError(
            f"The request to Roblox failed ({type(exc).__name__}).", kind="network"
        ) from exc

    status = response.status_code
    if status == 429:
        raise RobloxAPIError(
            "Roblox is rate-limiting requests (HTTP 429).",
            kind="rate_limited",
            retry_after=_retry_after_seconds(response),
        )
    if status == 404:
        raise RobloxAPIError("Roblox couldn't find that resource (HTTP 404).", kind="not_found")
    if status >= 500:
        raise RobloxAPIError(f"Roblox is having server problems (HTTP {status}).", kind="http")
    if status >= 400:
        raise RobloxAPIError(
            f"Roblox rejected the request (HTTP {status}).{_error_detail(response)}", kind="http"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RobloxAPIError(
            "Roblox returned a response that wasn't valid JSON.", kind="bad_response"
        ) from exc
    if not isinstance(payload, dict):
        raise RobloxAPIError("Roblox returned an unexpected response.", kind="bad_response")
    return payload


@st.cache_data(ttl=6 * 60 * 60, max_entries=512, show_spinner=False)
def lookup_universe_id(place_id: int) -> int:
    """Convert a Place ID to its Universe ID. Only successful lookups are cached."""
    payload = get_json(PLACE_TO_UNIVERSE_URL.format(place_id=place_id))
    universe_id = to_int(payload.get("universeId"))
    if universe_id is None or universe_id <= 0:
        raise RobloxAPIError(f"No game exists for Place ID {place_id}.", kind="not_found")
    return universe_id


def parse_game_stats(item: Any) -> GameStats | None:
    """Validate one game object from the API. Returns None if it is unusable."""
    if not isinstance(item, dict):
        return None
    universe_id = to_int(item.get("id"))
    visits = to_int(item.get("visits"))
    playing = to_int(item.get("playing"))
    if universe_id is None or visits is None or playing is None:
        return None
    creator = item.get("creator") if isinstance(item.get("creator"), dict) else {}
    name = item.get("name")
    creator_name = creator.get("name")
    creator_type = creator.get("type")
    return {
        "universe_id": universe_id,
        "root_place_id": to_int(item.get("rootPlaceId")),
        "name": name.strip()
        if isinstance(name, str) and name.strip()
        else f"Universe {universe_id}",
        "creator_name": creator_name.strip()
        if isinstance(creator_name, str) and creator_name.strip()
        else "Unknown creator",
        "creator_type": creator_type if isinstance(creator_type, str) else "",
        "visits": visits,
        "playing": playing,
        "favorites": to_int(item.get("favoritedCount")),
        "max_players": to_int(item.get("maxPlayers")),
    }


@st.cache_data(ttl=STATS_CACHE_SECONDS, max_entries=256, show_spinner=False)
def fetch_games(universe_ids: tuple[int, ...]) -> FetchResult:
    """Fetch live stats for several games in one API call.

    Games Roblox does not return (private, deleted, under review or unknown IDs)
    are simply absent from the result. The short cache is shared by every
    viewer of the app, which keeps the request rate low.
    """
    ids = ",".join(str(int(universe_id)) for universe_id in universe_ids)
    payload = get_json(f"{GAMES_URL}?universeIds={ids}")
    data = payload.get("data")
    if not isinstance(data, list):
        raise RobloxAPIError(
            "Roblox returned an empty or unexpected response.", kind="bad_response"
        )
    games: dict[int, GameStats] = {}
    for item in data:
        stats = parse_game_stats(item)
        if stats is not None:
            games[stats["universe_id"]] = stats
    return {"fetched_at": time.time(), "games": games}


def resolve_game(parsed: ParsedInput, id_type: str) -> tuple[GameStats, float, str | None]:
    """Find the game the user meant. Returns (stats, fetched_at, optional note)."""
    number = parsed["id"]
    kind: InputKind = parsed["kind"]
    if kind == "unknown":
        kind = {ID_TYPE_PLACE: "place", ID_TYPE_UNIVERSE: "universe"}.get(id_type, "unknown")  # type: ignore[assignment]

    if kind == "place":
        try:
            universe_id = lookup_universe_id(number)
        except RobloxAPIError as exc:
            if exc.kind == "not_found":
                raise InputError(
                    f"No Roblox game was found for Place ID {number:,}. Check the number or URL."
                ) from exc
            raise
        result = fetch_games((universe_id,))
        stats = result["games"].get(universe_id)
        if stats is None:
            raise InputError(
                f"Place ID {number:,} exists, but Roblox returned no public statistics "
                "for it. The game may be private, under review or unavailable."
            )
        return stats, result["fetched_at"], None

    if kind == "universe":
        result = fetch_games((number,))
        stats = result["games"].get(number)
        if stats is None:
            raise InputError(
                f"Roblox returned no public game for Universe ID {number:,}. It may be "
                "private, unavailable, or actually a Place ID."
            )
        return stats, result["fetched_at"], None

    # Bare number with auto-detect: try it as a Place ID first (that is the number
    # in game URLs), then as a Universe ID, using a single stats request for both.
    try:
        via_place_id: int | None = lookup_universe_id(number)
    except RobloxAPIError as exc:
        if exc.kind != "not_found":
            raise
        via_place_id = None
    candidates = tuple(sorted({number} | ({via_place_id} if via_place_id else set())))
    result = fetch_games(candidates)
    place_match = result["games"].get(via_place_id) if via_place_id else None
    universe_match = result["games"].get(number)

    if place_match is not None:
        note = None
        if (
            universe_match is not None
            and universe_match["universe_id"] != place_match["universe_id"]
        ):
            note = (
                f"{number:,} was read as a Place ID. It is also the Universe ID of "
                f"“{universe_match['name']}”. If you meant that game, set ID type to "
                "“Universe ID” and add it again."
            )
        return place_match, result["fetched_at"], note
    if universe_match is not None:
        return universe_match, result["fetched_at"], None
    raise InputError(
        f"No public Roblox game matches {number:,} as a Place ID or a Universe ID. "
        "The game may be private or unavailable."
    )


# --------------------------------------------------------------------------- #
# Session state and monitoring
# --------------------------------------------------------------------------- #


def new_poll_state() -> PollState:
    return {
        "last_attempt": 0.0,
        "last_success": None,
        "failures": 0,
        "blocked_until": 0.0,
        "error": None,
        "error_kind": None,
    }


def init_state() -> None:
    state = st.session_state
    state.setdefault("games", {})  # dict[int, TrackedGame], keyed by Universe ID
    state.setdefault("poll", new_poll_state())
    games: dict[int, TrackedGame] = state["games"]
    if state.get("selected_game") not in games:
        state["selected_game"] = next(iter(games), None)


def make_snapshot(stats: GameStats, ts: float) -> Snapshot:
    return {
        "ts": ts,
        "visits": stats["visits"],
        "playing": stats["playing"],
        "favorites": stats["favorites"],
    }


def new_tracked_game(stats: GameStats, ts: float) -> TrackedGame:
    snapshot = make_snapshot(stats, ts)
    return {
        "stats": stats,
        "first": snapshot,
        "history": [snapshot],
        "peak_playing": stats["playing"],
        "peak_ts": ts,
        "error": None,
    }


def record_snapshot(game: TrackedGame, stats: GameStats, ts: float) -> bool:
    """Store new API values for a game. Returns True if a snapshot was added."""
    game["stats"] = stats
    game["error"] = None
    history = game["history"]
    if history and ts <= history[-1]["ts"]:
        return False  # same (cached) API response as the previous snapshot
    history.append(make_snapshot(stats, ts))
    if len(history) > MAX_SNAPSHOTS:
        del history[: len(history) - MAX_SNAPSHOTS]
    if stats["playing"] > game["peak_playing"]:
        game["peak_playing"] = stats["playing"]
        game["peak_ts"] = ts
    return True


def backoff_seconds(kind: ErrorKind, failures: int, retry_after: float | None) -> float:
    """How long to wait before the next automatic attempt after a failure."""
    if retry_after:
        return min(max(retry_after, REFRESH_SECONDS), MAX_BACKOFF_SECONDS)
    base = 60 if kind == "rate_limited" else REFRESH_SECONDS
    return min(base * 2 ** max(failures - 1, 0), MAX_BACKOFF_SECONDS)


def poll_games(
    *, manual: bool
) -> Literal["updated", "waiting", "cooldown", "rate_limited", "failed"]:
    """Refresh every monitored game with one API request, if a refresh is due."""
    state = st.session_state
    games: dict[int, TrackedGame] = state["games"]
    poll: PollState = state["poll"]
    if not games:
        return "waiting"

    now = time.time()
    since_last = now - poll["last_attempt"]
    if manual:
        if poll["error_kind"] == "rate_limited" and now < poll["blocked_until"]:
            return "rate_limited"
        if since_last < STATS_CACHE_SECONDS:
            return "cooldown"
    elif now < poll["blocked_until"] or since_last < REFRESH_SECONDS - TIMER_TOLERANCE_SECONDS:
        return "waiting"

    poll["last_attempt"] = now
    try:
        result = fetch_games(tuple(sorted(games)))
    except RobloxAPIError as exc:
        poll["failures"] += 1
        poll["error"] = str(exc)
        poll["error_kind"] = exc.kind
        wait = backoff_seconds(exc.kind, poll["failures"], exc.retry_after)
        poll["blocked_until"] = now + wait - TIMER_TOLERANCE_SECONDS
        return "failed"

    poll.update(
        failures=0,
        error=None,
        error_kind=None,
        blocked_until=0.0,
        last_success=result["fetched_at"],
    )
    for universe_id, game in games.items():
        stats = result["games"].get(universe_id)
        if stats is None:
            game["error"] = (
                "Roblox didn't return this game in the latest update. It may have been "
                "made private, taken down or put under review. Earlier data is kept below."
            )
        else:
            record_snapshot(game, stats, result["fetched_at"])
    return "updated"


def visits_per_minute(history: list[Snapshot]) -> float | None:
    """Average visits gained per minute between the first and last snapshot given."""
    if len(history) < 2:
        return None
    elapsed = history[-1]["ts"] - history[0]["ts"]
    if elapsed <= 0:
        return None
    return (history[-1]["visits"] - history[0]["visits"]) / elapsed * 60


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #


def add_game_callback(input_key: str, type_key: str) -> None:
    state = st.session_state
    games: dict[int, TrackedGame] = state["games"]
    try:
        parsed = parse_game_input(state.get(input_key, ""))
        stats, fetched_at, note = resolve_game(parsed, state.get(type_key, ID_TYPE_AUTO))
        universe_id = stats["universe_id"]
        if universe_id not in games and len(games) >= MAX_GAMES:
            raise InputError(
                f"This session is already monitoring {MAX_GAMES} games. Remove one to add another."
            )
    except InputError as exc:
        state["add_feedback"] = ("error", str(exc))
        return
    except RobloxAPIError as exc:
        state["add_feedback"] = ("error", f"{exc} Nothing was added — please try again shortly.")
        return

    if universe_id in games:
        record_snapshot(games[universe_id], stats, fetched_at)
        message = f"“{stats['name']}” is already being monitored — it is now selected."
    else:
        games[universe_id] = new_tracked_game(stats, fetched_at)
        message = f"Now monitoring “{stats['name']}”."
    state["selected_game"] = universe_id
    state[input_key] = ""
    state["add_feedback"] = ("success", message if note is None else f"{message} {note}")


def remove_game_callback(universe_id: int) -> None:
    st.session_state["games"].pop(universe_id, None)
    st.session_state["selected_game"] = next(iter(st.session_state["games"]), None)


def reset_history_callback(universe_id: int) -> None:
    game: TrackedGame | None = st.session_state["games"].get(universe_id)
    if game is not None:
        latest = game["history"][-1]
        st.session_state["games"][universe_id] = new_tracked_game(game["stats"], latest["ts"])


def request_manual_refresh() -> None:
    st.session_state["manual_refresh_requested"] = True


# --------------------------------------------------------------------------- #
# UI: adding and choosing games
# --------------------------------------------------------------------------- #


def render_add_form(prefix: str) -> None:
    input_key, type_key = f"{prefix}_game_input", f"{prefix}_id_type"
    with st.form(f"{prefix}_add_form", border=False):
        st.text_input(
            "Roblox game URL or ID",
            key=input_key,
            placeholder="https://www.roblox.com/games/…",
            help="Paste the address of the game's page, a Creator Hub link, a Place ID "
            "or a Universe ID.",
        )
        st.selectbox(
            "ID type",
            ID_TYPES,
            key=type_key,
            help="Only used when you enter a bare number. Auto-detect tries it as a "
            "Place ID first (the number in a game URL), then as a Universe ID.",
        )
        st.form_submit_button(
            "Add game",
            type="primary",
            icon=":material/add:",
            width="stretch",
            on_click=add_game_callback,
            args=(input_key, type_key),
        )
    feedback = st.session_state.pop("add_feedback", None)
    if feedback:
        level, message = feedback
        if level == "error":
            st.error(escape_md(message), icon=":material/error:")
        else:
            st.success(escape_md(message), icon=":material/check_circle:")


def render_sidebar() -> None:
    games: dict[int, TrackedGame] = st.session_state["games"]
    with st.sidebar:
        st.markdown(f"### :material/monitoring: {APP_TITLE}")
        if not games:
            st.caption("Add a game on the right to start monitoring.")
            return

        st.radio(
            "Monitored games",
            options=list(games),
            key="selected_game",
            format_func=lambda uid: escape_md(games[uid]["stats"]["name"]),
            captions=[escape_md(game["stats"]["creator_name"]) for game in games.values()],
        )
        selected = st.session_state.get("selected_game")
        if selected in games:
            left, right = st.columns(2)
            left.button(
                "Remove",
                icon=":material/delete:",
                width="stretch",
                on_click=remove_game_callback,
                args=(selected,),
                help="Stop monitoring the selected game and discard its session history.",
            )
            right.button(
                "Reset",
                icon=":material/restart_alt:",
                width="stretch",
                on_click=reset_history_callback,
                args=(selected,),
                help="Clear the selected game's session history and start measuring "
                "again from the latest values.",
            )

        st.divider()
        st.markdown("**Add another game**")
        render_add_form("sidebar")
        st.divider()
        st.caption(
            f"Games and history are saved **in this browser session only** "
            f"(up to {MAX_GAMES} games). Reloading the page clears them."
        )


def render_welcome() -> None:
    st.title(APP_TITLE)
    st.markdown(
        "Track live visits, concurrent players and favorites for any public Roblox "
        "game, using Roblox's public API. No login or cookie is needed."
    )
    form_column, info_column = st.columns([3, 2], gap="large")
    with form_column:
        with st.container(border=True):
            st.markdown("**Add your first game**")
            render_add_form("welcome")
    with info_column:
        st.markdown("**What you can paste**")
        st.markdown(
            "- A game page address: `roblox.com/games/920587237/Adopt-Me`\n"
            "- A Creator Hub address: `create.roblox.com/…/experiences/383310974/…`\n"
            "- A Place ID or Universe ID on its own: `920587237`"
        )
    render_data_notes(expanded=True)


# --------------------------------------------------------------------------- #
# UI: live dashboard
# --------------------------------------------------------------------------- #


def render_header(game: TrackedGame) -> None:
    stats = game["stats"]
    title_column, action_column = st.columns([3, 2], vertical_alignment="center")
    with title_column:
        st.header(escape_md(stats["name"]), anchor=False)
        details = [f"by **{escape_md(stats['creator_name'])}**"]
        if stats["creator_type"]:
            details[0] += f" ({escape_md(stats['creator_type'].lower())})"
        if stats["max_players"] is not None:
            details.append(f"max server size **{stats['max_players']:,}**")
        details.append(f"Universe ID `{stats['universe_id']}`")
        if stats["root_place_id"]:
            details.append(f"Place ID `{stats['root_place_id']}`")
        st.caption(" · ".join(details))
    with action_column:
        with st.container(
            horizontal=True, horizontal_alignment="right", vertical_alignment="center"
        ):
            st.button(
                "Refresh now",
                icon=":material/refresh:",
                on_click=request_manual_refresh,
                help="Fetch the latest values from Roblox immediately.",
            )
            if stats["root_place_id"]:
                st.link_button(
                    "Open on Roblox",
                    GAME_PAGE_URL.format(place_id=stats["root_place_id"]),
                    icon=":material/open_in_new:",
                )


def render_status(game: TrackedGame, poll: PollState, tz: tzinfo, outcome: str) -> None:
    latest = game["history"][-1]
    if outcome == "cooldown":
        st.toast(
            f"Values were fetched moments ago. Roblox data is requested at most once "
            f"every {STATS_CACHE_SECONDS} seconds.",
            icon=":material/schedule:",
        )
    elif outcome == "rate_limited":
        st.toast(
            "Roblox is rate-limiting requests. Waiting before trying again.",
            icon=":material/block:",
        )

    if poll["error"]:
        wait = max(0.0, poll["blocked_until"] - time.time())
        retry = (
            f"in about {format_duration(wait + TIMER_TOLERANCE_SECONDS)}" if wait > 0 else "shortly"
        )
        st.warning(
            f"**Update failed.** {poll['error']} Showing the last successful values from "
            f"{format_clock(latest['ts'], tz)}. Retrying automatically {retry}.",
            icon=":material/cloud_off:",
        )
    if game["error"]:
        st.warning(game["error"], icon=":material/visibility_off:")

    with st.container(horizontal=True, vertical_alignment="center"):
        age = time.time() - latest["ts"]
        if poll["error"] or game["error"] or age > STALE_AFTER_SECONDS:
            st.badge("Not updating", icon=":material/warning:", color="orange")
        else:
            st.badge("Live", icon=":material/sensors:", color="green")
        st.caption(
            f"Latest successful update: **{format_clock(latest['ts'], tz)}** · "
            f"monitoring since {format_clock(game['first']['ts'], tz)} · "
            f"refreshes every {REFRESH_SECONDS} s while this page is open"
        )


def metric_card(
    column: Any,
    label: str,
    value: str,
    *,
    change: int | None = None,
    change_text: str | None = None,
    description: str | None = None,
    help_text: str | None = None,
) -> None:
    """A bordered metric card.

    `change` sets the colour (green when positive, red when negative, grey when
    flat) and `change_text` is the text shown beside the arrow. With no
    `change_text` the card shows no delta line.
    """
    shown = change_text if change is not None else None
    column.metric(
        label,
        value,
        delta=shown,
        delta_color="green" if change and change > 0 else "red" if change else "off",
        delta_arrow="up" if change and change > 0 else "down" if change else "off",
        delta_description=description if shown is not None else None,
        border=True,
        height="stretch",
        help=help_text,
    )


def render_metrics(game: TrackedGame, tz: tzinfo) -> None:
    stats, first, history = game["stats"], game["first"], game["history"]
    latest = history[-1]
    previous = history[-2] if len(history) > 1 else None

    def since_previous(field: Literal["visits", "playing", "favorites"]) -> dict[str, Any]:
        """Change between the two most recent snapshots, as metric_card arguments."""
        if previous is None or latest[field] is None or previous[field] is None:
            return {}
        change = latest[field] - previous[field]
        return {
            "change": change,
            "change_text": format_signed(change),
            "description": "since previous update",
        }

    top = st.columns(4)
    metric_card(
        top[0],
        "Total visits",
        f"{stats['visits']:,}",
        **since_previous("visits"),
        help_text="Lifetime visits reported by Roblox. Roblox updates this counter in "
        "batches, so it can stay flat for a while and then jump.",
    )
    metric_card(
        top[1],
        "Playing now",
        f"{stats['playing']:,}",
        **since_previous("playing"),
        help_text="Concurrent players reported by Roblox at the latest update.",
    )
    metric_card(
        top[2],
        "Favorites",
        "—" if stats["favorites"] is None else f"{stats['favorites']:,}",
        **since_previous("favorites"),
        help_text="Total number of players who have favorited the game.",
    )
    metric_card(
        top[3],
        "Peak players observed",
        f"{game['peak_playing']:,}",
        help_text=f"Highest concurrent player count seen while monitoring, reached at "
        f"{format_clock(game['peak_ts'], tz)}. Peaks between two updates, or before "
        f"monitoring started, are not captured.",
    )

    st.markdown("##### Since monitoring started")
    visits_gained = latest["visits"] - first["visits"]
    players_change = latest["playing"] - first["playing"]
    favorites_change = (
        latest["favorites"] - first["favorites"]
        if latest["favorites"] is not None and first["favorites"] is not None
        else None
    )
    session_rate = visits_per_minute([first, latest])
    recent = [snap for snap in history if snap["ts"] >= latest["ts"] - 5 * 60]
    recent_rate = visits_per_minute(recent)
    monitored_for = format_duration(latest["ts"] - first["ts"])

    bottom = st.columns(4)
    metric_card(
        bottom[0],
        "Visits gained",
        format_signed(visits_gained),
        change=visits_gained,
        change_text=percent_change(visits_gained, first["visits"]),
        description="of starting total" if first["visits"] > 0 else None,
        help_text=f"Latest total visits minus the total at the first snapshot "
        f"({first['visits']:,}). Monitored for {monitored_for}.",
    )
    metric_card(
        bottom[1],
        "Visits per minute",
        format_rate(session_rate),
        change=None,
        help_text="Visits gained divided by minutes monitored, calculated from the first "
        "and latest snapshots. Needs at least two snapshots. "
        + (
            f"Over the last 5 minutes of snapshots: {format_rate(recent_rate)} per minute."
            if recent_rate is not None
            else ""
        ),
    )
    metric_card(
        bottom[2],
        "Player change",
        format_signed(players_change),
        change=players_change,
        change_text=percent_change(players_change, first["playing"]),
        description="vs. start" if first["playing"] > 0 else None,
        help_text=f"Players now minus players at the first snapshot ({first['playing']:,}).",
    )
    metric_card(
        bottom[3],
        "Favorites gained",
        "—" if favorites_change is None else format_signed(favorites_change),
        change=favorites_change,
        change_text=percent_change(favorites_change, first["favorites"] or 0)
        if favorites_change is not None
        else None,
        description="of starting total" if first["favorites"] else None,
        help_text="Favorites now minus favorites at the first snapshot.",
    )


def history_frame(history: list[Snapshot], tz: tzinfo) -> pd.DataFrame:
    """Snapshots as a DataFrame with UTC and viewer-local timestamps."""
    frame = pd.DataFrame(history, columns=["ts", "visits", "playing", "favorites"])
    milliseconds = (frame["ts"] * 1000).round().astype("int64")
    utc = pd.to_datetime(milliseconds, unit="ms", utc=True)
    frame["time_utc"] = utc
    frame["time"] = utc.dt.tz_convert(tz).dt.tz_localize(None)  # local wall-clock time
    frame["visits"] = frame["visits"].astype("int64")
    frame["playing"] = frame["playing"].astype("int64")
    frame["favorites"] = frame["favorites"].astype("Int64")
    return frame


def history_csv(frame: pd.DataFrame) -> bytes:
    export = pd.DataFrame(
        {
            "timestamp_utc": frame["time_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_visits": frame["visits"],
            "concurrent_players": frame["playing"],
            "favorites": frame["favorites"],
        }
    )
    return export.to_csv(index=False).encode("utf-8")


def line_chart(
    frame: pd.DataFrame, field: str, label: str, color: str, *, floor_zero: bool
) -> alt.LayerChart:
    """A single-series time chart with a hover crosshair and tooltip."""
    low, high = int(frame[field].min()), int(frame[field].max())
    padding = max(1, round((high - low) * 0.08))
    domain = [max(0, low - padding) if floor_zero else low - padding, high + padding]
    surface = st.get_option("theme.backgroundColor") or FALLBACK_SURFACE_COLOR

    # Label the time axis to suit the span shown: seconds, minutes, or days.
    span_seconds = (frame["time"].iloc[-1] - frame["time"].iloc[0]).total_seconds()
    if span_seconds <= 10 * 60:
        time_format, min_step_ms = "%-I:%M:%S %p", 1_000
    elif span_seconds <= 24 * 60 * 60:
        time_format, min_step_ms = "%-I:%M %p", 60_000
    else:
        time_format, min_step_ms = "%b %-d, %-I %p", 3_600_000

    base = alt.Chart(frame).encode(
        x=alt.X(
            "time:T",
            title=None,
            axis=alt.Axis(
                format=time_format, tickMinStep=min_step_ms, tickCount=6, labelOverlap=True
            ),
        ),
        y=alt.Y(
            f"{field}:Q",
            title=None,
            scale=alt.Scale(domain=domain, zero=False, nice=True),
            axis=alt.Axis(format=",.0f", tickMinStep=1, tickCount=5),
        ),
        tooltip=[
            alt.Tooltip("time:T", title="Time", format="%b %-d, %-I:%M:%S %p"),
            alt.Tooltip(f"{field}:Q", title=label, format=","),
        ],
    )
    nearest = alt.selection_point(
        fields=["time"], nearest=True, on="pointerover", clear="pointerout", empty=False
    )
    line = base.mark_line(color=color, strokeWidth=2, strokeJoin="round", strokeCap="round")
    crosshair = base.mark_rule(color="#8b93a1", strokeWidth=1).encode(
        y=alt.value(0),
        y2=alt.value("height"),
        opacity=alt.condition(nearest, alt.value(0.7), alt.value(0)),
    )
    marker = base.mark_point(
        color=color, fill=color, size=90, stroke=surface, strokeWidth=2, opacity=1
    ).encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
    hit_area = base.mark_point(size=90, opacity=0).add_params(nearest)
    return alt.layer(line, crosshair, marker, hit_area).properties(height=300)


def render_history(game: TrackedGame, tz: tzinfo, tz_name: str) -> None:
    stats, history = game["stats"], game["history"]
    span = history[-1]["ts"] - history[0]["ts"]

    with st.container(horizontal=True, vertical_alignment="center"):
        st.subheader("History", anchor=False, width="content")
        st.badge("Session-only", icon=":material/history_toggle_off:", color="orange")
    st.caption(
        "Recorded only while this page is open, starting when you added the game. "
        "It is kept in this session's memory and is erased when you reload the page, "
        "the session ends, or the app restarts or goes to sleep on Streamlit Community Cloud."
    )

    # Offer only the ranges this session has enough history to fill.
    options = [name for name, seconds in TIME_RANGES if span > seconds] + [ALL_RANGE]
    range_key = f"range_{stats['universe_id']}"
    if st.session_state.get(range_key) not in options:
        st.session_state[range_key] = ALL_RANGE
    chosen = st.segmented_control(
        "Time range",
        options,
        key=range_key,
        required=True,
        help="Ranges appear as this session collects enough history to fill them.",
    )
    seconds = dict(TIME_RANGES).get(chosen or ALL_RANGE)
    visible = (
        history
        if seconds is None
        else [s for s in history if s["ts"] >= history[-1]["ts"] - seconds]
    )
    frame = history_frame(visible, tz)

    charts = st.columns(2)
    for column, field, label in (
        (charts[0], "visits", "Total visits"),
        (charts[1], "playing", "Concurrent players"),
    ):
        with column.container(border=True):
            st.markdown(f"**{label} over time**")
            if len(frame) < 2:
                st.info(
                    f"Collecting data. The chart appears after the next update, "
                    f"about {REFRESH_SECONDS} seconds after monitoring starts.",
                    icon=":material/hourglass_top:",
                )
                continue
            st.altair_chart(
                line_chart(
                    frame, field, label, SERIES_COLORS[field], floor_zero=field == "playing"
                ),
                width="stretch",
                key=f"chart_{field}_{stats['universe_id']}",
            )
            first_value, last_value = int(frame[field].iloc[0]), int(frame[field].iloc[-1])
            if field == "visits":
                summary = (
                    f"{format_signed(last_value - first_value)} visits in this range · "
                    f"{format_rate(visits_per_minute(visible))} per minute"
                )
            else:
                summary = (
                    f"{format_signed(last_value - first_value)} players in this range · "
                    f"low {int(frame[field].min()):,} · high {int(frame[field].max()):,}"
                )
            st.caption(summary)

    st.caption(
        f"{len(visible):,} snapshots shown, covering "
        f"{format_duration(visible[-1]['ts'] - visible[0]['ts'])}. "
        f"Times are in {tz_name}. Growth figures are differences between recorded "
        "snapshots; nothing is estimated."
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    with st.container(horizontal=True):
        st.download_button(
            "Download shown range (CSV)"
            if seconds is not None
            else "Download session history (CSV)",
            data=history_csv(frame),
            file_name=f"roblox-{stats['universe_id']}-history-{stamp}.csv",
            mime="text/csv",
            icon=":material/download:",
            on_click="ignore",
            key=f"download_range_{stats['universe_id']}",
        )
        if seconds is not None:
            st.download_button(
                "Download full session (CSV)",
                data=history_csv(history_frame(history, tz)),
                file_name=f"roblox-{stats['universe_id']}-session-{stamp}.csv",
                mime="text/csv",
                icon=":material/download:",
                on_click="ignore",
                key=f"download_all_{stats['universe_id']}",
            )

    with st.expander("Snapshot table", icon=":material/table_rows:"):
        table = frame.sort_values("ts", ascending=False)[["time", "visits", "playing", "favorites"]]
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            height=320,
            column_config={
                "time": st.column_config.DatetimeColumn(
                    f"Time ({tz_name})", format="MMM D, h:mm:ss a"
                ),
                "visits": st.column_config.NumberColumn("Total visits", format="localized"),
                "playing": st.column_config.NumberColumn("Concurrent players", format="localized"),
                "favorites": st.column_config.NumberColumn("Favorites", format="localized"),
            },
        )


def render_data_notes(*, expanded: bool = False) -> None:
    with st.expander("About this data", icon=":material/info:", expanded=expanded):
        st.markdown(
            f"- **Values can be delayed.** Everything comes from Roblox's public API. "
            f"Roblox caches these numbers, and the visit counter in particular is "
            f"updated in batches, so the dashboard can trail what is happening in the "
            f"game by seconds to minutes.\n"
            f"- **Refreshing needs an open page.** Statistics refresh every "
            f"{REFRESH_SECONDS} seconds only while this page is open in a browser and "
            f"the Streamlit session is active. When the tab is closed, the computer "
            f"sleeps or the connection drops, collection stops and that time is "
            f"missing from the history.\n"
            f"- **History is session-only.** Snapshots are held in memory for this "
            f"browser session (up to {MAX_SNAPSHOTS:,} per game, about 24 hours). "
            f"Nothing is saved to a database, and nothing is shown for the time before "
            f"you added a game.\n"
            f"- **Peaks are observed peaks.** The highest player count is the highest "
            f"value seen at an update, not the game's true all-time or daily peak."
        )


@st.fragment(run_every=REFRESH_SECONDS)
def live_dashboard() -> None:
    """The auto-refreshing part of the page. Reruns on its own on a timer."""
    state = st.session_state
    manual = bool(state.pop("manual_refresh_requested", False))
    outcome = poll_games(manual=manual)

    games: dict[int, TrackedGame] = state["games"]
    game = games.get(state.get("selected_game"))
    if game is None:
        st.info("Select a game in the sidebar.", icon=":material/arrow_back:")
        return

    tz, tz_name = user_timezone()
    poll: PollState = state["poll"]
    render_header(game)
    render_status(game, poll, tz, outcome)
    render_metrics(game, tz)
    render_history(game, tz, tz_name)
    render_data_notes()


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=":material/monitoring:",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            "About": "Live Roblox game statistics from Roblox's public API. "
            "History is session-only."
        },
    )
    init_state()
    render_sidebar()
    if st.session_state["games"]:
        live_dashboard()
    else:
        render_welcome()


if __name__ == "__main__":
    main()
