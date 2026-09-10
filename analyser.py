"""
ESGLens — extraction engine.

Gemini reads the PDF natively (tables, charts, layout included) and returns
one structured, page-cited extraction per report, organised into
Environmental / Social / Governance plus an open-ended catch-all for
material disclosures that don't fit the fixed taxonomy — real reports
disclose things no fixed schema fully anticipates, and the goal is to
never silently drop something a report actually said.

Every field is a SourcedValue: text value + page + quote for citation,
optional `numeric` for anything chart-able, and optional `trend` for
metrics a report shows across multiple years in the same table. Page
citations follow the report's OWN printed page numbers (its footer),
which is what lets you say "see page 42" the way an analyst would — but
note that can differ from your PDF viewer's page counter if the document
has unnumbered cover/TOC pages before its internal numbering starts.

IMPORTANT — why this does ONE Gemini call per report, with NO
response_json_schema:
Two things turned out to be in direct conflict on Gemini's free tier:

1. Gemini's structured-output enforcement (`response_json_schema`) compiles
   the schema into an internal constraint state machine, and rejects
   schemas that are too large/complex with a bare `400 INVALID_ARGUMENT`
   ("Request contains an invalid argument") — a documented Gemini API
   limit. The full ESGExtraction schema has ~34 SourcedValue fields (~200+
   leaf properties once you count found/value/numeric/page/quote/trend on
   each), which is well past where that limit tends to bite.
2. gemini-3.8-flash's FREE TIER allows only ~20 requests PER DAY, PER
   PROJECT (confirmed against Google's own rate-limits docs and community
   reports — this dropped from 250/day in Dec 2025). Splitting extraction
   into several smaller calls (an earlier version of this file did 5 —
   Overview, two Environmental sub-groups, Social, Governance) sidesteps
   problem #1, but directly worsens problem #2: 5 calls per report means a
   2-report comparison alone burns half the day's entire quota, and any
   retries burn through the rest in minutes.

The fix that solves BOTH at once: don't use `response_json_schema`
(controlled generation) at all. Instead, the full desired JSON shape is
spelled out as plain instructions in the prompt text, Gemini is asked for
`response_mime_type="application/json"` (syntactically-valid JSON, but not
schema-enforced), and the result is validated client-side with Pydantic —
with a single, cheap, PDF-free "repair" call if validation fails. This
means ONE call per report in the normal case (two for a compare of two
reports), and no schema-complexity ceiling to hit at all, since nothing is
ever handed to Gemini's schema compiler.

On top of that: `KeyPool` supports rotating across multiple free-tier API
keys, since Google confirms rate limits are enforced per PROJECT — and
each separate Google account normally gets its own project — so N keys
from N of your own accounts gives you N independent 20/day budgets rather
than one.
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from typing import Callable, List, Optional

from pydantic import BaseModel, Field
from google import genai
from google.genai import types

MODEL_NAME = "gemini-3.8-flash"  # update here if Google renames/retires this model
MAX_FILE_MB = 50  # Gemini's current per-document limit (also ~1000 pages)
MAX_OUTPUT_TOKENS = 32768  # generous ceiling for the full-schema JSON response
# Free-tier gemini-3.8-flash allows ~20 requests/day/project — retrying a
# genuinely transient issue (per-minute throttle, dropped connection, or
# Google's own server-side "high demand" 503s) is worth it, but only a
# few times, since each attempt itself spends one of those 20.
MAX_TRANSIENT_RETRIES = 3


# --------------------------------------------------------------------------
# Schema (unchanged from earlier versions — still the single source of
# truth for both client-side validation AND the prompt's field manifest,
# built automatically from these Field(...) descriptions below so the two
# never drift out of sync).
# --------------------------------------------------------------------------

class SourcedValue(BaseModel):
    found: bool = Field(
        description="True only if this exact information is explicitly stated in the document."
    )
    value: str = Field(
        description="The extracted value or text, in words. 'Not disclosed' if found is False. "
                    "If only a related/partial figure exists (e.g. an absolute quantity when a "
                    "percentage was requested), report that figure here rather than 'Not disclosed'."
    )
    numeric: Optional[float] = Field(
        default=None,
        description="Just the number, in the unit specified for this field. Null for qualitative fields.",
    )
    page: Optional[str] = Field(
        default=None,
        description="The page number exactly as printed in the document's own footer/header — "
                    "NOT the PDF file's internal page position. E.g. '34' or '34-35'.",
    )
    quote: Optional[str] = Field(
        default=None,
        description="A short verbatim snippet (25 words or fewer) from that page supporting the value.",
    )
    trend: Optional[str] = Field(
        default=None,
        description="If the source table shows this metric across multiple years, summarise prior "
                    "years here as text, e.g. 'FY23: 77,465 · FY24: 86,954'. Null if only one year shown.",
    )


class ConflictNote(BaseModel):
    field: str = Field(description="Which metric has conflicting values in the document.")
    values_found: List[str] = Field(
        description="Each conflicting value with its page, e.g. '124,500 tCO2e (p.34)'."
    )
    note: str = Field(description="One short line explaining the discrepancy.")


class NamedDisclosure(BaseModel):
    topic: str = Field(description="Short name for this disclosure, e.g. 'Green portfolio revenue share'.")
    category: str = Field(description="'Environmental', 'Social', or 'Governance'.")
    found: bool = True
    value: str
    numeric: Optional[float] = None
    page: Optional[str] = None
    quote: Optional[str] = None


class Environmental(BaseModel):
    scope1_emissions: SourcedValue = Field(description="Scope 1 direct GHG emissions. numeric in tCO2e.")
    scope2_location_based: SourcedValue = Field(description="Scope 2 emissions, location-based method specifically. numeric in tCO2e.")
    scope2_market_based: SourcedValue = Field(description="Scope 2 emissions, market-based method specifically. numeric in tCO2e.")
    scope2_total_unspecified: SourcedValue = Field(description="If the report gives ONE blended Scope 2 figure without specifying location- vs market-based methodology, put it here. numeric in tCO2e.")
    scope3_total: SourcedValue = Field(description="Total Scope 3 emissions if disclosed as one figure. numeric in tCO2e.")
    scope3_categories_reported: SourcedValue = Field(description="Which GHG Protocol Scope 3 categories are individually reported. Qualitative, numeric null.")
    total_energy_consumption: SourcedValue = Field(description="Total energy consumption. numeric in GJ (convert if the report uses another unit; note the conversion in the quote).")
    renewable_energy_pct: SourcedValue = Field(description="Percent of energy from renewable sources, ONLY if a percentage is explicitly stated or directly computable from total + renewable-portion figures both given in the same table. numeric as plain 0-100.")
    renewable_energy_quantity: SourcedValue = Field(description="Absolute renewable energy generated/consumed, if disclosed (even without an accompanying percentage). numeric in GJ.")
    water_consumption: SourcedValue = Field(description="Total water withdrawal/consumption. numeric in kilolitres (convert if needed).")
    water_recycled_pct: SourcedValue = Field(description="Percent of water recycled or reused. numeric as plain 0-100.")
    total_waste: SourcedValue = Field(description="Total waste generated. numeric in tonnes.")
    waste_recycled_pct: SourcedValue = Field(description="Percent of waste recycled/reused/diverted from landfill. numeric as plain 0-100.")
    net_zero_target_year: SourcedValue = Field(description="Net-zero / carbon-neutrality target year. Qualitative.")
    sbti_status: SourcedValue = Field(description="Science Based Targets initiative commitment/validation status. Qualitative.")


class Social(BaseModel):
    total_employees: SourcedValue = Field(description="Total headcount. numeric as a whole number.")
    women_workforce_pct: SourcedValue = Field(description="Percent of total workforce who are women. If the report's own consolidated ESG/BRSR summary table gives a figure, prefer that over any narrative mention elsewhere. numeric as plain 0-100.")
    women_leadership_pct: SourcedValue = Field(description="Percent of senior management/leadership team (not the board) who are women. numeric as plain 0-100.")
    employee_turnover_pct: SourcedValue = Field(description="Employee attrition/turnover rate. numeric as plain 0-100.")
    ltifr: SourcedValue = Field(description="Lost Time Injury Frequency Rate specifically (LTIFR) — do not confuse with TRIR. numeric as the rate value.")
    trir: SourcedValue = Field(description="Total Recordable Incident Rate specifically (TRIR) — do not confuse with LTIFR. These are two distinct metrics; only fill each if that exact metric is named in the report.")
    fatality_rate: SourcedValue = Field(description="Workplace fatality rate or fatality count, if disclosed.")
    training_hours_per_employee: SourcedValue = Field(description="Average training hours per employee per year. numeric as hours.")
    community_investment: SourcedValue = Field(description="CSR spend / community investment amount. Include currency in the value text.")


class Governance(BaseModel):
    board_size: SourcedValue = Field(description="Number of members on the board of directors. numeric as a whole number.")
    board_independence_pct: SourcedValue = Field(description="Percent of the board who are independent directors. numeric as plain 0-100.")
    women_on_board_pct: SourcedValue = Field(description="Percent of the board who are women. numeric as plain 0-100.")
    anti_corruption_policy: SourcedValue = Field(description="Whether an anti-corruption/anti-bribery policy is disclosed, one line on it. Qualitative.")
    whistleblower_mechanism: SourcedValue = Field(description="Whether a whistleblower/ethics reporting mechanism is disclosed. Qualitative.")
    data_privacy_disclosures: SourcedValue = Field(description="Any data privacy/cybersecurity policy or breach disclosures. Qualitative.")
    esg_linked_compensation: SourcedValue = Field(description="Whether executive compensation is linked to ESG performance. Qualitative.")
    assurance_provider: SourcedValue = Field(description="Name of the external assurance provider for this report, if any.")


class ESGExtraction(BaseModel):
    company_name: str
    reporting_year: Optional[str] = None
    industry_sector: Optional[str] = Field(
        default=None,
        description="Best-effort classification of the company's primary industry/sector in a few words "
                    "(e.g. 'IT services', 'steel manufacturing', 'FMCG'), based on how the report describes the business.",
    )
    reporting_framework: SourcedValue = Field(description="Reporting framework(s) used, e.g. GRI, BRSR, ISSB/IFRS S2, CDP, TCFD.")
    material_topics: SourcedValue = Field(description="Top material ESG topics from the materiality assessment. Qualitative.")
    environmental: Environmental
    social: Social
    governance: Governance
    additional_disclosures: List[NamedDisclosure] = Field(
        default_factory=list,
        description="Any other material ESG disclosure with an actual reported figure or statement that "
                    "isn't captured by the fixed fields above — especially anything the report itself lists "
                    "as a material topic (e.g. green/sustainable revenue share, digitalisation initiatives, "
                    "biodiversity programs, supply chain sustainability, product safety/quality, tax "
                    "transparency, human rights due diligence, litigation or regulatory penalties, R&D spend). "
                    "Only add an entry when you found a specific value — not just a topic name with nothing behind it.",
    )
    conflicts: List[ConflictNote] = Field(default_factory=list)


OVERVIEW_FIELDS = [
    ("reporting_framework", "Reporting framework"),
    ("material_topics", "Material topics"),
]
ENV_FIELDS = [
    ("scope1_emissions", "Scope 1 emissions (tCO2e)"),
    ("scope2_location_based", "Scope 2 – location-based (tCO2e)"),
    ("scope2_market_based", "Scope 2 – market-based (tCO2e)"),
    ("scope2_total_unspecified", "Scope 2 – total, methodology unspecified (tCO2e)"),
    ("scope3_total", "Scope 3 total (tCO2e)"),
    ("scope3_categories_reported", "Scope 3 categories reported"),
    ("total_energy_consumption", "Total energy consumption (GJ)"),
    ("renewable_energy_pct", "Renewable energy (%)"),
    ("renewable_energy_quantity", "Renewable energy — absolute (GJ)"),
    ("water_consumption", "Water consumption (kilolitres)"),
    ("water_recycled_pct", "Water recycled/reused (%)"),
    ("total_waste", "Total waste generated (tonnes)"),
    ("waste_recycled_pct", "Waste recycled/diverted (%)"),
    ("net_zero_target_year", "Net-zero target year"),
    ("sbti_status", "SBTi status"),
]
SOCIAL_FIELDS = [
    ("total_employees", "Total employees"),
    ("women_workforce_pct", "Women in workforce (%)"),
    ("women_leadership_pct", "Women in senior leadership (%)"),
    ("employee_turnover_pct", "Employee turnover (%)"),
    ("ltifr", "LTIFR (Lost Time Injury Frequency Rate)"),
    ("trir", "TRIR (Total Recordable Incident Rate)"),
    ("fatality_rate", "Fatality rate"),
    ("training_hours_per_employee", "Training hours / employee"),
    ("community_investment", "Community investment"),
]
GOV_FIELDS = [
    ("board_size", "Board size"),
    ("board_independence_pct", "Independent directors (%)"),
    ("women_on_board_pct", "Women on board (%)"),
    ("anti_corruption_policy", "Anti-corruption policy"),
    ("whistleblower_mechanism", "Whistleblower mechanism"),
    ("data_privacy_disclosures", "Data privacy & security"),
    ("esg_linked_compensation", "ESG-linked exec. compensation"),
    ("assurance_provider", "External assurance provider"),
]


# --------------------------------------------------------------------------
# Prompt — the full schema is spelled out here as plain text (NOT passed to
# Gemini's response_json_schema), built from the Field(...) descriptions
# above so the prompt can never drift out of sync with what Pydantic will
# actually validate against.
# --------------------------------------------------------------------------

def _manifest(model_cls: type[BaseModel], fields: List[tuple]) -> str:
    lines = []
    for attr, label in fields:
        desc = model_cls.model_fields[attr].description or ""
        lines.append(f'  "{attr}": {{...}}   // {label} — {desc}')
    return "\n".join(lines)


_SOURCED_VALUE_SHAPE = (
    '{"found": bool, "value": string, "numeric": number|null, '
    '"page": string|null, "quote": string|null, "trend": string|null}'
)

_FRAMEWORK_DESC = ESGExtraction.model_fields["reporting_framework"].description
_TOPICS_DESC = ESGExtraction.model_fields["material_topics"].description
_ADDITIONAL_DESC = ESGExtraction.model_fields["additional_disclosures"].description

MASTER_PROMPT = f"""You are an experienced ESG/sustainability analyst extracting data from a \
corporate sustainability report (native PDF understanding — you can see tables, charts and \
layout, not just body text) for use by investors, lenders, customers and ESG analysts. Accuracy \
and completeness both matter — this is used for real financial and business decisions.

Extract only what is explicitly stated. Never estimate, calculate, or infer a number that isn't \
written down, and never fill in an industry-typical figure. Do not produce an overall ESG score, \
letter grade or rating of any kind.

For every metric field (every object matching the shape {_SOURCED_VALUE_SHAPE} below):
- Set found=true and give the value, page number, and a short supporting quote (25 words or \
fewer) ONLY if that exact information is stated somewhere in the document.
- If genuinely absent, set found=false and value="Not disclosed". If a closely related figure \
exists even though the exact requested breakdown doesn't (e.g. the report gives one blended \
Scope 2 number instead of a location/market split, or an absolute renewable-energy quantity \
instead of a percentage), report that figure in the relevant field instead of marking it "Not \
disclosed" — a real related number is always more useful than an empty field.
- Whenever a field is a number, ALSO populate `numeric` with just that number in the unit \
specified in that field's description below (percentages as plain 0-100 values, not "42%" or \
0.42). Leave numeric null for qualitative fields.
- If the source table shows this metric across multiple years (e.g. a 3-year ESG factsheet with \
FY23/FY24/FY25 columns), fill `trend` with the prior years as text, e.g. 'FY23: 77,465 · FY24: 86,954'.
- The report's own consolidated ESG/BRSR/sustainability summary table or factsheet (if one \
exists) is the single most authoritative source for headline figures — prefer it over scattered \
narrative mentions elsewhere in the document, even if a narrative mention appears earlier.
- If you find two genuinely different values for the same metric after applying that preference, \
report the more authoritative one in the main field and add a matching entry to the top-level \
"conflicts" list with both values and their page numbers.
- Cite the page number exactly as PRINTED in the document's own page footer/header — not the PDF \
file's internal page position, which is often offset by unnumbered cover/TOC pages.
- Do NOT confuse similarly-named safety metrics: LTIFR (Lost Time Injury Frequency Rate) and \
TRIR (Total Recordable Incident Rate) are different metrics if both appear — extract each into \
its own field, never merge them.

Return a SINGLE JSON object and NOTHING ELSE — no markdown code fences, no commentary before or \
after. The first character of your response must be '{{' and the last must be '}}'. It must match \
EXACTLY this shape (every field below is required; use these exact key names):

{{
  "company_name": string,
  "reporting_year": string or null,
  "industry_sector": string or null,   // {ESGExtraction.model_fields["industry_sector"].description}
  "reporting_framework": {_SOURCED_VALUE_SHAPE},   // {_FRAMEWORK_DESC}
  "material_topics": {_SOURCED_VALUE_SHAPE},   // {_TOPICS_DESC}
  "environmental": {{
{_manifest(Environmental, ENV_FIELDS)}
  }},
  "social": {{
{_manifest(Social, SOCIAL_FIELDS)}
  }},
  "governance": {{
{_manifest(Governance, GOV_FIELDS)}
  }},
  "additional_disclosures": [
    {{"topic": string, "category": "Environmental"|"Social"|"Governance", "found": true, \
"value": string, "numeric": number|null, "page": string|null, "quote": string|null}}
  ],
  "conflicts": [
    {{"field": string, "values_found": [string, ...], "note": string}}
  ]
}}

For "additional_disclosures": {_ADDITIONAL_DESC} After completing every fixed field above, \
review the report's own list of material topics (often in a materiality matrix or similar \
section, across all of Environmental, Social and Governance) and add anything with a real \
reported figure that isn't already captured — the goal is that a reader never has to wonder \
whether something material got left out."""

REPAIR_PROMPT_TEMPLATE = """The text below was supposed to be a single JSON object but failed \
validation against the required schema with this error:

{error}

Here is the invalid text:
{bad_json}

Return ONLY the corrected, complete JSON object matching that schema — no commentary, no \
markdown code fences. The first character must be '{{' and the last must be '}}'."""


# --------------------------------------------------------------------------
# API key(s) — env var(s) or st.secrets only. No manual-entry box: keeps
# keys out of the UI entirely, matching how they're actually configured
# (secrets.toml locally / Secrets on Streamlit Cloud).
#
# Multiple keys: set GEMINI_API_KEYS as a comma-separated string (or a TOML
# list in secrets.toml — app.py joins it). Since Gemini rate limits are
# enforced per PROJECT, not per key, and each separate Google account
# normally gets its own project, several personal free-tier keys give you
# several independent 20-requests/day budgets rather than fighting over one.
# --------------------------------------------------------------------------

def get_api_keys() -> List[str]:
    multi = os.environ.get("GEMINI_API_KEYS", "").strip()
    if multi:
        keys = [k.strip() for k in multi.split(",") if k.strip()]
        if keys:
            return keys
    single = os.environ.get("GEMINI_API_KEY", "").strip()
    return [single] if single else []


def get_api_key() -> str:
    """Back-compat single-key accessor (used for simple 'is anything configured' checks)."""
    keys = get_api_keys()
    return keys[0] if keys else ""


CLIENT_TIMEOUT_MS = 10 * 60 * 1000  # 10 minutes — gemini-3.8-flash is a thinking
# model; the google-genai SDK's default HTTP timeout is much shorter than a
# long-thinking request can take, and when it's exceeded on a non-streamed
# call the connection is torn down with a "Server disconnected without
# sending a response" error rather than a clean timeout. Streaming (see
# _stream_json) also protects against this, but we set a generous explicit
# ceiling too.


def get_client(api_key: str) -> genai.Client:
    if not api_key:
        raise RuntimeError(
            "No Gemini API key found. Set GEMINI_API_KEY (or GEMINI_API_KEYS for "
            "several) as an environment variable or in .streamlit/secrets.toml."
        )
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=CLIENT_TIMEOUT_MS))


# --------------------------------------------------------------------------
# Error classification — daily quota (don't bother retrying, it won't
# recover until tomorrow) vs per-minute/transient (worth a short wait).
# --------------------------------------------------------------------------

def _is_rate_limit(err: Exception) -> bool:
    msg = str(err)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower() or "rate limit" in msg.lower()


def _is_transient_disconnect(err: Exception) -> bool:
    """Covers the known google-genai behaviour where a long-thinking request
    gets its connection dropped mid-flight instead of returning a clean
    error: 'Server disconnected without sending a response'
    (httpx.RemoteProtocolError), read timeouts, and deadline-exceeded style
    messages. These are worth retrying just like a per-minute throttle —
    the request itself was fine, the connection just didn't survive the wait."""
    msg = str(err).lower()
    return any(s in msg for s in (
        "server disconnected", "remoteprotocolerror", "deadline exceeded",
        "read timed out", "readtimeout", "connection reset", "timed out",
    ))


def _is_server_overloaded(err: Exception) -> bool:
    """Covers Gemini's '503 UNAVAILABLE ... currently experiencing high
    demand' response — a capacity problem on Google's end, not the request
    itself. Google's own error text says spikes are usually temporary, so
    this is worth a short backoff-and-retry, same as a per-minute throttle
    or a dropped connection. (Distinct from _is_rate_limit: a 503 has no
    '429'/'RESOURCE_EXHAUSTED'/'quota' in it, so it was previously falling
    through both classifiers and failing on the very first attempt with no
    retry at all.)"""
    msg = str(err)
    return "UNAVAILABLE" in msg or ("503" in msg and "high demand" in msg.lower())


def _quota_kind(err: Exception) -> Optional[str]:
    """Returns 'day' for a daily-request-cap 429 (retrying is pointless
    until the reset), 'minute' for a per-minute/other 429 (worth a short
    backoff), or None if this isn't a rate-limit error at all."""
    if not _is_rate_limit(err):
        return None
    flat = re.sub(r"[\s_-]", "", str(err).lower())
    if "perday" in flat:
        return "day"
    if "perminute" in flat:
        return "minute"
    return "minute"  # unclassified 429 (e.g. spend-based) — treat as transient


def _retry_wait_seconds(err: Exception, attempt: int) -> float:
    m = re.search(r"retry in ([\d.]+)\s*s", str(err), re.IGNORECASE)
    if m:
        return float(m.group(1)) + 2.0
    return min(15.0 * (2 ** attempt), 90.0)


def _seconds_until_pacific_midnight() -> float:
    try:
        from zoneinfo import ZoneInfo
        import datetime
        now_pt = datetime.datetime.now(ZoneInfo("America/Los_Angeles"))
        reset_pt = (now_pt + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
        return max(0.0, (reset_pt - now_pt).total_seconds())
    except Exception:
        return 24 * 3600.0  # best-effort fallback if zoneinfo data is unavailable


# --------------------------------------------------------------------------
# Multi-key rotation
# --------------------------------------------------------------------------

class KeyPool:
    """Rotates across one or more Gemini API keys, skipping any that have
    hit today's free-tier daily cap. Exhaustion state is persisted to a
    small local JSON file so it survives a Streamlit restart within the
    same day — but this is best-effort only (never blocks the app if the
    file can't be written, e.g. on a read-only deployment)."""

    def __init__(self, keys: List[str], state_path: Optional[str] = None):
        if not keys:
            raise RuntimeError("KeyPool needs at least one API key.")
        self.keys = keys
        self._state_path = state_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".esglens_key_state.json"
        )
        self._state = self._load_state()
        self._idx = 0
        self._skip_to_usable()

    @staticmethod
    def _fp(key: str) -> str:
        return key[-8:] if len(key) > 8 else key  # short fingerprint only, never the full key

    def _load_state(self) -> dict:
        try:
            with open(self._state_path) as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_state(self) -> None:
        try:
            with open(self._state_path, "w") as f:
                json.dump(self._state, f)
        except Exception:
            pass  # best-effort persistence only

    def _exhausted_until(self, key: str) -> float:
        return float(self._state.get(self._fp(key), 0))

    def _is_exhausted(self, key: str) -> bool:
        return time.time() < self._exhausted_until(key)

    def _skip_to_usable(self) -> None:
        for _ in range(len(self.keys)):
            if not self._is_exhausted(self.keys[self._idx]):
                return
            self._idx = (self._idx + 1) % len(self.keys)

    @property
    def current_key(self) -> str:
        return self.keys[self._idx]

    @property
    def usable_count(self) -> int:
        return sum(1 for k in self.keys if not self._is_exhausted(k))

    def mark_current_exhausted(self) -> None:
        self._state[self._fp(self.current_key)] = time.time() + _seconds_until_pacific_midnight()
        self._save_state()

    def rotate(self) -> bool:
        """Advance to the next non-exhausted key. Returns False if every key is exhausted."""
        for _ in range(len(self.keys)):
            self._idx = (self._idx + 1) % len(self.keys)
            if not self._is_exhausted(self.current_key):
                return True
        return False

    def seconds_until_any_reset(self) -> float:
        vals = [self._exhausted_until(k) for k in self.keys]
        if not all(vals):
            return 0.0
        return max(0.0, min(vals) - time.time())


# --------------------------------------------------------------------------
# Core Gemini call
# --------------------------------------------------------------------------

def _wait_until_active(client: genai.Client, uploaded, timeout_s: int = 60):
    """Best-effort poll in case the Files API returns a file that's still
    processing. Guarded so it never breaks the flow if the SDK response
    doesn't expose a state field."""
    start = time.time()
    file_obj = uploaded
    while getattr(file_obj, "state", None) == "PROCESSING" and time.time() - start < timeout_s:
        time.sleep(1.5)
        try:
            file_obj = client.files.get(name=file_obj.name)
        except Exception:
            break
    return file_obj


def _upload(client: genai.Client, pdf_bytes: bytes, filename: str):
    try:
        uploaded = client.files.upload(
            file=io.BytesIO(pdf_bytes), config=dict(mime_type="application/pdf", display_name=filename)
        )
        return _wait_until_active(client, uploaded)
    except Exception as e:
        raise RuntimeError(f"[upload step] {e}") from e


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _stream_json(client: genai.Client, uploaded, prompt: str) -> str:
    """One streamed generate_content call, no response_json_schema (see
    module docstring for why). Streaming — rather than a single blocking
    call — matters because a long-thinking request on a plain
    generate_content() call can have its connection torn down as idle
    before any bytes come back; streaming keeps it alive with periodic
    chunks."""
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        max_output_tokens=MAX_OUTPUT_TOKENS,
        # gemini-3.8-flash defaults to "medium" thinking; "low" is the right
        # level for a bounded extraction task like this one (not open-ended
        # multi-step reasoning) and keeps the call quick.
        # ("minimal" isn't supported on this model.)
        thinking_config=types.ThinkingConfig(thinking_level="low"),
    )
    contents = [types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type), prompt] if uploaded else [prompt]
    chunks: List[str] = []
    for chunk in client.models.generate_content_stream(model=MODEL_NAME, contents=contents, config=config):
        if chunk.text:
            chunks.append(chunk.text)
    text = "".join(chunks)
    if not text.strip():
        raise RuntimeError("Empty response from Gemini.")
    return text


def _validate_or_repair(client: genai.Client, raw_text: str) -> ESGExtraction:
    cleaned = _strip_code_fences(raw_text)
    try:
        return ESGExtraction.model_validate_json(cleaned)
    except Exception as e1:
        # One cheap, PDF-free repair attempt — re-running rarely produces
        # the exact same malformed shape twice, and this costs far less
        # (no file re-read, no full re-extraction) than starting over.
        repair_prompt = REPAIR_PROMPT_TEMPLATE.format(error=e1, bad_json=cleaned[:8000])
        text2 = _stream_json(client, None, repair_prompt)
        cleaned2 = _strip_code_fences(text2)
        try:
            return ESGExtraction.model_validate_json(cleaned2)
        except Exception as e2:
            raise RuntimeError(f"[parsing step] Gemini's response wasn't valid JSON even after one repair attempt: {e2}") from e2


def analyse_report(
    pdf_bytes: bytes,
    filename: str,
    company_hint: str = "",
    on_progress: Optional[Callable[[str], None]] = None,
    on_retry: Optional[Callable[[float], None]] = None,
    on_key_switch: Optional[Callable[[int, int], None]] = None,
) -> ESGExtraction:
    """Runs the whole extraction as ONE Gemini call (see module docstring).
    on_progress(msg) fires at each major step; on_retry(wait_seconds) fires
    before a short transient-error backoff; on_key_switch(usable, total)
    fires when a key hits today's cap and rotation moves to the next one —
    all so a caller (e.g. Streamlit) can show live status."""
    keys = get_api_keys()
    if not keys:
        raise RuntimeError(
            "No Gemini API key found. Set GEMINI_API_KEY (or GEMINI_API_KEYS for "
            "several) as an environment variable or in .streamlit/secrets.toml."
        )
    pool = KeyPool(keys)

    prompt = MASTER_PROMPT
    if company_hint:
        prompt += f"\n\nThe company is likely called: {company_hint}."

    max_attempts = len(pool.keys) + MAX_TRANSIENT_RETRIES + 1
    last_err: Optional[Exception] = None

    for _attempt in range(max_attempts):
        client = get_client(pool.current_key)
        try:
            if on_progress:
                on_progress("Uploading report to Gemini…")
            uploaded = _upload(client, pdf_bytes, filename)

            if on_progress:
                on_progress("Extracting ESG data (one pass over the full report)…")
            raw_text = _stream_json(client, uploaded, prompt)
            return _validate_or_repair(client, raw_text)

        except Exception as e:
            last_err = e
            kind = _quota_kind(e)
            if kind == "day":
                pool.mark_current_exhausted()
                if pool.rotate():
                    if on_key_switch:
                        on_key_switch(pool.usable_count, len(pool.keys))
                    continue
                wait_h = pool.seconds_until_any_reset() / 3600.0
                raise RuntimeError(
                    f"[quota] All {len(pool.keys)} configured API key(s) have hit today's "
                    f"free-tier daily cap (~20 requests/day/key). Resets at midnight Pacific "
                    f"Time (~{wait_h:.1f}h from now)."
                ) from e
            if kind == "minute" or _is_transient_disconnect(e) or _is_server_overloaded(e):
                wait_s = _retry_wait_seconds(e, _attempt)
                if on_retry:
                    on_retry(wait_s)
                time.sleep(wait_s)
                continue
            raise
    raise last_err or RuntimeError("Exceeded max attempts.")  # pragma: no cover


# --------------------------------------------------------------------------
# Multi-report helpers (unchanged — operate on the final ESGExtraction
# shape regardless of how many Gemini calls produced it)
# --------------------------------------------------------------------------

def report_labels(reports: List[ESGExtraction]) -> List[str]:
    """Unique, human-readable label per report — handles the same company
    appearing twice (e.g. two different years) without collisions."""
    labels: List[str] = []
    seen: dict = {}
    for r in reports:
        base = r.company_name or "Report"
        if r.reporting_year:
            base = f"{base} ({r.reporting_year})"
        seen[base] = seen.get(base, 0) + 1
        labels.append(base if seen[base] == 1 else f"{base} #{seen[base]}")
    return labels


def check_comparability(reports: List[ESGExtraction]) -> List[str]:
    """Flags reasons a side-by-side comparison might mislead, instead of
    silently plotting numbers that aren't actually comparable."""
    notes: List[str] = []
    names = [r.company_name for r in reports]
    years = {r.reporting_year for r in reports if r.reporting_year}
    same_company = len(set(n for n in names if n)) == 1 and len(reports) > 1

    if same_company:
        year_txt = ", ".join(sorted(years)) if years else "years not identified"
        notes.append(
            f"These are multiple years of the same company's reporting ({year_txt}) "
            "— read this as a trend over time, not a peer comparison."
        )
    else:
        sectors = {r.industry_sector for r in reports if r.industry_sector}
        if len(sectors) == 1 and len(reports) > 1:
            notes.append(f"All companies are in the same sector ({next(iter(sectors))}) — this is a reasonably like-for-like comparison.")
        elif len(sectors) > 1:
            notes.append(
                f"These companies span different sectors ({', '.join(sorted(sectors))}) "
                "— compare cautiously; typical scale and materiality vary a lot by industry."
            )
        if len(years) > 1:
            notes.append(f"Reports cover different reporting years ({', '.join(sorted(years))}) — not a strict like-for-like comparison.")

    frameworks = {r.reporting_framework.value for r in reports if r.reporting_framework.found}
    if len(frameworks) > 1:
        notes.append(f"Reports use different disclosure frameworks ({', '.join(sorted(frameworks))}) — scope/boundary definitions may not align.")

    getters = (
        [(a, l, (lambda r, a=a: getattr(r.environmental, a))) for a, l in ENV_FIELDS]
        + [(a, l, (lambda r, a=a: getattr(r.social, a))) for a, l in SOCIAL_FIELDS]
        + [(a, l, (lambda r, a=a: getattr(r.governance, a))) for a, l in GOV_FIELDS]
    )
    for _attr, label, getter in getters:
        flags = [getter(r).found for r in reports]
        if any(flags) and not all(flags):
            missing = [r.company_name for r, f in zip(reports, flags) if not f]
            notes.append(f"{label} is not disclosed by {', '.join(missing)} — comparison for this metric is partial.")

    return notes


def to_rows(ex: ESGExtraction) -> List[dict]:
    """Flatten one extraction into Category/Metric/Value/Page/Quote/Trend rows."""
    rows: List[dict] = []
    groups = [
        ("Overview", OVERVIEW_FIELDS, lambda a: getattr(ex, a)),
        ("Environmental", ENV_FIELDS, lambda a: getattr(ex.environmental, a)),
        ("Social", SOCIAL_FIELDS, lambda a: getattr(ex.social, a)),
        ("Governance", GOV_FIELDS, lambda a: getattr(ex.governance, a)),
    ]
    for category, fields, getter in groups:
        for attr, label in fields:
            sv: SourcedValue = getter(attr)
            rows.append({
                "Category": category,
                "Metric": label,
                "Value": sv.value,
                "Page": sv.page or "",
                "Trend": sv.trend or "",
                "Quote": sv.quote or "",
            })
    for d in ex.additional_disclosures:
        rows.append({
            "Category": d.category,
            "Metric": d.topic,
            "Value": d.value,
            "Page": d.page or "",
            "Trend": "",
            "Quote": d.quote or "",
        })
    return rows


# --------------------------------------------------------------------------
# Copilot — free-text Q&A over one or more already-uploaded reports.
# Pure addition alongside analyse_report(): reuses the exact same
# get_api_keys/KeyPool/get_client/_upload/_quota_kind/_is_transient_disconnect/
# _is_server_overloaded/_retry_wait_seconds machinery, unmodified, so it
# inherits the same daily-cap detection, multi-key rotation, and
# transient-error backoff for free. Answers are plain text (not the JSON
# extraction schema) and capped much shorter, since a question's answer is
# normally a few sentences, not a 200-field report.
# --------------------------------------------------------------------------

COPILOT_MAX_OUTPUT_TOKENS = 2048

COPILOT_PROMPT_TEMPLATE = """You are an ESG/sustainability analyst assistant. Using ONLY what is \
explicitly stated in the attached report(s) (native PDF understanding — you can read tables, \
charts and layout, not just body text), answer the user's question as accurately and \
specifically as possible.

Rules:
- Only state what the document(s) actually say. Never guess, estimate, or use outside knowledge \
if the document doesn't cover it.
- Whenever you cite a figure or statement, mention the page number exactly as PRINTED in the \
document's own footer/header (not the PDF file's internal page position).
- If multiple reports are attached: if the question is about one company specifically, answer \
about that one; if it's comparative, compare using each report's own figures and name which \
report each figure came from.
- If the answer genuinely isn't in the document(s), say so plainly rather than guessing.
- Keep the answer concise and direct — a few sentences or a short list — unless the question \
itself asks for detail.

Question: {question}"""


def ask_about_reports(
    files: List[tuple],  # list of (pdf_bytes, filename)
    question: str,
    on_progress: Optional[Callable[[str], None]] = None,
    on_retry: Optional[Callable[[float], None]] = None,
    on_key_switch: Optional[Callable[[int, int], None]] = None,
) -> str:
    """The 'copilot' feature: a single free-text question answered against
    one or more reports, re-uploaded fresh for this call (uploads aren't
    counted against the free-tier daily generate_content cap, only this
    actual question is)."""
    keys = get_api_keys()
    if not keys:
        raise RuntimeError(
            "No Gemini API key found. Set GEMINI_API_KEY (or GEMINI_API_KEYS for "
            "several) as an environment variable or in .streamlit/secrets.toml."
        )
    if not files:
        raise RuntimeError("No report uploaded to ask about yet.")
    pool = KeyPool(keys)
    prompt = COPILOT_PROMPT_TEMPLATE.format(question=question.strip())

    max_attempts = len(pool.keys) + MAX_TRANSIENT_RETRIES + 1
    last_err: Optional[Exception] = None

    for _attempt in range(max_attempts):
        client = get_client(pool.current_key)
        try:
            if on_progress:
                on_progress("Uploading report(s)…" if len(files) > 1 else "Uploading report…")
            file_parts = []
            for pdf_bytes, filename in files:
                uploaded = _upload(client, pdf_bytes, filename)
                file_parts.append(types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type))

            if on_progress:
                on_progress("Reading the report(s) to answer…")
            config = types.GenerateContentConfig(
                max_output_tokens=COPILOT_MAX_OUTPUT_TOKENS,
                thinking_config=types.ThinkingConfig(thinking_level="low"),
            )
            chunks: List[str] = []
            for chunk in client.models.generate_content_stream(
                model=MODEL_NAME, contents=[*file_parts, prompt], config=config
            ):
                if chunk.text:
                    chunks.append(chunk.text)
            answer = "".join(chunks).strip()
            if not answer:
                raise RuntimeError("Empty response from Gemini.")
            return answer

        except Exception as e:
            last_err = e
            kind = _quota_kind(e)
            if kind == "day":
                pool.mark_current_exhausted()
                if pool.rotate():
                    if on_key_switch:
                        on_key_switch(pool.usable_count, len(pool.keys))
                    continue
                wait_h = pool.seconds_until_any_reset() / 3600.0
                raise RuntimeError(
                    f"[quota] All {len(pool.keys)} configured API key(s) have hit today's "
                    f"free-tier daily cap (~20 requests/day/key). Resets at midnight Pacific "
                    f"Time (~{wait_h:.1f}h from now)."
                ) from e
            if kind == "minute" or _is_transient_disconnect(e) or _is_server_overloaded(e):
                wait_s = _retry_wait_seconds(e, _attempt)
                if on_retry:
                    on_retry(wait_s)
                time.sleep(wait_s)
                continue
            raise
    raise last_err or RuntimeError("Exceeded max attempts.")  # pragma: no cover
