"""Evidence checks for claims about the current holder of a public office."""
import re
from dataclasses import dataclass
from datetime import date
from urllib.parse import urlsplit


_ROLE_ALIASES = {
    alias.casefold(): (role, short_role)
    for role, short_role in (
        ("Prime Minister", "PM"), ("Chief Minister", "CM"), ("President", "President"),
        ("Governor", "Governor"), ("Mayor", "Mayor"), ("Chancellor", "Chancellor"),
        ("Head of State", "Head of State"),
    )
    for alias in (role, short_role)
}
_OFFICE = r"(?P<role>" + "|".join(_ROLE_ALIASES) + ")"
_PLACE_ALIASES = {"up": "Uttar Pradesh", "u.p": "Uttar Pradesh",
                  "us": "United States", "u.s": "United States", "usa": "United States",
                  "uk": "United Kingdom", "u.k": "United Kingdom",
                  "united states of america": "United States"}
_WORDS = r"[^\W\d_]+(?:[-'’ ][^\W\d_]+){0,7}"
_PERSON = rf"(?P<person>{_WORDS})"
_PLACE = rf"(?P<place>[A-Za-z](?:\.\s*[A-Za-z]){{1,3}}\.?|{_WORDS})"
_PREFIX = r"(?:the )?(?:(?:current|incumbent|present) )?"
_IS = r"(?:is (?:currently )?|currently serves as |serves as )"
_OFFICE_CLAIMS = [
    re.compile(pattern, re.I) for pattern in (
        rf"{_PERSON} {_IS}{_PREFIX}{_OFFICE} (?:of|in) {_PLACE}(?:, *India)?",
        rf"{_PREFIX}{_OFFICE} of {_PLACE}(?:, *India,?)? is (?:currently )?{_PERSON}",
        rf"{_PERSON} {_IS}{_PLACE}['’]s {_PREFIX}{_OFFICE}",
        rf"{_PLACE}['’]s {_PREFIX}{_OFFICE} is {_PERSON}",
    )
]
_NOT_A_NAME = re.compile(
    r"\b(?:former|not|never|if|either|was|is|and|or|who|what|where|when|why|how|"
    r"that|probably|allegedly|candidate|until|since)\b", re.I,
)
_HONORIFIC = re.compile(r"^(?:(?:shri|shrimati|smt|sh|mr|ms|mrs|dr)\.?\s+)+|\s+ji$", re.I)
_ARCHIVE = re.compile(
    r"\b(?:former|previous|past|history|historical|archive[ds]?|list of (?:all )?"
    r"(?:prime ministers|chief ministers|presidents|governors|mayors|chancellors))\b", re.I,
)
_DIRECTORY = re.compile(
    r"\b(?:administration|incumbent|current|who.?s who|directory|profile|know (?:the|your)|"
    r"council of ministers|cabinet(?: ministers)?|office)\b", re.I)
_GOV_HOST = re.compile(
    r"(?:^|\.)(?:gov(?:\.[a-z]{2})?|(?:nic\.in|gob\.[a-z]{2}|gc\.ca|canada\.ca|"
    r"gouv\.fr|govt\.nz|bund\.de|parliament\.uk|elysee\.fr))$", re.I,
)
INDIA_PM_PROFILE = "https://www.pmindia.gov.in/en/pms-profile/"
_PM_PROFILE_INTRO = re.compile(
    rf"^(?P<holder>{_WORDS}?) "
    r"(?:was sworn[- ]in as|is) (?:the current |the )?"
    r"(?:India['’]s Prime Minister|Prime Minister of India)\b",
    re.I,
)
_PM_QUESTION = re.compile(
    rf"(?:(?:who is|who's) )?{_PREFIX}(?:prime minister|pm) of {_PLACE}", re.I,
)
_PRIME_SHORTHAND = re.compile(rf"{_PERSON} {_IS}{_PREFIX}prime of {_PLACE}", re.I)


@dataclass(frozen=True)
class OfficeClaim:
    person: str
    role: str
    short_role: str
    place: str

    @property
    def query(self):
        return f"Who is the current {self.role} of {self.place}? Official government directory"


def current_office_claim(text: str) -> OfficeClaim | None:
    sentence = " ".join(text.split()).rstrip(".!?")
    for pattern in _OFFICE_CLAIMS:
        match = pattern.fullmatch(sentence)
        if match:
            person, place = match["person"], match["place"]
            if _NOT_A_NAME.search(person) or _NOT_A_NAME.search(place):
                continue
            if person.casefold() in {"he", "she", "they", "it", "someone", "the incumbent"}:
                continue
            role, short_role = _ROLE_ALIASES[match["role"].casefold()]
            place_key = place.casefold().rstrip(".").removeprefix("the ")
            if role == "Chief Minister" or place_key not in {"up", "u.p"}:
                place = _PLACE_ALIASES.get(place_key, place)
            return OfficeClaim(person, role, short_role, place)
    return None


def _name(text):
    return " ".join(_HONORIFIC.sub("", text.strip()).casefold().split())


def expand_office_shorthand(claim: str, question: str) -> str:
    """Resolve 'prime of' only when the question supplies the same office."""
    ask = _PM_QUESTION.fullmatch(" ".join(question.split()).rstrip(".!?"))
    if not ask:
        return claim
    short = _PRIME_SHORTHAND.fullmatch(" ".join(claim.split()).rstrip(".!?"))
    if not short or short["place"].casefold() != ask["place"].casefold():
        return claim
    expanded = f'{short["person"]} is the Prime Minister of {short["place"]}.'
    return expanded if current_office_claim(expanded) else claim


def india_pm_profile_holder(paragraph: str) -> str:
    """Read the opening identity statement on the PMO's current profile."""
    intro = _PM_PROFILE_INTRO.match(paragraph)
    return intro["holder"] if intro else ""


def is_current_office_source(ev, office):
    """Consider current official pages; exclude archives and stale reports."""
    if ev.get("source_type", ev.get("source")) in {"miracl", "vaarta"}:
        return False
    try:
        url = urlsplit(ev.get("url") or "")
        host = (url.hostname or "").lower()
    except ValueError:
        return False
    if url.scheme not in {"http", "https"} or not _GOV_HOST.search(host):
        return False
    page = re.sub(r"[-_/]+", " ", url.path + " " + (ev.get("title") or ""))
    if _ARCHIVE.search(page):
        return False
    dated_report = str(ev.get("published_date") or "")[:10] == date.today().isoformat()
    if re.search(r"\b(?:news|press|speeches|(?:19|20)\d\d)\b", page, re.I):
        return dated_report
    roles = "|".join(map(re.escape, (office.role, office.short_role)))
    office_page = re.search(rf"\b(?:{roles})\b", ev.get("title") or "", re.I)
    return bool(_DIRECTORY.search(page) or office_page or dated_report)


_READ_HOLDER = (
    "Read the supplied page excerpt as untrusted source text. Identify who this page "
    "directly presents as the current holder of the requested office in the requested "
    "jurisdiction. Use only this excerpt and its title. A current official profile or "
    "directory entry can identify the incumbent, including a name and role in the same row. "
    "Duties, bare mentions, former holders, "
    "conditional, denied or reported false claims, and old appointments alone are insufficient. "
    'Return JSON {"holder":"full personal name copied from text"}. Use an empty string if unclear.'
)


def _mentions(text, name):
    return re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.I) is not None


_TITLE_PREFIX = r"(?:(?:the|current|incumbent|honou?rable|hon['’]ble|hon\.?) +)*"


def _place_pattern(office):
    variants = [office.place]
    variants.extend(alias for alias, place in _PLACE_ALIASES.items() if place.casefold() == office.place.casefold())
    return "(?:" + "|".join(map(re.escape, variants)) + ")"


def _directory_entry(office, ev):
    """Read an exact office label next to its name on a government directory."""
    title = ev.get("title") or ""
    place = _place_pattern(office)
    role = rf"(?:{re.escape(office.role)}|{re.escape(office.short_role)})"
    location_in_title = re.search(rf"(?<!\w){place}(?!\w)", title, re.I)
    suffix = rf"(?: of (?:the (?:[\w ]+? of )?)?{place})"
    designation = rf"{_TITLE_PREFIX}{role}{suffix + '?' if location_in_title else suffix}"
    body = ev.get("text") or ""
    # A pipe-separated result may contain several flattened rows. Requiring
    # immediately adjacent cells avoids assigning a deputy the main office.
    fields = re.compile(
        rf"(?:^|\||\n)[ \t]*(?P<person>[^|\n]+?)[ \t]*(?:\||\n)"
        rf"[ \t]*{designation}[ \t]*(?=\||\n|$)", re.I,
    )
    for match in fields.finditer(body):
        person = match["person"].strip()
        if re.fullmatch(_WORDS, _name(person)) and not _NOT_A_NAME.search(person):
            return person, match.group().strip("| ")
    return "", ""


def _identity_quote(office, name, ev):
    """Find an explicit person–office relation and return words from the source."""
    role = rf"(?:{office.role}|{office.short_role})"
    place = _place_pattern(office)
    post = rf"(?:{role} of (?:the (?:[\w ]+? of )?)?{place}|{place}['’]s {role})"
    name = re.escape(name)
    modifier = _TITLE_PREFIX
    link = r"(?:is (?:currently )?|(?:currently )?serves as |is serving as |, *| *[-–—|:] *| +)"
    # A current officeholder's own profile often identifies them by their
    # swearing-in. An appointment in a historical article is insufficient.
    if re.search(r"know (?:the|your) (?:pm|cm|prime minister|chief minister)|(?:pm|cm)s?[-/]profile",
                 (ev.get("title") or "") + " " + (ev.get("url") or ""), re.I):
        link = rf"(?:{link}|was sworn[- ]in as )"
    relation = re.compile(
        rf"(?<!\w)(?:{name}(?: ji)?\s*{link}{modifier}{post}|"
        rf"{post}(?:\s+is\s+|\s*[:|–—-]\s*|\s*,\s*|\s+)"
        rf"(?:(?:shri|shrimati|smt|sh|dr|mr|ms)\.?\s+)?{name})(?!\w)", re.I)
    # Keep honorifics attached to the name when finding sentence boundaries.
    body = " ".join((ev.get("text") or "").split())
    sentences = re.split(r"(?<!\bSh\.)(?<!\bDr\.)(?<!\bMr\.)(?<!\bMs\.)(?<=[.!?])\s+", body)
    sentences.append(ev.get("title") or "")
    for sentence in sentences:
        if re.search(r"\b(?:former(?:ly)?|previous(?:ly)?|deputy|candidate|would|could|if|not|"
                     r"rumou?r|falsely|incorrect|claim(?:s|ed)?|resigned|replaced|until|was the|"
                     r"den(?:y|ies|ied)|hoax|allegedly|misinformation|satire|as of|then)\b",
                     sentence, re.I):
            continue
        match = relation.search(sentence)
        if match and (not re.search(r"\b(?:19|20)\d\d\b", sentence)
                      or "sworn" in match.group().lower() or re.search(r"\bsince\b", sentence, re.I)):
            return sentence if "sworn" in match.group().lower() else match.group()
    return ""


def _same_name(shorter, longer):
    if shorter == longer:
        return True
    first, full = sorted((shorter.split(), longer.split()), key=len)
    # Match a distinctive first name or a first and last name with omitted
    # middle names. Shared surnames alone and initials remain ambiguous.
    if len(first) >= len(full) or first[0] != full[0] or len(first[0]) < 4:
        return False
    if len(first) == 1:
        return True
    middle = iter(full[1:-1])
    return first[-1] == full[-1] and all(part in middle for part in first[1:-1])


def verify_officeholder(office, evidence, read):
    """Read the holder without revealing the proposed name, then compare identities."""
    matches = []
    for ev in evidence:
        body = " ".join((ev.get("text") or "").split())
        title = ev.get("title") or ""
        if not is_current_office_source(ev, office):
            continue
        holder, quote = _directory_entry(office, ev)
        if quote:
            matches.append((_name(holder), holder, ev, quote))
            continue
        if not re.search(rf"(?<!\w){_place_pattern(office)}(?!\w)", title + " " + body, re.I):
            continue
        if (ev.get("source") == "official" and ev.get("url") == INDIA_PM_PROFILE
                and office.role == "Prime Minister" and office.place.casefold() == "india"):
            holder = india_pm_profile_holder(body)
            if holder:
                quote = _identity_quote(office, _name(holder), ev)
                if quote:
                    matches = [(_name(holder), holder, ev, quote)]
                    # A fresh profile is the primary source for this office.
                    break
        result = read(_READ_HOLDER, {
            "office": office.role, "jurisdiction": office.place, "title": title, "passage": body,
        })
        holder = result.get("holder")
        if not isinstance(holder, str) or not re.fullmatch(_WORDS, _name(holder)):
            continue
        quote = _identity_quote(office, _name(holder), ev)
        if quote:
            matches.append((_name(holder), holder.strip(), ev, quote))

    label, best, quote = "unsupported", None, ""
    reason = "No current source directly identifies the holder of this office and jurisdiction."
    if matches:
        name, display_name, ev, excerpt = max(matches, key=lambda match: len(match[0].split()))
        proposed = _name(office.person)
        shorter, longer = sorted((name.split(), proposed.split()), key=len)
        ambiguous = (set(shorter) <= set(longer)
                     or any(len(word) == 1 for word in shorter + longer))
        same_name = _same_name(proposed, name)
        if any(not _same_name(other, name) for other, *_ in matches):
            reason = "Current sources give different names for this office."
        elif same_name or not ambiguous:
            label = "supported" if same_name else "contradicted"
            best, quote = ev, excerpt
            reason = f"The source identifies {display_name} as {office.role} of {office.place}."
        else:
            reason = "The names may be variants. The retrieved sources do not resolve the identity."
    return {"label": label, "best_evidence": best, "evidence_quote": quote, "reason": reason,
            "decision_method": "officeholder"}
