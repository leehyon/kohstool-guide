import argparse
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache, wraps
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

try:
    import mistune
except ImportError:  # pragma: no cover
    mistune = None  # type: ignore[assignment]

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]

try:
    from waybackpy import WaybackMachineSaveAPI
except ImportError:  # pragma: no cover
    WaybackMachineSaveAPI = None  # type: ignore[assignment]


# -- configurations begin --
TOOL_REPO_NAME: str = "kohstool"
GUIDE_REPO_NAME: str = "kohstool-guide"

MAX_CONTENT_LENGTH: int = 32 * 1024  # 32KB
MIN_CONTENT_LENGTH: int = 200
MAX_RETRIES: int = 3

# data.json caches guide_markdown as a full GitHub blob URL (not the markdown content).
# You may override repo/ref detection via:
# - GUIDE_GITHUB_REPO=owner/repo
# - GUIDE_GITHUB_REF=main

HTTP_CONNECT_TIMEOUT_SECONDS: int = 5
HTTP_READ_TIMEOUT_SECONDS: int = 30
# -- configurations end --


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@lru_cache(maxsize=1)
def _get_requests_session():
    """Return a shared requests.Session for connection pooling."""
    if requests is None:
        return None
    try:
        return requests.Session()
    except Exception:  # noqa: BLE001
        return None


def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info("Entering %s", func.__name__)
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logging.info("Exiting %s - Elapsed time: %.4f seconds", func.__name__, elapsed_time)
        return result

    return wrapper


@dataclass
class ToolGuideEntry:
    month: str  # yyyyMM
    name: str
    url: str
    timestamp: int  # unix timestamp
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    platform: List[str] = field(default_factory=list)
    # Cached fields for downstream processing convenience.
    tldr: str = ""
    # NOTE: Despite the name, this stores a full GitHub blob URL to the guide markdown file.
    guide_markdown: str = ""

    def identity(self) -> Tuple[str, str, int]:
        return (self.month, self.name, self.timestamp)

    def to_dict(self) -> dict:
        return {
            "month": self.month,
            "name": self.name,
            "url": self.url,
            "timestamp": self.timestamp,
            "tags": list(self.tags),
            "categories": list(self.categories),
            "Platform": list(self.platform),
            "tldr": self.tldr,
            "guide_markdown": self.guide_markdown,
        }

    @staticmethod
    def from_dict(payload: dict) -> "ToolGuideEntry":
        guide_markdown = normalize_guide_markdown_url(payload.get("guide_markdown") or payload.get("guideMarkdown") or "")
        return ToolGuideEntry(
            month=payload["month"],
            name=payload["name"],
            url=payload["url"],
            timestamp=payload["timestamp"],
            tags=payload.get("tags") or [],
            categories=payload.get("categories") or payload.get("Categories") or [],
            platform=payload.get("Platform") or payload.get("platform") or [],
            tldr=(payload.get("tldr") or "").strip(),
            guide_markdown=guide_markdown,
        )


@dataclass
class ToolGuideContent:
    tldr: str
    scenarios: List[str]
    pain_points: List[str]
    design_principles: List[str]
    categories: List[str]
    similar_tools: List[str]
    tags: List[str]
    platform: List[str]


@dataclass
class RunOptions:
    backfill: bool
    dry_run: bool
    archive: bool
    collection_readme_path: Path
    max_new: int
    heuristic: bool
    fetch: bool


@dataclass
class IngestionResult:
    entry: ToolGuideEntry
    guide_markdown: str
    guide_path: Path
    tldr: str


CURRENT_DATE_AND_TIME: str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

SCRIPT_DIR = Path(__file__).resolve().parent

SUMMARY_ROOT = Path(GUIDE_REPO_NAME)
if not SUMMARY_ROOT.exists():
    SUMMARY_ROOT = Path(".")

DATA_PATH = SUMMARY_ROOT / "data.json"
SUMMARY_README_PATH = SUMMARY_ROOT / "README.md"
ALL_GUIDE_PATH = SUMMARY_ROOT / "all_guide.md"


CATEGORY_CHOICES: List[str] = [
    "Knowledge Management",
    "Reading & Information",
    "Text Input & Writing",
    "Developer Tools",
    "File Management",
    "System & Automation",
    "Communication",
    "Media & Creativity",
    "Security & Privacy",
    "Data & Analytics",
]


def normalize_categories(categories: Iterable[str], max_count: int = 3) -> List[str]:
    cleaned: List[str] = []
    seen: set[str] = set()
    canonical_by_lower = {choice.lower(): choice for choice in CATEGORY_CHOICES}
    aliases = {
        "knowledge": "Knowledge Management",
        "knowledge base": "Knowledge Management",
        "note-taking": "Knowledge Management",
        "notes": "Knowledge Management",
        "rss": "Reading & Information",
        "reader": "Reading & Information",
        "reading": "Reading & Information",
        "text expansion": "Text Input & Writing",
        "text expander": "Text Input & Writing",
        "writing": "Text Input & Writing",
        "terminal": "Developer Tools",
        "cli": "Developer Tools",
        "dev": "Developer Tools",
        "developer": "Developer Tools",
        "file manager": "File Management",
        "files": "File Management",
        "system": "System & Automation",
        "automation": "System & Automation",
        "chat": "Communication",
        "messaging": "Communication",
        "media": "Media & Creativity",
        "design": "Media & Creativity",
        "security": "Security & Privacy",
        "privacy": "Security & Privacy",
        "data": "Data & Analytics",
        "analytics": "Data & Analytics",
    }

    for item in categories:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if not value:
            continue
        key = value.lower()
        canonical = canonical_by_lower.get(key) or canonical_by_lower.get(aliases.get(key, "").lower())
        if not canonical:
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        cleaned.append(canonical)

    # Stable ordering, capped.
    ordered = [choice for choice in CATEGORY_CHOICES if choice in seen]
    return ordered[:max_count]


def normalize_tags(tags: Iterable[str], max_count: int = 5) -> List[str]:
    cleaned: List[str] = []
    seen: set[str] = set()
    for tag in tags:
        if not isinstance(tag, str):
            continue
        value = tag.strip()
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
        if len(cleaned) >= max_count:
            break
    return cleaned


def normalize_platforms(platforms: Iterable[str]) -> List[str]:
    """Normalize platform list, explicitly excluding Cross-platform."""
    cleaned: List[str] = []
    seen: set[str] = set()
    for platform in platforms:
        if not isinstance(platform, str):
            continue
        value = platform.strip()
        if not value:
            continue
        if value.lower() == "cross-platform":
            continue
        if value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return cleaned

def _read_git_remote_origin_url(repo_root: Path) -> Optional[str]:
    config_path = repo_root / ".git" / "config"
    if not config_path.exists():
        return None

    try:
        content = config_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return None

    in_origin = False
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_origin = line.lower() == '[remote "origin"]'
            continue
        if in_origin and line.lower().startswith("url") and "=" in line:
            return line.split("=", 1)[1].strip()
    return None


def _parse_github_owner_repo(remote_url: str) -> Optional[str]:
    if not remote_url:
        return None
    url = remote_url.strip()

    # git@github.com:owner/repo.git
    if url.startswith("git@github.com:"):
        slug = url.split(":", 1)[1].strip()
        if slug.endswith(".git"):
            slug = slug[:-4]
        return slug if "/" in slug else None

    # https://github.com/owner/repo(.git)
    if re.match(r"^https?://github\.com/", url, flags=re.IGNORECASE):
        slug = re.sub(r"^https?://github\.com/", "", url, flags=re.IGNORECASE).strip("/")
        if slug.endswith(".git"):
            slug = slug[:-4]
        parts = slug.split("/")
        if len(parts) >= 2:
            return "/".join(parts[:2])
        return None

    return None


@lru_cache(maxsize=1)
def _get_guide_repo_slug() -> Optional[str]:
    env_slug = os.getenv("GUIDE_GITHUB_REPO", "").strip()
    if env_slug:
        parsed = _parse_github_owner_repo(env_slug)
        return parsed or env_slug

    origin = _read_git_remote_origin_url(SCRIPT_DIR)
    return _parse_github_owner_repo(origin or "")


@lru_cache(maxsize=1)
def _get_guide_repo_ref() -> str:
    return os.getenv("GUIDE_GITHUB_REF", "main").strip() or "main"


def build_guide_markdown_blob_url(entry: ToolGuideEntry) -> str:
    """Build a full GitHub blob URL for the guide markdown file.

    Uses the same path logic as README links (in_readme_md=True).
    """
    rel_path = get_guide_file_path(
        name=entry.name,
        timestamp=entry.timestamp,
        month=entry.month,
        in_readme_md=True,
    ).as_posix()

    repo_slug = _get_guide_repo_slug()
    if not repo_slug:
        logging.warning("Could not detect GitHub repo slug for guide; set GUIDE_GITHUB_REPO=owner/repo.")
        return rel_path
    ref = _get_guide_repo_ref()
    return f"https://github.com/{repo_slug}/blob/{ref}/{rel_path}"


def normalize_guide_markdown_url(value: str) -> str:
    """Normalize cached guide_markdown.

    - Drops older cached full markdown content.
    - Keeps full blob URLs.
    - Upgrades repo-relative paths to full blob URLs when possible.
    """
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text:
        return ""

    # Legacy: stored markdown content.
    if "\n" in text or text.lstrip().startswith("# "):
        return ""

    if re.match(r"^https?://github\.com/[^/]+/[^/]+/blob/[^/]+/.+", text):
        return text

    # Treat it as a repo-relative path.
    repo_slug = _get_guide_repo_slug()
    if not repo_slug:
        return text
    ref = _get_guide_repo_ref()
    rel_path = text.lstrip("/")
    return f"https://github.com/{repo_slug}/blob/{ref}/{rel_path}"


def _hydrate_entry_cached_fields_from_file(entry: ToolGuideEntry, dry_run: bool = False) -> bool:
    """Populate/refresh entry.tldr and entry.guide_markdown (blob URL) from the guide .md file.

    Returns True if entry fields were changed in memory.
    """
    guide_path = get_guide_file_path(
        name=entry.name,
        timestamp=entry.timestamp,
        month=entry.month,
        in_readme_md=False,
    )
    if not guide_path.exists():
        return False

    changed = False

    link = build_guide_markdown_blob_url(entry)
    if link and link != (entry.guide_markdown or ""):
        entry.guide_markdown = link
        changed = True

    # TL;DR extraction is relatively expensive (file read + parsing). Only
    # hydrate from disk when it's missing in the cached entry.
    if not (entry.tldr or "").strip():
        extracted_tldr = extract_tldr_from_markdown(str(guide_path))
        extracted_tldr = (extracted_tldr or "").strip()
        if extracted_tldr and extracted_tldr != (entry.tldr or ""):
            entry.tldr = extracted_tldr
            changed = True

    if changed and dry_run:
        logging.info("Dry-run: would update cached tldr/guide_markdown for %s", entry.name)

    return changed


def repair_guide_markdown_file(
    path: Path,
    tool_name: str,
    categories: List[str],
    platforms: List[str],
    dry_run: bool = False,
) -> bool:
    """Repair guide metadata in a single read/write pass.

    This consolidates:
    - enforce_guide_title_and_tldr_style
    - enforce_guide_categories_line
    - enforce_guide_platform_line
    """
    normalized_categories = normalize_categories(categories, max_count=3)
    normalized_platforms = normalize_platforms(platforms)
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return False
    if not content:
        return False

    original = content
    lines = content.splitlines()

    # Title
    if lines and lines[0].startswith("# "):
        if lines[0] != f"# {tool_name}":
            lines[0] = f"# {tool_name}"

    # TL;DR prefix stripping (first non-empty line under TL;DR)
    new_lines: List[str] = []
    in_tldr = False
    tldr_fixed = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower() == "## tl;dr":
            in_tldr = True
            new_lines.append(line)
            continue
        if in_tldr:
            if stripped.startswith("## ") and stripped.lower() != "## tl;dr":
                in_tldr = False
                new_lines.append(line)
                continue
            if not tldr_fixed and stripped:
                for sep in ("：", ":"):
                    prefix = f"{tool_name}{sep}"
                    if stripped.startswith(prefix):
                        leading_ws_match = re.match(r"^(\s*)", line)
                        leading_ws = leading_ws_match.group(1) if leading_ws_match else ""
                        rest = stripped[len(prefix) :].lstrip()
                        line = f"{leading_ws}{rest}" if rest else leading_ws
                        break
                tldr_fixed = True
                new_lines.append(line)
                continue

        new_lines.append(line)

    lines = new_lines

    # Categories: update if present, else insert (only if we have categories)
    categories_line_index: Optional[int] = None
    for index, line in enumerate(lines[:40]):
        if line.startswith("- Categories:"):
            categories_line_index = index
            break

    if categories_line_index is not None:
        new_line = "- Categories: " + ", ".join(normalized_categories) if normalized_categories else "- Categories:"
        if lines[categories_line_index] != new_line:
            lines[categories_line_index] = new_line
    elif normalized_categories:
        insert_at: Optional[int] = None
        for index, line in enumerate(lines[:40]):
            if line.startswith("- Tags:"):
                insert_at = index + 1
                break
        if insert_at is None:
            for index, line in enumerate(lines[:40]):
                if line.startswith("- Added:"):
                    insert_at = index + 1
                    break
        if insert_at is None:
            for index, line in enumerate(lines[:40]):
                if line.startswith("- URL:"):
                    insert_at = index + 1
                    break
        if insert_at is not None:
            lines.insert(insert_at, "- Categories: " + ", ".join(normalized_categories))

    # Platform: update existing line only (keep behavior consistent)
    for index, line in enumerate(lines[:40]):
        if line.startswith("- Platform:"):
            new_line = "- Platform: " + ", ".join(normalized_platforms) if normalized_platforms else "- Platform:"
            if line != new_line:
                lines[index] = new_line
            break

    updated = "\n".join(lines) + ("\n" if original.endswith("\n") else "")
    if updated == original:
        return False
    if dry_run:
        logging.info("Dry-run: would repair guide metadata in %s", path)
        return True
    path.write_text(updated, encoding="utf-8")
    return True

def get_default_collection_readme_path() -> Path:
    candidates = [
        SCRIPT_DIR / TOOL_REPO_NAME / "README.md",
        SCRIPT_DIR.parent / TOOL_REPO_NAME / "README.md",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_COLLECTION_README_PATH = get_default_collection_readme_path()


def ensure_directory(path: Path, dry_run: bool = False) -> None:
    if dry_run:
        logging.info("Dry-run: would ensure directory %s", path)
        return
    path.mkdir(parents=True, exist_ok=True)


def write_text_file(path: Path, content: str, dry_run: bool = False) -> None:
    if dry_run:
        logging.info("Dry-run: would write %s (%d bytes)", path, len(content.encode("utf-8")))
        return
    ensure_directory(path.parent, dry_run=False)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)


def format_month(month: str) -> str:
    try:
        return datetime.strptime(month, "%Y%m").strftime("%Y-%m")
    except ValueError:
        return month


def slugify(text: str) -> str:
    invalid_fs_chars: str = '/\\:*?"<>|'
    return re.sub(r"[" + re.escape(invalid_fs_chars) + r"\s]+", "-", text.lower()).strip("-")


def canonicalize_tool_name(raw_name: str) -> str:
    """Normalize a tool display name to a short tool name.

    Examples:
    - "MrRSS - Modern Cross-Platform RSS Reader" -> "MrRSS"
    - "mgunyho/tere: Terminal file explorer" -> "tere"
    - "owner/repo" -> "repo"
    """
    name = (raw_name or "").strip()
    if not name:
        return name

    # Common patterns: "Name - Description" or "Name — Description"
    for sep in (" - ", " — ", " – "):
        if sep in name:
            name = name.split(sep, 1)[0].strip()
            break

    # Repo style: "owner/repo: something"
    if ":" in name:
        name = name.split(":", 1)[0].strip()

    # Repo style: "owner/repo" -> "repo"
    if "/" in name:
        name = name.rsplit("/", 1)[-1].strip()

    # Remove trailing parenthetical qualifiers: "Foo (Beta)" -> "Foo"
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()
    return name


def normalize_http_url(url: str) -> str:
    url = url.strip()
    if not url:
        raise ValueError("URL is empty")
    if re.match(r"^https?://", url, flags=re.IGNORECASE):
        return url
    return f"https://{url}"


def read_tool_collection_lines(collection_readme_path: Path) -> List[str]:
    if not collection_readme_path.exists():
        logging.warning(
            "'%s' not found; skipping new tool ingestion.",
            collection_readme_path,
        )
        return []
    with collection_readme_path.open("r", encoding="utf-8") as handle:
        return handle.readlines()


def extract_tool_links(lines: Iterable[str]) -> List[Tuple[str, str]]:
    """Extract (name, url) pairs from kohstool README.

    Default supported format: - [Name](URL)
    """
    results: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()

    started = False
    for line in lines:
        stripped = line.strip()
        # Common pattern in exported/curated READMEs: tool links appear as a list at
        # the top, followed by an "About" section. Once we hit a heading after
        # starting to collect links, stop to avoid pulling unrelated links.
        if started and re.match(r"^#{1,6}\s+", stripped):
            break

        match = re.search(r"-\s*\[(.*?)\]\((.*?)\)", line)
        if not match:
            continue
        name, url = match.groups()
        name = canonicalize_tool_name(name)
        url = url.strip()
        if not name or not url:
            continue
        started = True
        pair = (name, url)
        if pair in seen:
            continue
        seen.add(pair)
        results.append(pair)

    return results


def load_entries() -> List[ToolGuideEntry]:
    if not DATA_PATH.exists():
        logging.info("No data.json found at %s, starting with empty dataset.", DATA_PATH)
        return []

    with DATA_PATH.open("r", encoding="utf-8") as handle:
        raw_entries = json.load(handle)

    entries: List[ToolGuideEntry] = []
    for payload in raw_entries:
        entry = ToolGuideEntry.from_dict(payload)
        entry.name = canonicalize_tool_name(entry.name)
        entry.tags = normalize_tags(entry.tags, max_count=5)
        entry.categories = normalize_categories(entry.categories, max_count=3)
        entry.platform = normalize_platforms(entry.platform)
        entries.append(entry)
    return entries


def save_entries(entries: Iterable[ToolGuideEntry], dry_run: bool = False) -> None:
    normalized_entries: List[ToolGuideEntry] = []
    for entry in entries:
        entry.name = canonicalize_tool_name(entry.name)
        entry.tags = normalize_tags(entry.tags, max_count=5)
        entry.categories = normalize_categories(entry.categories, max_count=3)
        entry.platform = normalize_platforms(entry.platform)
        entry.tldr = (entry.tldr or "").strip()
        entry.guide_markdown = normalize_guide_markdown_url(entry.guide_markdown or "")
        normalized_entries.append(entry)

    payload = [entry.to_dict() for entry in normalized_entries]
    if dry_run:
        logging.info("Dry-run: would write %s with %d entries.", DATA_PATH, len(payload))
        return
    ensure_directory(DATA_PATH.parent, dry_run=False)
    with DATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def get_guide_file_path(
    name: str,
    timestamp: int,
    month: Optional[str] = None,
    in_readme_md: bool = False,
) -> Path:
    date_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
    guide_filename = f"{date_str}-{slugify(name)}.md"
    if month is None:
        month = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y%m")

    if in_readme_md:
        root = Path(month)
        guide_filename = f"{date_str}-{quote(slugify(name))}.md"
    else:
        root = SUMMARY_ROOT / month

    return root / guide_filename


def find_existing_guide_by_url(month: str, url: str, timestamp: int) -> Optional[Path]:
    """Locate an existing guide markdown file by matching its URL line.

    Used for migration when tool names change (and thus file names change).
    """
    month_dir = SUMMARY_ROOT / month
    if not month_dir.exists():
        return None

    date_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
    candidates = list(month_dir.glob(f"{date_str}-*.md"))

    normalized = normalize_http_url(url)
    for candidate in candidates:
        if candidate.name == "monthly-index.md":
            continue
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                # Only need the header area.
                head = handle.read(4096)
        except Exception:  # noqa: BLE001
            continue
        if f"- URL: {normalized}" in head:
            return candidate

    return None


def rewrite_guide_name_in_markdown(path: Path, old_name: str, new_name: str, dry_run: bool = False) -> bool:
    """Rewrite guide markdown so the H1 title matches new_name."""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return False

    if not content:
        return False

    original = content
    lines = content.splitlines()
    if lines and lines[0].startswith("# "):
        lines[0] = f"# {new_name}"

    # Strip legacy TL;DR prefixes like "OldName：" or "NewName:".
    new_lines: List[str] = []
    in_tldr = False
    tldr_stripped = False
    prefixes = []
    for sep in ("：", ":"):
        if old_name:
            prefixes.append(f"{old_name}{sep}")
        if new_name:
            prefixes.append(f"{new_name}{sep}")

    for line in lines:
        stripped = line.strip()
        if stripped.lower() == "## tl;dr":
            in_tldr = True
            new_lines.append(line)
            continue
        if in_tldr:
            if stripped.startswith("## ") and stripped.lower() != "## tl;dr":
                in_tldr = False
                new_lines.append(line)
                continue
            if not tldr_stripped and stripped:
                for prefix in prefixes:
                    if stripped.startswith(prefix):
                        leading_ws_match = re.match(r"^(\s*)", line)
                        leading_ws = leading_ws_match.group(1) if leading_ws_match else ""
                        rest = stripped[len(prefix) :].lstrip()
                        line = f"{leading_ws}{rest}" if rest else leading_ws
                        break
                new_lines.append(line)
                tldr_stripped = True
                continue

        new_lines.append(line)

    content = "\n".join(new_lines) + ("\n" if original.endswith("\n") else "")

    if content == original:
        return False
    if dry_run:
        logging.info("Dry-run: would rewrite guide title/TL;DR in %s", path)
        return True
    path.write_text(content, encoding="utf-8")
    return True


def enforce_guide_title_and_tldr_style(path: Path, tool_name: str, dry_run: bool = False) -> bool:
    """Ensure the guide's H1 uses the canonical tool name.

    For TL;DR, strip legacy "{tool_name}：" / "{tool_name}:" prefixes if present.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return False

    if not content:
        return False

    original = content
    lines = content.splitlines()
    if lines and lines[0].startswith("# "):
        if lines[0] != f"# {tool_name}":
            lines[0] = f"# {tool_name}"

    new_lines: List[str] = []
    in_tldr = False
    tldr_fixed = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower() == "## tl;dr":
            in_tldr = True
            new_lines.append(line)
            continue
        if in_tldr:
            if stripped.startswith("## ") and stripped.lower() != "## tl;dr":
                in_tldr = False
                new_lines.append(line)
                continue
            if not tldr_fixed and stripped:
                # Strip legacy name prefix if present.
                for sep in ("：", ":"):
                    prefix = f"{tool_name}{sep}"
                    if stripped.startswith(prefix):
                        # Preserve original indentation.
                        leading_ws_match = re.match(r"^(\s*)", line)
                        leading_ws = leading_ws_match.group(1) if leading_ws_match else ""
                        rest = stripped[len(prefix) :].lstrip()
                        line = f"{leading_ws}{rest}" if rest else leading_ws
                        break
                new_lines.append(line)
                tldr_fixed = True
                continue

        new_lines.append(line)

    content = "\n".join(new_lines) + ("\n" if original.endswith("\n") else "")
    if content == original:
        return False
    if dry_run:
        logging.info("Dry-run: would enforce guide title/TL;DR prefix in %s", path)
        return True
    path.write_text(content, encoding="utf-8")
    return True


def enforce_guide_platform_line(path: Path, platforms: List[str], dry_run: bool = False) -> bool:
    platforms = normalize_platforms(platforms)
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return False
    if not content:
        return False

    original = content
    lines = content.splitlines()
    updated = False
    for index, line in enumerate(lines[:40]):
        if line.startswith("- Platform:"):
            new_line = "- Platform: " + ", ".join(platforms) if platforms else "- Platform:"
            if line != new_line:
                lines[index] = new_line
                updated = True
            break

    if not updated:
        return False

    content = "\n".join(lines) + ("\n" if original.endswith("\n") else "")
    if content == original:
        return False
    if dry_run:
        logging.info("Dry-run: would enforce platform line in %s", path)
        return True
    path.write_text(content, encoding="utf-8")
    return True


def enforce_guide_categories_line(path: Path, categories: List[str], dry_run: bool = False) -> bool:
    categories = normalize_categories(categories, max_count=3)
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return False
    if not content:
        return False

    original = content
    lines = content.splitlines()

    # Update existing line if present.
    for index, line in enumerate(lines[:40]):
        if line.startswith("- Categories:"):
            new_line = "- Categories: " + ", ".join(categories) if categories else "- Categories:"
            if line != new_line:
                lines[index] = new_line
                content = "\n".join(lines) + ("\n" if original.endswith("\n") else "")
                if dry_run:
                    logging.info("Dry-run: would enforce categories line in %s", path)
                    return True
                path.write_text(content, encoding="utf-8")
                return True
            return False

    # If there's nothing to write, don't insert a new header line.
    if not categories:
        return False

    insert_at = None
    for index, line in enumerate(lines[:40]):
        if line.startswith("- Tags:"):
            insert_at = index + 1
            break
    if insert_at is None:
        for index, line in enumerate(lines[:40]):
            if line.startswith("- Added:"):
                insert_at = index + 1
                break
    if insert_at is None:
        for index, line in enumerate(lines[:40]):
            if line.startswith("- URL:"):
                insert_at = index + 1
                break
    if insert_at is None:
        return False

    lines.insert(insert_at, "- Categories: " + ", ".join(categories))
    content = "\n".join(lines) + ("\n" if original.endswith("\n") else "")
    if dry_run:
        logging.info("Dry-run: would insert categories line in %s", path)
        return True
    path.write_text(content, encoding="utf-8")
    return True


def migrate_entry_name_and_file(entry: ToolGuideEntry, dry_run: bool = False) -> bool:
    """Return True if any migration was performed for the entry."""
    canonical = canonicalize_tool_name(entry.name)
    if not canonical or canonical == entry.name:
        return False

    old_name = entry.name
    entry.name = canonical

    new_path = get_guide_file_path(
        name=entry.name,
        timestamp=entry.timestamp,
        month=entry.month,
        in_readme_md=False,
    )

    # If the file already exists under the new name, no need to rename.
    if new_path.exists():
        rewrite_guide_name_in_markdown(new_path, old_name=old_name, new_name=entry.name, dry_run=dry_run)
        logging.info("Migrated name '%s' -> '%s' (file already in place)", old_name, entry.name)
        return True

    existing = find_existing_guide_by_url(entry.month, entry.url, entry.timestamp)
    if existing is None:
        logging.warning(
            "Migrated name '%s' -> '%s' but could not locate existing guide file for url=%s",
            old_name,
            entry.name,
            entry.url,
        )
        return True

    if dry_run:
        logging.info("Dry-run: would rename %s -> %s", existing, new_path)
        return True

    ensure_directory(new_path.parent, dry_run=False)
    existing.rename(new_path)
    logging.info("Renamed guide file %s -> %s", existing.name, new_path.name)
    rewrite_guide_name_in_markdown(new_path, old_name=old_name, new_name=entry.name, dry_run=False)
    return True


def extract_tldr_from_markdown(file_path: str) -> str:
    def extract_tldr_with_regex(content: str) -> str:
        match = re.search(r"##\s*TL;DR\s+(.*?)\n##\s", content, re.DOTALL)
        if not match:
            match = re.search(r"##\s*TL;DR\s+(.*)", content, re.DOTALL)
        if not match:
            return ""
        extracted = match.group(1).strip()
        return re.sub(r"\s+", " ", extracted)

    @lru_cache(maxsize=1)
    def _get_mistune_markdown_parser():
        if mistune is None:
            return None
        try:
            return mistune.create_markdown(renderer=None)
        except Exception:  # noqa: BLE001
            return None

    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            content = handle.read()
    except Exception as error:  # noqa: BLE001
        logging.warning("Could not read TL;DR from %s: %s", file_path, error)
        return ""

    if not content:
        return ""

    # Fast path: most guides follow a predictable structure and regex extraction
    # is much cheaper than building a full AST.
    fast = extract_tldr_with_regex(content)
    if fast:
        return fast

    if mistune is None:
        return ""

    try:
        markdown = _get_mistune_markdown_parser()
        if markdown is None:
            return ""
        ast = markdown(content)
    except Exception as error:  # noqa: BLE001
        logging.warning(
            "Mistune parsing failed for %s: %s. Falling back to regex parser.",
            file_path,
            error,
        )
        return extract_tldr_with_regex(content)

    tldr_content: List[str] = []
    found_tldr = False

    for token in ast:
        if token["type"] == "heading" and token.get("attrs", {}).get("level") == 2:
            if "TL;DR" in str(token.get("children", [])):
                found_tldr = True
                continue
            if found_tldr:
                break
        elif found_tldr and token["type"] == "paragraph":
            def extract_text(children):
                parts: List[str] = []
                for child in children:
                    if child["type"] == "text":
                        parts.append(child["raw"])
                    elif "children" in child:
                        parts.extend(extract_text(child["children"]))
                return parts

            text_parts = extract_text(token.get("children", []))
            tldr_content.append("".join(text_parts))

    if not tldr_content:
        return extract_tldr_with_regex(content)

    return "\n".join(tldr_content).strip()


def render_entry_lines(entry: ToolGuideEntry, link: str, tldr: str) -> List[str]:
    date_str = datetime.fromtimestamp(entry.timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
    lines = [f"({date_str}) [{entry.name}]({link})"]
    if tldr:
        lines.append(f"- {tldr}")
    if entry.tags:
        lines.append(f"- Tags: {', '.join(entry.tags)}")
    if entry.categories:
        lines.append(f"- Categories: {', '.join(entry.categories)}")
    if entry.platform:
        lines.append(f"- Platform: {', '.join(entry.platform)}")
    return lines


def build_monthly_index_markdown(
    month: str,
    entries: List[ToolGuideEntry],
    tldr_lookup: Dict[Tuple[str, str, int], str],
) -> str:
    lines: List[str] = [f"# {format_month(month)} Tool Guide Index", ""]
    for entry in entries:
        link = get_guide_file_path(
            name=entry.name,
            timestamp=entry.timestamp,
            month=entry.month,
            in_readme_md=True,
        ).name
        lines.extend(render_entry_lines(entry, link, tldr_lookup.get(entry.identity(), "")))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _strip_name_prefix_from_tldr(tldr: str, name: str) -> str:
    text = (tldr or "").strip()
    if not text:
        return ""
    for sep in ("：", ":"):
        prefix = f"{name}{sep}"
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


def build_root_readme(
    entries: List[ToolGuideEntry],
    grouped: Dict[str, List[ToolGuideEntry]],
    tldr_lookup: Dict[Tuple[str, str, int], str],
) -> str:
    prefix = (
        "# Tool Guide\n"
        "自动读取 [kohstool](https://github.com/leehyon/kohstool) 仓库中的工具链接，通过 Jina Reader 获取网页文本内容，再借助 AI 生成总结。\n"
    )

    lines: List[str] = [prefix.rstrip(), ""]

    sorted_entries = sorted(entries, key=lambda e: e.timestamp, reverse=True)
    if sorted_entries:
        for entry in sorted_entries:
            link = get_guide_file_path(
                name=entry.name,
                timestamp=entry.timestamp,
                month=entry.month,
                in_readme_md=True,
            ).as_posix()
            tldr = _strip_name_prefix_from_tldr(
                tldr_lookup.get(entry.identity(), ""),
                entry.name,
            )
            if tldr:
                lines.append(f"- [{entry.name}]({link}) - {tldr}")
            else:
                lines.append(f"- [{entry.name}]({link})")
    else:
        lines.append("- _No guides available yet._")

    lines.append("")
    lines.append("## Monthly Archive")
    lines.append("")

    sorted_months = sorted(grouped.keys(), reverse=True)
    if sorted_months:
        for month in sorted_months:
            link = Path(month, "monthly-index.md").as_posix()
            lines.append(f"- [{format_month(month)}]({link}) ({len(grouped[month])} entries)")
    else:
        lines.append("- _Archive will appear after the first guide._")

    return "\n".join(lines).strip() + "\n"


def build_all_guide_md(
    entries: List[ToolGuideEntry],
    tldr_lookup: Dict[Tuple[str, str, int], str],
) -> str:
    lines: List[str] = ["# All Guides", ""]
    for entry in sorted(entries, key=lambda e: e.timestamp, reverse=True):
        date_str = datetime.fromtimestamp(entry.timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
        guide_path = get_guide_file_path(
            name=entry.name,
            timestamp=entry.timestamp,
            month=entry.month,
            in_readme_md=True,
        ).as_posix()
        lines.append(f"- ({date_str}) [{entry.name}]({guide_path})")
        tldr = tldr_lookup.get(entry.identity(), "").strip()
        if entry.tags:
            lines.append(f"  - Tags: {', '.join(entry.tags)}")
        if entry.platform:
            lines.append(f"  - Platform: {', '.join(entry.platform)}")
        if tldr:
            lines.append(f"  - TL;DR: {tldr}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def group_entries_by_month(entries: Iterable[ToolGuideEntry]) -> Dict[str, List[ToolGuideEntry]]:
    grouped: Dict[str, List[ToolGuideEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.month, []).append(entry)
    for month in grouped:
        grouped[month].sort(key=lambda e: e.timestamp, reverse=True)
    return grouped


def collect_tldrs(
    entries: Iterable[ToolGuideEntry],
    overrides: Optional[Dict[Tuple[str, str, int], str]] = None,
) -> Dict[Tuple[str, str, int], str]:
    overrides = overrides or {}
    lookup: Dict[Tuple[str, str, int], str] = {}
    for entry in entries:
        key = entry.identity()
        if key in overrides:
            lookup[key] = overrides[key]
            continue
        if (entry.tldr or "").strip():
            lookup[key] = entry.tldr.strip()
            continue
        guide_path = get_guide_file_path(
            name=entry.name,
            timestamp=entry.timestamp,
            month=entry.month,
            in_readme_md=False,
        )
        lookup[key] = extract_tldr_from_markdown(str(guide_path))
    return lookup


def write_monthly_indexes(
    grouped: Dict[str, List[ToolGuideEntry]],
    tldr_lookup: Dict[Tuple[str, str, int], str],
    dry_run: bool = False,
) -> None:
    for month in sorted(grouped.keys(), reverse=True):
        month_dir = SUMMARY_ROOT / month
        if not month_dir.exists():
            ensure_directory(month_dir, dry_run=dry_run)
        content = build_monthly_index_markdown(month, grouped[month], tldr_lookup)
        write_text_file(month_dir / "monthly-index.md", content, dry_run=dry_run)


@log_execution_time
def submit_to_wayback_machine(url: str) -> None:
    if WaybackMachineSaveAPI is None:
        logging.info("WaybackMachineSaveAPI not available; skipping submission for %s.", url)
        return

    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
    )
    try:
        save_api = WaybackMachineSaveAPI(url, user_agent)
        wayback_url = save_api.save()
        logging.info("Wayback Saved: %s", wayback_url)
    except Exception as error:  # noqa: BLE001
        logging.warning("submit to wayback machine failed, skipping, url=%s", url)
        logging.exception(error)


def preflight_check_url(url: str) -> Tuple[Optional[int], Optional[str]]:
    if requests is None:
        return None, "requests package not available"

    session = _get_requests_session()
    if session is None:
        return None, "requests session not available"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
        )
    }
    timeout = (HTTP_CONNECT_TIMEOUT_SECONDS, HTTP_READ_TIMEOUT_SECONDS)
    try:
        response: requests.Response = session.head(url, allow_redirects=True, headers=headers, timeout=timeout)
        status = response.status_code
        if status in (403, 405):
            response = session.get(url, allow_redirects=True, headers=headers, timeout=timeout, stream=True)
            status = response.status_code
            response.close()
        return status, None
    except requests.RequestException as error:
        return None, str(error)


@log_execution_time
def get_text_content(url: str) -> str:
    if requests is None:
        raise RuntimeError("requests package not available; cannot fetch content.")

    session = _get_requests_session()
    if session is None:
        raise RuntimeError("requests session not available; cannot fetch content.")

    url = normalize_http_url(url)
    status_code, preflight_error = preflight_check_url(url)
    if preflight_error:
        logging.warning("Preflight check failed for %s: %s", url, preflight_error)
    elif status_code is not None and status_code >= 400 and status_code not in (401, 403, 429):
        logging.warning("Origin URL returned HTTP %d for %s; content fetch may fail.", status_code, url)

    jina_url = f"https://r.jina.ai/{url}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
        )
    }
    timeout = (HTTP_CONNECT_TIMEOUT_SECONDS, HTTP_READ_TIMEOUT_SECONDS)

    for attempt in range(MAX_RETRIES):
        try:
            response: requests.Response = session.get(jina_url, headers=headers, timeout=timeout)
            if response.status_code >= 400:
                status = response.status_code
                error_msg = f"Jina fetch failed (HTTP {status}) - attempt {attempt + 1}/{MAX_RETRIES}"
                logging.warning(error_msg)
                should_retry = status in (429, 500, 502, 503, 504)
                if should_retry and attempt < MAX_RETRIES - 1:
                    wait_time = 2**attempt
                    logging.info("Retrying in %d seconds...", wait_time)
                    time.sleep(wait_time)
                    continue
                raise Exception(f"All {MAX_RETRIES} retry attempts failed. Last error: {error_msg}")

            content = response.text.strip()
            if len(content) < MIN_CONTENT_LENGTH:
                error_msg = f"Content too short ({len(content)} chars, minimum {MIN_CONTENT_LENGTH}) - attempt {attempt + 1}/{MAX_RETRIES}"
                logging.warning(error_msg)
                if attempt < MAX_RETRIES - 1:
                    wait_time = 2**attempt
                    logging.info("Retrying in %d seconds...", wait_time)
                    time.sleep(wait_time)
                    continue
                raise Exception(f"All {MAX_RETRIES} retry attempts failed. Last error: {error_msg}")

            if len(content) > MAX_CONTENT_LENGTH:
                logging.warning(
                    "Content length (%d) exceeds maximum (%d), truncating...",
                    len(content),
                    MAX_CONTENT_LENGTH,
                )
                content = content[:MAX_CONTENT_LENGTH]

            logging.info("Successfully fetched content with %d characters", len(content))
            return content
        except requests.RequestException as error:
            logging.warning("Request failed (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, error)
            if attempt < MAX_RETRIES - 1:
                wait_time = 2**attempt
                logging.info("Retrying in %d seconds...", wait_time)
                time.sleep(wait_time)
            else:
                raise Exception(f"All {MAX_RETRIES} retry attempts failed. Last error: {error}") from error


@log_execution_time
def call_openai_api(prompt: str, content: str) -> str:
    if requests is None:
        raise RuntimeError("requests package not available; cannot call OpenAI API.")

    session = _get_requests_session()
    if session is None:
        raise RuntimeError("requests session not available; cannot call OpenAI API.")

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in your environment before running, "
            "or run with --dry-run/--backfill."
        )

    model: str = os.environ.get("OPENAI_API_MODEL", "gpt-4o-mini")
    headers: dict = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ],
    }
    api_endpoint: str = os.environ.get("OPENAI_API_ENDPOINT", "https://api.openai.com/v1/chat/completions")

    logging.info("Calling OpenAI API with model: %s", model)
    logging.info("API endpoint: %s", api_endpoint)

    response: requests.Response = session.post(api_endpoint, headers=headers, data=json.dumps(data))
    logging.info("Response status code: %d", response.status_code)
    response_json = response.json()

    if response.status_code != 200:
        logging.error("OpenAI API request failed with status %s", response.status_code)
        logging.error("Error response: %s", response_json)
        raise Exception(f"OpenAI API request failed with status {response.status_code}")

    if "choices" not in response_json:
        raise Exception("Response does not contain 'choices' field")

    return response_json["choices"][0]["message"]["content"]


def _extract_first_json_object(text: str) -> str:
    """Extract first top-level JSON object from a model response."""
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("Unterminated JSON object in response")


def _detect_platforms(text: str) -> List[str]:
    lowered = text.lower()
    platforms: List[str] = []
    if any(token in lowered for token in ("windows", "win32", "win 10", "win 11")):
        platforms.append("Windows")
    if any(token in lowered for token in ("macos", "os x", "mac")):
        platforms.append("Mac")
    if "linux" in lowered:
        platforms.append("Linux")
    if any(token in lowered for token in ("ios", "iphone", "ipad")):
        platforms.append("iOS")
    if "android" in lowered:
        platforms.append("Android")
    if any(token in lowered for token in ("browser extension", "chrome extension", "firefox addon", "firefox add-on")):
        platforms.append("Browser Extension")
    if any(token in lowered for token in ("web", "saas", "in your browser", "browser-based")):
        platforms.append("Web")
    if not platforms:
        platforms.append("Web")
    # Stable ordering
    order = ["Web", "Browser Extension", "Windows", "Mac", "Linux", "iOS", "Android"]
    return [p for p in order if p in platforms]


def _heuristic_tags_and_category(name: str, url: str, page_text: str) -> Tuple[List[str], str]:
    haystack = f"{name}\n{url}\n{page_text}".lower()
    if "rss" in haystack:
        return ["RSS Reader", "Cross-platform"], "rss"
    if any(token in haystack for token in ("text expander", "snippet", "autocomplete", "autohotkey", "expanso")):
        return ["Productivity", "Text Expansion"], "text-expander"
    if any(token in haystack for token in ("terminal file explorer", "tui", "terminal", "file manager", "cli")):
        return ["Terminal", "File Manager"], "terminal-file"
    if any(token in haystack for token in ("note", "note-taking", "knowledge base", "zettelkasten")):
        return ["Note-taking", "Knowledge Base"], "notes"
    return ["Software", "Tool"], "generic"


def heuristic_tool_guide(name: str, url: str, page_text: str) -> ToolGuideContent:
    tags, category = _heuristic_tags_and_category(name, url, page_text)
    platforms = normalize_platforms(_detect_platforms(f"{name}\n{url}\n{page_text}"))

    if category == "rss":
        scenarios = [
            "跨平台订阅与阅读 RSS 源",
            "集中管理资讯来源，减少信息噪音",
            "离线阅读与稍后读",
        ]
        pain_points = [
            "信息源分散，难以统一订阅与跟踪",
            "社交/算法推荐造成信息噪音与注意力消耗",
            "想离线或稍后读但缺少合适工具",
        ]
        design_principles = [
            "以订阅源为中心的聚合阅读",
            "离线优先与阅读队列",
        ]
        categories = ["Reading & Information"]
        similar_tools = ["NetNewsWire", "Reeder", "Fluent Reader", "Inoreader", "Feedly"]
    elif category == "text-expander":
        scenarios = [
            "用短缩写快速展开常用文本",
            "统一管理模板、签名与代码片段",
            "提升重复输入场景的效率",
        ]
        pain_points = [
            "重复输入耗时且容易出错",
            "常用片段散落在各处，难维护与复用",
            "跨应用复用内容不顺畅",
        ]
        design_principles = [
            "缩写触发 → 模板展开的工作流",
            "可配置的变量/占位符与表单化输入",
        ]
        categories = ["Text Input & Writing", "System & Automation"]
        similar_tools = ["TextExpander", "aText", "AutoHotkey", "PhraseExpress", "Alfred Snippets"]
    elif category == "terminal-file":
        scenarios = [
            "在终端中快速浏览与定位文件",
            "键盘驱动的文件管理与跳转",
            "配合 git/grep/fzf 提升开发效率",
        ]
        pain_points = [
            "文件查找与跳转成本高，鼠标操作打断思路",
            "在远程/无 GUI 环境下管理文件不便",
            "需要更快的键盘驱动工作流",
        ]
        design_principles = [
            "键盘优先的交互与快捷键体系",
            "与终端生态（grep/fzf/git）可组合",
        ]
        categories = ["Developer Tools", "File Management"]
        similar_tools = ["ranger", "nnn", "lf", "fzf", "broot"]
    elif category == "notes":
        scenarios = [
            "个人知识管理与笔记沉淀",
            "将资料按主题链接与组织",
            "写作与研究的长期积累",
        ]
        pain_points = [
            "笔记零散难检索，知识无法沉淀与复用",
            "主题之间缺少连接，难形成知识网络",
            "写作/研究需要长期、可演进的结构",
        ]
        design_principles = [
            "双向链接与知识图谱",
            "以块/页面为单位的可重组内容",
        ]
        categories = ["Knowledge Management"]
        similar_tools = ["Obsidian", "Notion", "Logseq", "Roam Research", "Joplin"]
    else:
        scenarios = ["记录与管理工作/学习中的常见需求", "提升信息获取与整理效率", "作为某类任务的辅助工具"]
        pain_points = [
            "缺少趁手工具导致流程低效",
            "信息/任务分散，难以统一管理",
        ]
        design_principles = [
            "围绕核心任务的最小闭环",
            "可组合/可扩展的工作流",
        ]
        categories = ["System & Automation"]
        similar_tools = ["Notion", "Obsidian", "Raycast", "Alfred", "PowerToys"]

    # Keep TL;DR <= 100 chars (roughly). We keep it short.
    # Do NOT include tool name in TL;DR.
    tldr = f"一款用于 {tags[0]} 的工具，适合提升日常效率。"
    if len(tldr) > 100:
        tldr = tldr[:100]

    # De-dup and size-limit per spec
    def _uniq(items: List[str], limit: int) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
            if len(out) >= limit:
                break
        return out

    return ToolGuideContent(
        tldr=tldr,
        scenarios=_uniq(scenarios, 7),
        pain_points=_uniq(pain_points, 7),
        design_principles=_uniq(design_principles, 7),
        categories=normalize_categories(categories, max_count=3),
        similar_tools=_uniq(similar_tools, 7),
        tags=_uniq(tags, 5),
        platform=_uniq(platforms, 7),
    )


@log_execution_time
def generate_tool_guide(name: str, url: str, page_text: str) -> ToolGuideContent:
    prompt = """
你是一个中文软件工具指南编写助手。
你将得到：工具名称、工具官网 URL、以及网页正文文本（可能不完整）。

任务：生成一份结构化的工具 guide 信息，并从内容中推断分类标签与支持平台；同时补充该工具解决的用户痛点与其设计理念（产品思路）。

输出要求：
- 只输出一个 JSON 对象（不要 Markdown/不要解释/不要代码块）。
- JSON 字段必须严格为：tldr, scenarios, pain_points, design_principles, categories, similar_tools, tags, platform。
- tldr：简体中文，不超过 100 个字；中英字符间保留空格；不要包含工具名称；尽量以“一款/一个…”开头。
- scenarios：数组，3-7 条，每条为简体中文的“应用场景/用途”短句。
- pain_points：数组，2-6 条，每条为简体中文的“用户痛点/问题”短句，描述用户在没有该工具时的困难。
- design_principles：数组，1-4 条，每条为简体中文短语/短句，概括该工具的核心设计理念（例如“双向链接”“块编辑”“离线优先”“键盘优先”等）。
- categories：数组，从以下集合中选择（可多选，建议 1-3 个）：Knowledge Management, Reading & Information, Text Input & Writing, Developer Tools, File Management, System & Automation, Communication, Media & Creativity, Security & Privacy, Data & Analytics。
- similar_tools：数组，3-7 个同类软件名称（可中英文混排）。
- tags：数组，2-6 个标签，使用英文 Title Case 短语（例如 Note-taking, Free, Knowledge Base）。
- platform：数组，从以下集合中选择：Mac, Windows, Linux, iOS, Android, Web, Browser Extension, Cross-platform。
- 如果无法确定某字段，给空数组或空字符串，但仍要保留字段。
""".strip()

    user_content = (
        f"Tool Name: {name}\n"
        f"URL: {normalize_http_url(url)}\n\n"
        f"Page Text:\n{page_text}"
    )
    raw = call_openai_api(prompt, user_content)
    extracted = _extract_first_json_object(raw)
    payload = json.loads(extracted)

    tldr = (payload.get("tldr") or "").strip()
    scenarios = payload.get("scenarios") or []
    pain_points = payload.get("pain_points") or []
    design_principles = payload.get("design_principles") or []
    categories = payload.get("categories") or []
    similar_tools = payload.get("similar_tools") or []
    tags = payload.get("tags") or []
    platform = payload.get("platform") or []

    def _clean_list(items: list) -> List[str]:
        cleaned: List[str] = []
        for item in items:
            if not isinstance(item, str):
                continue
            value = item.strip()
            if not value:
                continue
            cleaned.append(value)
        return cleaned

    return ToolGuideContent(
        tldr=tldr,
        scenarios=_clean_list(scenarios),
        pain_points=_clean_list(pain_points),
        design_principles=_clean_list(design_principles),
        categories=normalize_categories(_clean_list(categories), max_count=3),
        similar_tools=_clean_list(similar_tools),
        tags=normalize_tags(_clean_list(tags), max_count=5),
        platform=normalize_platforms(_clean_list(platform)),
    )


def generate_tool_guide_with_options(
    name: str,
    url: str,
    page_text: str,
    heuristic: bool,
) -> ToolGuideContent:
    if heuristic:
        return heuristic_tool_guide(name, url, page_text)

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        logging.warning("OPENAI_API_KEY not set; falling back to heuristic guide generation.")
        return heuristic_tool_guide(name, url, page_text)

    return generate_tool_guide(name, url, page_text)


def build_guide_markdown(
    name: str,
    url: str,
    tldr: str,
    scenarios: List[str],
    pain_points: List[str],
    design_principles: List[str],
    categories: List[str],
    similar_tools: List[str],
    tags: List[str],
    platform: List[str],
) -> str:
    tag_line = f"- Tags: {', '.join(tags)}\n" if tags else ""
    normalized_categories = normalize_categories(categories, max_count=3)
    categories_line = f"- Categories: {', '.join(normalized_categories)}\n" if normalized_categories else ""
    platform_line = f"- Platform: {', '.join(platform)}\n" if platform else ""

    scenario_lines = "\n".join(f"- {item}" for item in scenarios) if scenarios else "- _暂无_"
    pain_lines = "\n".join(f"- {item}" for item in pain_points) if pain_points else "- _暂无_"
    principle_lines = "\n".join(f"- {item}" for item in design_principles) if design_principles else "- _暂无_"
    similar_lines = "\n".join(f"- {item}" for item in similar_tools) if similar_tools else "- _暂无_"

    return (
        f"# {name}\n"
        f"- URL: {normalize_http_url(url)}\n"
        f"- Added: {CURRENT_DATE_AND_TIME}\n"
        f"{tag_line}"
        f"{categories_line}"
        f"{platform_line}"
        "\n"
        "## TL;DR\n"
        f"{tldr}\n\n"
        "## 应用场景\n"
        f"{scenario_lines}\n\n"
        "## 用户痛点\n"
        f"{pain_lines}\n\n"
        "## 设计理念\n"
        f"{principle_lines}\n\n"
        "## 类似软件\n"
        f"{similar_lines}\n"
    )


def find_next_tool_to_process(
    tool_pairs: Iterable[Tuple[str, str]],
    processed_urls: Iterable[str],
) -> Optional[Tuple[str, str]]:
    processed = set(processed_urls)
    for name, url in tool_pairs:
        if url in processed:
            continue
        return name, url
    return None


def ingest_tool(
    name: str,
    url: str,
    archive: bool,
    heuristic: bool,
    fetch: bool,
) -> IngestionResult:
    name = canonicalize_tool_name(name)
    url = normalize_http_url(url)

    # Wayback submission is network-bound and independent of guide generation.
    # Overlap it with Jina/OpenAI calls to reduce overall wall-clock time.
    with ThreadPoolExecutor(max_workers=1) as executor:
        wayback_future = executor.submit(submit_to_wayback_machine, url) if archive else None

        page_text = ""
        if fetch:
            try:
                page_text = get_text_content(url)
            except Exception as error:  # noqa: BLE001
                logging.warning("Failed to fetch page content for %s; continuing with empty text.", url)
                logging.exception(error)

        guide = generate_tool_guide_with_options(name, url, page_text, heuristic=heuristic)

        if wayback_future is not None:
            try:
                wayback_future.result()
            except Exception as error:  # noqa: BLE001
                # submit_to_wayback_machine already logs; keep this guard to
                # avoid bubbling unexpected thread exceptions.
                logging.warning("Wayback submission failed (async), skipping, url=%s", url)
                logging.exception(error)

    timestamp = int(datetime.now(timezone.utc).timestamp())
    month = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y%m")

    entry = ToolGuideEntry(
        month=month,
        name=name,
        url=url,
        timestamp=timestamp,
        tags=normalize_tags(guide.tags, max_count=5),
        categories=normalize_categories(guide.categories, max_count=3),
        platform=normalize_platforms(guide.platform),
        tldr=(guide.tldr or "").strip(),
    )
    guide_path = get_guide_file_path(name=name, timestamp=timestamp, month=month)
    markdown = build_guide_markdown(
        name=name,
        url=url,
        tldr=guide.tldr,
        scenarios=guide.scenarios,
        pain_points=guide.pain_points,
        design_principles=guide.design_principles,
        categories=guide.categories,
        similar_tools=guide.similar_tools,
        tags=normalize_tags(guide.tags, max_count=5),
        platform=normalize_platforms(guide.platform),
    )

    entry.guide_markdown = build_guide_markdown_blob_url(entry)
    return IngestionResult(entry=entry, guide_markdown=markdown, guide_path=guide_path, tldr=guide.tldr)


def process_tools(options: RunOptions) -> None:
    backfill = options.backfill
    dry_run = options.dry_run
    archive = options.archive
    collection_readme_path = options.collection_readme_path
    max_new = options.max_new
    heuristic = options.heuristic
    fetch = options.fetch

    try:
        same_readme = collection_readme_path.resolve() == SUMMARY_README_PATH.resolve()
    except Exception:  # noqa: BLE001
        same_readme = collection_readme_path == SUMMARY_README_PATH

    if same_readme and not (dry_run or backfill):
        raise ValueError(
            "collection README path points to the same file as output README.md. "
            "Refusing to overwrite the source list. Use --dry-run/--backfill or point "
            "--collection-readme to a different file (e.g. ../kohstool/README.md)."
        )

    logging.info("Collection README: %s", collection_readme_path)

    entries = load_entries()

    migrated_any = False
    for entry in entries:
        if migrate_entry_name_and_file(entry, dry_run=dry_run):
            migrated_any = True

    # Even if names are already canonical, keep guide markdown titles consistent.
    repaired_any = False
    for entry in entries:
        guide_path = get_guide_file_path(
            name=entry.name,
            timestamp=entry.timestamp,
            month=entry.month,
            in_readme_md=False,
        )
        if guide_path.exists():
            if repair_guide_markdown_file(
                guide_path,
                tool_name=entry.name,
                categories=entry.categories,
                platforms=entry.platform,
                dry_run=dry_run,
            ):
                repaired_any = True

    # Hydrate cached fields (tldr/guide_markdown) from disk so data.json can be
    # used for downstream processing without re-reading guide files.
    hydrated_any = False
    for entry in entries:
        if _hydrate_entry_cached_fields_from_file(entry, dry_run=dry_run):
            hydrated_any = True

    processed_urls = [entry.url for entry in entries]

    overrides: Dict[Tuple[str, str, int], str] = {}

    tool_lines = read_tool_collection_lines(collection_readme_path)
    tool_pairs = extract_tool_links(tool_lines)

    ingestion_result: Optional[IngestionResult] = None
    if backfill:
        logging.info("Backfill mode enabled; rebuilding indexes from existing data only.")
    elif dry_run:
        logging.info(
            "Dry-run mode enabled; will detect next tool but skip network calls and writes."
        )

        next_tool = find_next_tool_to_process(tool_pairs, processed_urls)
        if next_tool:
            name, url = next_tool
            logging.info("Dry-run: next tool to process would be: %s (%s)", name, normalize_http_url(url))
        else:
            logging.info("Dry-run: no new tools to process.")
    elif requests is None:
        logging.warning(
            "requests dependency missing; cannot ingest new tools. Run with --backfill or install dependencies."
        )
    else:
        processed_count = 0
        while processed_count < max_new:
            next_tool = find_next_tool_to_process(tool_pairs, processed_urls)
            if not next_tool:
                break

            name, url = next_tool
            logging.info("Processing new tool: %s", name)
            ingestion_result = ingest_tool(
                name,
                url,
                archive=archive,
                heuristic=heuristic,
                fetch=fetch,
            )
            entries.append(ingestion_result.entry)
            processed_urls.append(ingestion_result.entry.url)
            processed_count += 1

            if dry_run:
                logging.info("Dry-run: skipping writes for %s", ingestion_result.guide_path)
            else:
                write_text_file(ingestion_result.guide_path, ingestion_result.guide_markdown, dry_run=False)

            # Ensure cached fields reflect what was written.
            ingestion_result.entry.guide_markdown = build_guide_markdown_blob_url(ingestion_result.entry)
            ingestion_result.entry.tldr = (ingestion_result.tldr or "").strip()

            overrides[ingestion_result.entry.identity()] = ingestion_result.tldr

        if processed_count == 0:
            logging.info("No new tools to process.")
        else:
            logging.info("Processed %d new tool(s).", processed_count)

    if hydrated_any and not dry_run:
        logging.info("Cached TL;DR/guide markdown hydrated into entries.")

    save_entries(entries, dry_run=dry_run)

    grouped = group_entries_by_month(entries)
    tldr_lookup = collect_tldrs(entries, overrides=overrides)

    write_monthly_indexes(grouped, tldr_lookup, dry_run=dry_run)

    readme_content = build_root_readme(entries, grouped, tldr_lookup)
    write_text_file(SUMMARY_README_PATH, readme_content, dry_run=dry_run)

    # all_guide.md has been removed; delete legacy file if present.
    if ALL_GUIDE_PATH.exists():
        if dry_run:
            logging.info("Dry-run: would delete legacy %s", ALL_GUIDE_PATH)
        else:
            try:
                ALL_GUIDE_PATH.unlink()
            except Exception as error:  # noqa: BLE001
                logging.warning("Failed to delete legacy %s: %s", ALL_GUIDE_PATH, error)

    if (migrated_any or repaired_any) and not dry_run:
        logging.info("Guide name repair completed; indexes regenerated.")

    if dry_run:
        logging.info("Dry-run complete; no files were written.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract tool links and generate tool guides.")
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Rebuild README and monthly indexes without ingesting new tools.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pipeline without writing changes to disk.",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Skip submission to Wayback Machine.",
    )
    parser.add_argument(
        "--collection-readme",
        type=str,
        default=str(DEFAULT_COLLECTION_README_PATH.as_posix()),
        help=(
            "Path to the README.md that contains the tool link list "
            "(default auto-detected: kohstool/README.md or ../kohstool/README.md)."
        ),
    )
    parser.add_argument(
        "--max-new",
        type=int,
        default=1,
        help="Process up to N new tools in a single run (default: 1).",
    )
    parser.add_argument(
        "--heuristic",
        action="store_true",
        help="Generate guide/tags/platform without calling OpenAI (use heuristics).",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip fetching page text via Jina Reader (guide generated from name/url only).",
    )
    args = parser.parse_args()

    env_dry_run = os.getenv("TOOL_GUIDE_DRY_RUN", "").lower() in ("1", "true", "yes")
    if env_dry_run:
        args.dry_run = True

    return args


def main() -> None:
    args = parse_args()
    options = RunOptions(
        backfill=args.backfill,
        dry_run=args.dry_run,
        archive=not args.no_archive,
        collection_readme_path=Path(args.collection_readme),
        max_new=max(0, int(args.max_new)),
        heuristic=bool(args.heuristic),
        fetch=not bool(args.no_fetch),
    )
    process_tools(options)


if __name__ == "__main__":
    main()
