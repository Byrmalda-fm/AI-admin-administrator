"""Telegram AI administrator for a Minecraft server managed over SFTP."""

from __future__ import annotations

import io
import json
import logging
import os
import posixpath
import re
from stat import S_ISDIR
import threading
import base64
import html
import shutil
import tempfile
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import paramiko
import requests
import telebot
from groq import Groq


MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODELS = (
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
)
MAX_TOOL_STEPS = 8
MAX_READ_BYTES = 1_000_000
# Telegram Bot API не позволяет боту скачать файл крупнее 20 МБ, поэтому
# для больших аддонов пользователь присылает прямую ссылку, и бот качает
# архив сам по HTTP(S), в обход этого лимита. Здесь — лимит на такую
# загрузку, чтобы не исчерпать память/диск бесплатного инстанса Render.
MAX_URL_DOWNLOAD_BYTES = int(os.environ.get("ADDON_MAX_DOWNLOAD_MB", "").strip() or "300") * 1_000_000
ADDON_EXTENSIONS = (".mcaddon", ".mcpack", ".zip")
URL_PATTERN = re.compile(r"https?://[^\s<>\"]+", re.IGNORECASE)
MAX_MEMORY_ITEMS = 12
# На Render без постоянного диска файловая система эфемерна и сбрасывается
# при каждом деплое. Если задан STATE_DIR (например, точка монтирования
# постоянного диска Render, см. render.yaml), файлы памяти и runtime-конфига
# хранятся там; иначе — рядом с исходным кодом, как раньше.
_STATE_DIR = Path(os.environ.get("STATE_DIR", "")).expanduser() if os.environ.get("STATE_DIR") else None
if _STATE_DIR is not None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_FILE = (_STATE_DIR / "memory.json") if _STATE_DIR else Path(__file__).with_name("memory.json")
RUNTIME_CONFIG_FILE = (
    (_STATE_DIR / "runtime_config.json") if _STATE_DIR else Path(__file__).with_name("runtime_config.json")
)
SYSTEM_PROMPT = """Ты — опытный администратор игрового сервера Minecraft. Твоя задача — помогать владельцу управлять файлами сервера через текстовые команды.
Когда пользователь дает комплексную команду (например, активировать все аддоны), сначала осмотрись на сервере (через sftp_list_dir), собери все нужные данные, сформируй обновленные конфигурационные файлы и запиши их обратно. Старайся делать тяжелые операции за минимальное количество шагов, группируя действия.
Действуй аккуратно, не удаляй лишние папки. Для проверки worldbehaviorpacks и worldresourcepacks используй быстрый рекурсивный результат sftp_list_dir и содержимое manifest.json, не обходи каждый вложенный файл отдельным вызовом. Не повторяй вызовы с теми же аргументами. Старайся завершить задачу максимум за 4 инструментальных шага. В конце напиши только краткий отчет на русском языке: максимум 5 коротких пунктов и 600 символов, без длинных рекомендаций."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "sftp_list_dir",
            "description": "Показать содержимое папки на Minecraft-сервере.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Путь к папке"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sftp_read_file",
            "description": "Прочитать текстовый файл на Minecraft-сервере.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Путь к файлу"}
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sftp_write_file",
            "description": "Полностью записать текст в файл на Minecraft-сервере.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Путь к файлу"},
                    "content": {"type": "string", "description": "Новое содержимое"},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sftp_remove_file_or_dir",
            "description": "Удалить файл или рекурсивно удалить папку на Minecraft-сервере.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Путь"}},
                "required": ["path"],
            },
        },
    },
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("minecraft-ai-admin")


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Не задана обязательная переменная окружения: {name}")
    return value


def load_json_file(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, ValueError):
        return default


def save_json_file(path: Path, value: Any) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def load_runtime_config() -> dict[str, Any]:
    config = load_json_file(RUNTIME_CONFIG_FILE, {})
    return config if isinstance(config, dict) else {}


def update_runtime_config(**updates: Any) -> dict[str, Any]:
    """Merge-update runtime_config.json instead of overwriting it, so SFTP
    host/port and GROQ keys (stored in the same file) don't clobber each other."""
    config = load_runtime_config()
    config.update(updates)
    save_json_file(RUNTIME_CONFIG_FILE, config)
    return config


def get_groq_api_keys() -> list[str]:
    """Keys set via the bot's "Изменить GROQ API ключ(и)" button take priority;
    otherwise fall back to GROQ_API_KEY (comma- or newline-separated for multiple)."""
    configured = load_runtime_config().get("groq_api_keys")
    keys: list[str] = []
    if isinstance(configured, list):
        keys = [str(key).strip() for key in configured if str(key).strip()]
    if not keys:
        env_value = os.environ.get("GROQ_API_KEY", "")
        keys = [part.strip() for part in re.split(r"[,\n]+", env_value) if part.strip()]
    if not keys:
        raise RuntimeError(
            "Не задан ни один GROQ API ключ. Настройте через кнопку "
            "«Изменить GROQ API ключ(и)» или переменную GROQ_API_KEY."
        )
    return keys


def is_groq_key_error(exc: Exception) -> bool:
    """Whether an error looks like it's specific to the current API key
    (rate limit / quota exhausted / invalid key) rather than e.g. a bad model name."""
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "rate limit",
            "rate_limit",
            "429",
            "quota",
            "insufficient_quota",
            "invalid api key",
            "invalid_api_key",
            "401",
            "unauthorized",
        )
    )


class RotatingGroqClient:
    """Wraps one or more Groq API keys and transparently rotates to the next
    key when the current one hits a rate limit, exhausted quota, or auth error."""

    def __init__(self, api_keys: list[str], update_status: Callable[[str], None]) -> None:
        self._api_keys = api_keys
        self._index = 0
        self._update_status = update_status
        self._client = Groq(api_key=api_keys[0])

    def _rotate(self) -> bool:
        if self._index + 1 >= len(self._api_keys):
            return False
        self._index += 1
        self._client = Groq(api_key=self._api_keys[self._index])
        self._update_status(
            f"🔑 Лимит текущего GROQ-ключа исчерпан, переключаюсь на ключ №{self._index + 1}..."
        )
        return True

    def call(self, action: Callable[[Groq], Any]) -> Any:
        while True:
            try:
                return action(self._client)
            except Exception as exc:
                if is_groq_key_error(exc) and self._rotate():
                    continue
                raise


def load_memory() -> list[dict[str, str]]:
    memory = load_json_file(MEMORY_FILE, [])
    if not isinstance(memory, list):
        return []
    return [item for item in memory if isinstance(item, dict)][-MAX_MEMORY_ITEMS:]


def remember_request(request: str, report: str) -> None:
    memory = load_memory()
    memory.append({"request": request[:500], "report": report[:900]})
    save_json_file(MEMORY_FILE, memory[-MAX_MEMORY_ITEMS:])


def compact_report(report: str, max_chars: int = 600) -> str:
    cleaned = re.sub(r"\n{3,}", "\n\n", report.strip())
    if len(cleaned) <= max_chars:
        return cleaned
    shortened = cleaned[: max_chars - 1].rstrip()
    boundary = shortened.rfind(" ")
    if boundary >= max_chars // 2:
        shortened = shortened[:boundary].rstrip()
    return shortened + "…"


def safe_archive_member(root: Path, member_name: str) -> Path:
    target = (root / member_name).resolve()
    if target != root.resolve() and root.resolve() not in target.parents:
        raise ValueError("Архив содержит небезопасный путь")
    return target


def unpack_addon_archive(archive_path: Path, output_dir: Path) -> list[Path]:
    """Unpack mcaddon/mcpack archives and return pack directories with manifests."""
    pending = [archive_path]
    extracted_archives: set[Path] = set()
    while pending:
        current = pending.pop()
        if current in extracted_archives:
            continue
        extracted_archives.add(current)
        destination = output_dir / f"part_{len(extracted_archives)}"
        destination.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(current) as archive:
                for member in archive.infolist():
                    target = safe_archive_member(destination, member.filename)
                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
        except zipfile.BadZipFile as exc:
            raise ValueError(
                "Скачанный файл — не архив. Обычно так бывает, если ссылка "
                "ведёт на страницу предпросмотра, а не на сам файл (например, "
                "не прямая ссылка на скачивание). Проверьте, что ссылка "
                "скачивает файл напрямую, без открытия страницы в браузере."
            ) from exc
        for nested in destination.rglob("*"):
            if nested.is_file() and nested.suffix.lower() in {".mcaddon", ".mcpack", ".zip"}:
                pending.append(nested)

    manifests = list(output_dir.rglob("manifest.json"))
    if not manifests:
        raise ValueError("В архиве не найден manifest.json")
    return sorted({manifest.parent for manifest in manifests})


def pack_kind(manifest: dict[str, Any]) -> str:
    module_types = {
        str(module.get("type", "")).lower()
        for module in manifest.get("modules", [])
        if isinstance(module, dict)
    }
    if "resources" in module_types or "resource" in module_types:
        return "resource"
    if "data" in module_types or "script" in module_types or "behavior" in module_types:
        return "behavior"
    raise ValueError("В manifest.json не найден тип behavior или resource")


def pack_identity(manifest: dict[str, Any]) -> tuple[str, list[int]]:
    header = manifest.get("header")
    if not isinstance(header, dict) or not header.get("uuid"):
        raise ValueError("В manifest.json отсутствует header.uuid")
    version = header.get("version", [1, 0, 0])
    if not isinstance(version, list) or len(version) != 3:
        raise ValueError("В manifest.json некорректный header.version")
    return str(header["uuid"]), [int(part) for part in version]


MAX_DISCOVERY_DIRS = 600
# Каталоги, в которые точно не имеет смысла спускаться при поиске
# behavior_packs/resource_packs — экономит десятки SFTP-запросов на
# больших серверах.
DISCOVERY_SKIP_NAMES = {
    ".git",
    ".cache",
    ".npm",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "cache",
    "logs",
    "log",
    "backup",
    "backups",
    "tmp",
    "proc",
    "sys",
    "dev",
}


def discover_remote_paths(
    sftp: paramiko.SFTPClient,
) -> tuple[list[str], list[str], list[str]]:
    behavior_dirs: list[str] = []
    resource_dirs: list[str] = []
    world_configs: list[str] = []
    # Полный обход "/" на некоторых хостингах — это тысячи каталогов и
    # минуты SFTP-запросов. Ограничиваемся типичными для Minecraft-хостинга
    # корнями; ".", как правило, и есть корень SFTP-логина сервера.
    queue: list[tuple[str, int]] = [
        (root, 0) for root in (".", "/data", "/opt", "/home", "/srv")
    ]
    visited: set[str] = set()
    while queue:
        if len(visited) >= MAX_DISCOVERY_DIRS:
            break
        current, depth = queue.pop(0)
        if current in visited or depth > 5:
            continue
        visited.add(current)
        try:
            entries = sftp.listdir_attr(current)
        except OSError:
            continue
        for entry in entries:
            path = posixpath.join(current, entry.filename)
            name = entry.filename.lower()
            if name.startswith(".") or name in DISCOVERY_SKIP_NAMES:
                continue
            if name == "world_behavior_packs.json":
                world_configs.append(path)
            elif name == "world_resource_packs.json":
                world_configs.append(path)
            if S_ISDIR(entry.st_mode):
                is_behavior_dir = name in {
                    "behavior_packs",
                    "behaviorpacks",
                    "worldbehaviorpacks",
                    "world_behavior_packs",
                }
                is_resource_dir = name in {
                    "resource_packs",
                    "resourcepacks",
                    "worldresourcepacks",
                    "world_resource_packs",
                    "worldresoucepacks",
                }
                if is_behavior_dir:
                    behavior_dirs.append(path)
                elif is_resource_dir:
                    resource_dirs.append(path)
                # Внутрь самих behavior_packs/resource_packs спускаться не
                # нужно — там лежат уже установленные паки, а не другие
                # world_*.json или вложенные папки паков.
                if not is_behavior_dir and not is_resource_dir:
                    queue.append((path, depth + 1))
    return behavior_dirs, resource_dirs, world_configs


def discover_remote_paths_cached(
    sftp: paramiko.SFTPClient, update_status: Callable[[str], None]
) -> tuple[list[str], list[str], list[str]]:
    """Same as discover_remote_paths, but remembers the result in
    runtime_config.json so subsequent installs skip the slow full scan."""
    cached = load_runtime_config().get("addon_paths")
    if isinstance(cached, dict):
        behavior_dirs = [str(p) for p in cached.get("behavior_dirs") or []]
        resource_dirs = [str(p) for p in cached.get("resource_dirs") or []]
        world_configs = [str(p) for p in cached.get("world_configs") or []]
        probe_paths = (behavior_dirs[:1] + resource_dirs[:1])
        if probe_paths:
            try:
                for path in probe_paths:
                    sftp.stat(path)
                return behavior_dirs, resource_dirs, world_configs
            except OSError:
                pass  # Кэш устарел (например, сменился SFTP-хост) — ищем заново.
    update_status(
        "🔎 Ищу папки behavior_packs/resource_packs на сервере "
        "(разовая операция, дальше установка будет быстрее)..."
    )
    behavior_dirs, resource_dirs, world_configs = discover_remote_paths(sftp)
    if behavior_dirs or resource_dirs:
        update_runtime_config(
            addon_paths={
                "behavior_dirs": behavior_dirs,
                "resource_dirs": resource_dirs,
                "world_configs": world_configs,
            }
        )
    return behavior_dirs, resource_dirs, world_configs


def upload_directory(
    sftp: paramiko.SFTPClient, local_dir: Path, remote_dir: str
) -> None:
    try:
        sftp.mkdir(remote_dir)
    except OSError:
        pass
    for item in local_dir.iterdir():
        remote_item = posixpath.join(remote_dir, item.name)
        if item.is_dir():
            upload_directory(sftp, item, remote_item)
        else:
            sftp.put(str(item), remote_item)


def install_addon_archive(
    sftp: paramiko.SFTPClient,
    archive_path: Path,
    update_status: Callable[[str], None],
) -> str:
    with tempfile.TemporaryDirectory(prefix="minecraft-addon-") as temporary_dir:
        pack_dirs = unpack_addon_archive(archive_path, Path(temporary_dir))
        behavior_dirs, resource_dirs, world_configs = discover_remote_paths_cached(
            sftp, update_status
        )
        if not behavior_dirs and not resource_dirs:
            raise FileNotFoundError(
                "Не найдены папки behavior_packs/resource_packs на сервере"
            )
        installed: list[str] = []
        links: dict[str, list[dict[str, Any]]] = {
            "behavior": [],
            "resource": [],
        }
        for pack_dir in pack_dirs:
            manifest_path = pack_dir / "manifest.json"
            try:
                with manifest_path.open("r", encoding="utf-8") as manifest_file:
                    manifest = json.load(manifest_file)
                kind = pack_kind(manifest)
                pack_id, version = pack_identity(manifest)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Некорректный manifest.json: {exc}") from exc
            destination_roots = behavior_dirs if kind == "behavior" else resource_dirs
            if not destination_roots:
                raise FileNotFoundError(
                    f"Не найдена папка для {kind}-пака на сервере"
                )
            header = manifest.get("header", {})
            raw_name = str(header.get("name") or "").strip()
            # Многие Bedrock-аддоны указывают в header.name не настоящее
            # название, а лейбл локализации вида "pack.name" (реальное имя
            # берётся из texts/en_US.lang внутри resource-пака). Если взять
            # такой лейбл как есть, у разных аддонов совпадёт имя папки на
            # сервере, и один аддон затрёт файлы другого.
            is_placeholder_name = not raw_name or bool(
                re.fullmatch(r"[a-z0-9]+(?:\.[a-z0-9_]+)+", raw_name)
            )
            sanitized = re.sub(
                r"[^A-Za-z0-9_.-]+", "_", raw_name if not is_placeholder_name else pack_id
            ).strip("._")
            sanitized = sanitized[:60] or pack_id
            # UUID пака гарантированно уникален для каждого аддона, поэтому
            # добавляем его короткий суффикс к имени папки — так два разных
            # аддона никогда не окажутся в одной и той же папке, даже если
            # у обоих одинаковое (или одинаково пустое/generic) название.
            folder_name = f"{sanitized}_{pack_id.replace('-', '')[:8]}"
            display_name = raw_name if not is_placeholder_name else f"{pack_id[:8]}…"
            update_status(
                f"📦 Устанавливаю {('behavior' if kind == 'behavior' else 'resource')} pack..."
            )
            remote_pack_dir = posixpath.join(destination_roots[0], folder_name)
            upload_directory(sftp, pack_dir, remote_pack_dir)
            links[kind].append({"pack_id": pack_id, "version": version})
            installed.append(f"{kind}: {display_name}")

        for config_path in world_configs:
            config_name = posixpath.basename(config_path).lower()
            kind = "behavior" if "behavior" in config_name else "resource"
            if not links[kind]:
                continue
            try:
                with sftp.open(config_path, "rb") as config_file:
                    current = json.loads(config_file.read().decode("utf-8"))
                if not isinstance(current, list):
                    current = []
            except (OSError, ValueError, json.JSONDecodeError):
                current = []
            existing_ids = {
                str(item.get("pack_id"))
                for item in current
                if isinstance(item, dict) and item.get("pack_id")
            }
            current.extend(
                item for item in links[kind] if item["pack_id"] not in existing_ids
            )
            with sftp.open(config_path, "wb") as config_file:
                config_file.write(
                    json.dumps(current, ensure_ascii=False, indent=2).encode("utf-8")
                )
            update_status("📝 Обновляю список паков мира...")
        return compact_report("Установлено: " + ", ".join(installed))


def extract_first_url(text: str) -> str | None:
    match = URL_PATTERN.search(text)
    return match.group(0).rstrip(").,;!") if match else None


def _normalize_dropbox_url(url: str) -> str:
    """Dropbox "Поделиться" links (dl=0) return an HTML preview page, not the
    file itself. Forcing dl=1 makes Dropbox serve the raw file content."""
    if "dropbox.com" not in url or "dropboxusercontent.com" in url:
        return url
    if "dl=0" in url:
        return url.replace("dl=0", "dl=1")
    if re.search(r"[?&]dl=1\b", url):
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}dl=1"


def _google_drive_file_id(url: str) -> str | None:
    match = re.search(r"drive\.google\.com/file/d/([^/]+)", url)
    if match:
        return match.group(1)
    match = re.search(r"[?&]id=([^&]+)", url)
    if match and "drive.google.com" in url:
        return match.group(1)
    return None


def _filename_from_response(url: str, response: requests.Response) -> str:
    disposition = response.headers.get("Content-Disposition", "")
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', disposition)
    if match:
        candidate = match.group(1).strip()
        if candidate:
            return candidate
    path_name = posixpath.basename(re.sub(r"[?#].*$", "", url))
    return path_name or "addon.zip"


def _stream_to_file(
    response: requests.Response, destination: Path, update_status: Callable[[str], None]
) -> None:
    total = 0
    last_reported_mb = 0
    with destination.open("wb") as file:
        for chunk in response.iter_content(chunk_size=1 << 20):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_URL_DOWNLOAD_BYTES:
                raise ValueError(
                    "Файл превышает лимит "
                    f"{MAX_URL_DOWNLOAD_BYTES // 1_000_000} МБ для загрузки по ссылке "
                    "(настраивается через переменную ADDON_MAX_DOWNLOAD_MB)."
                )
            file.write(chunk)
            current_mb = total // (5 * 1_000_000)
            if current_mb > last_reported_mb:
                last_reported_mb = current_mb
                update_status(f"⬇️ Скачано {total // 1_000_000} МБ...")
    if total == 0:
        raise ValueError("По ссылке получен пустой файл.")


def download_archive_from_url(url: str, update_status: Callable[[str], None]) -> Path:
    """Download an addon archive from a direct link (or a Google Drive/Dropbox share link)."""
    update_status("🌐 Скачиваю архив по ссылке...")
    url = _normalize_dropbox_url(url)
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (compatible; MinecraftAIAdminBot/1.0)"
    drive_id = _google_drive_file_id(url)
    temp_dir = Path(tempfile.mkdtemp(prefix="minecraft-url-download-"))
    try:
        if drive_id:
            # Google Drive требует подтверждения для файлов, которые не проходят
            # антивирусную проверку (обычно всё, что крупнее ~25 МБ). Делаем
            # запрос, ищем токен подтверждения и повторяем запрос с ним.
            probe_url = "https://drive.google.com/uc?export=download"
            response = session.get(probe_url, params={"id": drive_id}, stream=True, timeout=30)
            token = None
            for key, value in response.cookies.items():
                if key.startswith("download_warning"):
                    token = value
            if token is None and "text/html" in response.headers.get("Content-Type", ""):
                match = re.search(r'confirm=([0-9A-Za-z_-]+)', response.text)
                if match:
                    token = match.group(1)
            if token:
                response.close()
                response = session.get(
                    probe_url,
                    params={"id": drive_id, "confirm": token},
                    stream=True,
                    timeout=30,
                )
            response.raise_for_status()
            filename = f"{drive_id}.zip"
        else:
            response = session.get(url, stream=True, timeout=30, allow_redirects=True)
            response.raise_for_status()
            filename = _filename_from_response(url, response)

        with response:
            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type and Path(filename).suffix.lower() not in ADDON_EXTENSIONS:
                raise ValueError(
                    "По ссылке вернулась веб-страница, а не файл архива. "
                    "Нужна прямая ссылка на скачивание .mcaddon/.mcpack/.zip "
                    "(для Google Drive подходит обычная ссылка «Поделиться»)."
                )
            if Path(filename).suffix.lower() not in ADDON_EXTENSIONS:
                filename = filename + ".zip"
            destination = temp_dir / (re.sub(r"[^A-Za-z0-9_.-]+", "_", filename) or "addon.zip")
            _stream_to_file(response, destination, update_status)
        return destination
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def normalize_private_key(raw_key: str) -> str:
    """Normalize common env-var paste formats without logging key material."""
    key_text = raw_key.strip().replace("\\r\\n", "\n").replace("\\n", "\n")
    key_text = key_text.replace("\r\n", "\n")
    if key_text.startswith("SSH_PRIVATE_KEY="):
        key_text = key_text.split("=", 1)[1].strip()
    if len(key_text) >= 2 and key_text[0] in {"'", '"'} and key_text[-1] == key_text[0]:
        key_text = key_text[1:-1].strip()

    # Некоторые сценарии сохраняют OpenSSH private key как Base64 без
    # строк BEGIN/END. Восстанавливаем стандартную OpenSSH PEM-обёртку.
    compact_key = re.sub(r"\s+", "", key_text)
    try:
        decoded_key = base64.b64decode(compact_key, validate=True)
    except (ValueError, base64.binascii.Error):
        decoded_key = b""
    if decoded_key.startswith(b"openssh-key-v1\x00"):
        key_text = base64.b64encode(decoded_key).decode("ascii")
        key_text = (
            "-----BEGIN OPENSSH PRIVATE KEY-----\n"
            + "\n".join(
                key_text[index : index + 64]
                for index in range(0, len(key_text), 64)
            )
            + "\n-----END OPENSSH PRIVATE KEY-----\n"
        )

    header_match = re.search(r"-----BEGIN [^-]+ PRIVATE KEY-----", key_text)
    footer_match = re.search(r"-----END [^-]+ PRIVATE KEY-----", key_text)
    if header_match and footer_match and footer_match.start() > header_match.end():
        header = header_match.group(0)
        footer = footer_match.group(0)
        body = key_text[header_match.end() : footer_match.start()]
        body = re.sub(r"\s+", "", body)
        wrapped_body = "\n".join(
            body[index : index + 64] for index in range(0, len(body), 64)
        )
        return f"{header}\n{wrapped_body}\n{footer}\n"
    return key_text


def connect_sftp() -> tuple[paramiko.SSHClient, paramiko.SFTPClient]:
    runtime_config = load_runtime_config()
    host = str(runtime_config.get("host") or required_env("SFTP_HOST")).strip()
    user = required_env("SFTP_USER")
    port = int(runtime_config.get("port") or os.environ.get("SFTP_PORT", "22") or "22")
    key_text = normalize_private_key(required_env("SSH_PRIVATE_KEY"))

    private_key: paramiko.PKey | None = None
    key_loaders = [
        ("RSA", paramiko.RSAKey),
        ("Ed25519", paramiko.Ed25519Key),
        ("ECDSA", paramiko.ECDSAKey),
    ]
    # DSS/DSA удалён из новых версий Paramiko, поэтому добавляем его
    # только если конкретная версия библиотеки ещё его предоставляет.
    dss_key = getattr(paramiko, "DSSKey", None)
    if dss_key is not None:
        key_loaders.append(("DSA", dss_key))
    errors: list[str] = []
    for key_name, key_type in key_loaders:
        try:
            private_key = key_type.from_private_key(io.StringIO(key_text))
            break
        except (paramiko.SSHException, ValueError) as exc:
            errors.append(f"{key_name}: {exc}")
    if private_key is None:
        raise RuntimeError(
            "SSH_PRIVATE_KEY не удалось прочитать. "
            "Вставьте полный многострочный ключ, начиная с "
            "`-----BEGIN ... PRIVATE KEY-----`, без имени переменной."
        ) from RuntimeError("; ".join(errors))

    ssh = paramiko.SSHClient()
    # Аутентификация выполняется только RSA-ключом из переменной окружения.
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        hostname=host,
        port=port,
        username=user,
        pkey=private_key,
        look_for_keys=False,
        allow_agent=False,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    return ssh, ssh.open_sftp()


def remove_recursive(sftp: paramiko.SFTPClient, path: str) -> None:
    try:
        attributes = sftp.stat(path)
    except OSError as exc:
        raise FileNotFoundError(f"Путь не найден: {path}") from exc

    if S_ISDIR(attributes.st_mode):
        for entry in sftp.listdir_attr(path):
            child = posixpath.join(path, entry.filename)
            remove_recursive(sftp, child)
        sftp.rmdir(path)
    else:
        sftp.remove(path)


def make_sftp_tools(
    sftp: paramiko.SFTPClient, on_tool_status: Callable[[str], None]
) -> dict[str, Callable[..., str]]:
    def list_dir(path: str) -> str:
        on_tool_status("⚙️ Изучаю структуру папок...")
        target_scan = any(
            folder_name in posixpath.basename(path).lower()
            for folder_name in ("worldbehaviorpacks", "worldresourcepacks", "worldresoucepacks")
        )

        def collect(folder: str, depth: int) -> list[dict[str, Any]]:
            collected: list[dict[str, Any]] = []
            for entry in sftp.listdir_attr(folder):
                child = posixpath.join(folder, entry.filename)
                is_dir = S_ISDIR(entry.st_mode)
                item: dict[str, Any] = {
                    "path": child,
                    "name": entry.filename,
                    "type": "dir" if is_dir else "file",
                    "size": entry.st_size,
                }
                if (
                    not is_dir
                    and entry.filename.lower() in {"manifest.json", "pack_manifest.json"}
                    and entry.st_size <= 32_768
                ):
                    try:
                        with sftp.open(child, "rb") as manifest_file:
                            item["content"] = manifest_file.read().decode(
                                "utf-8", errors="replace"
                            )
                    except OSError as exc:
                        item["content_error"] = str(exc)
                collected.append(item)
                if is_dir and depth > 0:
                    collected.extend(collect(child, depth - 1))
            return collected

        result = collect(path, 3 if target_scan else 0)
        return json.dumps({"path": path, "entries": result}, ensure_ascii=False)

    def read_file(file_path: str) -> str:
        on_tool_status("📖 Читаю конфигурацию сервера...")
        with sftp.open(file_path, "rb") as remote_file:
            content = remote_file.read(MAX_READ_BYTES + 1)
        truncated = len(content) > MAX_READ_BYTES
        if truncated:
            content = content[:MAX_READ_BYTES]
        return json.dumps(
            {
                "file_path": file_path,
                "content": content.decode("utf-8", errors="replace"),
                "truncated": truncated,
            },
            ensure_ascii=False,
        )

    def write_file(file_path: str, content: str) -> str:
        on_tool_status("💾 Записываю изменения...")
        with sftp.open(file_path, "wb") as remote_file:
            remote_file.write(content.encode("utf-8"))
        return json.dumps(
            {"ok": True, "file_path": file_path, "bytes": len(content.encode("utf-8"))},
            ensure_ascii=False,
        )

    def remove_file_or_dir(path: str) -> str:
        if posixpath.normpath(path) in {".", "/"}:
            raise PermissionError("Удаление корневой папки SFTP запрещено")
        on_tool_status("🗑️ Удаляю выбранные файлы...")
        remove_recursive(sftp, path)
        return json.dumps({"ok": True, "removed": path}, ensure_ascii=False)

    return {
        "sftp_list_dir": list_dir,
        "sftp_read_file": read_file,
        "sftp_write_file": write_file,
        "sftp_remove_file_or_dir": remove_file_or_dir,
    }


def format_status(request: str, status: str, history: list[str] | None = None) -> str:
    lines = [
        "🤖 <b>ИИ-Администратор принял задачу</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f'📥 <i>Запрос:</i> "{html.escape(request)}"',
    ]
    if history:
        lines.append("")
        lines.extend(html.escape(item) for item in history[-5:])
    lines.append(f"⏳ <i>Статус:</i> {html.escape(status)}")
    return "\n".join(lines)


def run_ai_task(
    request: str, update_status: Callable[[str], None]
) -> tuple[str, bool]:
    ssh: paramiko.SSHClient | None = None
    sftp: paramiko.SFTPClient | None = None
    step_history: list[str] = []
    try:
        update_status("🔐 Подключаюсь к SFTP-серверу...")
        ssh, sftp = connect_sftp()

        def tool_status(status: str) -> None:
            step_history.append(status)
            update_status(status)

        tools = make_sftp_tools(sftp, tool_status)
        groq = RotatingGroqClient(get_groq_api_keys(), update_status)
        configured_model = os.environ.get("GROQ_MODEL", MODEL).strip() or MODEL
        model_candidates = [configured_model] + [
            model for model in FALLBACK_MODELS if model != configured_model
        ]
        try:
            available_model_ids = [
                str(model.id)
                for model in groq.call(lambda c: c.models.list()).data
                if getattr(model, "id", None)
            ]
            if available_model_ids:
                preferred_ids = [
                    configured_model,
                    "openai/gpt-oss-120b",
                    "openai/gpt-oss-20b",
                    "llama-4-scout-17b-16e-instruct",
                    "llama-3.1-8b-instant",
                ]
                model_candidates = [
                    model
                    for model in preferred_ids
                    if model in available_model_ids
                ]
                tool_capable_ids = [
                    model_id
                    for model_id in available_model_ids
                    if any(
                        marker in model_id.lower()
                        for marker in ("gpt-oss", "llama", "qwen", "mixtral")
                    )
                ]
                for model_id in tool_capable_ids:
                    if model_id not in model_candidates:
                        model_candidates.append(model_id)
        except Exception:
            logger.warning("Не удалось получить список моделей Groq; использую fallback-список")
        if not model_candidates:
            raise RuntimeError(
                "В списке Groq нет доступной модели для Tool Use. "
                "Проверьте права GROQ_API_KEY."
            )
        active_model = model_candidates[0]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request},
        ]
        previous_requests = load_memory()
        if previous_requests:
            history_text = "\n".join(
                f"- Запрос: {item.get('request', '')}\n  Итог: {item.get('report', '')}"
                for item in previous_requests[-5:]
            )
            messages.insert(
                1,
                {
                    "role": "system",
                    "content": "Краткая память предыдущих задач. Учитывай её только если это помогает "
                    "понять текущий запрос:\n"
                    + history_text,
                },
            )

        tool_steps = 0
        model_rounds = 0
        duplicate_calls = 0
        tool_cache: dict[str, str] = {}
        observations: list[str] = []
        while tool_steps < MAX_TOOL_STEPS and model_rounds < 12:
            model_rounds += 1
            if tool_steps >= 6 or duplicate_calls >= 2:
                final_messages: list[dict[str, Any]] = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                        + "\nСейчас инструменты недоступны. Сформируй итоговый отчет только по данным ниже.",
                    },
                    {"role": "user", "content": request},
                    {
                        "role": "user",
                        "content": "Результаты уже выполненных проверок SFTP:\n"
                        + "\n\n".join(observations)[-60_000:],
                    },
                ]
                final_response = groq.call(
                    lambda c: c.chat.completions.create(
                        model=active_model,
                        messages=final_messages,
                        temperature=0.1,
                    )
                )
                return (
                    compact_report(
                        final_response.choices[0].message.content
                        or "Проверка завершена, но ИИ не сформировал текстовый отчет."
                    ),
                    False,
                )
            response = None
            last_model_error: Exception | None = None
            for model_name in model_candidates:
                try:
                    completion_args: dict[str, Any] = {
                        "model": model_name,
                        "messages": messages,
                        "temperature": 0.1,
                    }
                    completion_args["tools"] = TOOLS
                    completion_args["tool_choice"] = "auto"
                    response = groq.call(lambda c: c.chat.completions.create(**completion_args))
                    if model_name != active_model:
                        active_model = model_name
                        update_status(f"🧠 Использую доступную модель: {active_model}")
                    break
                except Exception as exc:
                    error_text = str(exc).lower()
                    if "modelnotfound" not in error_text and "404" not in error_text:
                        raise
                    last_model_error = exc
            if response is None:
                raise RuntimeError(
                    "Groq не предоставил доступную модель. "
                    "Проверьте GROQ_API_KEY или задайте переменную GROQ_MODEL."
                ) from last_model_error
            assistant_message = response.choices[0].message
            tool_calls = assistant_message.tool_calls or []
            if not tool_calls:
                return compact_report(
                    assistant_message.content or "ИИ не вернул отчет о выполнении."
                ), False

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in tool_calls
                    ],
                }
            )
            for call in tool_calls:
                if tool_steps >= MAX_TOOL_STEPS:
                    warning = "⚠️ Превышен лимит шагов ИИ (8 шагов). Действие остановлено для безопасности!"
                    update_status(warning)
                    return warning, True
                try:
                    cache_key = json.dumps(
                        {
                            "name": call.function.name,
                            "arguments": json.loads(call.function.arguments or "{}"),
                        },
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                except json.JSONDecodeError:
                    cache_key = f"{call.function.name}:{call.function.arguments}"
                function = tools.get(call.function.name)
                if cache_key in tool_cache:
                    result = tool_cache[cache_key]
                    duplicate_calls += 1
                    update_status("♻️ Использую уже полученные данные...")
                elif function is None:
                    result = json.dumps({"error": "Неизвестный инструмент"})
                else:
                    tool_steps += 1
                    try:
                        arguments = json.loads(call.function.arguments or "{}")
                        result = function(**arguments)
                    except Exception as exc:  # Send the error back to the model.
                        logger.exception("Tool %s failed", call.function.name)
                        result = json.dumps(
                            {"error": f"{type(exc).__name__}: {exc}"},
                            ensure_ascii=False,
                        )
                    tool_cache[cache_key] = result
                observations.append(
                    f"Инструмент {call.function.name} ({call.function.arguments}):\n{result}"
                )
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )

        warning = "⚠️ Превышен лимит шагов ИИ (8 шагов). Действие остановлено для безопасности!"
        update_status(warning)
        return warning, True
    finally:
        if sftp is not None:
            sftp.close()
        if ssh is not None:
            ssh.close()


class _HealthCheckHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler so Render's free Web Service sees an open port.

    Render's free tier only runs Web Services (a Background Worker requires
    a paid plan), and a Web Service must bind to $PORT to be considered
    healthy. The bot itself talks to Telegram via long polling and doesn't
    need HTTP, so this handler just answers "ok" to any request.
    """

    def do_GET(self) -> None:  # noqa: N802 (stdlib method name)
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Minecraft AI Admin Bot is running")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Suppress default per-request logging; Render's own health checks
        # and any keep-alive pings would otherwise spam the log.
        pass


def start_health_check_server() -> None:
    """Bind to $PORT (Render's requirement for Web Services) in a daemon thread."""
    port = int(os.environ.get("PORT", "").strip() or "10000")
    server = ThreadingHTTPServer(("0.0.0.0", port), _HealthCheckHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Health-check HTTP server listening on port %s", port)


def main() -> None:
    start_health_check_server()
    token = required_env("TELEGRAM_BOT_TOKEN")
    allowed_user_raw = (
        os.environ.get("TELEGRAM_ALLOWED_USER_ID", "").strip()
        or os.environ.get("TELEGRAM_ALLOWED_USER", "").strip()
    )
    if not allowed_user_raw:
        raise RuntimeError(
            "Не задана обязательная переменная TELEGRAM_ALLOWED_USER_ID "
            "(допускается также TELEGRAM_ALLOWED_USER)"
        )
    try:
        allowed_user_id = int(allowed_user_raw)
    except ValueError as exc:
        raise RuntimeError("Telegram user ID должен быть целым числом") from exc
    bot = telebot.TeleBot(token, parse_mode="HTML")
    user_lock = threading.Lock()
    busy = False
    awaiting_sftp_settings = False
    awaiting_groq_keys = False

    def install_addon_from_source(
        chat_id: int,
        label: str,
        fetch_archive: Callable[[Callable[[str], None]], Path],
    ) -> None:
        """Shared install flow for both a Telegram document upload and a direct link."""
        nonlocal busy
        with user_lock:
            if busy:
                bot.send_message(chat_id, "⏳ Предыдущая задача ещё выполняется.")
                return
            busy = True
        status_message = bot.send_message(chat_id, format_status(label, "Начинаю установку..."))
        history: list[str] = []

        def update_status(status: str) -> None:
            if status not in history:
                history.append(status)
            try:
                bot.edit_message_text(
                    format_status(label, status, history),
                    chat_id,
                    status_message.message_id,
                    parse_mode="HTML",
                )
            except Exception:
                logger.exception("Не удалось обновить статус установки")

        ssh: paramiko.SSHClient | None = None
        sftp: paramiko.SFTPClient | None = None
        temp_root: Path | None = None
        try:
            archive_path = fetch_archive(update_status)
            temp_root = archive_path.parent
            update_status("🔍 Проверяю manifest.json и структуру архива...")
            ssh, sftp = connect_sftp()
            report = install_addon_archive(sftp, archive_path, update_status)
            remember_request(f"Установка: {label}", report)
            update_status("✅ Успешно выполнено!")
            bot.edit_message_text(
                format_status(label, "✅ Успешно выполнено!", history)
                + f"\n\n📋 <b>Отчет:</b>\n{html.escape(report)}",
                chat_id,
                status_message.message_id,
                parse_mode="HTML",
            )
        except Exception as exc:
            logger.exception("Установка аддона завершилась ошибкой")
            update_status(f"❌ Ошибка: {exc}")
        finally:
            if sftp is not None:
                sftp.close()
            if ssh is not None:
                ssh.close()
            if temp_root is not None:
                shutil.rmtree(temp_root, ignore_errors=True)
            with user_lock:
                busy = False

    def sftp_keyboard() -> telebot.types.InlineKeyboardMarkup:
        keyboard = telebot.types.InlineKeyboardMarkup()
        keyboard.add(
            telebot.types.InlineKeyboardButton(
                "⚙️ Изменить SFTP Host / Port", callback_data="sftp_settings"
            )
        )
        keyboard.add(
            telebot.types.InlineKeyboardButton(
                "🔑 Изменить GROQ API ключ(и)", callback_data="groq_settings"
            )
        )
        keyboard.add(
            telebot.types.InlineKeyboardButton(
                "🔄 Пересканировать папки на сервере", callback_data="rescan_paths"
            )
        )
        return keyboard

    @bot.message_handler(commands=["start", "help"])
    def handle_start(message: telebot.types.Message) -> None:
        nonlocal awaiting_sftp_settings, awaiting_groq_keys
        if not message.from_user or message.from_user.id != allowed_user_id:
            bot.reply_to(message, "Доступ ограничен")
            return
        awaiting_sftp_settings = False
        awaiting_groq_keys = False
        bot.reply_to(
            message,
            "🤖 ИИ-Администратор Minecraft готов.\n\n"
            "Опишите задачу обычным языком, например:\n"
            "• Удали аддоны\n"
            "• Активируй все аддоны и ресурс паки\n"
            "• Измени настройки в server.properties\n\n"
            "Чтобы установить аддон, пришлите файл .mcaddon/.mcpack/.zip "
            "(до 20 МБ) или просто пришлите прямую ссылку на скачивание "
            "(в том числе Google Drive/Dropbox) — размер файла ограничен только "
            "лимитом сервера.\n\n"
            "Кнопками ниже можно сменить SFTP host/port, GROQ API ключ(и) "
            "(можно задать сразу несколько — бот сам переключится на следующий, "
            "если у текущего кончится лимит), а также сбросить запомненные "
            "пути к папкам паков, если вы сами что-то поменяли в структуре "
            "сервера.",
            reply_markup=sftp_keyboard(),
        )

    @bot.callback_query_handler(func=lambda call: call.data == "sftp_settings")
    def handle_sftp_settings(call: telebot.types.CallbackQuery) -> None:
        nonlocal awaiting_sftp_settings
        if not call.from_user or call.from_user.id != allowed_user_id:
            bot.answer_callback_query(call.id, "Доступ ограничен")
            return
        awaiting_sftp_settings = True
        bot.answer_callback_query(call.id)
        bot.send_message(
            call.message.chat.id if call.message else call.from_user.id,
            "Отправьте новые параметры одной строкой:\n"
            "<code>host port</code>\n\nНапример: <code>mc.example.com 22</code>",
            parse_mode="HTML",
        )

    @bot.callback_query_handler(func=lambda call: call.data == "groq_settings")
    def handle_groq_settings(call: telebot.types.CallbackQuery) -> None:
        nonlocal awaiting_groq_keys
        if not call.from_user or call.from_user.id != allowed_user_id:
            bot.answer_callback_query(call.id, "Доступ ограничен")
            return
        awaiting_groq_keys = True
        bot.answer_callback_query(call.id)
        current_count = len(load_runtime_config().get("groq_api_keys") or [])
        current_note = (
            f" Сейчас настроено ключей: {current_count}."
            if current_count
            else " Сейчас используется ключ из переменной окружения GROQ_API_KEY."
        )
        bot.send_message(
            call.message.chat.id if call.message else call.from_user.id,
            "Отправьте один или несколько GROQ API ключей — каждый с новой строки "
            "(можно и через запятую). Все текущие ключи будут заменены присланными."
            + current_note
            + "\n\nБот будет пробовать ключи по порядку и автоматически переключаться "
            "на следующий, если у текущего закончится лимит/квота.",
            parse_mode="HTML",
        )

    @bot.callback_query_handler(func=lambda call: call.data == "rescan_paths")
    def handle_rescan_paths(call: telebot.types.CallbackQuery) -> None:
        if not call.from_user or call.from_user.id != allowed_user_id:
            bot.answer_callback_query(call.id, "Доступ ограничен")
            return
        update_runtime_config(addon_paths=None)
        bot.answer_callback_query(call.id, "Кэш путей сброшен")
        bot.send_message(
            call.message.chat.id if call.message else call.from_user.id,
            "🔄 Запомненные пути к behavior_packs/resource_packs сброшены. "
            "При следующей установке аддона бот заново просканирует сервер "
            "(это будет чуть дольше обычного) и запомнит актуальные пути.",
        )

    @bot.message_handler(content_types=["document"])
    def handle_document(message: telebot.types.Message) -> None:
        if not message.from_user or message.from_user.id != allowed_user_id:
            bot.reply_to(message, "Доступ ограничен")
            return
        document = message.document
        filename = document.file_name or "addon.zip"
        if Path(filename).suffix.lower() not in ADDON_EXTENSIONS:
            bot.reply_to(message, "Поддерживаются файлы .mcaddon, .mcpack и .zip.")
            return

        def fetch_from_telegram(update_status: Callable[[str], None]) -> Path:
            update_status("📥 Скачиваю архив аддона из Telegram...")
            try:
                telegram_file = bot.get_file(document.file_id)
                archive_bytes = bot.download_file(telegram_file.file_path)
            except Exception as exc:
                if "file is too big" in str(exc).lower():
                    raise ValueError(
                        "Telegram не позволяет боту скачать файл крупнее 20 МБ. "
                        "Загрузите архив куда-нибудь (Google Drive, Яндекс.Диск и т.п.) "
                        "и пришлите мне прямую ссылку на скачивание — я скачаю его сам."
                    ) from exc
                raise
            directory = Path(tempfile.mkdtemp(prefix="minecraft-upload-"))
            archive_path = directory / filename
            archive_path.write_bytes(archive_bytes)
            return archive_path

        install_addon_from_source(message.chat.id, filename, fetch_from_telegram)

    @bot.message_handler(content_types=["text"])
    def handle_message(message: telebot.types.Message) -> None:
        nonlocal busy, awaiting_sftp_settings, awaiting_groq_keys
        if not message.from_user or message.from_user.id != allowed_user_id:
            bot.reply_to(message, "Доступ ограничен")
            return

        request = (message.text or "").strip()
        if not request:
            return
        if awaiting_groq_keys:
            keys = [part.strip() for part in re.split(r"[,\n]+", request) if part.strip()]
            if not keys:
                bot.reply_to(message, "Не нашёл ни одного ключа в сообщении. Попробуйте ещё раз.")
                return
            update_runtime_config(groq_api_keys=keys)
            awaiting_groq_keys = False
            bot.reply_to(
                message,
                f"✅ Сохранено ключей: {len(keys)}. Буду использовать их по порядку, "
                "переключаясь на следующий при исчерпании лимита.",
                reply_markup=sftp_keyboard(),
            )
            return
        if awaiting_sftp_settings:
            parts = request.split()
            if len(parts) == 1 and ":" in parts[0]:
                host, port_text = parts[0].rsplit(":", 1)
            elif len(parts) == 2:
                host, port_text = parts
            else:
                bot.reply_to(
                    message,
                    "Формат не распознан. Используйте: <code>host port</code>",
                    parse_mode="HTML",
                )
                return
            try:
                port = int(port_text)
                if not host or not 1 <= port <= 65535:
                    raise ValueError
            except ValueError:
                bot.reply_to(message, "Укажите корректные host и port (1–65535).")
                return
            update_runtime_config(host=host, port=port, addon_paths=None)
            awaiting_sftp_settings = False
            bot.reply_to(
                message,
                "✅ Параметры SFTP сохранены. Новые значения будут использованы "
                "при следующем запросе.",
                reply_markup=sftp_keyboard(),
            )
            return

        addon_url = extract_first_url(request)
        if addon_url:
            def fetch_from_url(update_status: Callable[[str], None]) -> Path:
                return download_archive_from_url(addon_url, update_status)

            install_addon_from_source(message.chat.id, addon_url, fetch_from_url)
            return

        with user_lock:
            if busy:
                bot.reply_to(message, "⏳ Предыдущая задача ещё выполняется. Попробуйте через несколько секунд.")
                return
            busy = True

        status_message = bot.send_message(
            message.chat.id, format_status(request, "Анализирую файлы сервера...")
        )
        history: list[str] = []

        def update_status(status: str) -> None:
            if status not in history:
                history.append(status)
            try:
                bot.edit_message_text(
                    format_status(request, status, history),
                    message.chat.id,
                    status_message.message_id,
                    parse_mode="HTML",
                )
            except Exception:
                logger.exception("Не удалось обновить статусное сообщение")

        try:
            report, limit_hit = run_ai_task(request, update_status)
            remember_request(request, report)
            if limit_hit:
                final_text = (
                    "⚠️ Превышен лимит шагов ИИ (8 шагов). "
                    "Действие остановлено для безопасности!"
                )
                update_status(final_text)
            else:
                update_status("✅ Успешно выполнено!")
                bot.edit_message_text(
                    format_status(request, "✅ Успешно выполнено!", history)
                    + f"\n\n📋 <b>Отчет:</b>\n{html.escape(report)}",
                    message.chat.id,
                    status_message.message_id,
                    parse_mode="HTML",
                )
        except Exception as exc:
            logger.exception("Задача завершилась ошибкой")
            update_status(f"❌ Ошибка: {exc}")
        finally:
            with user_lock:
                busy = False

    logger.info("Minecraft AI Admin Bot запущен")
    bot.infinity_polling(
        skip_pending=True, allowed_updates=["message", "callback_query"]
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error("Бот не запущен: %s", exc)
        raise
