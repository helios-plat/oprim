"""oprim._content_fetcher — 多源内容抓取 + 付费墙绕过引擎 (3O 内化 qiaomu 策略).

铁律: 仅做网络请求 + 内容提取, 不写文件/不操作数据库.
集成方式: oprim.fetch_url_with_bypass(url, strategy="auto")

策略层级(来自 qiaomu/fetch_url.sh + Bypass Paywalls Clean):
  0. r.jina.ai / defuddle.md (代理提取服务)
  1. Bot UA 绕过 (Googlebot/Bingbot + X-Forwarded-For, ~50 站点)
  2. Referer 伪装 (Google/Facebook/Twitter)
  3. AMP 页面提取 (部分站点 AMP 墙较弱)
  4. archive.today 存档
  5. Google Web Cache
  6. agent-fetch (本地 npx 回退)

源适配:
  - 微信公众号 (mp.weixin.qq.com) — jina.ai 代理抓取 (无需浏览器)
  - X/Twitter (x.com/twitter.com) — jina.ai/agent-fetch 级联
  - 付费墙 (nytimes/wsj/ft/economist/bloomberg/medium...) — Bot UA + AMP + archive 级联

⚠ 付费墙绕过为知识管线的灰色地带
仅供个人学习研究, 不作商业分发.
"""

from __future__ import annotations

import contextlib
import logging
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from obase.http.dns_pinned_transport import make_ssrf_safe_opener

log = logging.getLogger(__name__)

# ── 付费墙域名策略列表 (来自 BPC) ──────────────────────────────────

_GOOGLEBOT_DOMAINS = frozenset(
    (
        "wsj.com",
        "barrons.com",
        "ft.com",
        "economist.com",
        "theaustralian.com.au",
        "thetimes.co.uk",
        "telegraph.co.uk",
        "zeit.de",
        "handelsblatt.com",
        "leparisien.fr",
        "nzz.ch",
        "usatoday.com",
        "quora.com",
        "lefigaro.fr",
        "lemonde.fr",
        "spiegel.de",
        "sueddeutsche.de",
        "frankfurter-allgemeine.de",
        "wires.com",
        "brisbanetimes.com.au",
        "smh.com.au",
        "theage.com.au",
    )
)

_BINGBOT_DOMAINS = frozenset(
    (
        "haaretz.com",
        "nzherald.co.nz",
        "stratfor.com",
        "themarker.com",
    )
)

_FB_REF_DOMAINS = frozenset(
    (
        "law.com",
        "ftm.nl",
        "law360.com",
        "sloanreview.mit.edu",
    )
)

_AMP_DOMAINS = frozenset(
    (
        "wsj.com",
        "bostonglobe.com",
        "latimes.com",
        "chicagotribune.com",
        "seattletimes.com",
        "theatlantic.com",
        "wired.com",
        "newyorker.com",
        "washingtonpost.com",
        "smh.com.au",
        "theage.com.au",
        "brisbanetimes.com.au",
    )
)

_PAYWALL_DOMAINS = frozenset(
    (
        "nytimes.com",
        "wsj.com",
        "ft.com",
        "economist.com",
        "bloomberg.com",
        "washingtonpost.com",
        "newyorker.com",
        "wired.com",
        "theatlantic.com",
        "medium.com",
        "businessinsider.com",
        "technologyreview.com",
        "scmp.com",
        "seattletimes.com",
        "bostonglobe.com",
        "latimes.com",
        "chicagotribune.com",
        "theglobeandmail.com",
        "afr.com",
        "thetimes.co.uk",
        "telegraph.co.uk",
        "spiegel.de",
        "zeit.de",
        "sueddeutsche.de",
        "barrons.com",
        "forbes.com",
        "foreignaffairs.com",
        "foreignpolicy.com",
        "harvard.edu",
        "newscientist.com",
        "scientificamerican.com",
        "theinformation.com",
        "statista.com",
        "handelsblatt.com",
        "nzz.ch",
        "leparisien.fr",
        "lefigaro.fr",
        "lemonde.fr",
        "haaretz.com",
        "nzherald.co.nz",
        "theaustralian.com.au",
        "smh.com.au",
        "theage.com.au",
        "quora.com",
        "usatoday.com",
    )
)

_WECHAT_DOMAINS = frozenset(("mp.weixin.qq.com",))
_TWITTER_DOMAINS = frozenset(("x.com", "twitter.com", "t.co"))

_TIMEOUT = 15
_MAX_BYTES = 5 * 1024 * 1024
_STEALTH_TIMEOUT = 30  # Crawl4AI stealth 模式超时


# ── 工具函数 ────────────────────────────────────────────────────────


def _domain_of(url: str) -> str:
    """Extract hostname without port."""
    parsed = urllib.parse.urlparse(url)
    return (parsed.hostname or "").lower()


def _matches(url: str, domains: frozenset[str]) -> bool:
    host = _domain_of(url)
    return any(host == d or host.endswith("." + d) for d in domains)


def _has_content(text: str) -> bool:
    if not text:
        return False
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        return False
    if len(text) < 500:
        return False
    # 过滤常见墙/错误页
    deny = (
        "Don't miss what's happening",
        "Access Denied",
        "404 Not Found",
        "403 Forbidden",
        "subscribe to",
        "paywall",
        "premium_content",
        "remaining free articles",
        "to continue reading",
    )
    tlc = text.lower()
    return not any(w in tlc for w in deny)


def _extract_jsonld_article(html: str) -> str | None:
    """从 JSON-LD 提取完整 articleBody (BPC 策略 #6)."""
    m = re.search(r'"articleBody"\s*:\s*"((?:[^"\\]|\\.)*)"', html)
    if not m:
        return None
    raw = m.group(1)
    if len(raw) < 200:
        return None
    # 反转义
    raw = raw.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
    return raw


def _extract_title(html: str) -> str:
    m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.I)
    return m.group(1).strip() if m else ""


def _html_to_text(html: str) -> str:
    """简易 HTML→纯文本 (strip tags + decode entities)."""
    t = html
    t = re.sub(r"<script[^>]*>.*?</script>", "", t, flags=re.I | re.S)
    t = re.sub(r"<style[^>]*>.*?</style>", "", t, flags=re.I | re.S)
    t = re.sub(r"<nav[^>]*>.*?</nav>", "", t, flags=re.I | re.S)
    t = re.sub(r"<footer[^>]*>.*?</footer>", "", t, flags=re.I | re.S)
    t = re.sub(r"<header[^>]*>.*?</header>", "", t, flags=re.I | re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    for ent, ch in (
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", '"'),
        ("&#39;", "'"),
        ("&nbsp;", " "),
    ):
        t = t.replace(ent, ch)
    # 去空白行
    lines = [line.strip() for line in t.splitlines()]
    return "\n".join(line for line in lines if line)


def _do_fetch(
    url: str, headers: dict[str, str] | None = None, proxy: str | None = None
) -> str | None:
    """SSRF 安全抓取, 返回 body_text 或 None."""
    if proxy:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"https": proxy, "http": proxy})
        )
    else:
        try:
            opener = make_ssrf_safe_opener(timeout=_TIMEOUT)
        except Exception:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        req = urllib.request.Request(url, headers=headers or {})
        req.add_header(
            "User-Agent",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36",
        )
        with opener.open(req, timeout=_TIMEOUT) as resp:
            body = resp.read(_MAX_BYTES)
            return body.decode("utf-8", errors="replace")
    except Exception:
        return None


# ── 策略层 ──────────────────────────────────────────────────────────


def _try_proxy_service(url: str) -> dict[str, Any] | None:
    """1a. r.jina.ai / 1b. defuddle.md 代理提取服务."""
    for prefix in (f"https://r.jina.ai/{url}", f"https://defuddle.md/{url}"):
        body = _do_fetch(prefix)
        if body and _has_content(body):
            return {
                "strategy": "proxy_service",
                "title": _extract_title(body),
                "content": body,
                "source_url": url,
            }
    return None


def _try_bot_ua(url: str) -> dict[str, Any] | None:
    """2a/2b. Googlebot / Bingbot UA 绕过 (BPC 核心策略)."""
    bots = []
    if _matches(url, _GOOGLEBOT_DOMAINS) or _matches(url, _PAYWALL_DOMAINS):
        bots.append(
            {
                "ua": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
                "xfi": "66.249.66.1",
                "ref": "https://www.google.com/",
            }
        )
    if _matches(url, _BINGBOT_DOMAINS) or _matches(url, _PAYWALL_DOMAINS):
        bots.append(
            {
                "ua": "Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)",
                "ref": "https://www.bing.com/",
            }
        )
    for bot in bots:
        hdrs = {
            "User-Agent": bot["ua"],
            "Referer": bot["ref"],
            "Accept": "text/html,application/xhtml+xml",
        }
        if "xfi" in bot:
            hdrs["X-Forwarded-For"] = bot["xfi"]
        body = _do_fetch(url, headers=hdrs)
        if not body:
            continue
        # JSON-LD articleBody 提取优先
        article = _extract_jsonld_article(body)
        if article and _has_content(article):
            return {
                "strategy": "bot_ua_jsonld",
                "title": _extract_title(body),
                "content": f"# {_extract_title(body)}\n\nSource: {url}\n\n{article}",
                "source_url": url,
            }
        text = _html_to_text(body)
        if _has_content(text):
            return {
                "strategy": "bot_ua_html",
                "title": _extract_title(body),
                "content": f"# {_extract_title(body)}\n\nSource: {url}\n\n{text}",
                "source_url": url,
            }
    return None


def _try_referer_spoof(url: str) -> dict[str, Any] | None:
    """3. Referer 伪装 (Google/Facebook/Twitter)."""
    referrers = []
    if _matches(url, _FB_REF_DOMAINS):
        referrers.append("https://www.facebook.com/")
    referrers.append("https://www.google.com/")
    referrers.append("https://t.co/")
    for ref in referrers:
        hdrs = {"Referer": ref, "Accept": "text/html,application/xhtml+xml"}
        body = _do_fetch(url, headers=hdrs)
        if body and _has_content(body):
            text = _html_to_text(body)
            if _has_content(text):
                return {
                    "strategy": "referer_spoof",
                    "title": _extract_title(body),
                    "content": f"# {_extract_title(body)}\n\nSource: {url}\n\n{text}",
                    "source_url": url,
                }
    return None


def _try_amp(url: str) -> dict[str, Any] | None:
    """3e. AMP 页面提取."""
    if not _matches(url, _AMP_DOMAINS):
        return None
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.rstrip("/")
    suffixes = ["/amp", "?outputType=amp", ".amp.html", "?amp"]
    # .html → .amp.html 模式
    if path.endswith(".html") and not path.endswith(".amp.html"):
        suffixes.append(path[:-5] + ".amp.html")
    for suf in suffixes:
        amp = urllib.parse.urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                path + suf if suf.startswith("/") else path,
                parsed.params,
                suf if suf.startswith("?") else parsed.query,
                parsed.fragment,
            )
        )
        body = _do_fetch(amp)
        if not body:
            continue
        article = _extract_jsonld_article(body)
        if article and _has_content(article):
            return {
                "strategy": "amp_jsonld",
                "title": _extract_title(body),
                "content": f"# {_extract_title(body)}\n\nSource: {url}\n\n{article}",
                "source_url": url,
            }
        text = _html_to_text(body)
        if _has_content(text):
            return {
                "strategy": "amp_html",
                "title": _extract_title(body),
                "content": f"# {_extract_title(body)}\n\nSource: {url}\n\n{text}",
                "source_url": url,
            }
    return None


def _try_archive(url: str) -> dict[str, Any] | None:
    """4. archive.today 存档."""
    archive_url = f"https://archive.today/newest/{url}"
    body = _do_fetch(archive_url)
    if body and _has_content(body):
        text = _html_to_text(body)
        if _has_content(text):
            return {
                "strategy": "archive",
                "title": _extract_title(body),
                "content": (
                    f"# {_extract_title(body)}\n\n"
                    f"Source: {url} (via archive.today)\n\n{text}"
                ),
                "source_url": url,
            }
    return None


def _try_google_cache(url: str) -> dict[str, Any] | None:
    """5. Google Web Cache."""
    cache_url = f"https://webcache.googleusercontent.com/search?q=cache:{url}"
    body = _do_fetch(cache_url)
    if body and _has_content(body):
        text = _html_to_text(body)
        if _has_content(text):
            return {
                "strategy": "google_cache",
                "title": _extract_title(body),
                "content": (
                    f"# {_extract_title(body)}\n\n"
                    f"Source: {url} (via Google Cache)\n\n{text}"
                ),
                "source_url": url,
            }
    return None


def _try_crawl4ai_stealth(url: str) -> dict[str, Any] | None:
    """Crawl4AI stealth 模式抓取 (Playwright + 隐身指纹).

    用于: 高级付费墙/强反爬站点 (Wired/Medium/GitHub/Notion...)
    策略: enable_stealth=True + text_mode=True + 浏览器池
    """
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    except ImportError:
        log.debug("crawl4ai not installed, skipping stealth fetch")
        return None

    import asyncio

    async def _crawl():
        browser_cfg = BrowserConfig(
            headless=True,
            text_mode=True,  # 优化文本提取
            viewport_width=1920,  # 完整桌面视口
            viewport_height=1080,
            enable_stealth=True,  # 🔑 关键: playwright-stealth 隐身模式
        )
        run_cfg = CrawlerRunConfig(
            # 使用默认 wait_for (避免超时)
            page_timeout=_STEALTH_TIMEOUT * 1000,  # ms
            stream=False,
        )

        try:
            async with AsyncWebCrawler(config=browser_cfg) as crawler:
                result = await crawler.arun(url=url, config=run_cfg)

            # 提取 Markdown
            md = getattr(result, "markdown", "") or ""
            if isinstance(md, dict):
                md = md.get("raw_markdown", "") or md.get("markdown", "") or ""
            if not isinstance(md, str):
                md = ""

            if not result.success or not md:
                return None

            if not _has_content(md):
                return None

            # 提取标题
            title = ""
            with contextlib.suppress(Exception):
                title = getattr(result, "title", "") or ""

            return {
                "strategy": "crawl4ai_stealth",
                "title": title or _extract_title(md) or url,
                "content": md,
                "source_url": url,
                "content_type": "generic",
            }
        except Exception as e:
            log.debug("crawl4ai_stealth exception for %s: %s", url[:60], e)
            return None

    try:
        # 使用 asyncio.run() 创建新循环 (兼容 Python 3.10+)
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_crawl())
        finally:
            loop.close()
    except Exception as e:
        log.debug("crawl4ai_stealth loop error for %s: %s", url[:60], e)
        return None


# ── 源专用适配 ──────────────────────────────────────────────────────


# 增强 SSRF-safe 直取 (使用真实浏览器指纹)
def _do_fetch_enhanced(url: str) -> dict[str, Any] | None:
    """增强直取: 使用真实浏览器指纹链.

    作为最终回退, 比普通直取更隐蔽.
    """
    body = _do_fetch(url, stealth=True)
    if body and _has_content(body):
        text = _html_to_text(body)
        if _has_content(text):
            return {
                "success": True,
                "strategy": "enhanced_direct",
                "title": _extract_title(body),
                "content": f"# {_extract_title(body)}\n\nSource: {url}\n\n{text}",
                "source_url": url,
                "content_type": "generic",
                "error": None,
            }
    return None


def _fetch_wechat(url: str) -> dict[str, Any] | None:
    """微信公众号: jina.ai 代理 (无需浏览器, 绕过反爬)."""
    proxy_url = f"https://r.jina.ai/{url}"
    body = _do_fetch(proxy_url)
    if body and _has_content(body):
        return {
            "strategy": "wechat_jina",
            "title": _extract_title(body),
            "content": body,
            "source_url": url,
            "content_type": "wechat_article",
        }
    return None


def _fetch_twitter(url: str) -> dict[str, Any] | None:
    """X/Twitter 线程: jina.ai → agent-fetch 级联."""
    for prefix in (f"https://r.jina.ai/{url}", f"https://defuddle.md/{url}"):
        body = _do_fetch(prefix)
        if body and _has_content(body):
            return {
                "strategy": "twitter_proxy",
                "title": _extract_title(body),
                "content": body,
                "source_url": url,
                "content_type": "twitter_thread",
            }
    # agent-fetch 回退
    try:
        result = subprocess.run(
            ["npx", "--yes", "agent-fetch", url, "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.stdout and _has_content(result.stdout):
            return {
                "strategy": "twitter_agent_fetch",
                "title": "",
                "content": result.stdout,
                "source_url": url,
                "content_type": "twitter_thread",
            }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


# ── 主入口 ──────────────────────────────────────────────────────────


def fetch_url_with_bypass(
    url: str,
    *,
    strategy: str = "auto",
    proxy: str | None = None,
) -> dict[str, Any]:
    """Fetch URL with multi-tier paywall bypass.

    Args:
        url: Target URL
        strategy: "auto" (tier cascade) | "direct" (no bypass) | "wechat" | "twitter"
        proxy: Optional HTTP proxy for direct fetches

    Returns:
        {
            "success": bool,
            "strategy": str,           # 实际使用的策略名
            "title": str,              # 提取的标题
            "content": str,            # Markdown/文本内容
            "source_url": str,         # 原始 URL
            "content_type": str,       # "generic" | "wechat_article" | "twitter_thread"
            "error": str | None,       # 失败原因
        }
    """
    # 源专用路由
    if strategy == "wechat" or (strategy == "auto" and _matches(url, _WECHAT_DOMAINS)):
        result = _fetch_wechat(url)
        if result:
            return result
        return {
            "success": False,
            "strategy": "wechat",
            "error": "WeChat fetch failed",
            "title": "",
            "content": "",
            "source_url": url,
            "content_type": "wechat_article",
        }

    if strategy == "twitter" or (strategy == "auto" and _matches(url, _TWITTER_DOMAINS)):
        result = _fetch_twitter(url)
        if result:
            return result
        return {
            "success": False,
            "strategy": "twitter",
            "error": "Twitter fetch failed",
            "title": "",
            "content": "",
            "source_url": url,
            "content_type": "twitter_thread",
        }

    # stealth: 仅使用 Crawl4AI stealth 模式
    if strategy == "stealth":
        result = _try_crawl4ai_stealth(url)
        if result:
            result["success"] = True
            result["error"] = None
            return result
        return {
            "success": False,
            "strategy": "stealth",
            "error": "Crawl4AI stealth failed",
            "title": "",
            "content": "",
            "source_url": url,
            "content_type": "generic",
        }

    if strategy == "direct":
        body = _do_fetch(url)
        if body and _has_content(body):
            return {
                "success": True,
                "strategy": "direct",
                "title": _extract_title(body),
                "content": _html_to_text(body),
                "source_url": url,
                "content_type": "generic",
                "error": None,
            }
        return {
            "success": False,
            "strategy": "direct",
            "error": "Empty/no content",
            "title": "",
            "content": "",
            "source_url": url,
            "content_type": "generic",
        }

    # auto: tier cascade
    tiers = [
        ("proxy_service", _try_proxy_service),
        ("bot_ua", _try_bot_ua),
        ("referer_spoof", _try_referer_spoof),
        ("amp", _try_amp),
        ("archive", _try_archive),
        ("google_cache", _try_google_cache),
    ]
    for name, fn in tiers:
        try:
            result = fn(url)
            if result:
                result["success"] = True
                result["error"] = None
                result.setdefault("content_type", "generic")
                return result
        except Exception as e:
            log.debug("bypass tier %s failed for %s: %s", name, url[:80], e)
            continue

    # 最终回退: SSRF-safe 增强直取 (真实浏览器指纹)
    result = _do_fetch_enhanced(url)
    if result:
        return result

    # 终极回退: 普通直取
    body = _do_fetch(url)
    if body and _has_content(body):
        return {
            "success": True,
            "strategy": "fallback_direct",
            "title": _extract_title(body),
            "content": _html_to_text(body),
            "source_url": url,
            "content_type": "generic",
            "error": None,
        }
    return {
        "success": False,
        "strategy": "all_failed",
        "error": f"All fetch methods failed for {url}",
        "title": "",
        "content": "",
        "source_url": url,
        "content_type": "generic",
    }
