"""
Crypto Sentiment & Whale-Tracker Dashboard
==========================================
Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000

Free APIs used (no key required):
  - Binance WebSocket: wss://stream.binance.com:9443/ws/...
  - CoinGecko REST:    https://api.coingecko.com/api/v3/...
  - CoinTelegraph RSS: https://cointelegraph.com/rss
  - Newsdata.io REST:  https://newsdata.io/api/1/news (free tier, sign-up optional)

Premium key slots (plug in to go live):
  WHALE_ALERT_API_KEY  -> whale-alert.io production endpoint
  NEWSDATA_API_KEY     -> newsdata.io (free tier: 200 req/day)
"""

import asyncio
import json
import logging
import random
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

import feedparser
import httpx
import websockets
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

# ─────────────────────────────────────────────
#  API KEY CONFIGURATION
#  Drop your keys here to unlock live feeds
# ─────────────────────────────────────────────
WHALE_ALERT_API_KEY: Optional[str] = None
# Production whale-alert endpoint (activate by setting key above):
# GET https://api.whale-alert.io/v1/transactions?api_key={WHALE_ALERT_API_KEY}
#     &min_value=500000&start={unix_timestamp}&cursor={cursor}

NEWSDATA_API_KEY: Optional[str] = None
# Production newsdata.io endpoint (free tier: 200 req/day after signup):
# GET https://newsdata.io/api/1/news?apikey={NEWSDATA_API_KEY}
#     &q=crypto+bitcoin&language=en&category=technology

# ─────────────────────────────────────────────
#  RUNTIME CONFIG
# ─────────────────────────────────────────────
COINGECKO_URL = (
    "https://api.coingecko.com/api/v3/simple/price"
    "?ids=bitcoin,ethereum,solana&vs_currencies=usd"
    "&include_24hr_change=true&include_market_cap=true"
)
BINANCE_WS_URL = (
    "wss://stream.binance.com:9443/ws/"
    "btcusdt@ticker/ethusdt@ticker/solusdt@ticker"
)
COINTELEGRAPH_RSS = "https://cointelegraph.com/rss"
NEWSDATA_FREE_URL = (
    "https://newsdata.io/api/1/news"
    "?category=technology&q=crypto+bitcoin&language=en"
)

PRICE_POLL_INTERVAL = 30        # seconds between CoinGecko polls (respect rate limit)
RSS_POLL_INTERVAL   = 180       # seconds between RSS fetches
MOCK_WHALE_INTERVAL = 8         # seconds between simulated whale alerts
MOCK_SENTIMENT_INTERVAL = 5     # seconds between simulated social posts

MAX_WHALE_HISTORY   = 50
MAX_PRICE_HISTORY   = 120
MAX_SENTIMENT_POSTS = 60

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("crypto_dashboard")

# ─────────────────────────────────────────────
#  IN-MEMORY CACHE  (thread-safe via asyncio lock)
# ─────────────────────────────────────────────
cache_lock = asyncio.Lock()

price_cache: dict = {
    "bitcoin":  {"usd": 0, "usd_24h_change": 0, "usd_market_cap": 0},
    "ethereum": {"usd": 0, "usd_24h_change": 0, "usd_market_cap": 0},
    "solana":   {"usd": 0, "usd_24h_change": 0, "usd_market_cap": 0},
}

price_history: deque = deque(maxlen=MAX_PRICE_HISTORY)   # [{ts, btc, eth, sol}]
whale_history: deque = deque(maxlen=MAX_WHALE_HISTORY)   # [whale_event dicts]
sentiment_posts: deque = deque(maxlen=MAX_SENTIMENT_POSTS)
sentiment_score: float = 0.0   # rolling average: -1.0 to +1.0
sentiment_history: deque = deque(maxlen=MAX_PRICE_HISTORY)  # [{ts, score}]

# SSE subscriber queues
sse_subscribers: list[asyncio.Queue] = []

# ─────────────────────────────────────────────
#  VADER SENTIMENT ENGINE
# ─────────────────────────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    # Add crypto-specific lexicon boosters
    _crypto_boosters = {
        "moon": 2.5, "moonshot": 3.0, "rug": -3.0, "rugpull": -3.5,
        "pump": 1.5, "dump": -2.0, "bullish": 2.0, "bearish": -2.0,
        "rekt": -3.0, "hodl": 1.5, "whale": 0.5, "scam": -3.0,
        "hack": -2.5, "exploit": -2.5, "liquidation": -1.5,
        "ath": 2.5, "breakout": 2.0, "crash": -2.5, "recovery": 1.5,
        "adoption": 2.0, "fud": -2.0, "fomo": 1.5, "dip": -1.0,
        "accumulate": 1.5, "sell": -0.5, "buy": 0.8, "shill": -1.0,
    }
    _vader.lexicon.update(_crypto_boosters)
    USE_VADER = True
    log.info("VADER sentiment engine loaded with crypto lexicon.")
except ImportError:
    USE_VADER = False
    log.warning("vaderSentiment not installed. Using fallback regex scorer.")


def analyze_sentiment(text: str) -> float:
    """Returns score between -1.0 (extreme fear) and +1.0 (extreme greed)."""
    if USE_VADER:
        scores = _vader.polarity_scores(text)
        return round(scores["compound"], 4)

    # Fallback: simple keyword scoring
    text_l = text.lower()
    positive = ["bull", "moon", "pump", "buy", "ath", "breakout", "adoption", "rally", "gain", "up", "green"]
    negative = ["bear", "crash", "dump", "sell", "hack", "scam", "rug", "rekt", "liquidation", "fear", "down", "red"]
    score = sum(1 for w in positive if w in text_l) - sum(1 for w in negative if w in text_l)
    return max(-1.0, min(1.0, score * 0.2))


# ─────────────────────────────────────────────
#  BROADCAST HELPER
# ─────────────────────────────────────────────
async def broadcast(event_type: str, data: dict):
    """Push SSE event to all connected frontend clients."""
    payload = json.dumps({"type": event_type, "data": data, "ts": int(time.time() * 1000)})
    dead = []
    for q in sse_subscribers:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        sse_subscribers.remove(q)


# ─────────────────────────────────────────────
#  PIPELINE 1: BINANCE WEBSOCKET (real-time prices)
# ─────────────────────────────────────────────
_BINANCE_SYMBOL_MAP = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "SOLUSDT": "solana",
}

async def binance_ws_pipeline():
    """
    Connects to Binance public WebSocket stream.
    No API key required. Handles reconnection automatically.
    """
    backoff = 2
    while True:
        try:
            log.info("Connecting to Binance WebSocket...")
            async with websockets.connect(BINANCE_WS_URL, ping_interval=20) as ws:
                backoff = 2
                log.info("Binance WebSocket connected.")
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                        symbol = msg.get("s", "")
                        coin = _BINANCE_SYMBOL_MAP.get(symbol)
                        if not coin:
                            continue

                        price = float(msg.get("c", 0))         # current price
                        change_24h = float(msg.get("P", 0))    # 24h % change
                        volume = float(msg.get("v", 0))        # base volume

                        async with cache_lock:
                            price_cache[coin]["usd"] = price
                            price_cache[coin]["usd_24h_change"] = change_24h

                        await broadcast("price_tick", {
                            "coin": coin,
                            "price": price,
                            "change_24h": change_24h,
                            "volume": volume,
                        })
                    except (KeyError, ValueError, json.JSONDecodeError):
                        continue

        except (websockets.ConnectionClosed, ConnectionRefusedError, OSError) as e:
            log.warning(f"Binance WS disconnected: {e}. Reconnecting in {backoff}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)
        except Exception as e:
            log.error(f"Binance WS unexpected error: {e}")
            await asyncio.sleep(backoff)


# ─────────────────────────────────────────────
#  PIPELINE 1b: COINGECKO FALLBACK + HISTORY SNAPSHOTS
# ─────────────────────────────────────────────
async def coingecko_snapshot_pipeline():
    """
    Polls CoinGecko every PRICE_POLL_INTERVAL seconds.
    Fills price_history for the chart and provides market cap data.
    Also acts as fallback if Binance WS fails.
    """
    async with httpx.AsyncClient(timeout=15) as client:
        while True:
            try:
                resp = await client.get(COINGECKO_URL)
                if resp.status_code == 200:
                    data = resp.json()
                    snapshot = {"ts": int(time.time() * 1000)}
                    async with cache_lock:
                        for coin_id in ["bitcoin", "ethereum", "solana"]:
                            if coin_id in data:
                                coin_data = data[coin_id]
                                # Only override if Binance hasn't populated (price==0)
                                if price_cache[coin_id]["usd"] == 0:
                                    price_cache[coin_id]["usd"] = coin_data.get("usd", 0)
                                    price_cache[coin_id]["usd_24h_change"] = coin_data.get("usd_24h_change", 0)
                                price_cache[coin_id]["usd_market_cap"] = coin_data.get("usd_market_cap", 0)

                        snapshot["btc"] = price_cache["bitcoin"]["usd"]
                        snapshot["eth"] = price_cache["ethereum"]["usd"]
                        snapshot["sol"] = price_cache["solana"]["usd"]
                        snapshot["sentiment"] = sentiment_score
                        price_history.append(snapshot)
                        sentiment_history.append({"ts": snapshot["ts"], "score": sentiment_score})

                    await broadcast("price_snapshot", snapshot)
                    log.info(f"CoinGecko snapshot: BTC=${snapshot['btc']:,.0f}")
                elif resp.status_code == 429:
                    log.warning("CoinGecko rate limited. Backing off 60s.")
                    await asyncio.sleep(60)
                    continue

            except httpx.RequestError as e:
                log.warning(f"CoinGecko request failed: {e}")

            await asyncio.sleep(PRICE_POLL_INTERVAL)


# ─────────────────────────────────────────────
#  PIPELINE 2: WHALE TRACKER
#  Mock by default. Set WHALE_ALERT_API_KEY to go live.
# ─────────────────────────────────────────────
_WHALE_MOCK_COINS = ["BTC", "ETH", "SOL", "USDT", "USDC", "BNB", "XRP"]
_WHALE_MOCK_CHAINS = ["Bitcoin", "Ethereum", "Solana", "BSC", "Tron"]
_WHALE_MOCK_EXCHANGES = [
    "Binance", "Coinbase", "Kraken", "OKX", "Bybit",
    "Unknown Wallet", "Cold Storage", "DeFi Protocol",
]

def _mock_address(chain: str) -> str:
    prefixes = {"Bitcoin": "bc1q", "Ethereum": "0x", "Solana": "", "BSC": "0x", "Tron": "T"}
    p = prefixes.get(chain, "0x")
    return p + uuid.uuid4().hex[:10] + "..." + uuid.uuid4().hex[:6]

def _generate_mock_whale() -> dict:
    coin = random.choice(_WHALE_MOCK_COINS)
    chain = random.choice(_WHALE_MOCK_CHAINS)
    usd_value = random.randint(500_000, 80_000_000)
    # Calculate coin amount based on rough price estimates
    price_est = {"BTC": 95000, "ETH": 3500, "SOL": 180, "USDT": 1, "USDC": 1, "BNB": 600, "XRP": 0.6}
    amount = usd_value / price_est.get(coin, 1)
    src_is_exchange = random.random() > 0.5
    return {
        "id": str(uuid.uuid4()),
        "ts": int(time.time() * 1000),
        "symbol": coin,
        "chain": chain,
        "amount": round(amount, 4),
        "usd_value": usd_value,
        "from_addr": random.choice(_WHALE_MOCK_EXCHANGES) if src_is_exchange else _mock_address(chain),
        "to_addr": _mock_address(chain) if src_is_exchange else random.choice(_WHALE_MOCK_EXCHANGES),
        "tx_hash": "0x" + uuid.uuid4().hex + uuid.uuid4().hex[:24],
        "from_is_exchange": src_is_exchange,
        "to_is_exchange": not src_is_exchange,
    }


async def whale_alert_pipeline():
    """
    Streams whale movement events.

    LIVE MODE: Set WHALE_ALERT_API_KEY at top of file.
    The production endpoint is:
      GET https://api.whale-alert.io/v1/transactions
          ?api_key={WHALE_ALERT_API_KEY}
          &min_value=500000
          &start={int(time.time()) - 3600}
          &cursor=0
    Returns JSON: {"result":"success","cursor":"...","count":N,"transactions":[...]}
    Each transaction has: blockchain, symbol, id, transaction_type,
    hash, from{address,owner_type,owner}, to{...}, amount, amount_usd, timestamp

    MOCK MODE (default): Generates realistic whale events locally.
    """
    global whale_history

    if WHALE_ALERT_API_KEY:
        # ── LIVE whale-alert.io path ──────────────────────────────
        log.info("Whale Alert: LIVE mode via whale-alert.io")
        cursor = 0
        async with httpx.AsyncClient(timeout=15) as client:
            while True:
                try:
                    url = (
                        f"https://api.whale-alert.io/v1/transactions"
                        f"?api_key={WHALE_ALERT_API_KEY}"
                        f"&min_value=500000"
                        f"&start={int(time.time()) - 300}"
                        f"&cursor={cursor}"
                    )
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        body = resp.json()
                        for tx in body.get("transactions", []):
                            event = {
                                "id": tx.get("id", str(uuid.uuid4())),
                                "ts": tx.get("timestamp", int(time.time())) * 1000,
                                "symbol": tx.get("symbol", "?").upper(),
                                "chain": tx.get("blockchain", "?"),
                                "amount": tx.get("amount", 0),
                                "usd_value": tx.get("amount_usd", 0),
                                "from_addr": tx.get("from", {}).get("owner", tx.get("from", {}).get("address", "?")),
                                "to_addr": tx.get("to", {}).get("owner", tx.get("to", {}).get("address", "?")),
                                "tx_hash": tx.get("hash", "?"),
                                "from_is_exchange": tx.get("from", {}).get("owner_type") == "exchange",
                                "to_is_exchange": tx.get("to", {}).get("owner_type") == "exchange",
                            }
                            async with cache_lock:
                                whale_history.append(event)
                            await broadcast("whale_alert", event)
                        cursor = body.get("cursor", 0)
                    elif resp.status_code == 429:
                        await asyncio.sleep(60)
                        continue
                except httpx.RequestError as e:
                    log.warning(f"Whale Alert API error: {e}")
                await asyncio.sleep(60)  # Whale Alert free tier: 60s min interval

    else:
        # ── MOCK path ─────────────────────────────────────────────
        log.info("Whale Alert: MOCK mode (set WHALE_ALERT_API_KEY for live data)")
        # Pre-populate with 15 historical events so first-connect sees data
        for _ in range(15):
            event = _generate_mock_whale()
            event["ts"] = int(time.time() * 1000) - random.randint(60_000, 3_600_000)
            whale_history.append(event)

        while True:
            await asyncio.sleep(MOCK_WHALE_INTERVAL + random.uniform(-2, 4))
            event = _generate_mock_whale()
            async with cache_lock:
                whale_history.append(event)
            await broadcast("whale_alert", event)
            log.debug(f"Mock whale: {event['symbol']} ${event['usd_value']:,}")


# ─────────────────────────────────────────────
#  PIPELINE 3: SOCIAL SENTIMENT
#  RSS (CoinTelegraph) + Newsdata.io + Mock tweets
# ─────────────────────────────────────────────
_MOCK_ACCOUNTS = [
    "@CryptoWhale", "@WhalePump", "@SatoshiLegacy", "@BlockchainBull",
    "@DeFiDegen", "@AltcoinAlpha", "@NakamotoFan", "@Web3Insider",
    "@CoinSignals", "@BearTrap", "@MarketWatcher", "@CryptoTrader",
]
_MOCK_POSTS = [
    "BTC looking extremely bullish on the 4H chart! Targeting new ATH soon 🚀",
    "ETH gas fees are insane right now. Bear market incoming?",
    "SOL is absolutely destroying it. Layer 1 dominance shifting fast.",
    "This market is looking like a massive rug setup. Be careful out there.",
    "Just watched a $50M whale move from Binance to cold storage. Bullish signal.",
    "Fed rate decision tomorrow. Expect high volatility in crypto markets.",
    "DeFi TVL just hit new highs. Summer is here early 🔥",
    "USDT depeg rumors spreading on CT. Don't panic but stay alert.",
    "Bitcoin dominance at 58%. Altcoin season might be over for now.",
    "If you're not accumulating in this dip, you're going to regret it.",
    "Major exchange just got hacked. $30M in ETH drained. Markets tanking.",
    "Institutional buying pressure confirmed. BlackRock added 12,000 BTC this week.",
    "On-chain data shows 85% of BTC supply is in profit. Historically bearish sign.",
    "Layer 2 solutions hitting record transaction volumes. Ethereum scaling works.",
    "Stablecoin supply up 15% in 30 days. Dry powder ready to deploy.",
    "This rally feels fake. Whales distributing to retail again.",
    "BTC miner reserves at all-time lows. Miners not selling = bullish.",
    "SEC approved another spot ETF filing. We are so back 🚀",
    "Crypto fear & greed index at 78. Getting greedy out here.",
    "Short squeeze incoming on ETH. Funding rates extremely negative.",
]

async def rss_feed_pipeline():
    """Fetches CoinTelegraph RSS and computes sentiment on headlines."""
    global sentiment_score

    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        seen_links: set = set()
        while True:
            try:
                resp = await client.get(COINTELEGRAPH_RSS)
                if resp.status_code == 200:
                    feed = feedparser.parse(resp.text)
                    new_count = 0
                    scores = []
                    for entry in feed.entries[:15]:
                        link = entry.get("link", "")
                        if link in seen_links:
                            continue
                        seen_links.add(link)
                        title = entry.get("title", "")
                        summary = entry.get("summary", "")
                        text = f"{title}. {summary}"
                        score = analyze_sentiment(text)
                        scores.append(score)
                        post = {
                            "id": str(uuid.uuid4()),
                            "ts": int(time.time() * 1000),
                            "source": "CoinTelegraph",
                            "author": "CT News",
                            "text": title[:180],
                            "score": score,
                            "url": link,
                        }
                        async with cache_lock:
                            sentiment_posts.append(post)
                        await broadcast("sentiment_post", post)
                        new_count += 1

                    if scores:
                        avg = sum(scores) / len(scores)
                        async with cache_lock:
                            sentiment_score = round((sentiment_score * 0.7 + avg * 0.3), 4)
                        await broadcast("sentiment_update", {"score": sentiment_score})
                        log.info(f"RSS: {new_count} new articles, avg sentiment={avg:.3f}")

            except Exception as e:
                log.warning(f"RSS pipeline error: {e}")

            await asyncio.sleep(RSS_POLL_INTERVAL)


async def mock_social_pipeline():
    """
    Simulates live X/Twitter-style crypto commentary.

    To go live with real X API v2:
      POST https://api.twitter.com/2/tweets/search/stream/rules  (set your rules)
      GET  https://api.twitter.com/2/tweets/search/stream
      Headers: {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}
    Each streamed event is JSON with: data.id, data.text, data.created_at, author_id
    """
    global sentiment_score

    # Pre-populate 10 posts
    for _ in range(10):
        text = random.choice(_MOCK_POSTS)
        score = analyze_sentiment(text)
        post = {
            "id": str(uuid.uuid4()),
            "ts": int(time.time() * 1000) - random.randint(60000, 1800000),
            "source": "X (Twitter)",
            "author": random.choice(_MOCK_ACCOUNTS),
            "text": text,
            "score": score,
            "url": None,
        }
        sentiment_posts.append(post)

    while True:
        await asyncio.sleep(MOCK_SENTIMENT_INTERVAL + random.uniform(-1, 3))
        text = random.choice(_MOCK_POSTS)
        score = analyze_sentiment(text)
        post = {
            "id": str(uuid.uuid4()),
            "ts": int(time.time() * 1000),
            "source": random.choice(["X (Twitter)", "Reddit", "Telegram"]),
            "author": random.choice(_MOCK_ACCOUNTS),
            "text": text,
            "score": score,
            "url": None,
        }
        async with cache_lock:
            sentiment_posts.append(post)
            # Exponential moving average of sentiment
            sentiment_score = round(sentiment_score * 0.85 + score * 0.15, 4)

        await broadcast("sentiment_post", post)
        await broadcast("sentiment_update", {"score": sentiment_score})


# ─────────────────────────────────────────────
#  LIFESPAN — start all background tasks
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks = [
        asyncio.create_task(binance_ws_pipeline(), name="binance_ws"),
        asyncio.create_task(coingecko_snapshot_pipeline(), name="coingecko_snapshot"),
        asyncio.create_task(whale_alert_pipeline(), name="whale_alert"),
        asyncio.create_task(rss_feed_pipeline(), name="rss_feed"),
        asyncio.create_task(mock_social_pipeline(), name="mock_social"),
    ]
    log.info("All background pipelines started.")
    yield
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    log.info("Background pipelines stopped.")


# ─────────────────────────────────────────────
#  FASTAPI APP
# ─────────────────────────────────────────────
app = FastAPI(title="Crypto Sentinel", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/state")
async def get_state():
    """Returns the full cached state for new client connections."""
    async with cache_lock:
        return {
            "prices": dict(price_cache),
            "price_history": list(price_history),
            "whale_history": list(whale_history)[-20:],
            "sentiment_posts": list(sentiment_posts)[-20:],
            "sentiment_score": sentiment_score,
            "sentiment_history": list(sentiment_history),
            "server_time": int(time.time() * 1000),
        }


@app.get("/stream")
async def sse_stream(request: Request):
    """
    Server-Sent Events endpoint.
    Frontend connects here; all pipeline broadcasts are funneled to connected clients.
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    sse_subscribers.append(queue)

    async def event_generator() -> AsyncGenerator[str, None]:
        # Send initial ping
        yield f"data: {json.dumps({'type': 'connected', 'ts': int(time.time()*1000)})}\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=25.0)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    # Heartbeat to keep connection alive
                    yield f"data: {json.dumps({'type': 'heartbeat', 'ts': int(time.time()*1000)})}\n\n"
        finally:
            if queue in sse_subscribers:
                sse_subscribers.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
