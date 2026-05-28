import pytest
from oprim.financial_metric_extraction import financial_metric_extraction, NewsItem
from oprim.policy_event_extraction import policy_event_extraction, PolicyNews, PolicyEvent
from oprim.industry_attribution import industry_attribution
from oprim.pattern_detection import pattern_detection, OHLCVInput

# --- Financial Metric Extraction Tests ---
def test_financial_extraction_normal():
    news = [NewsItem(content="该公司今年营业收入达到 100 亿，同比增长 20%。净利润为 10 亿。")]
    res = financial_metric_extraction(news=news)
    assert len(res) >= 2
    names = [r.metric_name for r in res]
    assert "revenue" in names
    assert "net_profit" in names
    # Check value normalization (100亿 -> 1e10)
    rev = next(r for r in res if r.metric_name == "revenue")
    assert rev.value == 100 * 1e8
    assert rev.sentiment_score > 0

def test_financial_extraction_empty():
    res = financial_metric_extraction(news=[])
    assert res == []

def test_financial_extraction_no_match():
    news = [NewsItem(content="天气不错。")]
    res = financial_metric_extraction(news=news)
    assert res == []

def test_financial_extraction_sentiment():
    news = [NewsItem(content="营收下降 10 亿，利空出现。")]
    res = financial_metric_extraction(news=news)
    assert res[0].sentiment_score < 0

# --- Policy Event Extraction Tests ---
def test_policy_extraction_normal():
    policies = [PolicyNews(content="央行今日宣布降准 0.5 个百分点，利好市场。")]
    res = policy_event_extraction(policies=policies)
    assert len(res) > 0
    assert res[0].event_type == "monetary"
    assert res[0].direction == "positive"

def test_policy_extraction_severity():
    policies = [PolicyNews(content="这是一项特大紧急通知：全国禁止某些交易。")]
    res = policy_event_extraction(policies=policies)
    assert res[0].severity == "critical"
    assert res[0].direction == "negative"

def test_policy_extraction_unknown():
    policies = [PolicyNews(content="普通新闻。")]
    res = policy_event_extraction(policies=policies)
    assert len(res) == 0

# --- Industry Attribution Tests ---
def test_industry_attribution_normal():
    events = [PolicyEvent(event_type="stimulus", severity="major", direction="positive", days_ago=0, source_excerpt="新能源行业迎来重大利好")]
    industry_map = {"新能源": "电力设备"}
    res = industry_attribution(events=events, industry=industry_map)
    assert len(res) == 1
    assert res[0].industry == "电力设备"
    assert res[0].impact_direction == "positive"

def test_industry_attribution_multi():
    events = [PolicyEvent(event_type="regulation", severity="moderate", direction="negative", days_ago=1, source_excerpt="加强房市与金融监管")]
    industry_map = {"房市": "房地产", "金融": "银行"}
    res = industry_attribution(events=events, industry=industry_map)
    assert len(res) == 2
    industries = [r.industry for r in res]
    assert "房地产" in industries
    assert "银行" in industries

def test_industry_attribution_no_match():
    events = [PolicyEvent(event_type="stimulus", severity="major", direction="positive", days_ago=0, source_excerpt="某某行业利好")]
    industry_map = {"新能源": "电力设备"}
    res = industry_attribution(events=events, industry=industry_map)
    assert len(res) == 0

# --- Pattern Detection Tests ---
def test_pattern_detection_hammer():
    # Hammer: body small, lower shadow long, upper shadow short
    ohlcv = OHLCVInput(
        open=[10.0],
        high=[10.02],
        low=[9.0],
        close=[9.8],
        volume=[1000]
    )
    res = pattern_detection(ohlcv=ohlcv)
    assert len(res) == 1
    assert res[0].name == "hammer"

def test_pattern_detection_engulfing():
    # Bullish engulfing
    ohlcv = OHLCVInput(
        open=[10, 9.5],
        high=[10.1, 10.6],
        low=[9.4, 9.4],
        close=[9.5, 10.5],
        volume=[1000, 1200]
    )
    res = pattern_detection(ohlcv=ohlcv)
    # Filter for engulfing
    eng = [r for r in res if r.name == "bullish_engulfing"]
    assert len(eng) == 1
    assert eng[0].bullish_score > 0

def test_pattern_detection_none():
    ohlcv = OHLCVInput(open=[10], high=[11], low=[9], close=[10.5], volume=[100])
    res = pattern_detection(ohlcv=ohlcv)
    # Not a hammer (upper shadow 0.5, body 0.5, lower shadow 1.0. Lower shadow is only 2x body, but upper shadow is 1x body > 0.2x)
    assert len(res) == 0
