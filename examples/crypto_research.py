from starforge import run


result = run(
    objective="analyze BTC trend and summarize key signals",
    context={
        "working_dir": ".",
        "constraints": ["summarize only from gathered evidence"],
        "api_requests": [
            {
                "url": "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
                "params": {
                    "vs_currency": "usd",
                    "days": "7",
                    "interval": "daily",
                },
            }
        ],
        "output_path": "btc_trend_summary.md",
    },
    config={
        "adapter": "api",
        "max_steps": 4,
        "mode": "autonomous",
    },
)

print(result)
