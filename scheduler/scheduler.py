from __future__ import annotations

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from db.mongo import init_mongo
from pipeline.embed import run_embed
from pipeline.ingest import run_ingest


async def run_daily_news_pipeline() -> None:
    await init_mongo()
    await run_ingest(limit=50)
    await run_embed(batch_size=64)


def start_scheduler() -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        run_daily_news_pipeline,
        "cron",
        hour=3,
        minute=0,
        id="daily-news-ingest",
        replace_existing=True,
    )
    scheduler.start()
    return scheduler


if __name__ == "__main__":
    start_scheduler()
    print("Scheduler started. Press Ctrl+C to stop.")
