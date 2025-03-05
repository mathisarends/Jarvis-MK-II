import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

async def main():
    browser_config = BrowserConfig(headless=True, java_script_enabled=True)  # Headless für Effizienz
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        css_selector=".markdown-body"
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url="https://github.com/bhancockio/deepseek-ai-web-crawler",
            config=run_config
        )
        
        print(result.markdown)

if __name__ == "__main__":
    asyncio.run(main())
