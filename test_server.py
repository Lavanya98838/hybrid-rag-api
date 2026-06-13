import asyncio
from aiohttp import web
import time

async def handler(request):
    return web.Response(text='Hello')

async def main():
    app = web.Application()
    app.router.add_get('/', handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8022)
    await site.start()
    print('Server started on 0.0.0.0:8022', flush=True)
    # Keep running
    while True:
        await asyncio.sleep(1)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Shutting down')