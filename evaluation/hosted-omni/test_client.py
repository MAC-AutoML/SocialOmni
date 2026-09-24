import json
import tempfile
import unittest
from pathlib import Path

from aiohttp import web
from client import StreamingClient


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.requests = 0
        self.finish = "stop"
        self.fail_first = False
        self.response_text = "hello"

        async def handler(request):
            self.requests += 1
            self.payload = await request.json()
            if self.fail_first and self.requests == 1:
                return web.Response(status=429)
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            events = [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": self.response_text},
                            "finish_reason": None,
                        }
                    ]
                },
                {"choices": [{"index": 0, "delta": {}, "finish_reason": self.finish}]},
                {"choices": [], "usage": {"total_tokens": 8}},
            ]
            body = (
                "".join("data: " + json.dumps(event) + "\n\n" for event in events)
                + "data: [DONE]\n\n"
            )
            for offset in range(0, len(body), 13):
                await response.write(body[offset : offset + 13].encode())
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        self.client = StreamingClient(
            "test-secret",
            self.folder.name,
            base_url=f"http://127.0.0.1:{self.port}/v1?version=1",
            max_attempts=2,
        )

    async def asyncTearDown(self):
        await self.runner.cleanup()
        self.folder.cleanup()

    async def test_stream_usage_and_resume(self):
        messages = [{"role": "user", "content": "hi"}]
        result = await self.client.complete(messages)
        self.assertEqual(result["text"], "hello")
        self.assertEqual(result["usage"], {"total_tokens": 8})
        self.assertEqual(await self.client.complete(messages), result)
        self.assertEqual(self.requests, 1)
        for path in Path(self.folder.name).rglob("*.json"):
            self.assertNotIn("test-secret", path.read_text())

    async def test_retry_keeps_both_attempts(self):
        self.fail_first = True
        await self.client.complete([])
        self.assertEqual(self.requests, 2)
        self.assertEqual(len(list(Path(self.folder.name).rglob("attempt-*.json"))), 2)

    async def test_text_only_interface_has_distinct_cache_identity(self):
        original = await self.client.complete([])
        self.assertEqual(self.payload["modalities"], ["text"])
        text_only = await self.client.complete([], include_modalities=False)
        self.assertNotIn("modalities", self.payload)
        self.assertNotEqual(original["request_sha256"], text_only["request_sha256"])
        await self.client.complete([], include_modalities=False)
        self.assertEqual(self.requests, 2)

    async def test_explicit_reasoning_setting_has_distinct_cache_identity(self):
        original = await self.client.complete([])
        self.assertNotIn("reasoning_effort", self.payload)
        explicit = await self.client.complete([], reasoning_effort="none")
        self.assertEqual(self.payload["reasoning_effort"], "none")
        self.assertNotEqual(original["request_sha256"], explicit["request_sha256"])

    async def test_truncation_not_cached_as_success(self):
        self.finish = "length"
        with self.assertRaises(RuntimeError):
            await self.client.complete([])
        self.assertFalse(list(Path(self.folder.name).rglob("result.json")))

    async def test_empty_candidate_is_preserved_but_judge_requires_text(self):
        self.response_text = ""
        candidate = await self.client.complete([])
        self.assertTrue(candidate["complete"])
        self.client.max_attempts = 1
        with self.assertRaises(RuntimeError):
            await self.client.complete([], require_text=True)
        self.response_text = "75"
        judge = await self.client.complete([], require_text=True)
        self.assertEqual(judge["text"], "75")
        self.assertEqual(self.requests, 3)
        self.assertEqual(len(list(Path(self.folder.name).rglob("attempt-*.json"))), 3)


if __name__ == "__main__":
    unittest.main()
