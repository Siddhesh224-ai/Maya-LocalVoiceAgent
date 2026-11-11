import unittest

from fastapi.testclient import TestClient

from main import app


class HealthEndpointTest(unittest.TestCase):
    def test_health_returns_ok_status(self) -> None:
        with TestClient(app) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
