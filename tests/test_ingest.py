import unittest

from ingest import _sanitize_text


class IngestTextSanitizationTest(unittest.TestCase):
    def test_replaces_pdf_nuls_without_concatenating_tokens(self) -> None:
        self.assertEqual(_sanitize_text("docker run -p 8080\x0080 nginx"), "docker run -p 8080 80 nginx")

    def test_leaves_normal_pdf_text_unchanged(self) -> None:
        text = "docker ps\ndocker images\n"
        self.assertEqual(_sanitize_text(text), text)
