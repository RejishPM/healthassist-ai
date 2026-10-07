import unittest

from app.ingestion.chunker import chunk_blocks, split_large_prose


class ChunkParameterValidationTests(unittest.TestCase):
    def test_chunker_imports_through_package(self):
        self.assertTrue(callable(chunk_blocks))

    def test_max_words_must_be_positive(self):
        with self.assertRaisesRegex(
            ValueError,
            "max_words must be greater than 0",
        ):
            chunk_blocks([], max_words=0, overlap_words=0)

    def test_overlap_words_cannot_be_negative(self):
        with self.assertRaisesRegex(
            ValueError,
            "overlap_words must be greater than or equal to 0",
        ):
            chunk_blocks([], max_words=10, overlap_words=-1)

    def test_overlap_words_must_be_less_than_max_words(self):
        with self.assertRaisesRegex(
            ValueError,
            "overlap_words must be greater than or equal to 0",
        ):
            chunk_blocks([], max_words=10, overlap_words=10)

    def test_split_large_prose_validates_direct_calls(self):
        with self.assertRaisesRegex(
            ValueError,
            "overlap_words must be greater than or equal to 0",
        ):
            split_large_prose("some text", max_words=1, overlap_words=1)


if __name__ == "__main__":
    unittest.main()
