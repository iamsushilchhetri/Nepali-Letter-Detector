from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nepali_letter_detector.labels import LABEL_MAP, display_name, glyph_for


class LabelTests(unittest.TestCase):
    def test_label_map_has_46_classes(self) -> None:
        self.assertEqual(len(LABEL_MAP), 46)

    def test_label_map_covers_consonants_and_digits(self) -> None:
        consonants = [name for name in LABEL_MAP if name.startswith("character_")]
        digits = [name for name in LABEL_MAP if name.startswith("digit_")]
        self.assertEqual(len(consonants), 36)
        self.assertEqual(len(digits), 10)

    def test_glyph_for_known_class(self) -> None:
        self.assertEqual(glyph_for("character_1_ka"), "\u0915")
        self.assertEqual(glyph_for("digit_0"), "\u0966")

    def test_display_name_includes_glyph_and_romanization(self) -> None:
        self.assertEqual(display_name("character_1_ka"), "\u0915 (ka)")


if __name__ == "__main__":
    unittest.main()
