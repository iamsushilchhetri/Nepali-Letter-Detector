from app.labels import LABEL_MAP, display_name, glyph_for


def test_label_map_has_46_classes():
    assert len(LABEL_MAP) == 46


def test_label_map_covers_consonants_and_digits():
    consonants = [k for k in LABEL_MAP if k.startswith("character_")]
    digits = [k for k in LABEL_MAP if k.startswith("digit_")]
    assert len(consonants) == 36
    assert len(digits) == 10


def test_glyph_for_known_class():
    assert glyph_for("character_1_ka") == "क"
    assert glyph_for("digit_0") == "०"


def test_display_name_includes_glyph_and_romanization():
    assert display_name("character_1_ka") == "क (ka)"
