"""Mapping between the dataset's folder names and displayable Nepali (Devanagari) glyphs.

Class order matches the folder naming in the Devanagari Handwritten Character Dataset
(UCI/Kaggle: 36 consonants + 10 digits), which torchvision's ImageFolder sorts
alphabetically -- that sorted order is also the model's output order.
"""

# folder name -> (Devanagari glyph, romanized name)
LABEL_MAP = {
    "character_1_ka": ("क", "ka"),
    "character_2_kha": ("ख", "kha"),
    "character_3_ga": ("ग", "ga"),
    "character_4_gha": ("घ", "gha"),
    "character_5_kna": ("ङ", "kna"),
    "character_6_cha": ("च", "cha"),
    "character_7_chha": ("छ", "chha"),
    "character_8_ja": ("ज", "ja"),
    "character_9_jha": ("झ", "jha"),
    "character_10_yna": ("ञ", "yna"),
    "character_11_taamatar": ("ट", "ta"),
    "character_12_thaa": ("ठ", "tha"),
    "character_13_daa": ("ड", "da"),
    "character_14_dhaa": ("ढ", "dha"),
    "character_15_adna": ("ण", "na"),
    "character_16_tabala": ("त", "ta"),
    "character_17_tha": ("थ", "tha"),
    "character_18_da": ("द", "da"),
    "character_19_dha": ("ध", "dha"),
    "character_20_na": ("न", "na"),
    "character_21_pa": ("प", "pa"),
    "character_22_pha": ("फ", "pha"),
    "character_23_ba": ("ब", "ba"),
    "character_24_bha": ("भ", "bha"),
    "character_25_ma": ("म", "ma"),
    "character_26_yaw": ("य", "ya"),
    "character_27_ra": ("र", "ra"),
    "character_28_la": ("ल", "la"),
    "character_29_waw": ("व", "wa"),
    "character_30_motosaw": ("श", "sha"),
    "character_31_petchiryakha": ("ष", "sha"),
    "character_32_patalosaw": ("स", "sa"),
    "character_33_ha": ("ह", "ha"),
    "character_34_chhya": ("क्ष", "kshya"),
    "character_35_tra": ("त्र", "tra"),
    "character_36_gya": ("ज्ञ", "gya"),
    "digit_0": ("०", "0"),
    "digit_1": ("१", "1"),
    "digit_2": ("२", "2"),
    "digit_3": ("३", "3"),
    "digit_4": ("४", "4"),
    "digit_5": ("५", "5"),
    "digit_6": ("६", "6"),
    "digit_7": ("७", "7"),
    "digit_8": ("८", "8"),
    "digit_9": ("९", "9"),
}


def glyph_for(folder_name: str) -> str:
    return LABEL_MAP[folder_name][0]


def display_name(folder_name: str) -> str:
    glyph, romanized = LABEL_MAP[folder_name]
    return f"{glyph} ({romanized})"
