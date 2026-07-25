from __future__ import annotations


LABEL_MAP = {
    "character_1_ka": ("\u0915", "ka"),
    "character_2_kha": ("\u0916", "kha"),
    "character_3_ga": ("\u0917", "ga"),
    "character_4_gha": ("\u0918", "gha"),
    "character_5_kna": ("\u0919", "nga"),
    "character_6_cha": ("\u091A", "cha"),
    "character_7_chha": ("\u091B", "chha"),
    "character_8_ja": ("\u091C", "ja"),
    "character_9_jha": ("\u091D", "jha"),
    "character_10_yna": ("\u091E", "nya"),
    "character_11_taamatar": ("\u091F", "tta"),
    "character_12_thaa": ("\u0920", "ttha"),
    "character_13_daa": ("\u0921", "dda"),
    "character_14_dhaa": ("\u0922", "ddha"),
    "character_15_adna": ("\u0923", "nna"),
    "character_16_tabala": ("\u0924", "ta"),
    "character_17_tha": ("\u0925", "tha"),
    "character_18_da": ("\u0926", "da"),
    "character_19_dha": ("\u0927", "dha"),
    "character_20_na": ("\u0928", "na"),
    "character_21_pa": ("\u092A", "pa"),
    "character_22_pha": ("\u092B", "pha"),
    "character_23_ba": ("\u092C", "ba"),
    "character_24_bha": ("\u092D", "bha"),
    "character_25_ma": ("\u092E", "ma"),
    "character_26_yaw": ("\u092F", "ya"),
    "character_27_ra": ("\u0930", "ra"),
    "character_28_la": ("\u0932", "la"),
    "character_29_waw": ("\u0935", "wa"),
    "character_30_motosaw": ("\u0936", "sha"),
    "character_31_petchiryakha": ("\u0937", "ssa"),
    "character_32_patalosaw": ("\u0938", "sa"),
    "character_33_ha": ("\u0939", "ha"),
    "character_34_chhya": ("\u0915\u094D\u0937", "kshya"),
    "character_35_tra": ("\u0924\u094D\u0930", "tra"),
    "character_36_gya": ("\u091C\u094D\u091E", "gya"),
    "digit_0": ("\u0966", "0"),
    "digit_1": ("\u0967", "1"),
    "digit_2": ("\u0968", "2"),
    "digit_3": ("\u0969", "3"),
    "digit_4": ("\u096A", "4"),
    "digit_5": ("\u096B", "5"),
    "digit_6": ("\u096C", "6"),
    "digit_7": ("\u096D", "7"),
    "digit_8": ("\u096E", "8"),
    "digit_9": ("\u096F", "9"),
}


def glyph_for(folder_name: str) -> str:
    return LABEL_MAP[folder_name][0]


def romanized_for(folder_name: str) -> str:
    return LABEL_MAP[folder_name][1]


def display_name(folder_name: str) -> str:
    glyph, romanized = LABEL_MAP.get(folder_name, (folder_name, folder_name))
    return f"{glyph} ({romanized})"


def sorted_label_names() -> list[str]:
    return sorted(LABEL_MAP)
