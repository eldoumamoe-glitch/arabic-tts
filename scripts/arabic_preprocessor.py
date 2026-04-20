"""
Arabic text preprocessing for XTTS-v2 inference.

Handles:
  1. Text cleaning (normalize whitespace, punctuation, special chars)
  2. Hamza normalization (correct common hamza placement errors)
  3. Tashkeel (add diacritics for pronunciation accuracy)
  4. Number-to-word conversion (Arabic numerals and percentages)
  5. Symbol expansion (Arabic-appropriate replacements)

Usage:
    from scripts.arabic_preprocessor import ArabicPreprocessor
    preprocessor = ArabicPreprocessor()
    clean_text = preprocessor.process("مرحبا بكم في 2026")
"""

import re
import unicodedata
from num2words import num2words

# Lazy-load tashkeel (heavy import)
_vocalizer = None


def _get_vocalizer():
    global _vocalizer
    if _vocalizer is None:
        import mishkal.tashkeel as tashkeel
        _vocalizer = tashkeel.TashkeelClass()
    return _vocalizer


# --- Hamza correction map ---
# Common words where hamza is frequently dropped or misplaced.
# Format: incorrect -> correct
HAMZA_CORRECTIONS = {
    # Alef with hamza above (أ)
    "ان": "أن",
    "انا": "أنا",
    "انت": "أنت",
    "انتم": "أنتم",
    "اكثر": "أكثر",
    "اقل": "أقل",
    "اول": "أول",
    "اي": "أي",
    "ايضا": "أيضاً",
    "اذا": "إذا",
    "امام": "أمام",
    "اصبح": "أصبح",
    "اصبحت": "أصبحت",
    "اخرى": "أخرى",
    "اخر": "آخر",
    "اكبر": "أكبر",
    "اكد": "أكد",
    "اعلن": "أعلن",
    "اهم": "أهم",
    "امر": "أمر",
    "اساس": "أساس",
    "اساسي": "أساسي",
    "امن": "أمن",
    "امل": "أمل",
    "اسلام": "إسلام",
    "ادارة": "إدارة",
    "انتاج": "إنتاج",
    "انسان": "إنسان",
    "اعلام": "إعلام",
    # Alef with hamza below (إ)
    "الى": "إلى",
    "اذ": "إذ",
    "اذن": "إذن",
    "انما": "إنما",
    "انه": "إنه",
    "انها": "إنها",
    "انهم": "إنهم",
    # Alef madda (آ)
    "الان": "الآن",
    "القران": "القرآن",
    "الالات": "الآلات",
    "الالة": "الآلة",
    "اخرون": "آخرون",
    # Hamza on waw (ؤ)
    "مسوول": "مسؤول",
    "مسوولية": "مسؤولية",
    "روية": "رؤية",
    "تاثير": "تأثير",
    "تاكد": "تأكد",
    "مساله": "مسألة",
    "سوال": "سؤال",
}

# --- Symbol expansion ---
SYMBOL_MAP = {
    "&": " و ",
    "@": " على ",
    "%": " بالمئة",
    "$": " دولار",
    "£": " جنيه",
    "€": " يورو",
    "°": " درجة",
    "+": " زائد ",
    "=": " يساوي ",
}


class ArabicPreprocessor:
    """Full Arabic text preprocessing pipeline for TTS inference."""

    def __init__(self, enable_tashkeel=False):
        self.enable_tashkeel = enable_tashkeel
        self._hamza_pattern = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in HAMZA_CORRECTIONS) + r")\b"
        )

    def clean_text(self, text):
        """Basic text cleaning: normalize whitespace, punctuation, unicode."""
        # Normalize unicode
        text = unicodedata.normalize("NFC", text)
        # Remove zero-width characters
        text = re.sub(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]", "", text)
        # Normalize Arabic-specific punctuation
        text = text.replace("٪", "%")
        text = text.replace("،", "،")  # keep Arabic comma
        text = text.replace("؛", "؛")  # keep Arabic semicolon
        # Normalize multiple spaces
        text = re.sub(r"\s+", " ", text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text

    def fix_hamza(self, text):
        """Correct common hamza placement errors in Arabic text."""
        def _replace(match):
            word = match.group(0)
            return HAMZA_CORRECTIONS.get(word, word)
        return self._hamza_pattern.sub(_replace, text)

    def expand_numbers(self, text):
        """Convert numbers to Arabic words."""
        def _number_to_words(match):
            num_str = match.group(0)
            try:
                num = float(num_str) if "." in num_str else int(num_str)
                return num2words(num, lang="ar")
            except (ValueError, OverflowError):
                return num_str

        # Handle percentages first (e.g., "70%" -> "سبعون بالمئة")
        def _percent_to_words(match):
            num_str = match.group(1)
            try:
                num = float(num_str) if "." in num_str else int(num_str)
                return num2words(num, lang="ar") + " بالمئة"
            except (ValueError, OverflowError):
                return match.group(0)

        text = re.sub(r"(\d+(?:\.\d+)?)\s*[%٪]", _percent_to_words, text)
        # Then standalone numbers
        text = re.sub(r"\d+(?:\.\d+)?", _number_to_words, text)
        return text

    def expand_symbols(self, text):
        """Replace symbols with Arabic words."""
        for symbol, replacement in SYMBOL_MAP.items():
            text = text.replace(symbol, replacement)
        return text

    def add_tashkeel(self, text):
        """Add diacritical marks using Mishkal."""
        if not self.enable_tashkeel:
            return text
        vocalizer = _get_vocalizer()
        return vocalizer.tashkeel(text)

    def process(self, text, tashkeel=None):
        """
        Full preprocessing pipeline.

        Args:
            text: Raw Arabic text.
            tashkeel: Override tashkeel setting (True/False/None=use default).

        Returns:
            Processed text ready for XTTS-v2 inference.
        """
        text = self.clean_text(text)
        text = self.fix_hamza(text)
        text = self.expand_numbers(text)
        text = self.expand_symbols(text)
        text = self.clean_text(text)  # clean again after expansions

        use_tashkeel = tashkeel if tashkeel is not None else self.enable_tashkeel
        if use_tashkeel:
            text = self.add_tashkeel(text)

        return text


# --- CLI for testing ---
if __name__ == "__main__":
    preprocessor = ArabicPreprocessor()

    test_texts = [
        "الذكاء الاصطناعي يتطور بسرعة كبيرة، ويدخل في كل مجالات الحياة.",
        "اكثر من 70% من الشركات الكبرى تستخدم الذكاء الاصطناعي اليوم.",
        "الالات اصبحت قادرة على التعلم، واتخاذ قرارات معقدة بمفردها.",
        "ان مستقبل البشرية سيتشكل بناءً على كيفية تعاملنا مع هذه التقنية.",
        "هذا المشروع يكلف 500$ و يحقق نمو 25%",
    ]

    print("=" * 70)
    print("Arabic Preprocessor Test")
    print("=" * 70)

    for text in test_texts:
        # Show each step
        cleaned = preprocessor.clean_text(text)
        hamza_fixed = preprocessor.fix_hamza(cleaned)
        numbers_expanded = preprocessor.expand_numbers(hamza_fixed)
        symbols_expanded = preprocessor.expand_symbols(numbers_expanded)
        final = preprocessor.process(text)

        print(f"\nOriginal:  {text}")
        if hamza_fixed != cleaned:
            print(f"Hamza:     {hamza_fixed}")
        if numbers_expanded != hamza_fixed:
            print(f"Numbers:   {numbers_expanded}")
        if symbols_expanded != numbers_expanded:
            print(f"Symbols:   {symbols_expanded}")
        print(f"Final:     {final}")
        print("-" * 70)
