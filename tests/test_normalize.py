from subcheck.normalize import _ordinal_words, int_to_words, normalize


def test_case_and_punctuation():
    assert normalize("Hello, WORLD!") == "hello world"


def test_apostrophes_and_contractions():
    assert normalize("Don't stop, it's fine.") == "dont stop its fine"


def test_whitespace_collapse():
    assert normalize("  a\n\n  b\t c  ") == "a b c"


def test_formatting_tags_stripped():
    assert normalize("<i>Hello</i> <b>there</b>") == "hello there"
    # tags are replaced by separators, so glued words become separate tokens
    assert normalize("{\\an8}Hi{\\i1}ya") == "hi ya"
    assert normalize("a<i>b</i>c") == "a b c"


def test_music_glyphs_removed():
    assert normalize("♪ la la ♫") == "la la"


def test_acoustic_brackets_removed():
    assert normalize("[music playing] Hello (applause) world [laughs]") == "hello world"


def test_plain_brackets_kept():
    # Brackets that are not sound cues are ordinary dialogue
    assert normalize("[really?] yes") == "really yes"


def test_numbers_expanded():
    assert normalize("It was 10 o'clock") == "it was ten oclock"
    assert normalize("Chapter 2: in 1945") == "chapter two in one thousand nine hundred forty five"


def test_ordinals_expanded():
    assert normalize("He was 2nd, she was 21st") == "he was second she was twenty first"


def test_digits_identical_both_sides():
    # both sides go through the same transform, so digit/number differences vanish
    assert normalize("10") == normalize("ten")


def test_unicode_letters_kept():
    assert normalize("Café déjà vu!") == "café déjà vu"


def test_int_to_words_range():
    assert int_to_words(0) == "zero"
    assert int_to_words(1000000) == "one million"
    assert int_to_words(10**10) is None  # out of range -> None
    assert _ordinal_words(3) == "third"
    assert _ordinal_words(12) == "twelfth"
    assert _ordinal_words(20) == "twentieth"
    assert _ordinal_words(21) == "twenty first"
