from storybot.retrieval.matching import MetadataMatcher, normalize

MD = {
    "title_map": {normalize(t): {"id": i, "title": t, "genre": g} for i, (t, g) in enumerate([
        ("The Clockmaker's Daughter", "Fantasy"),
        ("The Clockmaker's Daughter - Vol 2", "Fantasy"),
        ("Echoes of the Red Planet", "Science Fiction"),
        ("Murder at Willow Lane", "Mystery"),
        ("A Quiet Night", "Romance"),
    ])},
    "genres": ["Fantasy", "Science Fiction", "Mystery", "Romance", "Post-Apocalyptic"],
}


def test_title_exact_after_normalisation():
    m = MetadataMatcher(MD).match_title("the clockmakers daughter!!")
    assert m.matched and m.value == "The Clockmaker's Daughter" and m.score == 100 and m.story_id == 0
    # the other volume is offered as a candidate
    assert any(c["title"] == "The Clockmaker's Daughter - Vol 2" for c in m.candidates)


def test_title_typo_and_word_order():
    mm = MetadataMatcher(MD)
    assert mm.match_title("Echos of the Red Plannet").value == "Echoes of the Red Planet"
    assert mm.match_title("willow lane murder at").value == "Murder at Willow Lane"


def test_title_below_threshold_is_no_match():
    m = MetadataMatcher(MD).match_title("dragons of the deep sea")
    assert not m.matched and m.value is None and m.candidates     # candidates still listed
    # short partial queries do not grab a long title
    assert not MetadataMatcher(MD).match_title("the night").matched


def test_genre_fuzzy_and_fallback():
    mm = MetadataMatcher(MD)
    assert mm.match_genre("science-fiction").value == "Science Fiction"
    assert mm.match_genre("post apocalyptic").value == "Post-Apocalyptic"
    assert mm.match_genre("Mistery").value == "Mystery"
    assert not mm.match_genre("cooking").matched
