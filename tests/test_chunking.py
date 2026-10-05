from storybot.retrieval.chunking import build_chunks, chunk_story


def words(texts):
    return [len(t.split()) for t in texts]


def test_short_story_is_one_chunk():
    assert chunk_story("One. Two.", words, whole_story_max=10) == [(0, 9)]


def test_long_story_spans_cover_text_with_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    spans = chunk_story(text, words, chunk_tokens=50, overlap_tokens=10, whole_story_max=100)
    assert len(spans) > 5
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        assert s1 < s2 < e1 <= e2                   # each chunk starts inside the previous one
    assert max(words([text[s:e] for s, e in spans])) <= 50


def test_chunk_text_is_exact_substring(stories):
    chunks = build_chunks(stories, words, chunk_tokens=60, overlap_tokens=10, whole_story_max=100)
    by_id = dict(zip(stories.id, stories.story))
    for c in chunks:
        assert c.text == by_id[c.story_id][c.start:c.end].strip()
        assert c.context.startswith("Title: ") and "search_document" not in c.context
        assert isinstance(c.story_id, int)
