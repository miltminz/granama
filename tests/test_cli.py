import pytest

from granama.cli import main


@pytest.fixture
def wordlist(tmp_path):
    path = tmp_path / "words.txt"
    path.write_text("dirty\nroom\ndormitory\nlisten\nsilent\n", encoding="utf-8")
    return str(path)


def test_prints_one_anagram_per_line(capsys, wordlist):
    assert main(["dirty room", "--wordlist", wordlist]) == 0
    assert capsys.readouterr().out.splitlines() == ["dormitory", "dirty room"]


def test_options_are_forwarded(capsys, wordlist):
    main(["dirty room", "--wordlist", wordlist, "--min-words", "2"])
    assert capsys.readouterr().out.splitlines() == ["dirty room"]
    main(["tinsel", "--wordlist", wordlist, "--limit", "1", "--max-len", "6", "--min-len", "6"])
    assert capsys.readouterr().out.splitlines() == ["listen"]
    main(["dirty room", "--wordlist", wordlist, "--max-words", "1"])
    assert capsys.readouterr().out.splitlines() == ["dormitory"]


def test_invalid_options_exit_2(capsys, wordlist):
    with pytest.raises(SystemExit) as exc:
        main(["abc", "--wordlist", wordlist, "--min-words", "3", "--max-words", "2"])
    assert exc.value.code == 2
    assert "max_words must be >= min_words" in capsys.readouterr().err


def test_unknown_language_exit_2(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["abc", "--lang", "xx"])
    assert exc.value.code == 2


def test_missing_wordlist_exit_1(capsys, tmp_path):
    assert main(["abc", "--wordlist", str(tmp_path / "missing.txt")]) == 1
    assert "cannot read word list" in capsys.readouterr().err


def test_bundled_vocabulary(capsys):
    assert main(["dormitory", "--max-words", "2"]) == 0
    assert "dirty room" in capsys.readouterr().out.splitlines()


def test_list_dictionaries(capsys):
    assert main(["--list-dictionaries"]) == 0
    assert capsys.readouterr().out.startswith("scowl-60 (default): SCOWL v2")


def test_text_required_without_list(capsys):
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
    assert "required: text" in capsys.readouterr().err


def test_dictionary_option(capsys):
    assert main(["dormitory", "--dictionary", "scowl-60", "--max-words", "1"]) == 0
    assert capsys.readouterr().out.splitlines() == ["dormitory"]


def test_unknown_dictionary_exit_2(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["abc", "--dictionary", "nope"])
    assert exc.value.code == 2
    assert "unknown dictionary 'nope'" in capsys.readouterr().err


def test_dictionary_and_wordlist_are_exclusive(capsys, wordlist):
    with pytest.raises(SystemExit) as exc:
        main(["abc", "--dictionary", "scowl-60", "--wordlist", wordlist])
    assert exc.value.code == 2


def test_exclude_tags(capsys, tmp_path):
    path = tmp_path / "words.txt"
    path.write_text("tosser\nrestos\n", encoding="utf-8")  # tosser: vulgar-3
    main(["tosser", "--wordlist", str(path)])
    assert capsys.readouterr().out.splitlines() == ["restos"]
    main(["tosser", "--wordlist", str(path), "--exclude-tags", ""])
    assert capsys.readouterr().out.splitlines() == ["restos", "tosser"]
    main(["tosser", "--wordlist", str(path), "--exclude-tags", "offensive"])
    assert capsys.readouterr().out.splitlines() == ["restos", "tosser"]
    main(["tosser", "--wordlist", str(path), "--exclude-tags", " vulgar-3 , offensive"])
    assert capsys.readouterr().out.splitlines() == ["restos"]


def test_max_rarity(capsys):
    assert main(["dormitory", "--max-words", "1", "--max-rarity", "10"]) == 0
    assert capsys.readouterr().out.splitlines() == []  # too rare for a very low tier
    assert main(["dormitory", "--max-words", "1", "--max-rarity", "60"]) == 0
    assert "dormitory" in capsys.readouterr().out.splitlines()


def test_max_rarity_with_wordlist_exit_2(capsys, wordlist):
    with pytest.raises(SystemExit) as exc:
        main(["abc", "--wordlist", wordlist, "--max-rarity", "35"])
    assert exc.value.code == 2
    assert "requires a bundled dictionary" in capsys.readouterr().err


def test_rank_prints_best_first(capsys):
    assert main(["dormitory", "--rank", "--limit", "3"]) == 0
    assert capsys.readouterr().out.splitlines() == ["dormitory", "dirty moor", "dirty room"]


def test_rank_limit_applies_after_ranking(capsys):
    main(["clint eastwood", "--max-words", "2", "--limit", "1"])
    assert capsys.readouterr().out.splitlines() != ["dislocate town"]  # not first unranked
    main(["clint eastwood", "--max-words", "2", "--limit", "1", "--rank"])
    assert capsys.readouterr().out.splitlines() == ["dislocate town"]


def test_rank_with_wordlist_exit_2(capsys, wordlist):
    with pytest.raises(SystemExit) as exc:
        main(["dirty room", "--wordlist", wordlist, "--rank"])
    assert exc.value.code == 2
    assert "--rank requires a bundled dictionary" in capsys.readouterr().err


@pytest.fixture
def fake_neural(monkeypatch):
    from test_neural import FLUENCY, ranker

    loaded = []

    def neural_ranker(args):
        loaded.append(args)
        return ranker(
            FLUENCY,
            pool=args.pool,
            relevance_weight=args.relevance_weight,
            rarity_weight=args.rarity_weight,
        )

    monkeypatch.setattr("granama.cli.neural_ranker", neural_ranker)
    return loaded


def test_neural_rank_prints_phrases(capsys, wordlist, fake_neural):
    assert main(["dormitory", "--wordlist", wordlist, "--rank", "neural"]) == 0
    assert capsys.readouterr().out.splitlines() == ["dirty room"]
    assert fake_neural[0].fluency_model == "HuggingFaceTB/SmolLM2-135M"


def test_neural_scores(capsys, wordlist, fake_neural):
    main(["dormitory", "--wordlist", wordlist, "--rank", "neural", "--scores"])
    assert capsys.readouterr().out.splitlines() == [" +0.00    -5.0  0.000  dirty room"]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--scores"], "--scores requires --rank neural"),
        (["--rank", "neural", "--pool", "0"], "--pool must be >= 1"),
        (["--rank", "neural", "--max-orders", "0"], "--max-orders must be >= 1"),
    ],
)
def test_neural_option_errors(capsys, wordlist, args, message):
    with pytest.raises(SystemExit) as exc:
        main(["dormitory", "--wordlist", wordlist, *args])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_neural_without_extra_exit_2(capsys, monkeypatch, wordlist):
    def missing(args):
        raise ImportError("neural ranking needs the lm extra")

    monkeypatch.setattr("granama.cli.neural_ranker", missing)
    with pytest.raises(SystemExit) as exc:
        main(["dormitory", "--wordlist", wordlist, "--rank", "neural"])
    assert exc.value.code == 2
    assert "lm extra" in capsys.readouterr().err


def test_bare_rank_is_rarity(capsys):
    main(["dormitory", "--rank", "--max-words", "2", "--limit", "2"])
    assert capsys.readouterr().out.splitlines()[0] == "dormitory"  # the most common words
