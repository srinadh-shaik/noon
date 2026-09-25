"""Run: .venv/bin/python src/test_s2_knowledge.py"""
from s2_knowledge import number_relation


def n(v, frac=None, letter=None, sub=None, bis=None):
    return {"v": v, "frac": frac, "letter": letter, "sub": sub, "bis": bis}


def test():
    cases = [  # (S1 numbers, record numbers, expected class) — examples from DATA_NOTES §8 / §9c
        ([n(9), n(123)], [n(123), n(9)], "identical"),
        ([n(44)], [n(44, letter="d")], "letter"),
        ([n(204, frac="1/2")], [n(204, frac="1/9")], "fraction"),
        ([n(2620)], [n(262)], "truncation"),
        ([n(9)], [n(9), n(123)], "injected"),
        ([n(9), n(123)], [n(123)], "dropped"),
        ([n(10084), n(112)], [n(10105), n(112)], "shared_far"),
        ([n(10084), n(112)], [n(10090), n(112)], "shared_near"),
        ([n(109)], [n(111)], "disjoint_near"),
        ([n(109)], [n(5000)], "disjoint"),
        ([], [n(1)], "one_none"),
        ([], [], "both_none"),
    ]
    for a, b, want in cases:
        assert number_relation(a, b) == want, (a, b, number_relation(a, b), want)
    print("ok")


if __name__ == "__main__":
    test()
