from __future__ import annotations
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:
    pass

import inspect
import re
import traceback

import exact_fama.physics.physics_solvers.domains.physics_engine as pe


TESTS = [
    "A coil has 3000 turns uniformly wound on a 0.25 m tube. What is n in turns/m?",
    "A student repeats a measurement and obtains 36.1 A, 36.3 A, and 36.5 A. Find the mean and mean absolute deviation.",
    "For L = 100 mH and C = 2.2 μF, compute T = 2π√LC.",
    "A quantity actually equals 80 A, but a student measures 82 A. Find Δx and δ%.",
]


def show_codepoints(label: str, text: str) -> None:
    print(f"\n[{label}]")
    print("repr:", repr(text))
    print("chars:", " ".join(f"{ch}=U+{ord(ch):04X}" for ch in text if ch in "μµπ√Δδ?"))


def safe_call(name: str, *args):
    print(f"\n{name}:")
    fn = getattr(pe, name, None)
    if fn is None:
        print("  MISSING")
        return None
    try:
        out = fn(*args)
        print(" ", out)
        return out
    except Exception:
        print("  EXCEPTION")
        traceback.print_exc()
        return None


def regex_find(label: str, pattern: str, text: str, flags=re.I) -> None:
    print(f"\nREGEX {label}:")
    print("pattern:", pattern)
    try:
        matches = list(re.finditer(pattern, text, flags=flags))
    except Exception:
        print("  REGEX COMPILE/SEARCH ERROR")
        traceback.print_exc()
        return
    if not matches:
        print("  NO MATCH")
        return
    for i, m in enumerate(matches, 1):
        print(f"  #{i}: {m.group(0)!r}")
        print("     groups:", m.groupdict())


def main() -> None:
    print("FILE:", pe.__file__)
    print("HAS HIGH IMPACT:", hasattr(pe, "_solve_clean_high_impact_guards"))
    print("ENTRYPOINT HAS HIGH IMPACT:", "_solve_clean_high_impact_guards" in inspect.getsource(pe.solve_clean_physics_engine))

    print("\n--- FUNCTION OBJECT ---")
    print(pe._solve_clean_high_impact_guards)
    print("defined at line:", inspect.getsourcelines(pe._solve_clean_high_impact_guards)[1])

    print("\n--- FIRST 220 LINES OF _solve_clean_high_impact_guards SOURCE ---")
    src_lines, start = inspect.getsourcelines(pe._solve_clean_high_impact_guards)
    for offset, line in enumerate(src_lines[:220], start):
        print(f"{offset:5d}: {line.rstrip()}")

    src = "".join(src_lines)
    print("\n--- SOURCE CONTAINS CANARY TERMS ---")
    for term in [
        "turns/m",
        "mean absolute deviation",
        "actually equals",
        "2π√LC",
        "2??LC",
        "wound on",
        "student repeats",
    ]:
        print(f"{term!r}:", term in src)

    print("\n--- HELPER EXISTENCE ---")
    for name in [
        "_normalize_text",
        "_parse_number",
        "_to_si",
        "_result",
        "_make_result",
        "_eng_symbol_values",
        "_eng_inductance_values",
        "_eng_cap_values",
        "_solve_clean_high_impact_guards",
        "solve_clean_physics_engine",
    ]:
        print(name, "=>", hasattr(pe, name))

    print("\n--- BASIC PARSER TESTS ---")
    for raw in ["3000", "0.25", "36.1", "100", "2.2", "0.109", "2.2 μF", "2.2 ?F"]:
        try:
            print(raw, "=>", pe._parse_number(raw) if re.fullmatch(pe.VALUE_PATTERN, pe._normalize_text(raw).replace(" μF", "").replace(" ?F", "")) else "not-single-number")
        except Exception as exc:
            print(raw, "=> EXC", exc)

    for q in TESTS:
        print("\n" + "=" * 100)
        print("QUESTION:", q)

        show_codepoints("original", q)

        t = pe._normalize_text(q)
        low = t.lower()
        show_codepoints("normalized", t)
        print("normalized:", repr(t))
        print("lower:", repr(low))

        print("\nDIRECT HIGH IMPACT CALL:")
        try:
            print(pe._solve_clean_high_impact_guards(q))
        except Exception:
            traceback.print_exc()

        print("\nENTRYPOINT CALL:")
        try:
            print(pe.solve_clean_physics_engine(q))
        except Exception:
            traceback.print_exc()

        print("\n--- generic number/unit matches with VALUE_PATTERN + UNIT_PATTERN ---")
        regex_find(
            "VALUE + optional UNIT",
            rf"(?P<v>{pe.VALUE_PATTERN})\s*(?P<u>{pe.UNIT_PATTERN})?",
            t,
        )

        print("\n--- quantities/helpers ---")
        safe_call("_eng_inductance_values", q)
        safe_call("_eng_cap_values", q)
        safe_call("_eng_freqs", q)

        print("\n--- targeted canary regexes ---")

        regex_find(
            "turn density",
            rf"(?:coil|solenoid)[^.?!]*?(?P<N>{pe.VALUE_PATTERN})\s*turns?[^.?!]*?(?:on|over|along|length|tube)[^.?!]*?(?P<l>{pe.VALUE_PATTERN})\s*(?P<u>m|cm|mm)\b",
            t,
        )

        regex_find(
            "repeated measurement values",
            rf"obtains?\s+(?P<a>{pe.VALUE_PATTERN})\s*(?P<u>A|V|g|cm|m)?\s*,\s*(?P<b>{pe.VALUE_PATTERN})\s*(?P<u2>A|V|g|cm|m)?\s*,\s*(?:and\s+)?(?P<c>{pe.VALUE_PATTERN})\s*(?P<u3>A|V|g|cm|m)?",
            t,
        )

        regex_find(
            "actual measured",
            rf"actually\s+equals\s+(?P<actual>{pe.VALUE_PATTERN})\s*(?P<u>A|V|g|cm|m)?[^.?!]*?measures?\s+(?P<measured>{pe.VALUE_PATTERN})\s*(?P<u2>A|V|g|cm|m)?",
            t,
        )

        regex_find(
            "LC period symbols",
            rf"\bL\s*=\s*(?P<L>{pe.VALUE_PATTERN})\s*(?P<Lu>mH|μH|µH|uH|H|\?H)\b.*?\bC\s*=\s*(?P<C>{pe.VALUE_PATTERN})\s*(?P<Cu>microfarads?|μF|µF|uF|\?F|mF|nF|pF|F)\b",
            t,
        )

        regex_find(
            "compute T phrase",
            r"(?:compute|calculate|find|determine)\s*T|period|2\s*(?:π|pi|\?)*\s*(?:√|sqrt|\?)*\s*L\s*C",
            t,
        )


if __name__ == "__main__":
    main()