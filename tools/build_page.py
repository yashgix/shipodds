"""Render docs/index.html from results/metrics_v2.json and tools/page_template.html."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RULES = (
    "always_hold",
    "mean_1",
    "mean_4",
    "cmh_1",
    "cmh_4",
    "naive_peek",
    "sequential_pocock",
    "sequential_obf",
    "obf_futility_0.0",
    "obf_futility_0.5",
    "obf_futility_1.0",
    "obf_futility_1.5",
    "obf_futility_2.0",
)
TESTS = ("cmh_1", "cmh_4", "naive_peek", "sequential_pocock", "sequential_obf", "obf_futility_1.0")
FOLD_RULES = ("pick", "sequential_obf", "cmh_4", "cmh_1")


def pct(value):
    return f"{100 * value:.1f}%"


def bar_rows(v2, key, rules, top, flag=("naive_peek",)):
    rows = []
    for rule in rules:
        value = v2[key][v2["rules"].index(rule)]
        width = min(100 * value / top, 100)
        cls = " warn" if rule in flag else ""
        rows.append(
            f'<div class="b"><span class="name">{rule}</span><span class="track"><span class="fill{cls}" '
            f'style="width:{width:.2f}%"></span></span><span class="v"><b>{pct(value)}</b></span></div>'
        )
    return "\n".join(rows)


def panel(title, note, body, ticks):
    axis = "".join(f"<span>{t}</span>" for t in ticks)
    return (
        f'<div class="panel"><h3>{title}</h3><p class="small muted">{note}</p><div class="bars">{body}</div>'
        f'<div class="axis"><span></span><span class="ticks">{axis}</span><span></span></div></div>'
    )


def fold_table(v2):
    head = "".join(
        f"<th>{h}</th>" for h in ("Held out", "Rule", "Identical copy promoted", "Better promoted", "Cost, real pairs")
    )
    rows = []
    for fold in v2["folds"]:
        for name in FOLD_RULES:
            rule = fold["pick"] if name == "pick" else name
            i = v2["rules"].index(rule)
            label = f"{rule} (pick)" if name == "pick" else rule
            cells = (
                fold["held_out"],
                f"<code>{label}</code>",
                pct(fold["aa_promote"][i]),
                pct(fold["better_promote"][i]),
                pct(fold["cost_share"][i]),
            )
            rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def main():
    v2 = json.loads((ROOT / "results" / "metrics_v2.json").read_text())
    at = lambda key, rule: v2[key][v2["rules"].index(rule)]  # noqa: E731
    charts = {
        "CHART_FALSE": panel(
            "False promotions",
            "Share of identical copies promoted. Always wrong.",
            bar_rows(v2, "aa_promote", RULES, 0.5),
            ("0", "10%", "20%", "30%", "40%", "50%"),
        ),
        "CHART_POWER": panel(
            "Power",
            "Share of truly better candidates promoted.",
            bar_rows(v2, "better_promote", RULES, 1.0),
            ("0", "25%", "50%", "75%", "100%"),
        ),
        "CHART_COST": panel(
            "Cost",
            "Share of a full four-trial evaluation, real pairs.",
            bar_rows(v2, "cost_share", RULES, 1.0),
            ("0", "25%", "50%", "75%", "100%"),
        ),
        "CHART_PEEK": panel(
            "Zoom on the statistical tests",
            "Identical copies promoted, scale 0 to 7%. A correct test at the 5% two-sided level promotes about 2.5%.",
            bar_rows(v2, "aa_promote", TESTS, 0.07),
            ("0", "2%", "4%", "7%"),
        ),
        "FOLD_TABLE": fold_table(v2),
    }
    values = {
        "MEAN1_FALSE": pct(at("aa_promote", "mean_1")),
        "CMH1_FALSE": pct(at("aa_promote", "cmh_1")),
        "CMH1_POWER": pct(at("better_promote", "cmh_1")),
        "PEEK_FALSE": pct(at("aa_promote", "naive_peek")),
        "CMH4_FALSE": pct(at("aa_promote", "cmh_4")),
        "PEEK_POWER": pct(at("better_promote", "naive_peek")),
        "PEEK_COST": pct(at("cost_share", "naive_peek")),
        "OBF_POWER": pct(at("better_promote", "sequential_obf")),
        "CMH4_POWER": pct(at("better_promote", "cmh_4")),
        "OBF_COST": pct(at("cost_share", "sequential_obf")),
        "FUT_POWER": pct(at("better_promote", "obf_futility_1.0")),
        "FUT_COST": pct(at("cost_share", "obf_futility_1.0")),
        "FUT_COST_AA": pct(at("aa_cost_share", "obf_futility_1.0")),
        "FUT_FALSE": pct(at("aa_promote", "obf_futility_1.0")),
        "OBF_COST_AA": pct(at("aa_cost_share", "sequential_obf")),
        "DISTINCT": str(v2["distinct_comparisons"]),
        "PICK": v2["pick"],
    }
    page = (ROOT / "tools" / "page_template.html").read_text()
    for key, text in {**charts, **values}.items():
        page = page.replace("{{" + key + "}}", text)
    (ROOT / "docs" / "index.html").write_text(page)


if __name__ == "__main__":
    main()
