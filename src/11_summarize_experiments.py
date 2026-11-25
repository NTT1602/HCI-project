"""Gom kết quả 09_compare_models.py + 10_tune_policies.py thành reports/EXPERIMENTS_RESULTS.md và biểu đồ.

    python src/11_summarize_experiments.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

POLICY_VI = {"fixed": "C1 cố định", "adaptive": "C2 thích ứng", "fixed_abstention": "C3 abstention",
             "adaptive_abstention": "C4 kết hợp"}


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if pd.isna(v):
                    cells.append("—")
                elif "%" in c:
                    cells.append(f"{v:.1f}")
                else:
                    cells.append(f"{v:.0f}" if abs(v) >= 100 else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pct(x):
    return None if pd.isna(x) else round(100 * float(x), 1)


def plot_tradeoff(grid_dir: Path, runs: list[str], out: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, axes = plt.subplots(1, len(runs), figsize=(5.5 * len(runs), 4.2), squeeze=False)
    for ax, run in zip(axes[0], runs):
        path = grid_dir / f"{run}_val_grid.csv"
        if not path.exists():
            continue
        g = pd.read_csv(path)
        for policy, label in POLICY_VI.items():
            sub = g[g.policy == policy].sort_values("non_command_trigger_rate")
            if sub.empty:
                continue
            front = sub.assign(best=sub.clean_success_rate.cummax()).drop_duplicates("best")  # đường Pareto
            ax.step(100 * front.non_command_trigger_rate, 100 * front.best, where="post", label=label)
        ax.set_xlim(0, 5)
        ax.set_xlabel("Kích hoạt nhầm (% trial không phải lệnh)")
        ax.set_ylabel("Đúng lệnh, không kèm lệnh sai (%)")
        ax.set_title(f"{run} (validation)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="reports/models/classifier_summary.csv")
    parser.add_argument("--tuning-dir", default="reports/tuning")
    parser.add_argument("--out", default="reports/EXPERIMENTS_RESULTS.md")
    parser.add_argument("--budget", type=float, default=0.02, help="Ngân sách kích hoạt nhầm dùng cho bảng chính")
    args = parser.parse_args()

    tuning = Path(args.tuning_dir)
    models = pd.read_csv(args.models)
    selected_all = pd.read_csv(tuning / "selected.csv")
    if "latency_cap" not in selected_all:
        selected_all["latency_cap"] = float("nan")
    selected = selected_all[selected_all.latency_cap.isna()]
    by_latency = selected_all[selected_all.latency_cap.notna()]
    defaults = pd.read_csv(tuning / "app_defaults.csv")
    parts = ["# Kết quả thực nghiệm (tự sinh bởi src/11_summarize_experiments.py)", ""]

    m = models[["run", "params", "window_ms", "infer_ms", "epochs_run", "test_accuracy", "test_macro_f1",
                "f1_next", "f1_previous", "f1_none", "near_miss_to_swipe"]].sort_values("test_macro_f1", ascending=False)
    parts += ["## 1. Bộ phân loại (test Jester, cửa sổ căn đúng cử chỉ)", "",
              "`near_miss_to_swipe`: tỉ lệ Thumb Up/Down, Stop Sign bị đoán thành vuốt (thấp là tốt). "
              "`infer_ms`: 1 lần dự đoán qua tf.function trên CPU (cách app gọi).", "", md_table(m), ""]

    b = selected[selected.budget == args.budget].copy()
    b["policy"] = b.policy.map(POLICY_VI)
    view = pd.DataFrame({
        "model": b.run, "policy": b.policy,
        "đúng lệnh sạch %": b.test_clean_success_rate.map(pct),
        "sai chiều %": b.test_wrong_command_rate.map(pct),
        "bỏ sót %": b.test_command_miss_rate.map(pct),
        "nhầm (nền+near-miss) %": b.test_non_command_trigger_rate.map(pct),
        "nhầm nền %": b.test_background_trigger_rate.map(pct),
        "nhầm near-miss %": b.test_near_miss_trigger_rate.map(pct),
        "latency ms": b.test_median_latency_ms,
        "abstention %": b.test_abstention_rate.map(pct),
        "ngưỡng": b.threshold, "giữ ms": b.hold_ms, "cooldown ms": b.cooldown_ms,
    }).sort_values(["model", "policy"])
    parts += [f"## 2. C1–C4 sau khi chỉnh tham số (chọn trên validation với kích hoạt nhầm ≤ {args.budget:.0%}, số liệu TEST)", "",
              "Chỉ next/previous là lệnh (zoom gộp vào none, video zoom = near-miss), chặn chiều ngược 2.5 s và "
              "mô phỏng reset GRU bật cho mọi cấu hình.", "", md_table(view), ""]

    ci_path = tuning / "bootstrap_ci.csv"
    ci_all = pd.read_csv(ci_path) if ci_path.exists() else None
    if ci_all is not None and "latency_cap" not in ci_all:
        ci_all["latency_cap"] = float("nan")

    def ci_table(ci):

        def span(r, key, diff=False):
            pre = f"{key}_diff" if diff else f"{key}_mean"
            lo, hi = (f"{key}_diff_lo", f"{key}_diff_hi") if diff else (f"{key}_lo", f"{key}_hi")
            sign = "+" if diff and r[pre] > 0 else ""
            return f"{sign}{100 * r[pre]:.1f} [{100 * r[lo]:.1f}; {100 * r[hi]:.1f}]"

        return md_table(pd.DataFrame([{
            "model": r.run, **({"trễ ≤ ms": int(r.latency_cap)} if pd.notna(r.latency_cap) else {}),
            "policy": POLICY_VI.get(r.policy, r.policy),
            "đúng lệnh sạch % [CI 95%]": span(r, "clean"), "so với C1 (điểm %)": span(r, "clean", True),
            "kích hoạt nhầm % [CI 95%]": span(r, "trigger"), "nhầm so với C1 (điểm %)": span(r, "trigger", True),
        } for _, r in ci.iterrows()]))

    if ci_all is not None:
        parts += ["### Khoảng tin cậy 95% (bootstrap 2000 lần trên trial test, chênh lệch ghép cặp cùng trial)", "",
                  "Khoảng chênh lệch không chứa 0 thì khác biệt so với C1 là có ý nghĩa.", "",
                  ci_table(ci_all[ci_all.latency_cap.isna()]), ""]

    if len(by_latency):
        lat = by_latency.copy()
        lview = pd.DataFrame({
            "model": lat.run, "trễ ≤ ms": lat.latency_cap.astype(int), "policy": lat.policy.map(POLICY_VI),
            "đúng lệnh sạch %": lat.test_clean_success_rate.map(pct), "sai chiều %": lat.test_wrong_command_rate.map(pct),
            "nhầm %": lat.test_non_command_trigger_rate.map(pct), "latency ms": lat.test_median_latency_ms,
            "ngưỡng": lat.threshold, "giữ ms": lat.hold_ms,
        }).sort_values(["model", "trễ ≤ ms", "policy"])
        parts += [f"## 2b. So sánh ở CÙNG độ trễ (validation: nhầm ≤ {args.budget:.0%} và trễ trung vị ≤ mức; số liệu TEST)", "",
                  "Giữ cử chỉ lâu cũng là một cách \"chờ thêm bằng chứng\", nên so C2/C3/C4 với C1 phải ở cùng độ trễ.", "",
                  md_table(lview), ""]
        if ci_all is not None and ci_all.latency_cap.notna().any():
            parts += ["### Khoảng tin cậy 95% ở cùng độ trễ", "", ci_table(ci_all[ci_all.latency_cap.notna()]), ""]

    rows = []
    for (run, policy), g in selected.groupby(["run", "policy"], sort=False):
        for _, r in g.sort_values("budget").iterrows():
            rows.append({"model": run, "policy": POLICY_VI[policy], "ngân sách": f"{r.budget:.0%}",
                         "đúng lệnh sạch %": pct(r.test_clean_success_rate), "nhầm %": pct(r.test_non_command_trigger_rate),
                         "sai chiều %": pct(r.test_wrong_command_rate), "latency ms": r.test_median_latency_ms})
    parts += ["## 3. Theo từng ngân sách kích hoạt nhầm (TEST)", "", md_table(pd.DataFrame(rows)), ""]

    d = defaults.copy()
    d["policy"] = d.policy.map(POLICY_VI)
    dview = pd.DataFrame({"model": d.run, "cấu hình app": d.setting, "policy": d.policy,
                          "đúng lệnh sạch %": d.test_clean_success_rate.map(pct),
                          "sai chiều %": d.test_wrong_command_rate.map(pct),
                          "nhầm %": d.test_non_command_trigger_rate.map(pct),
                          "latency ms": d.test_median_latency_ms})
    parts += ["## 4. Mốc: tham số app đang dùng, chưa chỉnh (TEST)", "", md_table(dview), ""]

    best_runs = list(dict.fromkeys(["gru_L16"] + list(selected.sort_values("test_clean_success_rate", ascending=False).run.head(2))))
    fig = Path(args.out).with_name("tradeoff_policies.png")
    if plot_tradeoff(tuning, best_runs, fig):
        parts += ["## 5. Đường đánh đổi trên validation", "", f"![tradeoff]({fig.name})", ""]
    Path(args.out).write_text("\n".join(parts), encoding="utf-8")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
