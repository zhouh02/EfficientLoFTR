import argparse
import ast
import csv
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REQUIRED_KEYS = {"auc@5", "auc@10", "auc@20", "num_matches", "prec@1e-04"}


def parse_metric_dicts(text: str):
    """
    从完整日志中提取所有 Python dict，
    只保留包含最终评估指标的 dict。
    """
    metric_dicts = []

    # 匹配多行 {...}
    candidates = re.findall(r"\{[\s\S]*?\}", text)

    for cand in candidates:
        try:
            obj = ast.literal_eval(cand)
        except Exception:
            continue

        if not isinstance(obj, dict):
            continue

        if REQUIRED_KEYS.issubset(set(obj.keys())):
            metric_dicts.append(obj)

    return metric_dicts


def run_one_eval(cmd: str, run_idx: int, log_dir: Path):
    log_path = log_dir / f"run_{run_idx:03d}.log"

    print(f"\n========== Run {run_idx} ==========")
    print(f"Command: {cmd}")
    print(f"Log: {log_path}")

    start_time = time.time()

    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines = []
    with open(log_path, "w", encoding="utf-8") as f:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            f.write(line)
            lines.append(line)

    return_code = process.wait()
    duration = time.time() - start_time

    full_text = "".join(lines)
    metric_dicts = parse_metric_dicts(full_text)

    if len(metric_dicts) == 0:
        print(f"[Run {run_idx}] WARNING: No metric dict found.")
        return {
            "run": run_idx,
            "status": "no_metrics",
            "return_code": return_code,
            "duration_sec": duration,
            "log_path": str(log_path),
        }

    # 一般最终结果只会有一个；如果有多个，取最后一个
    metrics = metric_dicts[-1]

    result = {
        "run": run_idx,
        "status": "ok" if return_code == 0 else "failed_but_metrics_found",
        "return_code": return_code,
        "duration_sec": duration,
        "log_path": str(log_path),
        **metrics,
    }

    print(f"[Run {run_idx}] Parsed metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    return result


def save_results(results, out_dir: Path):
    json_path = out_dir / "all_eval_results.json"
    csv_path = out_dir / "all_eval_results.csv"
    best_path = out_dir / "best_auc5_result.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "run",
        "status",
        "return_code",
        "duration_sec",
        "auc@5",
        "auc@10",
        "auc@20",
        "num_matches",
        "prec@1e-04",
        "log_path",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    valid_results = [
        r for r in results
        if r.get("status") in {"ok", "failed_but_metrics_found"}
        and "auc@5" in r
    ]

    if len(valid_results) == 0:
        print("\nNo valid result found. Cannot select best auc@5.")
        return None, json_path, csv_path, best_path

    best = max(valid_results, key=lambda x: x["auc@5"])

    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    return best, json_path, csv_path, best_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rounds",
        type=int,
        default=20,
        help="评估轮数，默认 20",
    )
    parser.add_argument(
        "--cmd",
        type=str,
        default="bash scripts/reproduce_test/outdoor_full_auc.sh",
        help="每轮执行的评估命令",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="输出目录，默认自动生成",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="某一轮命令返回非 0 时是否立即停止",
    )

    args = parser.parse_args()

    if args.out_dir is None:
        time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"multi_eval_results_{time_tag}")
    else:
        out_dir = Path(args.out_dir)

    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for i in range(1, args.rounds + 1):
        result = run_one_eval(args.cmd, i, log_dir)
        results.append(result)

        # 每轮结束后都保存一次，防止中途断掉导致结果丢失
        save_results(results, out_dir)

        if args.stop_on_error and result.get("return_code", 0) != 0:
            print(f"\nStop because run {i} returned non-zero code.")
            break

    best, json_path, csv_path, best_path = save_results(results, out_dir)

    print("\n========== Summary ==========")
    print(f"All results JSON: {json_path}")
    print(f"All results CSV : {csv_path}")
    print(f"Best result JSON: {best_path}")

    if best is not None:
        print("\nBest result by auc@5:")
        print(json.dumps(best, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()