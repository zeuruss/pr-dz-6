import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

status_map = {
    "planned_installation": "planned_installation",
    "planned": "planned_installation",
    "ok": "operational",
    "op": "operational",
    "operational": "operational",
    "faulty": "faulty",
    "broken": "faulty",
    "maintenance_scheduled": "maintenance_scheduled",
    "maintenance": "maintenance_scheduled",
}

stage_names = {
    "read_files": "чтение",
    "warranty": "гарантия",
    "clinics": "клиники",
    "calibration": "калибровка",
    "summary": "сводная",
    "save_reports": "запись",
    "total": "итог",
}

stage_order = [
    "read_files",
    "warranty",
    "clinics",
    "calibration",
    "summary",
    "save_reports",
    "total",
]

report_by_stage = {
    "clinics": "clinics_problems.xlsx",
    "calibration": "calibration.xlsx",
    "summary": "clinic_equipment_summary.xlsx",
}


def discover_input_files(base_dir: Path) -> list[Path]:
    """найти входные xlsx файлы в папке проекта"""
    files = sorted(base_dir.glob("medical_diagnostic_devices_*.xlsx"))
    return files or sorted(base_dir.glob("*.xlsx"))


def load_data(file_path: Path) -> pd.DataFrame:
    """загрузить xlsx и привести данные к нужному виду"""
    df = pd.read_excel(file_path)

    date_cols = (
        "install_date",
        "warranty_until",
        "last_calibration_date",
        "last_service_date",
    )
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")

    numeric_cols = ("issues_reported_12mo", "failure_count_12mo", "uptime_pct")
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    normalized = df["status"].astype(str).str.strip().str.lower()
    df["status_normalized"] = normalized.map(status_map).fillna("unknown")

    invalid_calibration = (
        df["last_calibration_date"].notna()
        & df["install_date"].notna()
        & (df["last_calibration_date"] < df["install_date"])
    )
    df.loc[invalid_calibration, "last_calibration_date"] = pd.NaT

    return df


def build_warranty_parts(
    df: pd.DataFrame,
    today: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """разбить устройства по статусу гарантии"""
    expired = df[df["warranty_until"] < today].copy()
    active = df[df["warranty_until"] >= today].copy()
    unknown = df[df["warranty_until"].isna()].copy()

    for part in (expired, active, unknown):
        part.sort_values(["clinic_name", "warranty_until", "device_id"], inplace=True)

    return expired, active, unknown


def build_clinics_problems(df: pd.DataFrame) -> pd.DataFrame:
    """собрать рейтинг клиник по проблемам"""
    clinics = (
        df.groupby(["clinic_id", "clinic_name", "city"], dropna=False)
        .agg(
            devices_count=("device_id", "nunique"),
            issues_total_12mo=("issues_reported_12mo", "sum"),
            failures_total_12mo=("failure_count_12mo", "sum"),
            avg_uptime_pct=("uptime_pct", "mean"),
        )
        .reset_index()
    )

    cols = ["issues_total_12mo", "failures_total_12mo", "avg_uptime_pct"]
    clinics[cols] = clinics[cols].fillna(0)
    clinics = clinics.sort_values(
        ["issues_total_12mo", "failures_total_12mo"],
        ascending=False,
    )

    return clinics


def build_calibration_report(df: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """собрать отчет по срокам калибровки"""
    base_date = df["last_calibration_date"].fillna(df["install_date"])
    calibration = df[
        [
            "device_id",
            "clinic_id",
            "clinic_name",
            "city",
            "model",
            "install_date",
            "last_calibration_date",
            "status_normalized",
        ]
    ].copy()

    calibration["next_calibration_due"] = base_date + pd.DateOffset(months=12)
    calibration["days_to_due"] = (calibration["next_calibration_due"] - today).dt.days
    calibration["calibration_status"] = "ok"
    calibration.loc[
        calibration["next_calibration_due"].notna()
        & (calibration["days_to_due"] < 0),
        "calibration_status",
    ] = "overdue"
    calibration.loc[
        calibration["next_calibration_due"].notna()
        & calibration["days_to_due"].between(0, 30),
        "calibration_status",
    ] = "due_30_days"
    calibration = calibration.sort_values(
        ["calibration_status", "days_to_due", "clinic_name"]
    )

    return calibration


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """собрать сводную таблицу по клиникам и моделям"""
    summary = (
        df.groupby(["clinic_id", "clinic_name", "city", "model"], dropna=False)
        .agg(
            devices_count=("device_id", "nunique"),
            avg_uptime_pct=("uptime_pct", "mean"),
            issues_total_12mo=("issues_reported_12mo", "sum"),
            failures_total_12mo=("failure_count_12mo", "sum"),
        )
        .reset_index()
        .sort_values(["clinic_name", "model"])
    )

    return summary


def prepare_output_dir(reports_dir: Path) -> None:
    """подготовить папку отчетов и удалить старые xlsx"""
    reports_dir.mkdir(parents=True, exist_ok=True)

    for old_file in reports_dir.glob("*.xlsx"):
        old_file.unlink()


def save_report_file(file_path: Path, frame: pd.DataFrame) -> None:
    """сохранить один отчет в xlsx"""
    frame.to_excel(file_path, index=False)


def run_timed(func, *args):
    """запустить функцию и вернуть результат плюс время"""
    start = time.perf_counter()
    result = func(*args)
    elapsed = time.perf_counter() - start

    return result, elapsed


def put_stage_result(
    reports: dict[str, pd.DataFrame],
    stage_key: str,
    result,
) -> None:
    """добавить результат этапа в словарь отчетов"""
    if stage_key == "warranty":
        expired, active, unknown = result
        reports["warranty_expired.xlsx"] = expired
        reports["warranty_active.xlsx"] = active
        reports["warranty_unknown.xlsx"] = unknown

    elif stage_key in report_by_stage:
        reports[report_by_stage[stage_key]] = result


async def run_async_pipeline(
    input_files: list[Path],
    output_dir: Path,
) -> dict[str, float]:
    """запустить async вариант и замерить время по этапам"""
    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    read_start = time.perf_counter()
    read_tasks = [asyncio.to_thread(load_data, file_path) for file_path in input_files]
    frames = await asyncio.gather(*read_tasks)
    timings["read_files"] = time.perf_counter() - read_start

    df = pd.concat(frames, ignore_index=True)
    today = pd.Timestamp.today().normalize()

    build_tasks = {
        "warranty": asyncio.to_thread(run_timed, build_warranty_parts, df, today),
        "clinics": asyncio.to_thread(run_timed, build_clinics_problems, df),
        "calibration": asyncio.to_thread(
            run_timed,
            build_calibration_report,
            df,
            today,
        ),
        "summary": asyncio.to_thread(run_timed, build_summary_table, df),
    }

    build_results = await asyncio.gather(*build_tasks.values())

    reports: dict[str, pd.DataFrame] = {}

    for key, payload in zip(build_tasks.keys(), build_results):
        result, elapsed = payload
        timings[key] = elapsed
        put_stage_result(reports, key, result)

    prepare_output_dir(output_dir)

    save_start = time.perf_counter()
    save_tasks = [
        asyncio.to_thread(save_report_file, output_dir / file_name, frame)
        for file_name, frame in reports.items()
    ]
    await asyncio.gather(*save_tasks)
    timings["save_reports"] = time.perf_counter() - save_start

    timings["total"] = time.perf_counter() - total_start

    return timings


def run_threading_pipeline(
    input_files: list[Path],
    output_dir: Path,
) -> dict[str, float]:
    """запустить threading вариант и замерить время по этапам"""
    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    workers = max(2, min(8, len(input_files)))
    read_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        read_futures = [
            executor.submit(load_data, file_path)
            for file_path in input_files
        ]
        frames = [future.result() for future in read_futures]

    timings["read_files"] = time.perf_counter() - read_start

    df = pd.concat(frames, ignore_index=True)
    today = pd.Timestamp.today().normalize()

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {
            "warranty": executor.submit(run_timed, build_warranty_parts, df, today),
            "clinics": executor.submit(run_timed, build_clinics_problems, df),
            "calibration": executor.submit(
                run_timed, build_calibration_report, df, today
            ),
            "summary": executor.submit(run_timed, build_summary_table, df),
        }

        reports: dict[str, pd.DataFrame] = {}

        for key, future in future_map.items():
            result, elapsed = future.result()
            timings[key] = elapsed
            put_stage_result(reports, key, result)

    prepare_output_dir(output_dir)

    save_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=6) as executor:
        save_futures = [
            executor.submit(save_report_file, output_dir / file_name, frame)
            for file_name, frame in reports.items()
        ]

        for future in save_futures:
            future.result()

    timings["save_reports"] = time.perf_counter() - save_start
    timings["total"] = time.perf_counter() - total_start

    return timings


def _faster_mode(async_time: float, thread_time: float) -> tuple[str, float]:
    """определить какой режим быстрее на этапе"""
    if async_time == thread_time:
        return "равно", 1.0

    if async_time < thread_time:
        if async_time == 0:
            return "async", 0.0

        return "async", thread_time / async_time

    if thread_time == 0:
        return "threading", 0.0

    return "threading", async_time / thread_time


def save_timing_report(
    reports_dir: Path,
    async_timing: dict[str, float],
    threading_timing: dict[str, float],
    files_count: int,
) -> None:
    """сохранить подробное сравнение времени в txt"""
    lines = ["async vs threading", f"файлов: {files_count}", ""]

    for stage in stage_order:
        async_value = async_timing.get(stage, 0.0)
        thread_value = threading_timing.get(stage, 0.0)
        faster, ratio = _faster_mode(async_value, thread_value)

        lines.append(
            f"{stage_names[stage]}: "
            f"async={async_value:.4f}, "
            f"threading={thread_value:.4f}, "
            f"лучше={faster}, x{ratio:.2f}"
        )

    with open(
        reports_dir / "execution_time_comparison.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n".join(lines))


def print_timing(
    async_timing: dict[str, float],
    threading_timing: dict[str, float],
) -> None:
    """вывести подробное сравнение времени в консоль"""
    print("\nсравнение async и threading")

    for stage in stage_order:
        async_value = async_timing.get(stage, 0.0)
        thread_value = threading_timing.get(stage, 0.0)
        faster, ratio = _faster_mode(async_value, thread_value)

        print(
            f"{stage_names[stage]}: "
            f"async={async_value:.4f} | "
            f"threading={thread_value:.4f} | "
            f"лучше: {faster} (x{ratio:.2f})"
        )


def main() -> None:
    """точка входа для 6 проекта"""
    base_dir = Path(__file__).resolve().parent
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    input_files = discover_input_files(base_dir)

    if not input_files:
        raise FileNotFoundError("xlsx файлы не найдены в папке 2 сем/6")

    print("найдены входные файлы:")

    for file_path in input_files:
        print(f"- {file_path.name}")

    async_dir = reports_dir / "async"
    threading_dir = reports_dir / "threading"

    async_timing = asyncio.run(run_async_pipeline(input_files, async_dir))
    threading_timing = run_threading_pipeline(input_files, threading_dir)

    save_timing_report(reports_dir, async_timing, threading_timing, len(input_files))

    print_timing(async_timing, threading_timing)

    print("\nотчеты сохранены:")
    print(f"- {async_dir}")
    print(f"- {threading_dir}")
    print(f"- {reports_dir / 'execution_time_comparison.txt'}")


if __name__ == "__main__":
    main()
