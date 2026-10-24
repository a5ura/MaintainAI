"""Run with: python -m scripts.seed [--db path/to/file.db]."""

import argparse

from maintainai.database import connect, default_db_path, initialize
from maintainai.seed import seed_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Add synthetic MaintainAI demo data without overwriting existing records.")
    parser.add_argument("--db", default=default_db_path(), help="SQLite database path")
    args = parser.parse_args()
    with connect(args.db) as connection:
        initialize(connection)
        result = seed_demo(connection)
    print(f"Synthetic demo data: {result['equipment_added']} equipment and {result['issues_added']} issues added to {args.db}.")


if __name__ == "__main__":
    main()
