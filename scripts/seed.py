from app.core.database import SessionLocal
from app.seed_data import seed_database


def main() -> None:
    with SessionLocal() as db:
        seed_database(db)
    print("Seeded patients and heart recordings.")


if __name__ == "__main__":
    main()
