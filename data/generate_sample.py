"""Generate a deterministic, fictional subscription-retention dataset."""
import csv
import math
import random
from pathlib import Path


SEED = 20261017
ROWS = 800
PLANS = {
    "starter": (19, 0.43),
    "standard": (39, 0.35),
    "pro": (79, 0.22),
}
REGIONS = ("north", "south", "east", "west")


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


def generate(path: Path):
    random.seed(SEED)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["tenure_months", "monthly_charge", "support_tickets", "weekly_usage_hours", "plan", "region", "autopay", "churned"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index in range(ROWS):
            plan = random.choices(list(PLANS), weights=(.38, .44, .18), k=1)[0]
            price, baseline = PLANS[plan]
            tenure = random.randint(1, 60)
            tickets = random.choices(range(7), weights=(.28,.22,.18,.13,.09,.06,.04), k=1)[0]
            usage = round(max(.5, random.gauss(5.2 if plan == "starter" else 9 if plan == "standard" else 15, 3.5)), 1)
            autopay = random.random() < .67
            logit = math.log(baseline / (1 - baseline)) - .043 * (tenure - 12) + .31 * tickets - .12 * usage - (.28 if autopay else 0)
            probability = sigmoid(logit + random.gauss(0, .55))
            writer.writerow({"tenure_months":tenure,"monthly_charge":price,"support_tickets":tickets,"weekly_usage_hours":usage,"plan":plan,"region":random.choice(REGIONS),"autopay":str(autopay).lower(),"churned":int(random.random()<probability)})
    print(f"Wrote {ROWS} synthetic rows to {path}")


if __name__ == "__main__":
        generate(Path(__file__).parents[1] / "src" / "modelbench" / "data" / "subscriptions.csv")
