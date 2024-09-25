import argparse
import json
import random
from os import environ
from typing import Optional

from openai import OpenAI
from tqdm import tqdm


def cron_component_sampler(start: int, end: int, weights: Optional[list] = None):
    def star():
        return "*"

    def random_singleton():
        return str(random.choices([
            start,
            random.randint(start + 1, end)
        ], weights=[0.67, 0.33])[0])

    def random_range():
        rstart = random.randint(start, end - 1)
        rend = random.randint(rstart, end)
        return f"{rstart}-{rend}"

    def random_step():
        step_start = random.choices([
            start,
            random.randint(start + 1, end - 2)
        ], weights=[0.67, 0.33])[0]
        step_size = random.randint(1, end - step_start - 1)
        return f"{step_start}/{step_size}"

    def random_list():
        return ",".join([str(x) for x in sorted(random.choices(range(start, end + 1), k=random.randint(1, 3)))])

    def _sampler():
        return random.choices([star, random_singleton, random_range, random_step, random_list],
                              k=1,
                              weights=weights or [
                                  0.5,
                                  0.3,
                                  0.1,
                                  0.05,
                                  0.05,
                              ])[0]()

    return _sampler


cron_minute_sampler = cron_component_sampler(0, 59, weights=[0.2, 0.6, 0.1, 0.05, 0.05])  # Minutes are more likely to be 0
cron_hour_sampler = cron_component_sampler(0, 23)
cron_date_samper = cron_component_sampler(1, 31)
cron_month_sampler = cron_component_sampler(0, 11)
cron_day_sampler = cron_component_sampler(1, 7)


def random_cron():
    return " ".join([
        cron_minute_sampler(),
        cron_hour_sampler(),
        cron_date_samper(),
        cron_month_sampler(),
        cron_day_sampler(),
    ])


def format_cron(expr: str):
    minute, hour, date, month, day = expr.split()

    return f"""{expr}

- Minute: {minute}
- Hour: {hour}
- Date: {date}
- Month: {month}
- Day: {day}"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dataset for Cronformer")
    parser.add_argument("-n", "--num-samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("-o", "--output-file", type=str, default="cronformer.jsonl", help="Output file for the dataset")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI base URL")
    parser.add_argument("--model", type=str, default="gpt-4-turbo", help="OpenAI model")
    args = parser.parse_args()

    client = OpenAI(
        api_key=args.api_key if args.api_key is not None else environ["OPENAI_API_KEY"],
        base_url=args.base_url if args.base_url is not None else environ.get("OPENAI_BASE_URL"),
    )

    instructions = f"""\
Generate a user prompt that would result in the given cron expression.

Introduce a fictional scenario for a task that would run periodically under the provided schedule. The prompt must
precisely be tailored to the cron schedule, making sure each non-default component of the schedule is explicitly mentioned.

Your output should not have any formatting, without mentioning this meta-task.
Keep your response concise, and no more than 1 sentence.
"""
    k_shot_examples = [
        (
            "* * * * *",
            "Run a an uptime check every minute"
        ),
        (
            "0 0 * * *",
            "Send an email everyday at midnight"
        ),
        (
            "0 0 * * 0",
            "Run a weekly marketing report at midnight on Sunday"
        ),
        (
            "0 0 1 * *",
            "Send payments every midnight of the first day of the month"
        ),
        (
            "0 0 1 1 *",
            "Create a yearly report at midnight on the first day of the year"
        ),
        (
            "1-59/2 * * * *",
            "Sync data every 2 minutes",
        ),
        (
            "0 0 1,15 * *",
            "Send salaries the 1st and 15th of every month",
        ),
        (
            "0 0 * * 1-5",
            "Start each weekday by sending a daily digest",
        ),
        (
            "1-59/2 9-17 * * 1-5",
            "Track stock prices every 2 minutes between 9am and 5pm on weekdays",
        ),
        (
            "11,17,18 * * * *",
            "Update DNS records at the 11th, 17th, and 18th minute of every hour",
        ),
        (
            "15,30,42 9-17 * * 1-5",
            "Send reminders at 15, 30, and 42 minutes past the hour between 9am and 5pm on weekdays",
        ),
    ]

    examples: list[tuple[str, str]] = []

    with open(args.output_file, "w") as output_file:
        for _ in tqdm(range(args.num_samples)):
            output = random_cron()
            input = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": instructions},
                    *[
                        item for pair in (
                        [
                            {"role": "user", "content": format_cron(example[0])},
                            {"role": "assistant", "content": example[1]},
                        ]
                        for example in random.choices(k_shot_examples))
                        for item in pair
                    ],
                    {"role": "user", "content": output}
                ],
                max_tokens=512,
                temperature=0.9,
                top_p=0.9,
            ).choices[0].message.content

            examples.append((output, input))

            replace_index = random.randint(0, len(k_shot_examples) - 1)
            k_shot_examples[replace_index] = (output, input)

            output_file.write(json.dumps({"input": input, "output": output}) + "\n")
            output_file.flush()


