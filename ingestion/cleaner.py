from config.settings import DATA_DIR
import json
import re
from pathlib import Path


INPUT_PATH = (DATA_DIR / "raw/te_website_content.json")
OUTPUT_PATH = (DATA_DIR / "processed/te_website_clean.json")


# UI-only lines that add no useful knowledge
NOISE_LINES = {
    "تفاصيل أكتر",
    "تفاصيل اكثر",
    "تفاصيل أكثر",
    "المزيد",
    "اعرف المزيد",
    "أضف للمقارنة",
    "اضف للمقارنة",
    "اشترك",
    "إشترك",
    "إشترك الأن",
    "اشترك الآن",
    "(current)",
}


def normalize_whitespace(text: str) -> str:
    lines = []

    for line in text.splitlines():
        # Collapse repeated spaces/tabs
        line = re.sub(
            r"[ \t]+",
            " ",
            line,
        ).strip()

        if not line:
            continue

        lines.append(line)

    return "\n".join(lines).strip()


def remove_ui_noise(text: str) -> str:
    cleaned_lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if line in NOISE_LINES:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def clean_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = remove_ui_noise(text)
    text = normalize_whitespace(text)

    return text


def main():
    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        documents = json.load(file)

    cleaned_documents = []
    skipped_empty = 0

    for document in documents:
        original_text = document["text"].strip()

        # Skip genuinely empty source pages
        if not original_text:
            skipped_empty += 1

            print(
                f'{document["document_id"]}: '
                f'SKIPPED - empty page'
            )

            continue

        cleaned_text = clean_text(
            original_text
        )

        # Safety check after cleaning
        if not cleaned_text:
            skipped_empty += 1

            print(
                f'{document["document_id"]}: '
                f'SKIPPED - empty after cleaning'
            )

            continue

        cleaned_document = {
            **document,
            "text": cleaned_text,
        }

        cleaned_documents.append(
            cleaned_document
        )

        print(
            f'{document["document_id"]}: '
            f'{len(original_text):,} -> '
            f'{len(cleaned_text):,} chars'
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            cleaned_documents,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 80)

    print(
        "Original documents:",
        len(documents),
    )

    print(
        "Skipped empty documents:",
        skipped_empty,
    )

    print(
        "Clean documents:",
        len(cleaned_documents),
    )

    print(
        "Saved to:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()