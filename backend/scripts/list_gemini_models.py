import os
import sys
from pprint import pprint


def list_models(api_key: str):
    try:
        from google.genai.client import Client
    except ImportError as exc:
        raise ImportError(
            "google.genai package is not installed in the current environment. "
            "Install it with `pip install google-genai` or run this script in the project venv."
        ) from exc

    client = Client(api_key=api_key)
    models = client.models.list()
    return models


def main():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if len(sys.argv) > 1:
        api_key = sys.argv[1]

    if not api_key:
        print("Error: GEMINI_API_KEY or GOOGLE_API_KEY is required.")
        print("Usage: python list_gemini_models.py [API_KEY]")
        sys.exit(1)

    print("Using Gemini API key from environment." if len(sys.argv) == 1 else "Using API key from CLI argument.")
    try:
        models = list_models(api_key)
    except Exception as exc:
        print(f"Error listing Gemini models: {exc}")
        sys.exit(2)

    print("Found models:")
    page_number = 1
    while True:
        try:
            for model in models.page:
                # Some model objects may be dataclasses or dict-like
                try:
                    name = getattr(model, "name", None) or model.get("name")
                except Exception:
                    name = str(model)
                print(f"- {name}")
                try:
                    info = {
                        k: getattr(model, k, None) or (model.get(k) if isinstance(model, dict) else None)
                        for k in ["display_name", "description", "type", "available"]
                        if getattr(model, k, None) or (isinstance(model, dict) and model.get(k))
                    }
                    if info:
                        pprint(info, indent=4)
                except Exception:
                    pass
            try:
                if not models.next_page():
                    break
            except (IndexError, RuntimeError):
                break
            page_number += 1
        except IndexError:
            break


if __name__ == "__main__":
    main()
