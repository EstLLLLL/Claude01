"""One-time helper to mint a Gmail API refresh token.

Prereqs:
  1. Google Cloud project with Gmail API enabled.
  2. OAuth 2.0 client of type "Desktop app".
  3. Download the client JSON and save it next to this script as `credentials.json`.

Then run:
    pip install -r requirements.txt
    python scripts/auth_setup.py

A browser window opens; sign in with the Gmail account whose Drafts folder
should receive the digest. The script prints the three values you need to
add to GitHub secrets:

    GMAIL_CLIENT_ID
    GMAIL_CLIENT_SECRET
    GMAIL_REFRESH_TOKEN

Then delete `credentials.json` and `token.json` (they're gitignored anyway).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/gmail.compose"]


def main() -> int:
    creds_path = Path(__file__).parent.parent / "credentials.json"
    if not creds_path.exists():
        print(f"ERROR: {creds_path} not found. Download it from Google Cloud Console "
              f"(OAuth 2.0 Client → Desktop app → Download JSON).", file=sys.stderr)
        return 1

    flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent")

    raw = json.loads(creds_path.read_text())
    block = raw.get("installed") or raw.get("web") or {}
    client_id = block.get("client_id", "")
    client_secret = block.get("client_secret", "")

    print()
    print("=" * 60)
    print("Add these as GitHub repo secrets:")
    print("=" * 60)
    print(f"GMAIL_CLIENT_ID={client_id}")
    print(f"GMAIL_CLIENT_SECRET={client_secret}")
    print(f"GMAIL_REFRESH_TOKEN={creds.refresh_token}")
    print("=" * 60)
    print()
    print("Done. You can now delete credentials.json (it's gitignored).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
