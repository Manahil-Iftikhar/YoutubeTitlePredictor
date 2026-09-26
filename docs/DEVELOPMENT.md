# Development guide

## Environment preparation

The web workflow was tested with Python 3.12, Django 5.2.17, and requests 2.34.2. Use an isolated environment.

```bash
git clone https://github.com/Manahil-Iftikhar/YoutubeTitlePredictor.git
cd YoutubeTitlePredictor
python -m venv .venv
```

Activate in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or in macOS/Linux:

```bash
source .venv/bin/activate
```

Install the minimal web dependencies with `python -m pip install -r requirements.txt`. The web app requires only Django and requests.

There are also two historical ML dependency snapshots:

- `packages.txt`: the smaller snapshot, including Django, pandas, scikit-learn, Google API client, NLTK, requests, and aiohttp.
- `reqirements.txt`: the original spelling is retained; this broader environment export includes unrelated tools and Windows-specific packages.

Review and update these snapshots in an isolated environment before installing. They are not a maintained dependency lockfile or a security endorsement.

## Credentials

Read [SECURITY.md](../SECURITY.md) first. Both API scripts now read `YOUTUBE_API_KEY`; Django settings read `DJANGO_SECRET_KEY`. Missing variables raise `KeyError` rather than falling back to a committed credential.

Set values locally in your terminal. For example, in PowerShell:

```powershell
$env:YOUTUBE_API_KEY = "<your-own-key>"
$env:DJANGO_SECRET_KEY = "<new-local-secret>"
```

Or in bash:

```bash
export YOUTUBE_API_KEY="<your-own-key>"
export DJANGO_SECRET_KEY="<new-local-secret>"
```

`.env.example` is a reference only. No dotenv loader is configured. Never commit actual values.

## Run the web application

```bash
python manage.py check
python manage.py migrate
python manage.py runserver
```

Open http://127.0.0.1:8000/. The landing page does not call YouTube or train models. Submit a topic to retrieve up to ten video snippets. Missing keys, invalid inputs, empty results, timeouts, and provider failures are handled in the interface.

The interface shows retrieved titles, channels, descriptions, and video links. It does not predict engagement or generate titles. Competition and search-volume inputs were removed because the web workflow did not use them.

## Offline verification

```bash
python manage.py test analysis
```

Eight tests cover page loading, invalid submissions, escaped result text, empty results, error display, CSRF protection, HTTP methods, credential configuration, safe video links, request timeouts, and malformed provider responses. All network requests in service tests are mocked. GitHub Actions runs system checks and these tests.

Validation: Django system checks passed and all eight tests passed locally. No real YouTube key was used, and live API behavior has not been verified.

## Historical ML experiments

`analysis/youtube_seo_tool.py` and `extra files/seo_tool.py` remain historical experiments and are not imported by the web application. They still require separate repair: import-time side effects, missing competition/search-volume measurements, inconsistent CSV paths, and mismatched tags/stop-word handling. See the README for evaluation limitations. Do not run these scripts as part of the web setup.

## Keyless offline demo

Start Django as described above with a local `DJANGO_SECRET_KEY`; `YOUTUBE_API_KEY` may be unset. Open http://127.0.0.1:8000/demo/ or use the landing-page demo link.

The three records in `analysis/demo.py` are fictional and have no outbound video URLs. The demo is explicitly labelled and never substitutes for a failed live search. It makes no API calls. Internet access may still be needed for initial dependency installation.

Three additional tests verify deterministic keyless rendering, the landing-page link, and rejected demo POST requests. The complete suite now contains 11 tests.
