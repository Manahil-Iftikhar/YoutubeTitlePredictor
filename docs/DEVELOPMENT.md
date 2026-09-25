# Development guide

## Environment preparation

The historical bytecode filenames indicate Python 3.12. Use an isolated environment; dependency compatibility has not been revalidated during this documentation refresh.

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

There are two historical dependency snapshots:

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

## Runtime blockers found in source review

| Location | Issue | Required repair |
| --- | --- | --- |
| `analysis/urls.py` | Refers to `views.predict_engagement`, but the view is named `predict_engagement_view` | Align the route and callable |
| `analysis/views.py` | Wildcard import triggers API requests, file writes, NLTK download, and training | Make imports side-effect free and call services explicitly |
| `analysis/youtube_seo_tool.py` | Reads `competition` although collection does not create it | Define and validate the feature schema |
| `analysis/views.py` and `templates/results.html` | View passes `video_data`; template expects `video_title` and `prediction` | Define a consistent result contract |
| `extra files/seo_tool.py` | Reads `youtube_video_metadata.csv`, while collection writes other filenames | Use an explicit shared data path |
| `extra files/seo_tool.py` | Keyword analysis expects `tags`, which trending collection does not supply; stop words supplied as a set | Normalize schema and use supported vectorizer parameters |

After these issues are repaired, the intended Django workflow is `python manage.py check`, `python manage.py migrate`, then `python manage.py runserver`. These are future verification steps, not a claim that this revision passes them.

## Verification of this refresh

The changed Python files were syntax-checked without importing or executing them. Documentation links and credential substitutions were checked. No API requests, training runs, or end-to-end web tests were performed.

The original application behavior remains incomplete. This refresh improves documentation and credential configuration; it does not report the application as repaired.
