# Credential handling

The original source contained hardcoded YouTube API keys and a Django secret. The current source reads them from environment variables instead.

**Repository history may still contain the original values.** Removing them from the latest files does not revoke them.

If any original value was real or used:
1. Revoke or rotate the YouTube API key in its owning Google Cloud project.
2. Replace the Django secret in any environment that used it.
3. Configure API restrictions and quota limits appropriate to your application.
4. Inspect tracked bytecode and historical commits before any credential-cleanup or history-rewrite operation.

No original credential was tested or used during the documentation refresh. No history was rewritten.

Keep local secrets out of commits, screenshots, logs, and issue reports. `.env` files are ignored; `.env.example` contains placeholders only.

The existing Django settings enable development debugging. This repository is not configured or validated for production deployment.
