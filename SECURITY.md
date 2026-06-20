# Security Policy

## Supported Version

Security fixes are applied to the latest revision of the default branch.

## Reporting a Vulnerability

Use GitHub private vulnerability reporting when available. Do not include API keys, AWS credentials, dataset samples, or other private information in a public issue.

Include the affected component, reproduction steps, impact, and a minimal sanitized example. Allow maintainers time to investigate before public disclosure.

## Credential Handling

- Never commit `.env`, Streamlit secrets, cloud credentials, private keys, or signed certificates.
- Desktop and Streamlit settings are stored in `%APPDATA%\AutoLabel\.env` as plaintext for the current Windows user.
- Rotate a credential immediately if it is exposed in Git history, logs, screenshots, datasets, or issue attachments.
- Do not expose the local Streamlit server to an untrusted network without adding authentication and transport security.

## Dataset Privacy

Images, labels, metrics, checkpoints, generated reports, and workspace outputs are excluded from Git by default. Review every artifact separately before attaching it to an issue or publishing a release.
