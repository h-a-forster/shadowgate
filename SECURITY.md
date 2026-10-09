# Security policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |
| < 0.1   | No        |

## Reporting a vulnerability

Please do not report security vulnerabilities through public issues, discussions or pull
requests.

Report them privately through GitHub's private vulnerability reporting:
<https://github.com/h-a-forster/shadowgate/security/advisories/new> (or the repository's
**Security** tab, **Report a vulnerability**). This creates a draft security advisory visible
only to the maintainers. Include a description of the issue, steps to reproduce, the
affected version, and the impact you expect.

You can expect an initial response within a few days. Fixes are released as patch versions and
disclosed through a GitHub security advisory once a fix is available.

## Scope and handling notes

- **API keys.** shadowgate reads API keys only from environment variables named in config
  (`api_key_env`). A literal `api_key` field in config is rejected. Keys are never written to
  ledgers, response caches or logs. A path by which a key ends up in any of these is a
  vulnerability.
- **Ledgers and caches contain sensitive data.** The SQLite ledger, response cache, JSONL exports
  and generated reports store prompts and model outputs verbatim. Treat them with the same care
  as the data you send to the models: keep them out of version control and public artifacts, and
  restrict access to them.
- **Configs are code.** The `command` backend executes the commands named in a config, and other
  backends send data to the endpoints a config names. Only run configs you trust, exactly as you
  would only run scripts you trust. Running an untrusted config is out of scope for vulnerability
  reports.
- **Model outputs are untrusted input.** shadowgate treats model output as data; it does not
  execute it, and HTML reports must escape model-generated text. A report or export that allows
  injected markup or script is in scope.
