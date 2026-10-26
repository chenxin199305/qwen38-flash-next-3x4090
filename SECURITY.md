# Security

## Reporting

Please open a private security advisory on GitHub, or contact the maintainer
through the GitHub profile. Do not paste API keys, SSH keys, FRP passwords or
private hostnames into a public issue.

## Safe defaults

- The server binds to `127.0.0.1` by default.
- Binding to `0.0.0.0` without an API key is refused unless
  `ALLOW_INSECURE=1` is set explicitly.
- Put remote access behind a reverse proxy with TLS and authentication.
  This repository does not ship a tunnel, FRP configuration or public endpoint.
- Keep `.env` and the API-key file at mode `600`.
- `--api-key` is visible in the process list on some systems. Prefer a
  dedicated host account and restrict local access; vLLM also reads
  `VLLM_API_KEY` from the environment.

## Model weights

Weights are not distributed here. Do not commit checkpoints, generated
overlays or shard files. They are large and carry their own license.