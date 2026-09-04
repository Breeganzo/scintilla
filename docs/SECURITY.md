# Security

This document describes the security posture of Scintilla: what is enforced
today, what is deliberately deferred, and what would have to change before this
handled anything other than public data.

**Threat model in one line.** Scintilla serves publicly available arXiv metadata
through a read-only API. There is no user data, no authentication and nothing
confidential in the corpus. The realistic risks are therefore *availability*
(someone hammering the API or the upstream arXiv endpoint), *supply chain*
(a malicious dependency), and *the machine it runs on being used as a foothold*.

---

## Reporting a vulnerability

Open a GitHub issue describing the problem. If you believe the issue is
sensitive, open an issue asking for a private contact rather than including
details. This is a portfolio project with no SLA — expect a response in days,
not hours.

---

## What is enforced today

### Secrets

- No secret is committed. `.env` is gitignored; `.env.example` contains only
  placeholder values and explanatory comments.
- `DJANGO_SECRET_KEY` has no default in production settings — the application
  refuses to start without it rather than falling back to a known value.
- The Airflow admin password file
  (`airflow/simple_auth_manager_passwords.json.generated`) is gitignored, along
  with `airflow/airflow.cfg`, which can contain connection strings.
- CI has no repository secrets configured, so a malicious pull request cannot
  exfiltrate one.

### Input handling

- **XML parsing uses `defusedxml`, not the standard library.** arXiv responses
  are untrusted XML from the network. Python's `xml.etree` is vulnerable to
  entity-expansion attacks — the "billion laughs" class of denial of service.
  This is the single most important input-handling decision in the codebase.
- All arXiv HTTP requests set an explicit timeout. A hung upstream connection
  cannot pin a worker indefinitely.
- The API is **read-only**. Every write path is a management command run
  deliberately; `POST`, `PUT`, `PATCH` and `DELETE` return `405` on all
  resource endpoints.
- Database access goes through the Django ORM. There is no string-interpolated
  SQL anywhere in the codebase.
- The Airflow DAG validates that `OPENSEARCH_URL` uses an `http` or `https`
  scheme before dereferencing it, so a typo in the environment cannot turn a
  health check into a local file read.

### Transport and browser controls

- `CORS_ALLOWED_ORIGINS` is an explicit allowlist. **`*` is never used**, in
  any environment. A wildcard CORS policy on a credentialed API is one of the
  most common and most damaging web misconfigurations, and the habit of not
  reaching for it matters more than the fact that this particular API has
  nothing to steal.
- `ALLOWED_HOSTS` is set explicitly in production settings.
- Production settings disable `DEBUG`. Django's debug page discloses settings,
  installed apps and local variables from the stack frame.

### Supply chain

- Dependencies are pinned with version ranges in `requirements/`, and CI runs a
  **dependency audit job on every push**.
- Heavy ML dependencies live in `requirements/ml.txt`, separate from
  `requirements/base.txt`, so the deployed application and CI do not install
  PyTorch. Less installed is less attack surface.
- The container image runs as a **non-root user**.

### Verification

CI runs five jobs on every push — lint, tests, dependency audit, container
build, and Compose config validation. The full test suite runs **offline**: no
test touches the network, so a network failure cannot mask a real regression.

---

## Deliberately deferred

These are honest gaps, not oversights. Each is listed with what it would take
to close it.

| Gap | Why it is acceptable today | What closing it needs |
|---|---|---|
| **No authentication on the API** | The corpus is public arXiv metadata; there is nothing to authorise | Token or session auth, plus per-user rate limits |
| **No rate limiting** | Single-user portfolio deployment | DRF throttling, or rate limiting at the Cloudflare edge |
| **OpenSearch runs with its security plugin disabled locally** | It listens on `localhost` only, on a development machine | Enable the security plugin and set credentials in the production stack |
| **PostgreSQL uses passwordless local trust auth** | Local development, `localhost` socket only | Password or certificate auth in production |
| **No Content Security Policy** | No frontend is deployed yet | Set CSP headers when the React app ships |
| **No audit logging** | No authenticated actors to audit | Structured request logging with retention |
| **Container full-stack boot is unverified** | No Docker available on the development machine; the image builds and Compose config validates in CI | Boot the full stack on the deployment VM |

---

## Production deployment posture

Not yet deployed. The intended posture:

- **No inbound ports.** `cloudflared` establishes an outbound tunnel and holds
  it open. The VM has no inbound firewall rules and no publicly reachable
  listener. There is no port to scan and no service to brute-force.
- **No container publishes a port to the host.** A reverse proxy is the only
  entry point, and it is reachable only through the tunnel.
- **The Airflow UI is never publicly reachable.** It can trigger DAGs and
  exposes connection metadata, so it sits behind Cloudflare Access with a
  single-identity policy. Anyone else reaching that hostname gets an identity
  challenge they cannot pass.
- **Secrets are supplied as environment variables**, never baked into the
  image.
- Automatic TLS via Cloudflare.

---

## Dependency and data notes

Scintilla stores only what arXiv publishes openly: titles, abstracts, authors,
categories and identifiers. It does not fetch or store paper PDFs, and it stores
no personal data beyond author names as printed on public preprints.

The arXiv API is queried at **no more than one request every three seconds**, in
line with arXiv's stated expectations. Politeness to an upstream you depend on
is an availability control for both parties.
