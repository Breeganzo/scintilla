# Running with Docker

For a machine where Docker is available. This is the shorter path - one
command brings up PostgreSQL, OpenSearch and the API together, already
networked and configured.

If Docker is not available on your machine, see
[LOCAL_SETUP.md](LOCAL_SETUP.md) instead.

---

## Install Docker

**macOS** - Docker Desktop, or the lighter Colima:

```bash
# Docker Desktop
brew install --cask docker
open -a Docker

# or Colima: no GUI, no licence restrictions, lower memory use
brew install colima docker docker-compose
colima start --cpu 4 --memory 8
```

Colima is worth knowing about - Docker Desktop requires a paid licence for
larger companies, Colima does not.

**Linux**:

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker "$USER"    # then log out and back in
```

Confirm:

```bash
docker --version
docker compose version
docker run --rm hello-world
```

---

## Start the stack

```bash
cd scintilla
cp .env.example .env

# Generate a secret key and put it in .env
docker run --rm python:3.13-slim python -c \
  "import secrets; print(secrets.token_urlsafe(50))"

docker compose up --build
```

First run takes several minutes: it pulls the Postgres and OpenSearch images
and compiles Python dependencies. Later runs start in seconds because Docker
caches both.

When the logs settle:

- API: http://localhost:8000/api/
- Schema: http://localhost:8000/api/docs/
- OpenSearch: http://localhost:9200
- Airflow: http://localhost:8080 (from Phase 2)

---

## Daily commands

```bash
docker compose up                  # start
docker compose up -d               # start detached
docker compose logs -f api         # follow one service's logs
docker compose ps                  # what is running and healthy
docker compose down                # stop, keep data
docker compose down -v             # stop and DELETE all data
docker compose restart api         # restart one service
```

`down -v` removes the named volumes. That deletes your database and your
search index. It is the correct fix for a corrupted local state and a very
expensive mistake otherwise.

---

## Running commands inside a container

```bash
docker compose exec api python manage.py migrate
docker compose exec api python manage.py createsuperuser
docker compose exec api pytest
docker compose exec api bash                    # shell inside the container
docker compose exec postgres psql -U scintilla  # database shell
```

`exec` runs in an already-running container. Use `run --rm` if the stack is
not up:

```bash
docker compose run --rm api python manage.py migrate
```

---

## Rebuilding

Docker caches layers aggressively, which is usually what you want and
occasionally not.

```bash
docker compose build              # rebuild, using cache
docker compose build --no-cache   # rebuild from scratch
docker compose up --build         # rebuild then start
```

Rebuild when `requirements/*.txt` or the `Dockerfile` change. Application code
does not need a rebuild - it is bind-mounted into the container, so edits take
effect immediately.

---

## Why the compose file looks the way it does

**Health checks and `condition: service_healthy`.** Compose starts containers
in dependency order but does not wait for a service to be *ready*. Postgres
accepts connections several seconds after its container starts. Without the
health check, Django races it and crashes on first boot - intermittently,
which is the worst kind of bug.

**`OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m`.** OpenSearch sizes its heap from
available system RAM by default. On the free-tier VM this project deploys to,
that guess is wrong and the container gets killed by the OOM reaper. Pinning
it makes behaviour identical everywhere.

**`DISABLE_SECURITY_PLUGIN=true`.** Local development only. Certificate
management for a service reachable only from localhost is friction with no
benefit. Production enables it - see `docker-compose.prod.yml` and
`SECURITY.md`.

**Bind mount `.:/app`.** Code changes reload without a rebuild. Production
does the opposite and bakes code into the image, so the running container
cannot be modified.

**Non-root user in the Dockerfile.** A container process running as root that
gets compromised gives the attacker root inside the container, and a kernel
exploit away from root on the host.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot connect to the Docker daemon` | Docker not running | `open -a Docker` or `colima start` |
| `port is already allocated` | Something already on 5432/8000/9200 | `lsof -i :5432`, stop it, or change the host port in `docker-compose.yml` |
| OpenSearch exits with code 137 | Out of memory | Raise Docker's memory limit to 8 GB, or lower `-Xmx` |
| `max virtual memory areas too low` | Host `vm.max_map_count` | Linux: `sudo sysctl -w vm.max_map_count=262144` |
| Django cannot reach Postgres | Using `localhost` instead of the service name | Inside compose the host is `postgres`, not `localhost` |
| Changes not appearing | Edited a file outside the bind mount | `docker compose up --build` |
| Disk filling up | Old images and volumes | `docker system prune -a --volumes` (deletes unused data) |

---

## Deploying

Production deployment onto a free-tier VM behind Cloudflare Tunnel is covered
separately in the deployment guide. It uses the same compose file with
`docker-compose.prod.yml` layered on top:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```
