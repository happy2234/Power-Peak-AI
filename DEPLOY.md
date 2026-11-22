# Deploying Power Peak AI as a public, read-only dashboard

Goal: a free, public copy of the dashboard that keeps working when your laptop is off.

```
 GitHub (public repo)  ──push──▶  Streamlit Community Cloud  ──▶  public URL (read-only app)
        ▲
        │ daily commit of the data snapshot (GitHub Actions, no secrets)
```

* The **app** only reads files from the repository (`data/processed`, a few reports). It does not train,
  download or run models, so it needs no GPU and no PyTorch (`requirements.txt`).
* A **scheduled GitHub Actions job** runs the daily pipeline on CPU, tests the result and commits the
  refreshed data. Streamlit Community Cloud redeploys on each push.
* Your laptop keeps the full app (`POWER_PEAK_MODE=full`, Update data button) and is unchanged.

Cost: **free** if the repository is public (see "Cost" below). Nothing here needs a credit card.

---

## 1. What is in the repository (audited)

Everything else is ignored by `.gitignore` (an allow-list, not a block-list).

| Path | Why it is there | Size |
|---|---|---|
| `app/`, `src/`, `scripts/`, `.streamlit/config.toml` | the code | ~1 MB |
| `data/processed/*` (33 named files) | tables the app reads and the daily job updates: features, forecasts, scorecard, backtest, evaluation tables, run history | 60 MB* |
| `data/raw/weather/{obs,obsx,fc,fcx}_<city>.parquet` | Open-Meteo history the daily job extends | 12 MB |
| `data/raw/psp_latest/` (exactly ONE `DD.MM.YY_NLDC_PSP*.xls`) | the newest public Grid-India report, shown by the "Raw to processed" page; the daily job replaces it | 0.4 MB |
| `models/{lgbm,lstm,tft}/<latest>/` + `latest.txt` + `best_params*.yaml` | trained models, used only by the daily job | 5 MB |
| `reports/*.md`, `reports/training/*.csv` | what Model lab and Data quality show | 1 MB |
| `tests/` (+ 1 MB of public Grid-India sample files) | the test suite the job runs | 1 MB |

\* on disk; the whole repository is about 33 MB as committed (parquet is already compressed).

Largest single files: `backtest_processed.parquet` 9.2 MB, `weather_fc.parquet` 4.4 MB,
`lgbm/model.pkl` 4.0 MB, `timeseries_15min.parquet` 3.5 MB. GitHub warns above 50 MiB and blocks above
100 MiB per file: nothing is close.

**Deliberately not included** (checked with `git ls-files`, and by searching the tracked files, the parquet
string columns and the model files for `/home/`, user names, e-mail addresses, `key`, `secret`, `token`,
`password`: no matches):

* `data/raw/psp/` (355 MB of Grid-India downloads; only its newest file is copied to `data/raw/psp_latest/`), `data/raw/forecast/`, `data/raw/_index/`: the job
  fetches what it needs and the pipeline skips days already parsed into `daily_summary.parquet`
* `logs/` (pipeline log, lock file, run history with local paths), `*.log`, `run.log`
* experiment outputs: `data/processed/accuracy_round`, `weather_nasa`, `weather_coverage`,
  `baseline_v1`, model snapshots (`models/*_snapshot`, 45 MB), `reports/screenshots`
* `.env*`, `.streamlit/secrets.toml`, `*.pem`, `*.key`
* `RESUME.md` (private notes): left out by `scripts/export_public_repo.sh`

The project uses **no API keys**: Grid-India and Open-Meteo are public and keyless.

## 2. Run it on a clean machine

Python **3.12** (`.python-version`; Streamlit Community Cloud defaults to 3.12).

```bash
# the public app only (what Streamlit Community Cloud installs): no PyTorch
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
POWER_PEAK_READONLY=1 streamlit run app/Home.py
POWER_PEAK_READONLY=1 python scripts/smoke_app.py        # renders the pages, checks the footer

# the daily pipeline and the full test suite: CPU-only PyTorch, no CUDA, no GPU needed
pip install -r requirements-pipeline.txt                  # pulls torch==2.14.1+cpu from download.pytorch.org/whl/cpu
python -m pytest -q
python -m src.pipeline --trigger manual                   # downloads, rebuilds, forecasts
```

Both files pin exact versions of the direct dependencies. This was tested in fresh virtual environments
created from a clean `git clone` of the branch (results in the last section).

## 3. Read-only mode

| Variable | Effect |
|---|---|
| `POWER_PEAK_READONLY=1` | Hides every Update data button, never starts the pipeline, shows "Data snapshot: `<date>`" in the sidebar and the page badge, reads run history from `data/processed/pipeline_runs.jsonl` |
| *(unset or `0`)* | Your laptop: unchanged, with the Update data button |
| `POWER_PEAK_MODE=presentation` (default) / `full` | Which pages are shown; independent of read-only |

Streamlit Community Cloud documents settings as **Secrets**, not as environment variables, so `app/Home.py`
copies `POWER_PEAK_READONLY` and `POWER_PEAK_MODE` from `st.secrets` into the environment when they are not
already set. As a second safety net, the app also treats itself as read-only whenever it runs from
`/mount/src/…` (where Community Cloud runs apps) and the variable is unset. (That path is observed
behaviour, not documented: set the secret anyway.)

Every page footer shows: *Sources: Grid-India (NLDC) … Open-Meteo weather. Not an official forecast. Data
snapshot: 16 Nov 2026 (actuals through 14 Nov 2026).* The snapshot date is the forecast's own issue time
from `latest_meta.json`, not a file date, so it survives a git checkout.

## 4. Keeping the snapshot fresh: `.github/workflows/daily-snapshot.yml`

Runs daily at 06:23 UTC (11:53 IST) and on demand (Actions tab, "Run workflow"). Steps, in order:

1. checkout, Python 3.12, `pip install -r requirements-pipeline.txt` (CPU torch; pip cache)
2. `python -m src.pipeline --trigger timer`: the seven steps (downloads, parse, features, forecast, scorecard)
3. `scripts/ci_snapshot.py check`: **fails** unless the run ended `success` (a failed download counts as a
   failure, so a stale snapshot never looks fresh), the forecast has 7 complete days, the forecast was written
   by this run and the actuals are at most 3 days old
4. `python -m pytest -q` on the new data
5. `scripts/ci_snapshot.py finish` (copies the run history and data-quality report next to the data)
6. commit **only** the allow-listed paths (`data/processed`, `data/raw/weather`, `data/raw/psp_latest`, `reports/data_quality*`) and
   push. Any other changed file aborts the job before the commit.

If any step fails nothing is committed, the job shows red in the Actions tab, and the app keeps showing the previous snapshot. The next day's run retries.

**Secrets needed: none.** **Permissions:** `contents: write` for the automatic `GITHUB_TOKEN`, declared in the
workflow (the only permission it has). If pushes are rejected, check Settings → Actions → General →
Workflow permissions is not set to block writes. There are no repository, organisation or personal secrets.

**Open-Meteo usage per run:** one request per city and data kind: 11 cities × 4 kinds (observed, archived
forecast, extra observed, extra forecast) = 44 in the weather step, plus 11 live-forecast requests when the
forecast is made, = **55 requests per run**, measured in a clean-checkout run (the pipeline prints "Open-Meteo requests this run: N").
Open-Meteo's free tier allows 10,000 calls/day, 5,000/hour and 600/minute for non-commercial use
([terms](https://open-meteo.com/en/terms), [pricing](https://open-meteo.com/en/pricing)); one run is about
0.5 % of the daily limit. Limits are per IP address and GitHub runners share IPs, so a 429 is possible on a
bad day: the step fails, nothing is committed, tomorrow retries. Attribution (CC BY 4.0) is in the footer.

**Things to know about scheduled workflows** (GitHub docs):

* they only run on the **default branch** of the repository, so the public repo's `main` must contain this
  workflow;
* in a public repository GitHub **disables scheduled workflows after 60 days without repository activity**.
  The daily commit should count as activity, but this is not stated explicitly. If the Actions tab shows the
  workflow as disabled, click "Enable workflow";
* scheduled runs can be delayed (especially at the top of the hour) or dropped under load; hence 06:23;
* commits pushed with `GITHUB_TOKEN` do not start other workflows. This workflow does not need that.
  Whether Streamlit Community Cloud sees such a push and redeploys is **not documented**; if the app does not
  pick up the daily commit, press "Reboot app" once and tell me: the fallback is a Personal Access Token.

**Repository growth.** A day with new data rewrites several append-only parquet files (~25 MB in total if git
could not delta-compress them). In the clean-checkout test a run with no new data changed 7 files (the forecast, features and metadata files plus the new run-history file, a few hundred KB); growth over weeks is not measured: check after the first weeks. GitHub recommends keeping repositories "ideally less than 1 GB" and below 5 GB. Watch
`du -sh .git` after the first weeks; if it grows too fast, squash the history (the app does not need it):
`git checkout --orphan fresh && git add -A && git commit -m "snapshot" && git branch -M main && git push -f`.

## 5. Streamlit Community Cloud (first choice)

### One-time setup (by hand: your accounts)

1. **GitHub**: create an empty **public** repository, e.g. `power-peak-public` (no README, no licence file).
   Public matters: the free plan allows one private app, and the scheduled workflow is free on public repos.
2. **Export a clean copy** (no history, neutral author, no `RESUME.md`; writes only to a local folder):
   ```bash
   git switch deploy-public
   scripts/export_public_repo.sh ../power-peak-public
   cd ../power-peak-public
   git remote add origin https://github.com/<you>/power-peak-public.git
   git push -u origin main          # this is the first thing that leaves your machine
   ```
   Do not push the private repository itself: its history contains your git identity and every earlier
   commit.
3. **Actions**: open the repository's Actions tab and enable workflows. Run "Daily data snapshot" once with
   "Run workflow" and confirm it goes green (roughly 10 minutes: the pipeline took 250 s in my clean-checkout test, 237 s of it the weather step; the test suite takes about 2 minutes).
4. **Streamlit Community Cloud**: sign in at <https://share.streamlit.io> with GitHub, authorise access to
   the repository, then **Create app**:
   * Repository: `<you>/power-peak-public`, Branch: `main`, Main file path: `app/Home.py`
   * **Advanced settings → Python version: 3.12**
   * **Advanced settings → Secrets** (TOML):
     ```toml
     POWER_PEAK_READONLY = "1"
     POWER_PEAK_MODE = "presentation"
     ```
   * Deploy. Open the URL, and check: no Update data button, sidebar footer says "Data snapshot: …".

### Required file layout (already in place)

```
app/Home.py                 entry point ("Main file path")
requirements.txt            the ONLY dependency file Community Cloud should find (it reads the one next to
                            the entrypoint, then the repository root; do not add a second one such as
                            uv.lock, Pipfile or pyproject dependencies that would take precedence)
.streamlit/config.toml      theme; must be at the repository root
.python-version             for local tools; Community Cloud takes the version from the Advanced-settings dropdown
data/…  models/…  reports/… the snapshot (see section 1)
```

`pyproject.toml` lists dependencies too but Community Cloud's priority order is `uv.lock`, `Pipfile`,
`environment.yml`, `requirements.txt`, `pyproject.toml`, so `requirements.txt` wins. Community Cloud runs
`streamlit run` from the repository root, which this project expects.

### Limits (current official documentation, checked 16 Nov 2026)

| Limit | Value | Our use |
|---|---|---|
| Memory per app | 690 MB minimum, **2.7 GB maximum**; "may change at any time without notice" | ~390 MB measured while rendering all five pages (server idle: 62 MB) |
| CPU | 0.078 cores minimum, 2 cores maximum | the app only reads parquet and draws Plotly charts |
| Storage | 50 GB maximum | repository ≈ 33 MB |
| Repository size | **no limit stated** in the Community Cloud docs; GitHub: ≈1 GB recommended, 5 GB strongly recommended, 100 MiB per file | 33 MB |
| Sleep | apps with no traffic for 12 hours go to sleep; anyone can wake one ("Yes, get this app back up!") | first visitor after a quiet night waits for a restart |
| Redeploys from GitHub | at most five per minute; dependency changes trigger a full redeploy | one push a day |
| Python | all released versions still receiving security updates; default 3.12; to change it you must delete and redeploy the app | 3.12 |

Sources: [Community Cloud limits and hibernation](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app),
[status and limitations](https://docs.streamlit.io/deploy/streamlit-community-cloud/status),
[app dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies),
[file organisation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization),
[secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management),
[GitHub repository limits](https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits),
[large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
Community Cloud documents no way to set `--extra-index-url` for `requirements.txt`; that is why the app file has
no PyTorch (the CPU wheel index is only needed by the daily job, which uses pip in GitHub Actions).

## 6. Hugging Face Spaces (fallback): **not free for this app**

Checked against the current documentation (<https://huggingface.co/docs/hub/en/spaces-overview>):

* The Hub offers **three SDKs: Gradio, Docker and static HTML. Streamlit is no longer an SDK option.**
  A Streamlit app has to be a Docker Space.
* "Gradio and Docker Spaces run on compute and **require a paid plan to create**: PRO for personal accounts,
  Team or Enterprise for organizations." Only static Spaces are free for everyone, even though the default
  CPU Basic hardware (2 vCPU, 16 GB RAM, 50 GB non-persistent disk) has no hourly cost.
* Free hardware sleeps when unused (the docs give no duration; an older answer says 48 hours), and the disk is
  ephemeral, which suits a read-only app.

So a Hugging Face Space is a fallback only if you accept a PRO subscription (I did not verify the price: see
<https://huggingface.co/pricing>). If you want a free fallback, the realistic alternatives are a second
Community Cloud app on a different GitHub account, or running the same repository with `streamlit run` on any
small VM you already own. If you choose a PRO Space, the Dockerfile is:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV POWER_PEAK_READONLY=1 POWER_PEAK_MODE=presentation
EXPOSE 7860
CMD ["streamlit", "run", "app/Home.py", "--server.port=7860", "--server.address=0.0.0.0"]
```

with a `README.md` front-matter of `sdk: docker` and `app_port: 7860` (Spaces repositories are git
repositories; the 50 GB disk and 16 GB RAM are far above what this app needs). Because Spaces are a
separate git remote, the daily snapshot would have to be pushed there too (an extra workflow step with an HF
write token stored as a repository secret): not set up.

## 7. Cost

| Item | Cost |
|---|---|
| GitHub public repository + GitHub Actions on standard runners | free: "The use of standard GitHub-hosted runners is free … in public repositories" |
| Streamlit Community Cloud | no charge documented for Community Cloud; one private app allowed, public apps need a public repo |
| Open-Meteo free API | free, non-commercial, no key; attribution required |
| Grid-India (NLDC) reports | public downloads |
| Hugging Face Space (fallback) | **paid** (PRO) for Docker Spaces |

Expected total: **0**. A private repository would instead use the Free plan's included Actions credit (US$12
a month, about 2,000 Linux minutes at the standard rate; this job uses ~10 minutes a day, ~300 a month) and
would need the single private-app slot on Community Cloud.

## 8. Retrain or change the data by hand

The daily job never retrains. When you retrain on your laptop, copy the new `models/<kind>/<timestamp>/`
folders and `latest.txt` files and the updated `data/processed/*` into a checkout of the public repository,
commit, push. Delete old timestamped model folders first to keep the repository small.

## 9. What was tested (16 Nov 2026)

In a fresh `git clone` of this branch (no ignored files, so only what is committed), with new virtual
environments on Python 3.12.15:

| Check | Result |
|---|---|
| `pip install -r requirements.txt` (app only) | installs; `scripts/smoke_app.py` renders the five pages read-only, no PyTorch/LightGBM imported; `streamlit run` answers `/_stcore/health` with 200; peak memory 388 MB while rendering all pages |
| `pip install -r requirements-pipeline.txt` | `torch 2.14.1+cpu`, `cuda available: False`, no `nvidia-*`/`triton` packages; `pip check` clean |
| `python -m pytest -q` (GPU hidden) | 197 passed |
| `python -m src.pipeline --trigger timer` from the clean clone (real downloads) | success in 250 s, 55 Open-Meteo requests; `ci_snapshot.py check` passed; only 7 allow-listed files changed |

The clean-environment run found a real race in the pipeline lock (a reader could delete a lock that was
still being written); fixed in `src/pipeline.py` with a regression test. The GitHub Actions workflow itself
could not be run here (it needs the repository on GitHub): its steps were run by hand in the order above.
