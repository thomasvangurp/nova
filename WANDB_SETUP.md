# SN68 nanobody — W&B (CoreWeave Forge) setup

The logger `/root/wandb_log_epoch.py` is read-only and standalone (leaderboard
API + Hetzner MySQL + producer artifacts). It runs in a dedicated venv
`/root/wandb_logger_venv` (wandb + pymysql) so it never touches the live
producer's pixi envs or code.

## The one missing input

A **CoreWeave W&B API key is not available in this environment** (nothing in
the macOS keychain, and nothing on the boxes). To go live you must provide:

1. **API key** — from CoreWeave Forge: `forge.coreweave.com/settings` →
   "API keys" (the sign-in lives at `id.coreweave.com/login`; the same account
   as before the wandb.ai → Forge move works).
2. **Entity / username** — the account (or team) that will own the private
   project. Optional if you want the key owner's default entity.

### Where to put it (prod box — this is where the logger runs)

Create `/root/.wandb_env` with mode 0600. Never commit it, never print it.

```bash
# on prod (ssh -p 46916 root@8.17.147.152)
umask 077
cat > /root/.wandb_env <<'EOF'
export WANDB_API_KEY="<paste key here>"
export WANDB_ENTITY="<username-or-team>"     # optional
# export WANDB_BASE_URL="..."                # do NOT set for Forge, see below
EOF
chmod 600 /root/.wandb_env
```

The wrapper `/root/wandb_log_loop.sh` sources this file before each run, so
cron never holds the key. (If you prefer the macOS keychain: that is fine for
*your* copies, but the key must still reach the prod box in a 0600 file because
that is the host making the upload.)

## WANDB_BASE_URL — leave it unset

For CoreWeave's hosted W&B (Forge, multi-tenant) the **API endpoint is still
`https://api.wandb.ai`** — CoreWeave docs state "You don't need to update the
`WANDB_BASE_URL` environment variable". Only the web app moved:

| What | URL |
|---|---|
| Web app / project | `forge.coreweave.com/wandb/{entity}/{project}` |
| API (SDK base URL) | `https://api.wandb.ai` (unchanged, default) |
| API key page | `forge.coreweave.com/settings` |

The `/wandb` prefix is the app path only — **do NOT** set
`WANDB_BASE_URL=https://forge.coreweave.com/wandb`; there is no SDK API there
and it will break uploads. Leave the variable unset (or explicitly
`https://api.wandb.ai`).

## Privacy

A project created by `wandb.init()` is **private by default** (only the entity
/ team members can see it). W&B has no public SDK field to change visibility,
so the script best-effort upserts the project and reports its visibility. To
tighten further (Team / Restricted with explicit invites) use the Forge UI:
project page → visibility dropdown (scopes: Open, Public, Team, Restricted).

## Verify

```bash
# dry-run (no key needed):
/root/wandb_logger_venv/bin/python /root/wandb_log_epoch.py --dry-run

# once the key is in /root/.wandb_env:
/root/wandb_log_loop.sh once

# then open in the browser:
#   https://forge.coreweave.com/wandb/{entity}/sn68-nanobody
#   - confirm epoch runs appear (one run per epoch: "epoch-<N>")
#   - confirm project visibility shows Private
```

Scheduling: the box's cron daemon is NOT running (standard for vast.ai
images), so the scheduler is a detached tmux loop, matching how the miner
itself is launched:

    tmux new-session -d -s wandb-log '/bin/bash /root/wandb_log_loop.sh'

It polls every 30 min, out of the producer's hot path, and the logger skips
epochs it has already logged (state file `/root/wandb_log_state/{epoch}.done`).
If cron is later enabled on the box, add:

    */30 * * * * /bin/bash /root/wandb_log_loop.sh once >> /root/wandb_log_cron.log 2>&1
