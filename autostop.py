#     Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#     Licensed under the Apache License, Version 2.0 (the "License").
#     You may not use this file except in compliance with the License.
#     A copy of the License is located at
#
#         https://aws.amazon.com/apache-2-0/
#
#     or in the "license" file accompanying this file. This file is distributed
#     on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#     express or implied. See the License for the specific language governing
#     permissions and limitations under the License.

import getopt
import json
import os
import sys
import time as time_module
from datetime import datetime, timezone

import boto3
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------------------------
# Usage / help
# ---------------------------------------------------------------------------
usageInfo = """Usage:
This script checks whether a SageMaker notebook instance is idle and, if so, stops it:
python autostop.py --time <time_in_seconds> [--port <jupyter_port>] [--strategy <name>] [--ignore-connections] [--dry-run]
Type "python autostop.py -h" for available options.
"""

helpInfo = """-t, --time
    Idle grace period in seconds. The instance is only stopped once the idle
    condition of the chosen strategy holds AND the most recent kernel/terminal
    activity (or the Jupyter server start time when there has been none) is
    older than this many seconds.
-p, --port
    Jupyter port (default 8443)
-s, --strategy
    How "idle" is decided. One of:
      connections  (default) idle when there are NO browser connections to any
                   kernel, i.e. no notebook/console is open in any browser tab.
      notebooks    idle when there are NO open notebook sessions on the server
                   (a notebook closed with "Close and Shut Down Kernel", or whose
                   kernel was shut down, no longer counts as open).
      either       idle when EITHER of the above holds.
      activity     legacy behaviour: every session's kernel must be idle, have no
                   connections (unless -c) and be inactive for --time seconds.
    In every strategy a kernel that is currently executing code blocks shutdown.
-c, --ignore-connections
    (activity strategy only) ignore connected users when deciding idleness.
-n, --dry-run
    Evaluate and report the idle decision but never call StopNotebookInstance.
-h, --help
    Help information
"""

VALID_STRATEGIES = ("connections", "notebooks", "either", "activity")


# ---------------------------------------------------------------------------
# Time / logging helpers
# ---------------------------------------------------------------------------
def now_utc():
    return datetime.now(timezone.utc)


def fmt_ts(dt):
    """Format an aware datetime as e.g. 2026-09-24 07:41:03.512 UTC."""
    if dt is None:
        return "n/a"
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"


def fmt_age(seconds):
    """Format a duration in seconds as e.g. 1h 02m 03s (or 'n/a')."""
    if seconds is None:
        return "n/a"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return "%dh %02dm %02ds" % (h, m, s)
    if m:
        return "%dm %02ds" % (m, s)
    return "%ds" % s


def log(msg):
    print("[%s] %s" % (fmt_ts(now_utc()), msg), flush=True)


def parse_jupyter_time(value):
    """Parse the ISO-8601 UTC timestamps returned by the Jupyter REST API.

    Handles '2026-09-24T07:41:03.512345Z', '2026-09-24T07:41:03Z' and
    offset forms such as '2026-09-24T07:41:03.512345+00:00'.
    Returns an aware datetime in UTC, or None if the value cannot be parsed.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        dt = None
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
            try:
                dt = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        if dt is None:
            log("WARNING: could not parse timestamp %r" % value)
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def age_seconds(dt):
    if dt is None:
        return None
    return (now_utc() - dt).total_seconds()


def describe_time(dt):
    return "%s (age %s)" % (fmt_ts(dt), fmt_age(age_seconds(dt)))


# ---------------------------------------------------------------------------
# Command-line parameters
# ---------------------------------------------------------------------------
SCRIPT_START = now_utc()
idle_time = None
port = "8443"
strategy = "connections"
ignore_connections = False
dry_run = False
try:
    opts, args = getopt.getopt(
        sys.argv[1:],
        "ht:p:s:cn",
        ["help", "time=", "port=", "strategy=", "ignore-connections", "dry-run"],
    )
    if len(opts) == 0:
        raise getopt.GetoptError("No input parameters!")
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(helpInfo)
            exit(0)
        if opt in ("-t", "--time"):
            idle_time = int(arg)
        if opt in ("-p", "--port"):
            port = str(arg)
        if opt in ("-s", "--strategy"):
            strategy = str(arg).lower()
        if opt in ("-c", "--ignore-connections"):
            ignore_connections = True
        if opt in ("-n", "--dry-run"):
            dry_run = True
except getopt.GetoptError:
    print(usageInfo)
    exit(1)

missingConfiguration = False
if idle_time is None:
    print("Missing '-t' or '--time'")
    missingConfiguration = True
if strategy not in VALID_STRATEGIES:
    print("Invalid '-s' or '--strategy' value %r. Must be one of: %s" % (strategy, ", ".join(VALID_STRATEGIES)))
    missingConfiguration = True
if missingConfiguration:
    exit(2)

log("=" * 78)
log("autostop.py starting")
log("Script start time ........ %s" % fmt_ts(SCRIPT_START))
log("Idle grace period (-t) ... %d seconds (%s)" % (idle_time, fmt_age(idle_time)))
log("Jupyter port (-p) ........ %s" % port)
log("Idle strategy (-s) ....... %s" % strategy)
log("Ignore connections (-c) .. %s" % ignore_connections)
log("Dry run (-n) ............. %s" % dry_run)


# ---------------------------------------------------------------------------
# SageMaker helpers
# ---------------------------------------------------------------------------
def get_notebook_name():
    # SAGEMAKER_RESOURCE_METADATA lets the script be exercised off-instance.
    log_path = os.environ.get(
        "SAGEMAKER_RESOURCE_METADATA", "/opt/ml/metadata/resource-metadata.json"
    )
    with open(log_path, "r") as logs:
        _logs = json.load(logs)
    return _logs["ResourceName"]


def get_instance_last_modified_time():
    """Fallback reference time: when SageMaker last modified (e.g. started) the instance."""
    name = get_notebook_name()
    log("Calling SageMaker DescribeNotebookInstance for %r" % name)
    t0 = time_module.monotonic()
    client = boto3.client("sagemaker")
    desc = client.describe_notebook_instance(NotebookInstanceName=name)
    log("  -> DescribeNotebookInstance returned in %.0f ms (status=%s)"
        % ((time_module.monotonic() - t0) * 1000, desc.get("NotebookInstanceStatus")))
    return parse_jupyter_time(desc["LastModifiedTime"])


# ---------------------------------------------------------------------------
# Jupyter REST API helpers
# ---------------------------------------------------------------------------
def jupyter_get(path, optional=False):
    """GET a Jupyter Server API path on localhost and return the decoded JSON.

    `no_track_activity=1` asks Jupyter Server not to count this request as user
    activity, so the autostop poll itself never keeps the server "busy".
    Returns None (after logging) if `optional` and the endpoint is unavailable.
    """
    url = "https://localhost:%s%s" % (port, path)
    log("GET %s" % url)
    t0 = time_module.monotonic()
    try:
        response = requests.get(
            url, params={"no_track_activity": "1"}, verify=False, timeout=15
        )
    except requests.RequestException as exc:
        log("  -> request FAILED after %.0f ms: %s" % ((time_module.monotonic() - t0) * 1000, exc))
        if optional:
            return None
        raise
    elapsed_ms = (time_module.monotonic() - t0) * 1000
    log("  -> HTTP %d in %.0f ms (%d bytes)" % (response.status_code, elapsed_ms, len(response.content)))
    if response.status_code != 200:
        if optional:
            log("  -> endpoint unavailable, treating as optional")
            return None
        response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Gather state from Jupyter
# ---------------------------------------------------------------------------
log("-" * 78)
log("Collecting Jupyter server state")

# /api/status: {"started", "last_activity", "connections", "kernels"}
status = jupyter_get("/api/status", optional=True)
# /api/sessions: one entry per open notebook/console (a kernel bound to a file)
sessions = jupyter_get("/api/sessions")
# /api/kernels: every running kernel, including ones no longer bound to a session
kernels = jupyter_get("/api/kernels", optional=True)
if kernels is None:
    kernels = [s["kernel"] for s in sessions if s.get("kernel")]
# /api/terminals: running terminals (only present if the terminals extension is enabled)
terminals = jupyter_get("/api/terminals", optional=True) or []

log("-" * 78)
log("Jupyter server status")
server_started = None
server_last_activity = None
if status:
    server_started = parse_jupyter_time(status.get("started"))
    server_last_activity = parse_jupyter_time(status.get("last_activity"))
    log("  server started ............... %s" % describe_time(server_started))
    log("  server last_activity ......... %s  (includes browser API polling)" % describe_time(server_last_activity))
    log("  total browser connections .... %s" % status.get("connections"))
    log("  running kernels .............. %s" % status.get("kernels"))
else:
    log("  /api/status not available - deriving totals from /api/kernels")

log("Open sessions (notebooks/consoles): %d" % len(sessions))
for i, s in enumerate(sessions, 1):
    k = s.get("kernel") or {}
    log("  [%d] path=%r type=%r" % (i, s.get("path") or s.get("notebook", {}).get("path"), s.get("type", "notebook")))
    log("      kernel id=%s name=%s state=%s connections=%s"
        % (k.get("id"), k.get("name"), k.get("execution_state"), k.get("connections")))
    log("      kernel last_activity=%s" % describe_time(parse_jupyter_time(k.get("last_activity"))))

log("Running kernels: %d" % len(kernels))
for i, k in enumerate(kernels, 1):
    log("  [%d] id=%s name=%s state=%s connections=%s last_activity=%s"
        % (i, k.get("id"), k.get("name"), k.get("execution_state"), k.get("connections"),
           describe_time(parse_jupyter_time(k.get("last_activity")))))

log("Running terminals: %d" % len(terminals))
for i, t in enumerate(terminals, 1):
    log("  [%d] name=%s last_activity=%s" % (i, t.get("name"), describe_time(parse_jupyter_time(t.get("last_activity")))))

# ---------------------------------------------------------------------------
# Derived facts
# ---------------------------------------------------------------------------
busy_kernels = [k for k in kernels if k.get("execution_state") not in ("idle", None)]
if status and status.get("connections") is not None:
    total_connections = int(status["connections"])
else:
    total_connections = sum(int(k.get("connections") or 0) for k in kernels)
open_notebooks = len(sessions)

# Reference time for the grace period. We deliberately use kernel and terminal
# activity (plus the server start time) rather than the server-wide
# last_activity, because an open-but-untouched JupyterLab browser tab polls the
# REST API every few seconds and would otherwise never look idle.
activity_times = [parse_jupyter_time(k.get("last_activity")) for k in kernels]
activity_times += [parse_jupyter_time(t.get("last_activity")) for t in terminals]
activity_times = [t for t in activity_times if t is not None]
if activity_times:
    reference_time = max(activity_times)
    reference_source = "most recent kernel/terminal activity"
elif server_started is not None:
    reference_time = server_started
    reference_source = "Jupyter server start time"
else:
    reference_time = get_instance_last_modified_time()
    reference_source = "SageMaker instance LastModifiedTime"
reference_age = age_seconds(reference_time)
grace_elapsed = reference_age is not None and reference_age > idle_time

log("-" * 78)
log("Idle evaluation (strategy=%s) at %s" % (strategy, fmt_ts(now_utc())))
log("  busy kernels ................. %d %s" % (len(busy_kernels), [k.get("id") for k in busy_kernels] if busy_kernels else ""))
log("  browser connections .......... %d" % total_connections)
log("  open notebook sessions ....... %d" % open_notebooks)
log("  reference time ............... %s [%s]" % (describe_time(reference_time), reference_source))
log("  grace period ................. %s required, %s elapsed -> %s"
    % (fmt_age(idle_time), fmt_age(reference_age), "ELAPSED" if grace_elapsed else "NOT YET ELAPSED"))
if reference_time is not None and not grace_elapsed:
    remaining = idle_time - reference_age
    log("  grace period ends at ......... %s (in %s)"
        % (fmt_ts(datetime.fromtimestamp(now_utc().timestamp() + remaining, timezone.utc)), fmt_age(remaining)))


# ---------------------------------------------------------------------------
# Decide
# ---------------------------------------------------------------------------
def legacy_activity_strategy():
    """Original behaviour, kept for backwards compatibility (strategy=activity)."""
    if not sessions:
        lmt = get_instance_last_modified_time()
        lmt_age = age_seconds(lmt)
        log("  no sessions: instance LastModifiedTime=%s" % describe_time(lmt))
        if lmt_age is not None and lmt_age > idle_time:
            return True, "no sessions and instance untouched for longer than the grace period"
        return False, "no sessions but instance was modified/started within the grace period"
    for s in sessions:
        k = s.get("kernel") or {}
        label = s.get("path") or k.get("id")
        if k.get("execution_state") != "idle":
            return False, "session %r kernel state is %r" % (label, k.get("execution_state"))
        if not ignore_connections and int(k.get("connections") or 0) > 0:
            return False, "session %r has %s browser connection(s)" % (label, k.get("connections"))
        last = parse_jupyter_time(k.get("last_activity"))
        last_age = age_seconds(last)
        log("  session %r last_activity=%s" % (label, describe_time(last)))
        if last_age is None or last_age <= idle_time:
            return False, "session %r active within the grace period" % label
    return True, "every session is idle, unconnected and inactive beyond the grace period"


idle = False
reason = ""
if busy_kernels:
    idle = False
    reason = "%d kernel(s) are currently executing code" % len(busy_kernels)
elif strategy == "activity":
    idle, reason = legacy_activity_strategy()
else:
    no_connections = total_connections == 0
    no_notebooks = open_notebooks == 0
    if strategy == "connections":
        condition, cond_text = no_connections, "no browser connections to any kernel"
        blocked_text = "%d browser connection(s) still open" % total_connections
    elif strategy == "notebooks":
        condition, cond_text = no_notebooks, "no open notebook sessions"
        blocked_text = "%d notebook session(s) still open" % open_notebooks
    else:  # either
        condition = no_connections or no_notebooks
        cond_text = "no browser connections" if no_connections else "no open notebook sessions"
        blocked_text = "%d browser connection(s) and %d notebook session(s) still open" % (total_connections, open_notebooks)

    if not condition:
        idle = False
        reason = blocked_text
    elif not grace_elapsed:
        idle = False
        reason = "%s, but the %s grace period has not elapsed since %s" % (cond_text, fmt_age(idle_time), reference_source)
    else:
        idle = True
        reason = "%s and the %s grace period has elapsed since %s" % (cond_text, fmt_age(idle_time), reference_source)

log("-" * 78)
log("DECISION: notebook instance is %s -- %s" % ("IDLE" if idle else "NOT IDLE", reason))

# ---------------------------------------------------------------------------
# Act
# ---------------------------------------------------------------------------
if idle:
    name = get_notebook_name()
    if dry_run:
        log("DRY RUN: would call StopNotebookInstance for %r now, skipping" % name)
    else:
        log("Calling StopNotebookInstance for %r" % name)
        t0 = time_module.monotonic()
        client = boto3.client("sagemaker")
        client.stop_notebook_instance(NotebookInstanceName=name)
        log("  -> StopNotebookInstance accepted in %.0f ms; instance shutdown initiated at %s"
            % ((time_module.monotonic() - t0) * 1000, fmt_ts(now_utc())))
else:
    log("Notebook not idle. Pass.")

log("autostop.py finished (total runtime %s)" % fmt_age(age_seconds(SCRIPT_START)))
log("=" * 78)
