"""Tests for deploy.sh: the dependency-scan report (Step 1.5) and the step order.

The scan helpers are extracted from deploy.sh and run in real bash (with `set -e`,
like deploy.sh) against a fake `pip-audit`, so no network access is needed.
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

DEPLOY_SH = Path(__file__).resolve().parent.parent / "deploy.sh"
DEPLOY_TEXT = DEPLOY_SH.read_text()

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")

CHECK = "\u2713"  # the success tick deploy.sh prints


def run_scan(tmp_path, fake_pip_audit: str) -> subprocess.CompletedProcess:
    """Run deploy.sh's scan_requirements against a fake .venv/bin/pip-audit."""
    bin_dir = tmp_path / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    audit = bin_dir / "pip-audit"
    audit.write_text("#!/bin/bash\n" + fake_pip_audit)
    audit.chmod(0o755)
    (tmp_path / "requirements.txt").write_text("multidict==6.9.1\n")

    script = f"""
set -e
eval "$(sed -n '/^print_success()/,/^}}/p;/^print_warning()/,/^}}/p;/^scan_requirements()/,/^}}/p' "{DEPLOY_SH}")"
scan_requirements "test dependencies" requirements.txt "FIX-HINT"
echo "DEPLOY-CONTINUED"
"""
    return subprocess.run(
        ["bash", "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=60
    )


class TestDependencyScanReportsTheTruth:
    def test_clean_scan_prints_a_tick(self, tmp_path):
        result = run_scan(tmp_path, 'echo "No known vulnerabilities found"; exit 0')

        assert result.returncode == 0
        assert f"{CHECK} No known vulnerabilities found in test dependencies" in result.stdout
        assert "DEPLOY-CONTINUED" in result.stdout

    def test_vulnerabilities_are_listed_and_the_deploy_continues(self, tmp_path):
        fake = """
cat <<'EOF'
WARNING:pip_audit._cli:--no-deps is supported, but users are encouraged to fully hash their pinned dependencies
Found 1 known vulnerability in 1 package
Name      Version ID              Fix Versions
--------- ------- --------------- ------------
multidict 6.9.0   CVE-2026-104874 6.9.1
EOF
exit 1
"""
        result = run_scan(tmp_path, fake)

        assert result.returncode == 0, "a finding must not abort the deploy"
        assert "KNOWN VULNERABILITIES found in test dependencies" in result.stdout
        assert "CVE-2026-104874" in result.stdout and "6.9.1" in result.stdout
        assert "FIX-HINT" in result.stdout
        assert CHECK not in result.stdout
        assert "DEPLOY-CONTINUED" in result.stdout

    @pytest.mark.parametrize(
        "name,fake",
        [
            (
                "ensurepip crash (the original failure)",
                "echo \"subprocess.CalledProcessError: ... ensurepip ... died with <Signals.SIGABRT: 6>\" >&2; exit 1",
            ),
            ("killed by SIGABRT", "kill -ABRT $$"),
            ("exit 0 with no output", "exit 0"),
            ("unrecognised output", 'echo "something else entirely"; exit 0'),
            (
                "clean but a package was skipped",
                'printf "No known vulnerabilities found\\nName Skip Reason\\nfoo  not on PyPI\\n"; exit 0',
            ),
            ("invalid input", 'echo "ERROR:pip_audit._cli:invalid requirements input" >&2; exit 1'),
        ],
        ids=lambda v: v if isinstance(v, str) and " " in v and len(v) < 60 else None,
    )
    def test_a_scan_that_could_not_run_is_never_a_tick(self, tmp_path, name, fake):
        result = run_scan(tmp_path, fake)

        assert result.returncode == 0, "a broken scanner must not abort the deploy"
        assert "COULD NOT RUN" in result.stdout
        assert "vulnerability status UNKNOWN" in result.stdout
        assert CHECK not in result.stdout
        assert "DEPLOY-CONTINUED" in result.stdout

    def test_the_scanner_exit_status_is_not_taken_from_a_pipe(self):
        """Regression: `if pip-audit ... | tail; then` tested tail's exit status."""
        for line in DEPLOY_TEXT.splitlines():
            if "pip-audit" in line and not line.lstrip().startswith("#"):
                assert not re.search(r"\|\s*(tail|head|cat|tee)\b", line), line

    def test_the_known_good_flags_are_used(self):
        assert "--no-deps --disable-pip" in DEPLOY_TEXT
        assert "--progress-spinner off" in DEPLOY_TEXT


class TestStepOrder:
    @staticmethod
    def position(needle, start=0):
        index = DEPLOY_TEXT.find(needle, start)
        assert index >= 0, f"{needle!r} not found in deploy.sh"
        return index

    def test_gateway_and_memory_are_created_before_the_agents_are_deployed(self):
        role = self.position("scripts/create_agentcore_role.py")
        gateway = self.position("scripts/setup_gateway.py")
        memory = self.position("scripts/setup_memory.py")
        agents = self.position("scripts/deploy_agentcore.py")

        assert role < gateway < memory < agents

    def test_cdk_redeploy_that_links_the_orchestrator_follows_the_agents(self):
        agents = self.position("scripts/deploy_agentcore.py")
        first_cdk = self.position("npx cdk deploy")
        relink = self.position("npx cdk deploy", first_cdk + 1)

        assert first_cdk < agents < relink

    def test_print_step_numbers_increase_without_gaps_or_repeats(self):
        steps = re.findall(r'print_step "([0-9.]+)"', DEPLOY_TEXT)
        as_numbers = [float(s) for s in steps]

        assert steps[0] == "0"
        assert as_numbers == sorted(as_numbers)
        assert len(set(steps)) == len(steps)
        # whole-number steps are consecutive
        whole = [int(float(s)) for s in steps if "." not in s]
        assert whole == list(range(whole[0], whole[-1] + 1))

    def test_step_comments_match_the_printed_step_titles(self):
        comment_steps = re.findall(r"^# Step ([0-9.]+):", DEPLOY_TEXT, re.M)
        printed_steps = re.findall(r'print_step "([0-9.]+)"', DEPLOY_TEXT)

        assert comment_steps == printed_steps

    def test_skip_agents_still_creates_role_gateway_and_memory(self):
        """Only the agent deploy (Step 4) is skippable; role/gateway/memory (Step 3) do not depend on it."""
        step3 = DEPLOY_TEXT[DEPLOY_TEXT.index("# Step 3:"):DEPLOY_TEXT.index("# Step 4:")]
        step4 = DEPLOY_TEXT[DEPLOY_TEXT.index("# Step 4:"):DEPLOY_TEXT.index("# Step 5:")]

        for script in ("create_agentcore_role.py", "setup_gateway.py", "setup_memory.py"):
            assert script in step3
        assert "SKIP_AGENTS" not in step3
        assert 'if [ "$SKIP_AGENTS" = false ]' in step4
        assert "scripts/deploy_agentcore.py" in step4
