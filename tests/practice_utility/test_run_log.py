"""Tests for training-log parsing and run comparison."""

import pytest

from gear_sonic.research.practice_utility import run_log as R


def table(index, rewards, length, entropy=13.0, std=0.38, steps=26.0,
          collection=34.5, learning=23.5):
    """One Rich-style iteration block, as the trainer prints it."""
    return "\n".join([
        "╭──────────── Training Log ────────────╮",
        f"│            Learning iteration {index}     │",
        f"│  Computation: {steps} steps/s (Collection: {collection}s,  │",
        f"│ Learning {learning}s)                     │",
        f"│  Mean action noise std: {std}            │",
        f"│  Mean entropy: {entropy}                 │",
        f"│  Mean rewards: {rewards}                 │",
        f"│  Mean length: {length}                   │",
        "│ Env/Episode_Reward/action_rate_l2: -0.0313 │",
        "╰──────────────────────────────────────╯",
    ])


def write(tmp_path, blocks, name="run.log"):
    path = tmp_path / name
    path.write_text("\n".join(blocks))
    return path


class TestParsing:
    def test_parses_each_iteration(self, tmp_path):
        path = write(tmp_path, [table(i, 1.0 + i, 10 + i) for i in range(1, 4)])
        log = R.parse_run_log(path)
        assert log.indices == [1, 2, 3]

    def test_extracts_metrics(self, tmp_path):
        path = write(tmp_path, [table(1, 5.0561, 63.36364)])
        it = R.parse_run_log(path).iterations[0]
        assert it.metrics["Mean rewards"] == pytest.approx(5.0561)
        assert it.metrics["Mean length"] == pytest.approx(63.36364)

    def test_extracts_namespaced_metrics(self, tmp_path):
        path = write(tmp_path, [table(1, 1.0, 2.0)])
        it = R.parse_run_log(path).iterations[0]
        assert it.metrics["Env/Episode_Reward/action_rate_l2"] == pytest.approx(-0.0313)

    def test_extracts_timing(self, tmp_path):
        path = write(tmp_path, [table(1, 1.0, 2.0, steps=26.0, collection=34.5, learning=23.5)])
        it = R.parse_run_log(path).iterations[0]
        assert it.steps_per_second == pytest.approx(26.0)
        assert it.wall_seconds == pytest.approx(58.0)

    def test_keeps_the_last_render_of_a_repeated_iteration(self, tmp_path):
        """The trainer re-renders the table many times per iteration."""
        path = write(tmp_path, [table(1, 1.0, 10.0), table(1, 9.0, 90.0)])
        log = R.parse_run_log(path)
        assert len(log.iterations) == 1
        assert log.iterations[0].metrics["Mean rewards"] == pytest.approx(9.0)

    def test_strips_ansi_sequences(self, tmp_path):
        block = table(1, 5.0, 60.0)
        noisy = "\x1b[32m" + block.replace("Mean rewards", "\x1b[1mMean rewards\x1b[0m") + "\x1b[0m"
        path = write(tmp_path, [noisy])
        assert R.parse_run_log(path).iterations[0].metrics["Mean rewards"] == pytest.approx(5.0)

    def test_handles_carriage_returns(self, tmp_path):
        path = tmp_path / "cr.log"
        path.write_text(table(1, 5.0, 60.0).replace("\n", "\r"))
        assert R.parse_run_log(path).iterations[0].metrics["Mean rewards"] == pytest.approx(5.0)

    def test_ignores_lines_before_the_first_iteration(self, tmp_path):
        path = tmp_path / "pre.log"
        path.write_text("Mean rewards: 999.0\n" + table(1, 5.0, 60.0))
        assert R.parse_run_log(path).iterations[0].metrics["Mean rewards"] == pytest.approx(5.0)

    def test_empty_log_yields_nothing(self, tmp_path):
        path = tmp_path / "empty.log"
        path.write_text("no iterations here\n")
        log = R.parse_run_log(path)
        assert log.iterations == [] and log.median_steps_per_second() is None

    def test_series_accessor(self, tmp_path):
        path = write(tmp_path, [table(i, float(i), 10.0) for i in (1, 2, 3)])
        assert R.parse_run_log(path).series("Mean rewards") == {1: 1.0, 2: 2.0, 3: 3.0}


class TestThroughput:
    def test_median_skips_warmup(self, tmp_path):
        """Iteration 1 carries graph capture and allocator warm-up."""
        blocks = [table(1, 1.0, 10.0, steps=1.0, collection=500.0, learning=100.0)]
        blocks += [table(i, 1.0, 10.0, steps=26.0, collection=34.0, learning=24.0)
                   for i in (2, 3, 4)]
        log = R.parse_run_log(write(tmp_path, blocks))
        assert log.median_steps_per_second() == pytest.approx(26.0)
        assert log.median_iteration_seconds() == pytest.approx(58.0)

    def test_report_derives_branch_cost(self, tmp_path):
        blocks = [table(i, 1.0, 10.0, collection=30.0, learning=30.0) for i in range(1, 5)]
        report = R.throughput_report(R.parse_run_log(write(tmp_path, blocks)), num_envs=256)
        assert report["transitions_per_iteration"] == 256 * 24
        assert report["median_iteration_seconds"] == pytest.approx(60.0)
        assert report["hours_per_128_iteration_branch"] == pytest.approx(128 * 60 / 3600)
        assert report["env_steps_per_second"] == pytest.approx(256 * 24 / 60)

    def test_report_handles_an_unparsed_log(self, tmp_path):
        path = tmp_path / "x.log"
        path.write_text("nothing")
        report = R.throughput_report(R.parse_run_log(path), num_envs=64)
        assert report["iterations_parsed"] == 0
        assert "hours_per_128_iteration_branch" not in report


class TestComparison:
    def identical(self, tmp_path):
        blocks = [table(i, 1.0 + i, 10.0 + i) for i in range(1, 5)]
        return (R.parse_run_log(write(tmp_path, blocks, "a.log")),
                R.parse_run_log(write(tmp_path, blocks, "b.log")))

    def test_identical_runs_pass_at_zero_tolerance(self, tmp_path):
        left, right = self.identical(tmp_path)
        result = R.compare_runs(left, right)
        assert result.passes is True
        assert all(v == 0.0 for v in result.max_abs_difference.values())

    def test_divergent_runs_fail(self, tmp_path):
        left = R.parse_run_log(write(tmp_path, [table(1, 1.0, 10.0)], "a.log"))
        right = R.parse_run_log(write(tmp_path, [table(1, 2.0, 10.0)], "b.log"))
        result = R.compare_runs(left, right)
        assert result.passes is False
        assert result.max_abs_difference["Mean rewards"] == pytest.approx(1.0)

    def test_tolerance_can_admit_small_drift(self, tmp_path):
        left = R.parse_run_log(write(tmp_path, [table(1, 1.000, 10.0)], "a.log"))
        right = R.parse_run_log(write(tmp_path, [table(1, 1.001, 10.0)], "b.log"))
        assert R.compare_runs(left, right, tolerance=0.0).passes is False
        assert R.compare_runs(left, right, tolerance=0.01).passes is True

    def test_tolerance_is_recorded_in_the_result(self, tmp_path):
        left, right = self.identical(tmp_path)
        assert R.compare_runs(left, right, tolerance=0.25).to_dict()["tolerance"] == 0.25

    def test_compares_only_shared_iterations(self, tmp_path):
        left = R.parse_run_log(
            write(tmp_path, [table(i, float(i), 10.0) for i in (1, 2, 3)], "a.log"))
        right = R.parse_run_log(
            write(tmp_path, [table(i, float(i), 10.0) for i in (2, 3, 4)], "b.log"))
        assert R.compare_runs(left, right).shared_iterations == [2, 3]

    def test_skip_iterations_excludes_warmup(self, tmp_path):
        left = R.parse_run_log(write(tmp_path, [table(1, 1.0, 10.0), table(2, 5.0, 10.0)], "a.log"))
        right = R.parse_run_log(write(tmp_path, [table(1, 9.0, 10.0), table(2, 5.0, 10.0)], "b.log"))
        assert R.compare_runs(left, right).passes is False
        assert R.compare_runs(left, right, skip_iterations=2).passes is True

    def test_no_shared_iterations_does_not_silently_pass(self, tmp_path):
        left = R.parse_run_log(write(tmp_path, [table(1, 1.0, 10.0)], "a.log"))
        right = R.parse_run_log(write(tmp_path, [table(9, 1.0, 10.0)], "b.log"))
        result = R.compare_runs(left, right)
        assert result.passes is False and result.missing_keys == ["no shared iterations"]

    def test_missing_metric_does_not_silently_pass(self, tmp_path):
        left, right = self.identical(tmp_path)
        result = R.compare_runs(left, right, keys=("Nonexistent metric",))
        assert result.passes is False and "Nonexistent metric" in result.missing_keys
