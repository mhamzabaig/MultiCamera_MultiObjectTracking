"""Configuration layering, validation, and derived-path tests.

The precedence chain is the part of this module most likely to break silently
during a refactor, so every hop is asserted explicitly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mcmot.core.errors import ConfigError
from mcmot.core.settings import Settings, load_settings


class TestDefaults:
    def test_code_defaults_apply_with_empty_config_dir(self, config_dir: Path) -> None:
        settings = load_settings(config_dir=config_dir)

        assert settings.env == "dev"
        assert settings.identity.tau_accept == 0.55
        assert settings.detector.conf_threshold == 0.35
        assert settings.pipeline.queue_size == 1

    def test_shipped_default_yaml_is_valid(self, repo_config_dir: Path) -> None:
        """The committed default.yaml must parse and satisfy every constraint.

        This is the test that catches a typo in config/default.yaml: with
        extra="forbid", an unknown key raises rather than being ignored.
        """
        settings = load_settings(config_dir=repo_config_dir)

        assert settings.detector.name == "yolo11"
        assert settings.tracker.name == "bytetrack"
        assert settings.reid.name == "osnet_ain_x1_0"
        assert settings.detector.classes == [0]


class TestPrecedence:
    def test_default_yaml_overrides_code_defaults(self, config_dir: Path, write_yaml) -> None:
        write_yaml(config_dir / "default.yaml", "identity:\n  tau_accept: 0.70\n")

        assert load_settings(config_dir=config_dir).identity.tau_accept == 0.70

    def test_local_yaml_overrides_default_yaml(self, config_dir: Path, write_yaml) -> None:
        write_yaml(config_dir / "default.yaml", "identity:\n  tau_accept: 0.70\n")
        write_yaml(config_dir / "local.yaml", "identity:\n  tau_accept: 0.80\n")

        assert load_settings(config_dir=config_dir).identity.tau_accept == 0.80

    def test_env_overrides_yaml(
        self, config_dir: Path, write_yaml, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_yaml(config_dir / "default.yaml", "identity:\n  tau_accept: 0.70\n")
        write_yaml(config_dir / "local.yaml", "identity:\n  tau_accept: 0.80\n")
        monkeypatch.setenv("MCMOT__IDENTITY__TAU_ACCEPT", "0.90")

        assert load_settings(config_dir=config_dir).identity.tau_accept == 0.90

    def test_explicit_arguments_override_env(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MCMOT__ENV", "prod")

        assert load_settings(config_dir=config_dir, env="test").env == "test"

    def test_layers_deep_merge_within_a_section(self, config_dir: Path, write_yaml) -> None:
        """Overriding one key must not reset its siblings.

        Shallow merging would leave tau_margin at its code default while
        appearing to work, which is exactly the kind of bug that only shows up as
        a slightly-wrong identity threshold in production.
        """
        write_yaml(
            config_dir / "default.yaml",
            "identity:\n  tau_accept: 0.70\n  tau_margin: 0.20\n  top_k: 25\n",
        )
        write_yaml(config_dir / "local.yaml", "identity:\n  tau_accept: 0.90\n")

        identity = load_settings(config_dir=config_dir).identity
        assert identity.tau_accept == 0.90
        assert identity.tau_margin == 0.20
        assert identity.top_k == 25

    def test_nested_sub_section_merge(self, config_dir: Path, write_yaml) -> None:
        write_yaml(config_dir / "default.yaml", "reid:\n  quality:\n    min_width: 64\n")

        quality = load_settings(config_dir=config_dir).reid.quality
        assert quality.min_width == 64
        assert quality.min_height == 96  # untouched code default


class TestTypoRejection:
    def test_unknown_top_level_key_is_fatal(self, config_dir: Path, write_yaml) -> None:
        write_yaml(config_dir / "default.yaml", "identtiy:\n  tau_accept: 0.7\n")

        with pytest.raises(ConfigError):
            load_settings(config_dir=config_dir)

    def test_unknown_nested_key_is_fatal(self, config_dir: Path, write_yaml) -> None:
        write_yaml(config_dir / "default.yaml", "identity:\n  tau_acept: 0.7\n")

        with pytest.raises(ConfigError):
            load_settings(config_dir=config_dir)

    def test_malformed_yaml_is_reported_with_the_path(self, config_dir: Path, write_yaml) -> None:
        path = write_yaml(config_dir / "default.yaml", "identity:\n  tau_accept: [unclosed\n")

        with pytest.raises(ConfigError) as excinfo:
            load_settings(config_dir=config_dir)

        # The offending file must be named: "invalid configuration" alone sends
        # the operator hunting through three layers.
        assert excinfo.value.details["path"] == str(path)
        assert "reason" in excinfo.value.details

    def test_non_mapping_yaml_is_fatal(self, config_dir: Path, write_yaml) -> None:
        write_yaml(config_dir / "default.yaml", "- just\n- a\n- list\n")

        with pytest.raises(ConfigError):
            load_settings(config_dir=config_dir)

    def test_missing_local_yaml_is_not_an_error(self, config_dir: Path) -> None:
        assert not (config_dir / "local.yaml").exists()
        assert load_settings(config_dir=config_dir).identity.tau_accept == 0.55


class TestValueValidation:
    @pytest.mark.parametrize(
        ("section", "key", "value"),
        [
            ("identity", "tau_accept", 5.0),
            ("identity", "top_k", 0),
            ("detector", "conf_threshold", -0.1),
            ("detector", "image_size", 641),
            ("pipeline", "jpeg_quality", 101),
            ("camera", "target_fps", 0),
            ("device", "prefer", "tpu"),
            ("logging", "level", "TRACE"),
            ("storage", "vector_index_type", "ivfpq"),
        ],
    )
    def test_out_of_range_values_are_rejected(
        self, config_dir: Path, section: str, key: str, value: object
    ) -> None:
        with pytest.raises(ConfigError):
            load_settings(config_dir=config_dir, **{section: {key: value}})

    def test_empty_detector_class_list_is_rejected(self, config_dir: Path) -> None:
        with pytest.raises(ConfigError):
            load_settings(config_dir=config_dir, detector={"classes": []})

    @pytest.mark.parametrize(
        ("section", "payload"),
        [
            ("identity", {"vote_window": 5, "vote_min_agree": 9}),
            ("identity", {"top_k": 3, "per_person_top": 8}),
            ("identity", {"reid_interval_frames": 30, "locked_reid_interval_frames": 10}),
            ("tracker", {"track_high_thresh": 0.3, "track_low_thresh": 0.6}),
            ("camera", {"reconnect_initial_delay_s": 60.0, "reconnect_max_delay_s": 5.0}),
            ("reid", {"quality": {"min_aspect_ratio": 0.9, "max_aspect_ratio": 0.5}}),
        ],
    )
    def test_cross_field_constraints_are_enforced(
        self, config_dir: Path, section: str, payload: dict[str, object]
    ) -> None:
        """Individually valid values that are nonsensical together must fail.

        A locked track sampled *more* often than an unlocked one, or a low
        association threshold above the high one, would produce a system that
        runs and quietly misbehaves.
        """
        with pytest.raises(ConfigError):
            load_settings(config_dir=config_dir, **{section: payload})


class TestDerivedPaths:
    def test_relative_paths_resolve_against_base_dir(
        self, tmp_path: Path, config_dir: Path
    ) -> None:
        base = tmp_path / "workspace"
        settings = load_settings(config_dir=config_dir, paths={"base_dir": str(base)})

        assert settings.paths.data_dir == (base / "data").resolve()
        assert settings.paths.logs_dir == (base / "logs").resolve()
        assert settings.paths.data_dir.is_absolute()

    def test_absolute_paths_are_preserved(self, tmp_path: Path, config_dir: Path) -> None:
        elsewhere = tmp_path / "elsewhere"
        settings = load_settings(
            config_dir=config_dir,
            paths={"base_dir": str(tmp_path), "data_dir": str(elsewhere)},
        )

        assert settings.paths.data_dir == elsewhere.resolve()

    def test_data_dir_placeholder_is_expanded(self, settings: Settings) -> None:
        data_dir = settings.paths.data_dir.as_posix()

        assert "{data_dir}" not in settings.storage.database_url
        assert data_dir in settings.storage.database_url
        assert settings.vector_index_file.is_absolute()
        assert settings.blob_directory.is_absolute()

    def test_unknown_placeholder_is_rejected(self, config_dir: Path) -> None:
        with pytest.raises(ConfigError):
            load_settings(
                config_dir=config_dir,
                storage={"database_url": "sqlite:///{home_dir}/x.db"},
            )

    def test_model_weights_are_anchored_to_models_dir(self, settings: Settings) -> None:
        assert settings.detector.weights.parent == settings.paths.models_dir
        assert settings.detector.weights.name == "yolo11n.pt"
        assert settings.reid.weights.parent == settings.paths.models_dir

    def test_absolute_weights_path_is_left_alone(self, tmp_path: Path, config_dir: Path) -> None:
        custom = tmp_path / "custom" / "yolo11m.pt"
        settings = load_settings(config_dir=config_dir, detector={"weights": str(custom)})

        assert settings.detector.weights == custom

    def test_log_file_path_sits_under_logs_dir(self, settings: Settings) -> None:
        assert settings.log_file_path.parent == settings.paths.logs_dir
        assert settings.log_file_path.name == "mcmot.jsonl"

    def test_ensure_directories_creates_them(self, settings: Settings) -> None:
        assert not settings.paths.data_dir.exists()

        settings.paths.ensure_directories()

        assert settings.paths.data_dir.is_dir()
        assert settings.paths.models_dir.is_dir()
        assert settings.paths.logs_dir.is_dir()

    def test_ensure_directories_is_idempotent(self, settings: Settings) -> None:
        settings.paths.ensure_directories()
        settings.paths.ensure_directories()

        assert settings.paths.logs_dir.is_dir()


class TestSummary:
    def test_summary_reports_operational_values(self, settings: Settings) -> None:
        summary = settings.summary()

        assert summary["env"] == "test"
        assert summary["identity"]["tau_accept"] == 0.55
        assert summary["detector"]["weights"] == "yolo11n.pt"

    def test_summary_redacts_database_credentials(self, config_dir: Path) -> None:
        settings = load_settings(
            config_dir=config_dir,
            storage={"database_url": "postgresql://admin:s3cret@db.internal:5432/mcmot"},
        )

        rendered = settings.summary()["storage"]["database"]
        assert "s3cret" not in rendered
        assert "admin" not in rendered
        assert "db.internal:5432/mcmot" in rendered

    def test_summary_is_json_serialisable(self, settings: Settings) -> None:
        import json

        assert json.loads(json.dumps(settings.summary()))["env"] == "test"
