from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "scripts" / "python" / "year2_final_campaign.py"
SPEC = importlib.util.spec_from_file_location("year2_final_campaign", MODULE_PATH)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def test_frozen_manifests_are_exact_cartesian_products():
    htpmd = campaign.validate_manifest("htpmd", campaign.HTPMD_MANIFEST)
    eaip = campaign.validate_manifest("eaip", campaign.EAIP_MANIFEST)
    assert len(htpmd) == 375
    assert len(eaip) == 50
    assert htpmd == campaign.expected_rows("htpmd")
    assert eaip == campaign.expected_rows("eaip")
    assert len({row["canonical_run_id"] for row in htpmd + eaip}) == 425
    assert len({row["output_dir"] for row in htpmd + eaip}) == 425


def test_split_mapping_and_initialization_are_frozen():
    expected = {0: 42, 1: 43, 2: 44, 3: 45, 4: 46}
    for dataset, path in (("htpmd", campaign.HTPMD_MANIFEST), ("eaip", campaign.EAIP_MANIFEST)):
        rows = campaign.validate_manifest(dataset, path)
        assert [int(row["array_index"]) for row in rows] == list(range(len(rows)))
        assert all(int(row["split_seed"]) == expected[int(row["split_index"])] for row in rows)
        assert all(int(row["init_seed"]) == 42 for row in rows)


def test_all_commands_are_unique_and_frozen():
    htpmd = campaign.validate_manifest("htpmd", campaign.HTPMD_MANIFEST)
    eaip = campaign.validate_manifest("eaip", campaign.EAIP_MANIFEST)
    commands = campaign.validate_commands(htpmd + eaip, ROOT)
    assert len(commands) == 425
    assert len(set(commands)) == 425
    assert all("--max_batches" not in command for command in commands)
    assert all("--init_seed 42" in command for command in commands)
    assert all("--batch_size 64" in command for command in commands)
    assert all("--max_epochs 300" in command for command in commands)
    assert all("--patience 30" in command for command in commands)
    assert all("--device cuda" in command for command in commands)
    assert all("results/year2_corrected_final_2026/" in command for command in commands)
    assert all("checkpoints/" not in command and "predictions/" not in command for command in commands)
    assert all("--wd_convention corrected" in command for command in commands[:375])
    assert all("htpmd_chemistry_disjoint.json" in command for command in commands[:375])
    assert all("ea_ip_chemistry_disjoint.json" in command for command in commands[375:])


def test_expected_sample_run_ids_and_fastest_split_dimension():
    htpmd = campaign.expected_rows("htpmd")
    eaip = campaign.expected_rows("eaip")
    assert htpmd[0]["canonical_run_id"] == "htpmd__dmpnn__conda_native__conductivity__split00_seed42__init42"
    assert htpmd[1]["canonical_run_id"] == "htpmd__dmpnn__conda_native__conductivity__split01_seed43__init42"
    assert htpmd[125]["canonical_run_id"] == "htpmd__gin__condc_dop_molality__conductivity__split00_seed42__init42"
    assert eaip[0]["canonical_run_id"] == "eaip__dmpnn__ea__split00_seed42__init42"
    assert eaip[1]["canonical_run_id"] == "eaip__dmpnn__ea__split01_seed43__init42"
    assert eaip[40]["canonical_run_id"] == "eaip__wd_published_xn8__ea__split00_seed42__init42"


def test_pbs_wrappers_have_required_guards_and_resources():
    specifications = {
        "run_year2_final_htpmd_array.pbs": (
            "#PBS -l walltime=12:00:00", "#PBS -l mem=32GB",
            "--resolve htpmd", "htpmd_cells.tsv",
        ),
        "run_year2_final_eaip_array.pbs": (
            "#PBS -l walltime=24:00:00", "#PBS -l mem=64GB",
            "--resolve eaip", "eaip_cells.tsv",
        ),
    }
    for filename, required in specifications.items():
        text = (ROOT / "scripts" / "pbs" / filename).read_text()
        for value in required:
            assert value in text
        assert "#PBS -P ng76" in text
        assert "#PBS -q gpuvolta" in text
        assert "#PBS -J" not in text
        assert "#PBS -l ncpus=12" in text
        assert "#PBS -l ngpus=1" in text
        assert "YEAR2_CELL_INDEX is required" in text
        assert "scratch/um09+gdata/dk92" in text
        assert 'if [[ -e "$CELL_DIR" ]]' in text
        assert "exit 20" in text
        assert 'mkdir "$CELL_DIR"' in text
        assert "launch_manifest.json" in text
        assert "trap on_exit EXIT" in text
        assert '"${PYTHON_COMMAND[@]}"' in text
        assert "--max_batches" not in text
