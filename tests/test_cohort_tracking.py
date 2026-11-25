"""Exercise real tiny-video batches, source guards, and reuse after failed diagnostics."""

from dataclasses import asdict
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

from nvsl_offline.background import BackgroundConfig
from nvsl_offline.detection import DetectionConfig
from nvsl_offline.pipeline import producer
from nvsl_offline.recording import Calibration, Recording, identity
from test_baseline import recording

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cohort_tracking', ROOT/'scripts/run_cohort_tracking.py')
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def inputs(tmp_path):
    video = recording(tmp_path)
    rec = Recording.load(video)
    frozen = dict(producer=producer(), detector=asdict(DetectionConfig(
        mode='normalized', illumination='global_offset', segmentation='body_core')),
        background=asdict(BackgroundConfig()),
        audit=dict(layout='large2', margin=24, samples=129, hash_video=False))
    job = dict(index=0, video=str(video), sources=rec.sources(), slots=4, frames=rec.shape[1],
               calibration=asdict(Calibration.from_recording(rec,240,240,layout='large2',margin=24)),
               acquisition_note=None, disposition='ready_for_audit', warnings=[])
    batch.save(tmp_path/'freeze.json', frozen)
    batch.save(tmp_path/'cohort.json', dict(name='synthetic', freeze=str(tmp_path/'freeze.json'),jobs=[job,
        dict(index=1,video='held.avi', disposition='hold',hold_reasons=['Incomplete acquisition'])]))
    return rec, frozen, job


def invoke(tmp_path, *extra):
    return subprocess.run([sys.executable,str(ROOT/'scripts/run_cohort_tracking.py'),
        '--cohort',str(tmp_path/'cohort.json'),'--output',str(tmp_path/'batch'),
        '--workers','1',*extra],cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT/'src')),
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)


def test_real_batch_resume_and_changed_source_refusal(tmp_path):
    rec, frozen, job = inputs(tmp_path)
    result = invoke(tmp_path,'--dry-run')
    assert result.returncode == 0, result.stdout
    assert not (tmp_path/'batch').exists()
    result = invoke(tmp_path)
    assert result.returncode == 0, result.stdout
    state = batch.read(tmp_path/'batch/status.json')
    assert state['complete'] and len(state['jobs']) == 1 and len(state['held']) == 1
    assert state['aggregate']['fly_slots'] == 4
    companion = Path(state['jobs'][0]['companion'])
    original_identity = identity(companion,content_hash=True)
    assert rec.sources() == job['sources']
    result = invoke(tmp_path,'--resume')
    assert result.returncode == 0, result.stdout
    assert batch.read(Path(state['jobs'][0]['directory'])/'progress.json')['attempts'] == 1
    assert identity(companion,content_hash=True) == original_identity
    with rec.trx_path.open('ab') as stream:
        stream.write(b'changed-source')
    result = invoke(tmp_path,'--resume')
    assert result.returncode != 0
    assert not batch.read(tmp_path/'batch/status.json')['complete']
    assert identity(companion,content_hash=True) == original_identity


def test_resume_reuses_published_companion_after_diagnostic_failure(tmp_path,monkeypatch):
    rec, frozen, job = inputs(tmp_path)
    directory = tmp_path/'job'; directory.mkdir()
    batch.save(directory/'input.json',job)
    real_run = batch.subprocess.run
    def failed_comparison(*args,**kwargs):
        if any('compare_jump_trajectories.py' in str(value) for value in args[0]):
            raise RuntimeError('Diagnostic failure after tracking')
        return real_run(*args,**kwargs)
    monkeypatch.setattr(batch.subprocess,'run',failed_comparison)
    # producer() also uses subprocess.run internally through check_output;
    # preserve runtime verification while simulating only diagnostic failure.
    monkeypatch.setattr(batch,'verify_runtime',lambda config: None)
    with pytest.raises(RuntimeError,match='Diagnostic failure'):
        batch.execute_job(directory)
    progress = batch.read(directory/'progress.json')
    assert progress['stage'] == 'failed'
    companion_identity = identity(Path(progress['companion']),content_hash=True)
    audit_identity = identity(Path(progress['audit'])/'background.npz',content_hash=True)
    monkeypatch.setattr(batch.subprocess,'run',real_run)
    batch.execute_job(directory)
    progress = batch.read(directory/'progress.json')
    assert progress['stage'] == 'complete' and progress['attempts'] == 2
    assert identity(Path(progress['companion']),content_hash=True) == companion_identity
    assert identity(Path(progress['audit'])/'background.npz',content_hash=True) == audit_identity


def test_drifted_tracker_is_rejected_before_output_creation(tmp_path):
    rec,frozen,job = inputs(tmp_path)
    frozen['producer']['package_sha256'] = 'different-tracker'
    batch.save(tmp_path/'freeze.json',frozen)
    result = invoke(tmp_path)
    assert result.returncode != 0 and 'Frozen runtime mismatch' in result.stdout
    assert not (tmp_path/'batch').exists()


def test_job_reuses_verified_external_audit_and_companion(tmp_path, monkeypatch):
    rec, frozen, job = inputs(tmp_path)
    audit_dir = tmp_path/'prior-audit'
    companion = tmp_path/'prior.offline.trx'
    batch.audit(rec.video, audit_dir, **frozen['audit'])
    batch.track(audit_dir, companion, config=DetectionConfig(**frozen['detector']))
    expected = identity(companion, content_hash=True)
    directory = tmp_path/'job'
    directory.mkdir()
    batch.save(directory/'input.json', dict(job, reuse_audit=str(audit_dir),
                                           reuse_companion=str(companion)))

    def must_reuse(*args, **kwargs):
        pytest.fail('Inputs should be reused without decoding or tracking again')

    monkeypatch.setattr(batch, 'audit', must_reuse)
    monkeypatch.setattr(batch, 'track', must_reuse)
    batch.execute_job(directory)
    progress = batch.read(directory/'progress.json')
    assert progress['stage'] == 'complete'
    assert progress['audit'] == str(audit_dir)
    assert progress['companion_identity'] == expected
