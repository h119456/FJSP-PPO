"""Check independent fault-only test generation and its audit safeguards."""
import hashlib
import random
import shutil

import pytest

from ppo_rs import load_instances
from scripts import generate_fault_test100 as generator


def test_fault_conditioned_rounding():
    assert generator.quotas() == {'breakdown_only': 67, 'both': 33}
    assert generator.quotas(3) == {'breakdown_only': 2, 'both': 1}


@pytest.fixture(scope='module')
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp('fault100')
    # Integration exercises real generation/loading; source inventory is tested
    # against all real source files during the production data build.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(generator, 'inventory', lambda root: (set(), {'test_inventory': True}))
        first, second = root / 'first', root / 'second'
        report = generator.generate(first)
        generator.generate(second)
    return first, second, report


def test_all_scales_load_reproduce_and_refuse_overwrite(generated):
    first, second, report = generated
    assert report['passed'] and report['total_instances'] == 300
    assert (first/'manifest.json').read_bytes() == (second/'manifest.json').read_bytes()
    for dataset in (1, 2, 3):
        instances = load_instances(first/f'set{dataset}/test')
        assert len(instances) == 100
        for path in (first/f'set{dataset}/test').glob('*.json'):
            assert path.read_bytes() == (second/f'set{dataset}/test'/path.name).read_bytes()
    with pytest.raises(FileExistsError):
        generator.generate(first)


def test_audit_detects_changed_data(generated, tmp_path):
    first, _, _ = generated
    target = tmp_path/'changed'; shutil.copytree(first, target)
    path = target/'set1/test/instance_0001.json'
    raw = generator.read(path); raw['stress_relief']['cost'] = 0.1
    generator.write(path, raw)
    with pytest.raises(ValueError, match='checksum mismatch'):
        generator.validate(target, existing=set())
    manifest = generator.read(target/'manifest.json')
    manifest['sets']['set1']['files_sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    generator.write(target/'manifest.json', manifest)
    with pytest.raises(ValueError, match='expected cost0'):
        generator.validate(target, existing=set())


def test_overlap_rejects_entire_build_before_publication(tmp_path, monkeypatch):
    base_seed = generator.SEED + 1_000_003
    existing = generator.fingerprint(generator.candidate(1, random.Random(base_seed+1), 1))
    monkeypatch.setattr(generator, 'inventory', lambda root: ({existing}, {}))
    target = tmp_path/'rejected'
    with pytest.raises(ValueError, match='overlapping scheduling instance'):
        generator.generate(target)
    assert not target.exists()
    assert not list(tmp_path.glob('.fault-test100-*'))
