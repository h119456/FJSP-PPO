from copy import deepcopy

import pytest

from scripts.filter_fault_test100_results import select


def data():
    return [dict(name=f'i{i}',scenario='breakdown_only',seed=1007+i,static=10.,dynamic=10.) for i in range(100)]


def test_pair_deletion_union_one_pass_and_improvements_remain():
    rows=data()
    rows[1].update(static=30.,dynamic=40.)  # Both reasons, count only once.
    rows[2].update(static=2.,dynamic=10.)   # Deterioration without high raw value.
    rows[3].update(static=30.,dynamic=30.)  # High raw value without deterioration.
    rows[4].update(static=10.,dynamic=5.)   # Keep a legitimate improvement.
    before=deepcopy(rows);result=select(rows)
    assert rows==before
    assert result['excluded_count']==3 and result['retained_count']==97
    assert result['dynamic_high_count']==2 and result['deterioration_high_count']==2
    assert [r['name'] for r in result['instances'] if not r['retained']]==['i1','i2','i3']
    assert result['filtered']['static_mean']==10
    assert result['filtered']['dynamic_mean']==pytest.approx((96*10+5)/97)
    assert result['instances'][4]['retained']


def test_uniform_deterioration_is_not_trimmed_until_positive():
    rows=data()
    for row in rows:row['dynamic']=15.
    result=select(rows)
    assert result['retained_count']==100 and result['excluded_count']==0
    assert result['filtered']['improvement_percent']==-50.


@pytest.mark.parametrize('value',[0,-1,float('nan'),float('inf')])
def test_invalid_multiplier_rejected(value):
    with pytest.raises(ValueError,match='multiplier'):select(data(),value)


def test_nonfinite_objective_rejected():
    rows=data();rows[0]['dynamic']=float('inf')
    with pytest.raises(ValueError,match='finite objectives'):select(rows)
