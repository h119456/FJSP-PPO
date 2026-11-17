import numpy as np
import torch

from scripts.analyze_physical_gate import fixed_gates, gate_summary
from ppo_rs.models.physical_gate import PhysicalEdgeGate


def test_fixed_intervention_changes_only_applicable_relations():
    graph = dict(node_features=torch.zeros(2, 13),
                 edge_types=torch.tensor([0, 4, 5, 6, 7, 8, 9, 4]),
                 physical_edge_mask=torch.tensor([False, True, True, True, True, True, True, False]))
    table = torch.tensor([[.1]*4, [.5]*4, [.8]*4])
    gates = fixed_gates(graph, table)
    torch.testing.assert_close(gates, torch.tensor([[1.]*4, [.1]*4, [.1]*4, [.5]*4,
                                                   [.5]*4, [.8]*4, [.8]*4, [1.]*4]))


def test_constant_gate_statistics_exclude_nonapplicable_ones_and_have_zero_sensitivity():
    module = PhysicalEdgeGate(4)
    with torch.no_grad():
        for head in module.gate_heads:
            head[-1].weight.zero_()
            head[-1].bias.zero_()
    features = np.random.default_rng(7).normal(size=(30, 6)).astype(np.float32)
    groups = np.repeat(np.arange(3), 10)
    statistics = gate_summary(features, np.full((30, 4), .5), groups, module)
    for group in statistics.values():
        assert group['count'] == 10
        np.testing.assert_allclose(group['mean'], .5)
        np.testing.assert_allclose(group['std'], 0.)
        np.testing.assert_allclose(group['standardized_sensitivity_head_by_feature'], 0.)
