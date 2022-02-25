import jax
import pytest


@pytest.fixture
def setup():
    from jaxmapp.env.instance import InstanceGeneratorCircleObs

    key = jax.random.PRNGKey(0)
    generator = InstanceGeneratorCircleObs(
        num_agents_min=5,
        num_agents_max=5,
        max_speeds_cands=[0.25],
        rads_cands=[0.01],
        map_size=160,
        num_obs=0,
    )
    ins = generator.generate(key)
    return key, ins


def test_pp(setup):
    from jaxmapp.planner.prioritized_planning import PrioritizedPlanning as PP
    from jaxmapp.roadmap.random_sampler import RandomSampler

    key, ins = setup

    sampler = RandomSampler(share_roadmap=True, num_samples=100, max_T=64)
    trms = sampler.construct_trms(key, ins)
    planner = PP(verbose=1)
    res = planner.solve(ins.to_numpy(), trms)
    print(f"{res=}")
