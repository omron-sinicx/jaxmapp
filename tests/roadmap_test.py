import jax
import pytest


@pytest.fixture
def setup():
    from jaxmapp.env.instance import InstanceGeneratorCircleObs

    key = jax.random.PRNGKey(0)
    generator = InstanceGeneratorCircleObs(
        num_agents_min=10,
        num_agents_max=10,
        max_speeds_cands=[0.04],
        rads_cands=[0.02],
        map_size=160,
        num_obs=0,  # no obstacles are preferred to test roadmaps
    )
    ins = generator.generate(key)
    return key, ins


def test_random_sampler(setup):
    from jaxmapp.roadmap.random_sampler import RandomSampler

    key, ins = setup

    sampler = RandomSampler(share_roadmap=False, num_samples=100, max_T=64)
    vertices = sampler.sample_vertices(key, 100, ins)
    assert vertices.shape == (ins.num_agents, 100, 2)
    edges = sampler.check_connectivity(vertices, ins)
    assert edges.shape == (ins.num_agents, 100, 100)
    trms = sampler.construct_trms(key, ins)

    sampler = RandomSampler(share_roadmap=True, num_samples=100, max_T=64)
    vertices = sampler.sample_vertices(key, 100, ins)
    assert vertices.shape == (100, 2)
    edges = sampler.check_connectivity(vertices, ins)
    assert edges.shape == (100, 100)
    trms = sampler.construct_trms(key, ins)


def test_grid_sampler(setup):
    from jaxmapp.roadmap.grid_sampler import GridSampler

    key, ins = setup

    sampler = GridSampler(share_roadmap=False, num_samples=100, max_T=64)
    vertices = sampler.sample_vertices(key, 100, ins)
    assert vertices.shape == (ins.num_agents, 100, 2)
    edges = sampler.check_connectivity(vertices, ins)
    assert edges.shape == (ins.num_agents, 100, 100)
    trms = sampler.construct_trms(key, ins)

    sampler = GridSampler(share_roadmap=True, num_samples=100, max_T=64)
    vertices = sampler.sample_vertices(key, 100, ins)
    assert vertices.shape == (100, 2)
    edges = sampler.check_connectivity(vertices, ins)
    assert edges.shape == (100, 100)
    trms = sampler.construct_trms(key, ins)


def test_learned_sampler(setup):
    import hydra
    from flax.training.checkpoints import restore_checkpoint
    from jaxmapp.roadmap.learned_sampler import CTRMSampler
    from omegaconf import OmegaConf

    key, ins = setup

    sampler = CTRMSampler()
    model = hydra.utils.instantiate(OmegaConf.load("scripts/config/model/ctrm.yaml"))
    params = restore_checkpoint("data_example/model/training_hetero_k05/", None)[
        "params"
    ]
    sampler.set_model_and_params(model, params)
    vertices = sampler.sample_vertices(key, 25, ins)
    assert vertices.shape == (ins.num_agents, sampler.max_T, 25, 2)
    edges = sampler.check_connectivity(vertices, ins)
    assert edges.shape == (ins.num_agents, sampler.max_T, 25, 25)
    trms = sampler.construct_trms(key, ins.to_jnumpy())
