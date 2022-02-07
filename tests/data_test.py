import jax
import pytest


@pytest.fixture
def setup():
    from jaxmapp.env.instance import InstanceGeneratorCircleObs

    key = jax.random.PRNGKey(0)
    generator = InstanceGeneratorCircleObs(
        num_agents_min=11,
        num_agents_max=11,
        max_speeds_cands=[0.1],
        rads_cands=[0.025],
        map_size=160,
        num_obs=10,
    )
    ins = generator.generate(key)
    return key, ins


def test_load_save(setup):
    import os

    from jaxmapp.utils.data import load_instance, save_instance

    _, ins = setup
    try:
        save_instance(ins, "test.pkl")
        ins_ = load_instance("test.pkl")
    except:
        pass
    finally:
        os.remove("test.pkl")
