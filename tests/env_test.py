import jax


def test_instance_generation_circle_obs():
    from jaxmapp.env.instance import InstanceGeneratorCircleObs

    key = jax.random.PRNGKey(0)
    generator = InstanceGeneratorCircleObs(
        num_agents_min=4,
        num_agents_max=8,
        max_speeds_cands=[0.1, 0.2],
        rads_cands=[0.025, 0.05],
        map_size=160,
        num_obs=10,
    )
    ins = generator.generate(key)


def test_instance_generation_circle_obs_large_num_of_agents():
    from jaxmapp.env.instance import InstanceGeneratorCircleObs

    key = jax.random.PRNGKey(0)
    generator = InstanceGeneratorCircleObs(
        num_agents_min=31,
        num_agents_max=40,
        max_speeds_cands=[0.03],
        rads_cands=[0.001],
        map_size=160,
        num_obs=10,
    )
    ins = generator.generate(key)


def test_instance_generation_circle_img():
    import numpy as np
    from jaxmapp.env.instance import InstanceGeneratorImageInput

    key = jax.random.PRNGKey(0)
    image = np.zeros((160, 160))
    image[40:80, 40:80] = 1
    generator = InstanceGeneratorImageInput(
        num_agents_min=4,
        num_agents_max=8,
        max_speeds_cands=[0.1, 0.2],
        rads_cands=[0.025, 0.05],
        image=image,
    )
    ins = generator.generate(key)
