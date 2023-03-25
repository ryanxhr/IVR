from typing import Tuple
import jax.numpy as jnp
from common import PRNGKey
import policy
import jax

from common import Batch, InfoDict, Model, Params


def update_v(critic: Model, value: Model, batch: Batch,
             alpha: float, alg: str) -> Tuple[Model, InfoDict]:

    q1, q2 = critic(batch.observations, batch.actions)
    q = jnp.minimum(q1, q2)

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply({'params': value_params}, batch.observations)
        if alg == 'SQL':
            sp_term = (q - v) / (2 * alpha) + 1.0
            sp_weight = jnp.where(sp_term > 0, 1., 0.)
            value_loss = (sp_weight * (sp_term**2) + v / alpha).mean()
        elif alg == 'EQL':
            sp_term = (q - v) / alpha
            sp_term = jnp.minimum(sp_term, 5.0)
            max_sp_term = jnp.max(sp_term, axis=0)
            max_sp_term = jnp.where(max_sp_term < -1.0, -1.0, max_sp_term)
            max_sp_term = jax.lax.stop_gradient(max_sp_term)
            value_loss = (jnp.exp(sp_term - max_sp_term) + jnp.exp(-max_sp_term) * v / alpha).mean()
        else:
            raise NotImplementedError('please choose SQL or EQL')
        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
            'q-v': (q - v).mean(),
        }

    new_value, info = value.apply_gradient(value_loss_fn)

    return new_value, info


def update_q(critic: Model, value: Model,
             batch: Batch,  discount: float) -> Tuple[Model, InfoDict]:
    next_v = value(batch.next_observations)
    target_q = batch.rewards + discount * batch.masks * next_v
    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply({'params': critic_params}, batch.observations,
                              batch.actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean()
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info