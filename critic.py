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
            sp_term = (q - v) / alpha + 0.5
            weight = jnp.where(sp_term > 0, 1., 0.)
            loss_v = weight * (sp_term**2)
            value_loss = (loss_v + v / alpha).mean()
        elif alg == 'EQL':
            diff = (q - v) / alpha
            diff = jnp.minimum(diff, 5.0)
            max_z = jnp.max(diff, axis=0)
            max_z = jnp.where(max_z < -1.0, -1.0, max_z)
            max_z = jax.lax.stop_gradient(max_z)
            loss_v = jnp.exp(diff - max_z) - diff * jnp.exp(-max_z) - jnp.exp(-max_z)
            value_loss = loss_v.mean()
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
    # def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
    #     q1, q2 = critic.apply({'params': critic_params}, batch.observations,
    #                           batch.actions)
    #
    #     def mse_loss(q, q_target):
    #         loss_dict = {}
    #
    #         x = q-q_target
    #         loss = huber_loss(x, delta=20.0)  # x**2
    #         loss_dict['critic_loss'] = loss.mean()
    #
    #         return loss.mean()
    #
    #     critic_loss = (mse_loss(q1, target_q) + mse_loss(q2, target_q)).mean()
    #
    #     return critic_loss, {
    #         'critic_loss': critic_loss,
    #         'q1': q1.mean()
    #     }
    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply({'params': critic_params}, batch.observations,
                              batch.actions)
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean()
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info


def huber_loss(x, delta: float = 1.):
    """Huber loss, similar to L2 loss close to zero, L1 loss away from zero.
    See "Robust Estimation of a Location Parameter" by Huber.
    (https://projecteuclid.org/download/pdf_1/euclid.aoms/1177703732).
    Args:
    x: a vector of arbitrary shape.
    delta: the bounds for the huber loss transformation, defaults at 1.
    Note `grad(huber_loss(x))` is equivalent to `grad(0.5 * clip_gradient(x)**2)`.
    Returns:
    a vector of same shape of `x`.
    """
    # 0.5 * x^2                  if |x| <= d
    # 0.5 * d^2 + d * (|x| - d)  if |x| > d
    abs_x = jnp.abs(x)
    quadratic = jnp.minimum(abs_x, delta)
    # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
    linear = abs_x - quadratic
    return 0.5 * quadratic**2 + delta * linear