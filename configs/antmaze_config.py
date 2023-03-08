import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 2e-4
    config.value_lr = 2e-4
    config.critic_lr = 2e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99

    # config.temperature = 10
    config.dropout_rate = 0.0
    config.value_dropout_rate = 0.5

    config.tau = 0.005  # For soft target updates.

    return config
