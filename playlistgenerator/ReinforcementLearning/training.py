from __future__ import absolute_import, division, print_function

import json
import os

import pandas as pd
import reverb
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.policies import policy_saver, py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

from playlistgenerator.ReinforcementLearning.constants import (
    batch_size,
    collect_steps_per_iteration,
    eval_interval,
    log_interval,
    num_eval_episodes,
    num_iterations,
)
from playlistgenerator.ReinforcementLearning.environment import MusicPlaylistEnv

tf.config.optimizer.set_jit(True)


# Compute the average return of the policy over a number of episodes
def compute_avg_return(environment, policy, num_episodes=10):
    """Computes the average return of the RL policy on a testing environment."""
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    print(total_return)

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def dense_layer(num_units):
    """Helper function to create Dense layers configured with the right activation and kernel initializer."""
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_in", distribution="truncated_normal"
        ),
    )


def initialize_agent(
    env,
    train_env2,
    learning_rate=1e-3,
    epsilon_initial=1.0,
    epsilon_final=0.1,
    epsilon_decay_steps=10000,
):
    """Initialize the DQN agent with the given environment and hyperparameters."""
    fc_layer_params = (1024, 512, 256, 128)
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    # Create the Q-network
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation="relu",
        kernel_initializer=tf.keras.initializers.RandomUniform(minval=-2, maxval=2),
        bias_initializer=tf.keras.initializers.Constant(0.0),
    )
    q_net = sequential.Sequential(dense_layers + [q_values_layer])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    # Define the epsilon decay schedule
    epsilon = tf.compat.v1.train.polynomial_decay(
        learning_rate=epsilon_initial,
        global_step=train_step_counter,
        decay_steps=epsilon_decay_steps,
        end_learning_rate=epsilon_final,
    )

    # Create the DQN agent
    agent = dqn_agent.DqnAgent(
        train_env2.time_step_spec(),
        train_env2.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        target_update_period=1000,  # Update target network every 1000 steps
        td_errors_loss_fn=common.element_wise_huber_loss,
        gamma=0.99,  # Discount factor
        train_step_counter=train_step_counter,
        epsilon_greedy=epsilon,  # Decaying exploration rate
    )

    agent.initialize()

    return agent


def generate_replay_buffer(agent, table_name, replay_buffer_max_length=100000):
    """Generate the replay buffer for experience replay"""
    replay_buffer_signature = tensor_spec.from_spec(agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)

    # Create the replay buffer table
    table = reverb.Table(
        table_name,
        max_size=replay_buffer_max_length,
        sampler=reverb.selectors.Prioritized(
            priority_exponent=0.8
        ),  # Use prioritized experience replay
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature,
    )

    reverb_server = reverb.Server([table])

    # Create the replay buffer
    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        table_name=table_name,
        sequence_length=2,
        local_server=reverb_server,
    )

    return replay_buffer


def train_models(ml_data):
    """Train the RL Music Playlist Generator with the provided data."""
    # Create training and evaluation environments
    train_py_env = MusicPlaylistEnv(ml_data)
    eval_py_env = MusicPlaylistEnv(ml_data)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    env = MusicPlaylistEnv(ml_data)

    # Initialize the agent
    agent = initialize_agent(env, train_env)

    table_name = "uniform_table"

    # Generate the replay buffer
    replay_buffer = generate_replay_buffer(agent, table_name)

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
        replay_buffer.py_client, table_name, sequence_length=2
    )

    # Collect initial experience
    py_driver.PyDriver(
        env,
        py_tf_eager_policy.PyTFEagerPolicy(agent.collect_policy, use_tf_function=True),
        [rb_observer],
        max_steps=collect_steps_per_iteration,
    ).run(train_py_env.reset())

    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2
    ).prefetch(3)

    iterator = iter(dataset)

    global_best = 0
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step.
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    # Reset the environment.
    time_step = train_py_env.reset()

    # Create a driver to collect experience.
    collect_driver = py_driver.PyDriver(
        env,
        py_tf_eager_policy.PyTFEagerPolicy(agent.collect_policy, use_tf_function=True),
        [rb_observer],
        max_steps=collect_steps_per_iteration,
    )

    train_loss_array = []
    global_best_array = []
    # Training loop
    for iteration_num in range(num_iterations):
        print(f"Iteration: {iteration_num}")
        # Collect a few steps and save to the replay buffer.
        time_step, _ = collect_driver.run(time_step)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        # Log training loss
        if step % log_interval == 0:
            print("step = {0}: loss = {1}".format(step, train_loss))
            print(train_loss.numpy())
            train_loss_array.append(str(train_loss.numpy()))

        # Evaluate the agent's policy periodically
        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            global_best = max(global_best, avg_return)
            print("step = {0}: Average Return = {1}".format(step, avg_return))
            print(f"Global Best: {global_best}")
            global_best_array.append(str(global_best))
            returns.append(avg_return)

            policy_dir = os.path.join("outputs/ModelOutputs", "policy")
            tf_policy_saver = policy_saver.PolicySaver(agent.policy)
            tf_policy_saver.save(policy_dir)

        with open("outputs/ModelOutputs/train_loss.json", "w") as f1:
            json.dump(train_loss_array, f1, indent=4)

        with open("outputs/ModelOutputs/global_best.json", "w") as f2:
            json.dump(global_best_array, f2, indent=4)

    return agent


if __name__ == "__main__":
    print("\n\n\n")
    print("Starting Training!")
    print("\n\n\n")

    reccs = pd.read_csv("playlistgenerator/Datasets/spotify_recommendations.csv")
    model_output = pd.read_csv("playlistgenerator/Datasets/full_recommendations.csv")
    reccs = reccs.rename(
        columns={
            "track_name": "name",
            "artist_name": "artists",
            "sentiment": "sentiment_score",
        }
    )

    for col in model_output.columns:
        if col not in reccs.columns:
            reccs[col] = 0

    # Reorder `model_outputs` to start with the columns in `reccs`
    ordered_columns = list(reccs.columns) + [
        col for col in model_output.columns if col not in reccs.columns
    ]
    result = pd.concat([reccs, model_output]).fillna(0)
    result = result[ordered_columns]

    result.to_csv("playlistgenerator/Datasets/combined_recommendation.csv", index=False)
    agent = train_models(result)
