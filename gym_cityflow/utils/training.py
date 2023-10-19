import numpy as np
from utils.helper import timing_decorator, update_model


# Pre-training the agent without an adversary
@timing_decorator # Decorator to measure the execution time of the function
def pretrain(agent, agent_env, file_config, params):

    # Setting perturbation probability to 0 for pre-training
    agent_env.pert_prob = 0 
    
    # Training the agent for a specified number of timesteps without resetting timesteps
    agent.learn(total_timesteps=agent_env.steps_per_episode*params["pretrain_episodes"], reset_num_timesteps=False, tb_log_name="pretrain_log_every")

    # Saving the pre-trained agent model
    agent.save(file_config["pretrain_checkpoint"])


@timing_decorator
def main_train(agent, agent_env, adv, adv_env, config):

    # Extract configurations and parameters from the provided config
    file_config = config["file_config"]
    params = config["params"]


    # If starting from a checkpoint (not from the pretrained agent)
    if file_config["load_models"] and not file_config["load_pretrain"]:
        with open(file_config["current_episode"], "r") as file:
            start_episode = int(file.read())
    else: 
        start_episode = 0

    # updates model if pretrain agent has different params 
    if params["pretrain"]:
        agent = update_model(agent, agent_env)

    # Loop through the specified total number of episodes 
    for episode in range(start_episode, params["total_episodes"]):
        
        # Gradually increase the perturbation probability
        if params["curriculum_learning"]:
            pert_prob = np.linspace(
                params["initial_perturbation_prob"],
                params["max_perturbation_prob"],
                params["total_episodes"]
            )[episode]

            # Set the computed perturbation probability for the agent's environment
            agent_env.pert_prob = pert_prob

            # Logging the current perturbation probability
            print(f"Agent perturbation probability: {agent_env.pert_prob}")
        else:
            # Keeps perturbation probability static
            agent_env.pert_prob = 1

         # Save model checkpoints and update current episode at specified intervals
        if file_config["save_model"]:
            if episode % file_config["save_model_every"] == 0 and episode > 0:
                agent.save(file_config["agent_checkpoint_path"] + str(episode))
                adv.save(file_config["adv_checkpoint_name"] + str(episode))
                with open(file_config["current_episode"], "w") as file:
                    file.write(str(episode))

        # Reset the LSTM's internal states for both agent and adversary
        agent.policy.reset_states()
        adv.policy.reset_states()

        # Determine the training entity (agent or adversary) based on the current episode
        is_adv_training = (episode // params["episodes_per_round"]) % 2 == 1

        # Set the training mode based on the determined training entity and train the active model
        if is_adv_training:
            print("Training the adversary in episode:", episode)
            adv_env.set_mode(True, agent)
            adv.learn(total_timesteps=agent_env.steps_per_episode, reset_num_timesteps=False, tb_log_name=file_config["log_adv"])
        else:
            print("Training the agent in episode:", episode)
            agent_env.set_mode(False, agent, adv)
            agent.learn(total_timesteps=agent_env.steps_per_episode, reset_num_timesteps=False, tb_log_name=file_config["log_agent"])
    return None
