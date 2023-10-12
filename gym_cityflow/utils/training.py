import numpy as np
from utils.helper import timing_decorator


# Phase 1: Pre-training the agent without an adversary
@timing_decorator
def pretrain(agent, agent_env, file_config):

    agent_env.no_perturb = True

    for episode in range(file_config["pretrain_episodes"]):
        print(f"Episode {episode}")
        agent.policy.reset_states()
        obs = agent_env.reset().reshape(1, 5, 33)
        for step in range(agent_env.steps_per_episode):
            action, _ = agent.predict(obs)
            obs, reward, done, info = agent_env.step(action)
            if done:
                break
        agent.learn(total_timesteps=agent_env.steps_per_episode, reset_num_timesteps=False, tb_log_name="pretrain_log")
    
    agent_env.no_perturb = False
    agent.save(file_config["pretrain_checkpoint"] + "_checkpoint_" + f"{pre_training_episodes}")

# Phase 2: Gradual Adversarial Introduction
@timing_decorator
def main_training(agent, agent_env, adv, adv_env, config):

    file_config = config["file_config"]
    params = config["params"]

    for episode in range(params["start_episode"], params["total_episodes"]):
        # Gradually increase perturbation probability
        pert_prob = np.linspace(
            params["initial_perturbation_prob"],
            params["max_perturbation_prob"],
            params["total_episodes"]
        )[episode]

        agent_env.pert_prob = pert_prob
        print(f"Agent perturbation probability {agent_env.pert_prob}")
        
        if episode % file_config["save_model_every"] == 0 and episode > 0:
            agent.save(file_config["agent_checkpoint"] + str(episode))
            adv.save(file_config["adv_checkpoint"] + str(episode))
            with open(file_config["current_episode"], "w") as file:
                file.write(str(episode))

        # Reset the LSTM's internal states for both agent and adversary
        agent.policy.reset_states()
        adv.policy.reset_states()

        done = False

        # Determine the training entity (agent or adversary) based on the current episode
        is_adv_training = (episode // params["episodes_per_round"]) % 2 == 1

        # Set the training mode based on the determined training entity and display a message
        if is_adv_training:
            print("Training the adversary in episode:", episode)
            adv_env.set_mode(True, agent)
        else:
            print("Training the agent in episode:", episode)
            agent_env.set_mode(False, agent, adv)
        
        # Reset the environments and get initial observations
        agent_obs = agent_env.reset().reshape(1, 5, 33)
        adv_obs = adv_env.reset().reshape(1, 5, 33)

        for step in range(agent_env.steps_per_episode):
            # If the adversary is training, predict its action and take a step in its environment
            if is_adv_training:
                # Train the adversary
                action, _ = adv.predict(adv_obs)
                next_obs, reward, done, _ = adv_env.step(action)
                
                # Update adversary observation
                adv_obs = next_obs
                
                # Sync the perturbed state to the agent's environment
                agent_env.perturbed_state = adv_env.perturbed_state
            # If the agent is training, predict its action and take a step in its environment
            else:

                action, _ = agent.predict(agent_obs)
                agent_obs, reward, done, info = agent_env.step(action)
                
        # Train the active model (agent or adversary) based on the experience collected during this episode
        if is_adv_training:
            adv_env.set_mode(True, agent)
            adv.learn(total_timesteps=agent_env.steps_per_episode,reset_num_timesteps=False, tb_log_name=file_config["log_adv"])
        else:            
            agent_env.set_mode(False, agent, adv)
            agent.learn(total_timesteps=agent_env.steps_per_episode,reset_num_timesteps=False, tb_log_name=file_config["log_agent"])

    return None