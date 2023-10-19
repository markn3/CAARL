import json
import random

hyperparams_grid = {
    'learning_rate': [1e-3, 1e-4, 1e-5],
    'n_steps' : [1024, 2048],
    'batch_size':[32, 64],
    'n_epochs': [5, 10, 15], 
    'gamma': [0.98, 0.99, 0.995], 
    'clip_range': [0.1, 0.2, 0.3],
    'clip_range_vf': [None, 0.2, 0.3],
    'normalize_advantage': [True],
    'ent_coef': [0.0, 0.01, 0.001],
    'vf_coef': [0.5],
    'max_grad_norm': [0.5],
}


# TODO 
def grid_search(hyperparams_grid, env):

    results = {}  # A dictionary to store results for each combination

    for lr in hyperparams_grid['learning_rate']:
        for n_steps in hyperparams_grid['n_steps']:
            for batch_size in hyperparams_grid['batch_size']:
                for n_epochs in hyperparams_grid['n_epochs']:
                    for gamma in hyperparams_grid['gamma']:
                        for gae_lambda in hyperparams_grid['gae_lambda']:
                            for clip_range in hyperparams_grid['clip_range']:
                                for clip_range_vf in hyperparams_grid['clip_range_vf']:
                                    for normalize_advantage in hyperparams_grid['normalize_advantage']:
                                        for ent_coef in hyperparams_grid['ent_coef']:
                                            agent = PPO(
                                                CustomLSTMPolicy,
                                                env,
                                                learning_rate=lr,
                                                n_steps=n_steps,
                                                batch_size=batch_size,
                                                n_epochs=n_epochs,
                                                gamma=gamma,
                                                gae_lambda=gae_lambda,
                                                clip_range=clip_range,
                                                clip_range_vf=clip_range_vf,
                                                normalize_advantage=normalize_advantage,
                                                ent_coef=ent_coef,
                                                vf_coef=hyperparams_grid['vf_coef'][0],  # Single value
                                                max_grad_norm=hyperparams_grid['max_grad_norm'][0]  # Single value
                                            )
                                            # training here
                                            main_train(agent)

                                            # evaluate and log rewards in dictionary
                                            reward = evaluate(model, env)
                                            key = (lr, n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range, clip_range_vf, normalize_advantage, ent_coef)
                                            results[key] = reward
    # Save results to a JSON file
    with open("./logs/grid_search", "w") as file:
        json.dump(results, file)

def rand_search(hyperparams_grid, env):
    # Number of random combinations to try
    num_iterations = 100  

    results_random_search = {}

    for _ in range(num_iterations):
    # Randomly sample hyperparameters
    lr = random.choice(hyperparams_grid['learning_rate'])
    n_steps = random.choice(hyperparams_grid['n_steps'])
    batch_size = random.choice(hyperparams_grid['batch_size'])
    n_epochs = random.choice(hyperparams_grid['n_epochs'])
    gamma = random.choice(hyperparams_grid['gamma'])
    gae_lambda = random.choice(hyperparams_grid['gae_lambda'])
    clip_range = random.choice(hyperparams_grid['clip_range'])
    clip_range_vf = random.choice(hyperparams_grid['clip_range_vf'])
    normalize_advantage = random.choice(hyperparams_grid['normalize_advantage'])
    ent_coef = random.choice(hyperparams_grid['ent_coef'])
    
    # Train and evaluate the model with the sampled hyperparameters
    agent = PPO(
        CustomLSTMPolicy,
        env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        clip_range_vf=clip_range_vf,
        normalize_advantage=normalize_advantage,
        ent_coef=ent_coef,
        vf_coef=hyperparams_grid['vf_coef'][0],  # Single value
        max_grad_norm=hyperparams_grid['max_grad_norm'][0]  # Single value
    )


    # training here
    main_train(agent)

    # evaluate and log rewards in dictionary
    reward = evaluate(model, env)
    key = (lr, n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range, clip_range_vf, normalize_advantage, ent_coef)
    results_random_search[key] = reward

    # Save results to a JSON file
    with open("./logs/rand_search", "w") as file:
        json.dump(results, file)