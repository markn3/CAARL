from utils.training import pretrain, main_training
from utils.helper import load_parameters, create_models, timing_decorator

if __name__ == "__main__":

    print("Loading parameters")
    config = load_parameters("./utils/configs.json")

    print("Creating models")
    agent, agent_env, adv, adv_env = create_models(config)


    if config["params"]["pretrain"]:
        print("pretraining model without adversary")
        agent = pretrain(agent, agent_env, config["file_config"])
    
    print("Main training")    
    main_training(agent, agent_env, adv, adv_env, config)

    #TODO model parameter loading


    


