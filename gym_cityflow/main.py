from utils.training import pretrain, main_train
from utils.helper import get_config, create_models, timing_decorator

if __name__ == "__main__":

    print("Getting configs")
    config = get_config("./utils/configs.json")

    print("Creating models")
    agent, agent_env, adv, adv_env = create_models(config)


    if config["params"]["pretrain"]:
        print("pretraining model without adversary")
        agent = pretrain(agent, agent_env, config["file_config"], config["params"])
    
    print("\nMain training")    
    main_train(agent, agent_env, adv, adv_env, config)

    #TODO model parameter loading


    


