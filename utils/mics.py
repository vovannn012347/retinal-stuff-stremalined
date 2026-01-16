


def check_optimizer_model_match(optimizer, model):
    model_params = set(p for p in model.parameters())
    optimizer_params = set(p for group in optimizer.param_groups for p in group['params'])

    if model_params == optimizer_params:
        print("Optimizer parameters match the model parameters.")
        return True
    else:
        print("Mismatch detected!")
        extra_in_model = model_params - optimizer_params
        extra_in_optimizer = optimizer_params - model_params

        if extra_in_model:
            print(f"Parameters in model but not in optimizer: {len(extra_in_model)}")
        if extra_in_optimizer:
            print(f"Parameters in optimizer but not in model: {len(extra_in_optimizer)}")
        return False


def check_model_state_corruption(model, state_dict):
    model_params = model.state_dict()
    for name, param in model_params.items():
        if name not in state_dict:
            print(f"Missing parameter in state dict: {name}")
        elif state_dict[name].shape != param.shape:
            print(f"Shape mismatch for {name}: model {param.shape}, state dict {state_dict[name].shape}")
    for name in state_dict.keys():
        if name not in model_params:
            print(f"Extra parameter in state dict: {name}")


def check_optimizer_state_corruption(optimizer, state_dict):
    # Check param_groups length
    if len(optimizer.param_groups) != len(state_dict['param_groups']):
        print("Mismatch in param_groups length.")
        return False

    # Check individual params in param_groups
    for group_idx, (group, saved_group) in enumerate(zip(optimizer.param_groups, state_dict['param_groups'])):
        if len(group['params']) != len(saved_group['params']):
            print(f"Mismatch in param count for group {group_idx}.")
            return False

    # Validate state values (this does not work as it should)
    '''for param_id, state in state_dict['state'].items():
        if param_id not in optimizer.state:
            print(f"Missing parameter state for ID {param_id}.")
        else:
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    if value.shape != optimizer.state[param_id][key].shape:
                        print(f"Shape mismatch for state {key} in param ID {param_id}.")'''
    print("Optimizer state appears valid.")
    return True


