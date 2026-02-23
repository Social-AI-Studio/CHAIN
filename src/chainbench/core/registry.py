# The registry of environment config
ENVIRONMENT_CONFIG_REGISTRY = {}

# The registry of environment class
ENVIRONMENT_REGISTRY = {}

# The registry of task config
TASK_CONFIG_REGISTRY = {}

# The registry of task class
TASK_REGISTRY = {}

# The registry of agent class
AGENT_REGISTRY = {}

def register_environment_config(kind: str):
    def deco(cls):
        ENVIRONMENT_CONFIG_REGISTRY[kind] = cls
        return cls
    return deco

def register_environment(kind: str):
    def deco(cls):
        ENVIRONMENT_REGISTRY[kind] = cls
        return cls
    return deco

def register_task_config(kind: str):
    def deco(cls):
        TASK_CONFIG_REGISTRY[kind] = cls
        return cls
    return deco

def register_task(kind: str):
    def deco(cls):
        TASK_REGISTRY[kind] = cls
        return cls
    return deco

def register_agent(kind: str):
    def deco(cls):
        AGENT_REGISTRY[kind] = cls
        return cls
    return deco