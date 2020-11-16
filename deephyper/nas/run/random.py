import random

def run(config):
    random.seed(config.get('seed'))
    if 'arch_seq' in config:
        return sum(config['arch_seq']) + random.random()
    else:
        return sum(config.values()) + random.random()

