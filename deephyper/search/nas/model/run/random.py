import random

def run(config):
    random.seed(config.get('seed'))
    return sum(config['arch_seq']) + random.random()
