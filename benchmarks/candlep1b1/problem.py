from collections import OrderedDict
class Problem():
    def __init__(self):
        space = OrderedDict()
        space['epochs'] = (100, 200)
        space['activation'] = ["relu", "sigmoid", "tanh"]
        space['alpha_dropout'] = [False, True]
        space['model'] = ["ae", "vae", "cvae"]
        space['dense'] = [ 
                          [2000, 600], 
                          [978, 978], 
                          [978, 978, 978], 
                          [978, 978, 978, 978] 
                          [978, 978, 978, 978, 978], 
                          [978, 978, 978, 978, 978, 978], 
                          ]
        space['latent_dim'] = [2, 8, 32, 128, 512]
        space['residual'] = [False, True]
        
        space['batch_size'] = [32, 64, 128, 256, 512, 1024]
        space['drop'] = (0.0, 0.9)

        space['optimizer'] = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']
        #space['clipnorm'] = (1e-04, 1e01)
        #space['clipvalue'] = (1e-04, 1e01)

        # optimizer parameters
        space['learning_rate'] = (1e-04, 1e01)
        space['reduce_lr'] = [False, True]
        space['warmup_lr'] = [False, True]
        space['decay_lr'] =  (0, 1e01)
        space['decay_schedule_lr'] =  (0, 1e01)
        space['nesterov_sgd'] = [False, True]
        space['momentum_sgd'] = (0.0, 1e01)
        space['rho'] = (1e-04, 1e01)
        space['epsilon'] = (1e-08, 1e01)
        space['beta_1'] = (1e-04, 1e01)
        space['beta_2'] = (1e-04, 1e01)

        self.space = space
        self.params = self.space.keys()
        self.starting_point = []

if __name__ == '__main__':
    instance = Problem()
    print(instance.space)
    print(instance.params)
