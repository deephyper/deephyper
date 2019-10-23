from deephyper.post.pipeline import train
from deephyper.search.nas.model.run.alpha import run
from deephyper.benchmark.nas.linearRegMultiLoss import Problem


def test_multi_loss():
    print(Problem)

    config = Problem.space
    config['hyperparameters']['verbose'] = 1

    # Baseline
    config['arch_seq'] = [0.8844696727478868, 0.5947199629108353, 0.7022291539528956, 0.16433200901517664, 0.3210369300814784]

    run(config)

if __name__ == '__main__':
    test_multi_loss()
