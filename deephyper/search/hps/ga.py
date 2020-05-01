import signal
import random

from deephyper.search.hps.optimizer import GAOptimizer
from deephyper.search import Search
from deephyper.search import util

logger = util.conf_logger("deephyper.search.hps.ga")

SERVICE_PERIOD = 2  # Delay (seconds) between main loop iterations
CHECKPOINT_INTERVAL = 10  # How many jobs to complete between optimizer checkpoints
EXIT_FLAG = False


def on_exit(signum, stack):
    global EXIT_FLAG
    EXIT_FLAG = True


class GA(Search):
    def __init__(self, problem, run, evaluator, **kwargs):
        super().__init__(problem, run, evaluator, **kwargs)
        logger.info("Initializing GA")
        self.optimizer = GAOptimizer(self.problem, self.evaluator.num_workers, self.args)

    @staticmethod
    def _extend_parser(parser):
        parser.add_argument(
            "--ga_num_gen",
            default=100,
            type=int,
            help="number of generation for genetic algorithm",
        )
        parser.add_argument(
            "--individuals_per_worker",
            default=5,
            type=int,
            help="number of individuals per worker",
        )
        return parser

    def main(self):
        # opt = GAOptimizer(cfg)
        # evaluator = evaluate.create_evaluator(cfg)
        logger.info(f"Starting new run")

        timer = util.DelayTimer(max_minutes=None, period=SERVICE_PERIOD)
        timer = iter(timer)
        elapsed_str = next(timer)

        logger.info("Hyperopt GA driver starting")
        logger.info(f"Elapsed time: {elapsed_str}")

        if self.optimizer.pop is None:
            logger.info("Generating initial population")
            logger.info(f"{self.optimizer.INIT_POP_SIZE} individuals")
            self.optimizer.pop = self.optimizer.toolbox.population(
                n=self.optimizer.INIT_POP_SIZE
            )
            individuals = self.optimizer.pop
            self.evaluate_fitnesses(
                individuals,
                self.optimizer,
                self.evaluator,
                self.args.eval_timeout_minutes,
            )
            self.optimizer.record_generation(num_evals=len(self.optimizer.pop))

            with open("ga_logbook.log", "w") as fp:
                fp.write(str(self.optimizer.logbook))
            print("best:", self.optimizer.halloffame[0])

        while self.optimizer.current_gen < self.optimizer.NGEN:
            self.optimizer.current_gen += 1
            logger.info(
                f"Generation {self.optimizer.current_gen} out of {self.optimizer.NGEN}"
            )
            logger.info(f"Elapsed time: {elapsed_str}")

            # Select the next generation individuals
            offspring = self.optimizer.toolbox.select(
                self.optimizer.pop, len(self.optimizer.pop)
            )
            # Clone the selected individuals
            offspring = list(map(self.optimizer.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.optimizer.CXPB:
                    self.optimizer.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.optimizer.MUTPB:
                    self.optimizer.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            logger.info(f"Evaluating {len(invalid_ind)} invalid individuals")
            self.evaluate_fitnesses(
                invalid_ind,
                self.optimizer,
                self.evaluator,
                self.args.eval_timeout_minutes,
            )

            # The population is entirely replaced by the offspring
            self.optimizer.pop[:] = offspring

            self.optimizer.record_generation(num_evals=len(invalid_ind))

            with open("ga_logbook.log", "w") as fp:
                fp.write(str(self.optimizer.logbook))
            print("best:", self.optimizer.halloffame[0])

    def evaluate_fitnesses(self, individuals, opt, evaluator, timeout_minutes):
        points = map(opt.space_encoder.decode_point, individuals)
        points = [
            {key: x for key, x in zip(self.problem.space.keys(), point)}
            for point in points
        ]
        evaluator.add_eval_batch(points)
        logger.info(f"Waiting on {len(points)} individual fitness evaluations")
        results = evaluator.await_evals(points, timeout=timeout_minutes * 60)

        for ind, (x, fit) in zip(individuals, results):
            ind.fitness.values = (-fit,)  # ! "-" maximizing


if __name__ == "__main__":
    args = GA.parse_args()
    search = GA(**vars(args))
    # signal.signal(signal.SIGINT, on_exit)
    # signal.signal(signal.SIGTERM, on_exit)
    search.run()
