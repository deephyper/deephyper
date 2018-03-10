def main():
    resultsList = []
    eval_counter = 0
    start_time = time.time()
    pop = toolbox.population(n=50)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40
    for x in pop:
        task = {}
        task['x'] = space_enc.decode_point(x)
        print(task['x'])
    # Evaluate the entire population
    #fitnesses = map(toolbox.evaluate_ga, pop)
    fitnesses = []
    for x in pop:
        task = {}
        task['x'] = space_enc.decode_point(x)
        task['eval_counter'] = eval_counter
        task['start_time'] = float(time.time() - start_time)
        fitness = toolbox.evaluate_ga(task['x'], task['eval_counter'])
        task['end_time'] = float(time.time() - start_time)
        eval_counter = eval_counter + 1
        fitnesses.append(fitness)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = map(toolbox.clone, offspring)

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = []
        for x in invalid_ind:
            task = {}
            task['x'] = space_enc.decode_point(x)
            task['eval_counter'] = eval_counter
            task['start_time'] = float(time.time() - start_time)
            fitness = toolbox.evaluate_ga(task['x'], task['eval_counter'])
            task['end_time'] = float(time.time() - start_time)
            eval_counter = eval_counter + 1
            fitnesses.append(fitness)
            resultsList.append(task)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            print(ind.fitness.values)

        # The population is entirely replaced by the offspring
        pop[:] = offspring
    saveResults(resultsList, results_json_fname, results_csv_fname)
    return pop

if __name__ == '__main__':
    main()
