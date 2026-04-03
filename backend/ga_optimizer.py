import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from cuda_module import gpu_rmse

class GAOptimizer:

    def __init__(self, X, y, pop_size=10, generations=5):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.generations = generations

    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            individual = {
                "n_estimators": np.random.randint(10, 100),
                "max_depth": np.random.randint(3, 15)
            }
            population.append(individual)
        return population

    def fitness(self, individual):
        model = RandomForestRegressor(
            n_estimators=individual["n_estimators"],
            max_depth=individual["max_depth"]
        )
        model.fit(self.X, self.y)
        preds = model.predict(self.X)

        # GPU RMSE
        return gpu_rmse(self.y, preds)

    def selection(self, population, fitnesses):
        idx = np.argsort(fitnesses)
        return [population[i] for i in idx[:2]]

    def crossover(self, parent1, parent2):
        return {
            "n_estimators": parent1["n_estimators"],
            "max_depth": parent2["max_depth"]
        }

    def mutation(self, individual):
        individual["n_estimators"] += np.random.randint(-5, 5)
        individual["max_depth"] += np.random.randint(-2, 2)
        return individual

    def run(self):
        population = self.initialize_population()

        for _ in range(self.generations):
            fitnesses = [self.fitness(ind) for ind in population]

            parents = self.selection(population, fitnesses)

            new_population = parents.copy()

            while len(new_population) < self.pop_size:
                child = self.crossover(parents[0], parents[1])
                child = self.mutation(child)
                new_population.append(child)

            population = new_population

        best = min(population, key=lambda ind: self.fitness(ind))
        return best