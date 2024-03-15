from monopoly_simulator.gameplay import play_game
import numpy as np

if __name__ == "__main__":
    # play_game('/', seed_val=4)
    winner_arr = []
    times = []
    for i in range(0, 5000):
        winner, time = play_game('feasability_experiment_1/', seed_val=i)
        winner_arr.append(winner)
        times.append(time)

    np.save('feasability_experiment_1/times.npy', times)
    np.save('feasability_experiment_1/winner.npy', winner_arr)