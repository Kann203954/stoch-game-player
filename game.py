import math
import random
from typing import Union, List, Tuple, Optional
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt


class SGame:

    def __init__(self,
                 payoff_matrices: List[np.ndarray],
                 transition_matrices: List[np.ndarray],
                 discount_factors: Union[np.ndarray, float, int],
                 ) -> None:
        """Inputs:

        payoff_matrices:      list of array-like, one for each state: payoff_matrices[s][p,A]

        transition_matrices:  list of array-like, one for each from_state: transition_matrices[s][A,s']

        discount_factors:     array-like: discount_factors[p]
                              or numeric (-> common discount factor)

        Different numbers of actions across states and players are allowed.
        Inputs should be of relevant dimension and should NOT contain nan.
        """

        # bring payoff_matrices to list of np.ndarray, one array for each state
        payoff_matrices = [np.array(payoff_matrices[s], dtype=np.float64) for s in range(len(payoff_matrices))]

        # read out game shape
        self.num_states = len(payoff_matrices)
        self.num_players = payoff_matrices[0].shape[0]

        self.nums_actions = np.zeros((self.num_states, self.num_players), dtype=np.int32)
        for s in range(self.num_states):
            for p in range(self.num_players):
                self.nums_actions[s, p] = payoff_matrices[s].shape[1 + p]

        self.num_actions_max = self.nums_actions.max()
        self.num_actions_total = self.nums_actions.sum()

        # generate array representing u [s,p,A]
        self.u = np.zeros((self.num_states, self.num_players, *[self.num_actions_max] * self.num_players))
        for s in range(self.num_states):
            for p in range(self.num_players):
                for A in np.ndindex(*self.nums_actions[s]):
                    self.u[(s, p) + A] = payoff_matrices[s][(p,) + A]

        # generate array representing discount factors [p]
        if isinstance(discount_factors, (list, tuple, np.ndarray)):
            self.delta = np.array(discount_factors, dtype=np.float64)
        else:
            self.delta = discount_factors * np.ones(self.num_players)

        # bring transition_matrices to list of np.ndarray, one array for each state
        transition_matrices = [np.array(transition_matrices[s], dtype=np.float64) for s in range(self.num_states)]

        # build big transition matrix [s,A,s'] from list of small transition matrices [A,s'] for each s
        transition_matrix = np.zeros((self.num_states, *[self.num_actions_max] * self.num_players, self.num_states))
        for s0 in range(self.num_states):
            for A in np.ndindex(*self.nums_actions[s0]):
                for s1 in range(self.num_states):
                    transition_matrix[(s0,) + A + (s1,)] = transition_matrices[s0][A + (s1,)]

        self.phi = transition_matrix

        digits = len(str(self.num_states))
        self.state_labels = [f's{s:0{digits}}' for s in range(self.num_states)]
        digits = len(str(self.num_players))
        self.player_labels = [f'p{s:0{digits}}' for s in range(self.num_players)]
        digits = len(str(self.num_actions_max))
        self.action_labels = [f'a{a:0{digits}}' for a in range(self.num_actions_max)]

        self.QL = np.zeros((self.num_players, self.num_states, self.num_actions_max), dtype=np.float64)
        self.QLS = [[[0 for a in range(self.nums_actions[s, p])] for s in range(self.num_states)] for p in
                    range(self.num_players)]

    @classmethod
    def random_game(cls, num_states, num_players, num_actions, delta=0.95, seed=None) -> 'SGame':
        """Creates an SGame of given size, with random payoff- and transition arrays.
        num_actions can be specified in the following ways:
        - integer: all agents have this same fixed number of actions
        - list/tuple of 2 integers: number of actions is randomized, the input determining [min, max]
        - array of dimension [num_states, num_actions]: number of actions for each agent
        Similarly for delta:
        - float: value used for all players
        - tuple/list of 2 floats: randomized per player with (delta_min, delta_max)
        - list or array of length num_players: values used for all players

        Passing a seed to the random number generator ensures that the game can be recreated at a
        later occasion or by other users.
        """
        rng = np.random.default_rng(seed=seed)

        # if num_actions passed as int -> fixed number for all agents:
        if isinstance(num_actions, (int, float)):
            num_actions = np.ones((num_states, num_players), dtype=int) * num_actions
        # if given as (min, max) -> randomize accordingly
        elif isinstance(num_actions, (list, tuple)) and np.array(num_actions).shape == (2,):
            num_actions = rng.integers(low=num_actions[0], high=num_actions[1],
                                       size=(num_states, num_players), endpoint=True)
        # else, assume it is an array that fully specifies the game size
        num_actions = np.array(num_actions, dtype=np.int32)

        u = []
        for s in range(num_states):
            a = rng.random((num_players, *num_actions[s, :]))
            u.append(a)

        phi = [rng.exponential(scale=1, size=(*num_actions[s, :], num_states)) for s in range(num_states)]
        for s in range(num_states):
            for index, value in np.ndenumerate(np.sum(phi[s], axis=-1)):
                phi[s][index] *= 1 / value

        if isinstance(delta, (int, float)):
            delta = np.ones(num_players) * delta
        elif isinstance(delta, (list, tuple)) and len(delta) == 2:
            delta = rng.uniform(delta[0], delta[1], size=num_players)

        return cls(u, phi, delta)

    @classmethod
    def random_identical_interest_game(cls, num_states, num_players, num_actions, delta=0.95, seed=None) -> 'SGame':
        """Creates an SGame of given size, with random payoff- and transition arrays.
        num_actions can be specified in the following ways:
        - integer: all agents have this same fixed number of actions
        - list/tuple of 2 integers: number of actions is randomized, the input determining [min, max]
        - array of dimension [num_states, num_actions]: number of actions for each agent
        Similarly for delta:
        - float: value used for all players
        - tuple/list of 2 floats: randomized per player with (delta_min, delta_max)
        - list or array of length num_players: values used for all players

        Passing a seed to the random number generator ensures that the game can be recreated at a
        later occasion or by other users.
        """
        rng = np.random.default_rng(seed=seed)

        # if num_actions passed as int -> fixed number for all agents:
        if isinstance(num_actions, (int, float)):
            num_actions = np.ones((num_states, num_players), dtype=int) * num_actions
        # if given as (min, max) -> randomize accordingly
        elif isinstance(num_actions, (list, tuple)) and np.array(num_actions).shape == (2,):
            num_actions = rng.integers(low=num_actions[0], high=num_actions[1],
                                       size=(num_states, num_players), endpoint=True)
        # else, assume it is an array that fully specifies the game size
        num_actions = np.array(num_actions, dtype=np.int32)

        u = []
        for s in range(num_states):
            a = rng.random((num_players, *num_actions[s, :]))
            for p in range(num_players):
                a[p] = a[0]
            u.append(a)

        phi = [rng.exponential(scale=1, size=(*num_actions[s, :], num_states)) for s in range(num_states)]
        for s in range(num_states):
            for index, value in np.ndenumerate(np.sum(phi[s], axis=-1)):
                phi[s][index] *= 1 / value

        if isinstance(delta, (int, float)):
            delta = np.ones(num_players) * delta
        elif isinstance(delta, (list, tuple)) and len(delta) == 2:
            delta = rng.uniform(delta[0], delta[1], size=num_players)

        return cls(u, phi, delta)

    @classmethod
    def random_zero_sum_game(cls, num_states, num_players, num_actions, delta=0.95, seed=None) -> 'SGame':
        """Only allow 2 players"""

        rng = np.random.default_rng(seed=seed)

        # if num_actions passed as int -> fixed number for all agents:
        if isinstance(num_actions, (int, float)):
            num_actions = np.ones((num_states, num_players), dtype=int) * num_actions
        # if given as (min, max) -> randomize accordingly
        elif isinstance(num_actions, (list, tuple)) and np.array(num_actions).shape == (2,):
            num_actions = rng.integers(low=num_actions[0], high=num_actions[1],
                                       size=(num_states, num_players), endpoint=True)
        # else, assume it is an array that fully specifies the game size
        num_actions = np.array(num_actions, dtype=np.int32)

        u = []
        for s in range(num_states):
            a = rng.random((num_players, *num_actions[s, :]))
            for p in range(num_players):
                a[p] = -a[0]
            u.append(a)

        phi = [rng.exponential(scale=1, size=(*num_actions[s, :], num_states)) for s in range(num_states)]
        for s in range(num_states):
            for index, value in np.ndenumerate(np.sum(phi[s], axis=-1)):
                phi[s][index] *= 1 / value

        if isinstance(delta, (int, float)):
            delta = np.ones(num_players) * delta
        elif isinstance(delta, (list, tuple)) and len(delta) == 2:
            delta = rng.uniform(delta[0], delta[1], size=num_players)

        return cls(u, phi, delta)

    def get_next_state(self, s: int, A: tuple):
        p = self.phi[(s,) + A]
        s1 = random.choices(range(self.num_states), p)
        return s1[0]

    # def select_action(self, eps: float, p, s):
    #     q_values = self.QL[p, s, 0:self.nums_actions[s, p]]
    #     nb_actions = q_values.shape[0]
    #
    #     if np.random.uniform() < eps:
    #         action = np.random.random_integers(0, nb_actions - 1)
    #     else:
    #         action = np.argmax(q_values)
    #
    #     return action

    def update_q(self, s, ns, A):
        for p in range(self.num_players):
            self.QL[p, s, A[p]] = self.u[(s, p) + A] + self.delta[p] * np.max(self.QL[p, ns, :])
            self.QLS[p][s] = self.QL[p, s, 0:self.nums_actions[s, p]]


def game_to_table(game: SGame) -> pd.DataFrame:
    """Convert SGame to a DataFrame in the tabular format."""
    state_labels = game.state_labels
    player_labels = game.player_labels
    action_labels = game.action_labels
    # if action-labels is given as single list for all agents, create a full nested version:
    if isinstance(action_labels[0], (str, int, float)):
        action_labels = [[action_labels[:game.nums_actions[s, i]]
                          for i in range(game.num_players)] for s in range(game.num_states)]

    # table header:
    a_cols = [f'a_{p}' for p in player_labels]
    u_cols = [f'u_{p}' for p in player_labels]
    phi_cols = [f'to_{s}' for s in state_labels]
    df = pd.DataFrame(columns=['state'] + a_cols + u_cols + phi_cols)
    # delta-row:
    df.loc[0] = ['delta'] + len(a_cols) * [np.nan] + game.delta.tolist() + len(phi_cols) * [np.nan]

    for s in range(game.num_states):
        for index, action_profile in zip(np.ndindex(*game.nums_actions[s]), itertools.product(*action_labels[s])):
            u = game.u[(s, slice(None)) + index].tolist()
            phi = game.phi[(s,) + index + (slice(None),)].tolist()
            df.loc[len(df)] = [state_labels[s]] + list(action_profile) + u + phi

    return df


def q_to_table(game: SGame, p: int) -> pd.DataFrame:
    state_labels = game.state_labels
    action_labels = game.action_labels

    # table header:
    a_cols = [f'{a}' for a in action_labels]
    s_rows = [f'{s}' for s in state_labels]

    df = pd.DataFrame(game.QLS[p], columns=a_cols[0:game.nums_actions[0][p]], index=s_rows)

    # player_labels = game.player_labels
    # title = [str(player_labels[p])]
    # for i in range(game.nums_actions[0][p] - 1):
    #     title.append('')
    # df.columns = pd.MultiIndex.from_tuples(
    #     zip(title, df.columns))

    return df


def learn_and_plot(game: SGame, episode=1000, start_state=0, policy=None, eps=0.1):
    if policy is None:
        policy = EpsGreedy()

    s = start_state
    path = [[[] for ss in range(game.num_states)] for p in range(game.num_players)]  # path[p][s] = [0,1,2,0,2,1,...]
    frq_path = [[[[] for a in range(game.nums_actions[s, p])] for ss in range(game.num_states)] for p in
                range(game.num_players)]  # frq_path[p][s][a] = [0.11,0.12,0.13,...]

    # loop for learning episode times
    for t in range(episode):
        A = []

        for p in range(game.num_players):
            A.append(policy.select_action(game, eps, p, s, t + 1, episode))
            path[p][s].append(A[p])
            for a in range(game.nums_actions[s, p]):
                frq_path[p][s][a].append(path[p][s].count(a) / len(path[p][s]))

        At = tuple(A)
        ns = game.get_next_state(s, At)
        game.update_q(s, ns, At)
        s = ns

    # plot
    for s in range(game.num_states):
        for p in range(game.num_players):
            for a in range(game.nums_actions[s, p]):
                r = range(len(frq_path[p][s][a]))
                r = [x + 1 for x in r]
                plt.subplot(game.num_states, game.num_players, game.num_players * s + p + 1)
                plt.plot(r, frq_path[p][s][a])
                plt.text(r[-1], frq_path[p][s][a][-1], f'a{a}')

                if s == 0:
                    plt.title(f'player{p}')
                if p == 0:
                    plt.ylabel(f'state{s}')
                if s == game.num_states - 1:
                    plt.xlabel('path length')


def print_q(game: SGame):
    for p in range(game.num_players):
        qdf = q_to_table(game, p=p)
        print('Q-values of player ' + str(game.player_labels[p]))
        print(qdf)
        print()


def print_game(game: SGame):
    df = game_to_table(game)
    pd.set_option('display.width', None)  # no limit width
    pd.set_option('display.max_rows', None)  # no limit rows
    print(df)
    print("")


class EpsGreedy:

    @staticmethod
    def select_action(game: SGame, eps: float, p, s, t, epi):
        q_values = game.QL[p, s, 0:game.nums_actions[s, p]]
        nb_actions = q_values.shape[0]
        # eps = 1/(t+1)

        if np.random.uniform() < eps:
            action = np.random.random_integers(0, nb_actions - 1)
        else:
            action = np.argmax(q_values)

        return action


class LogitBestResponse:

    @staticmethod
    def select_action(game: SGame, eps: float, p, s, t, epi):
        q_values = game.QL[p, s, 0:game.nums_actions[s, p]]
        nb_actions = q_values.shape[0]
        p = []
        e = 2.718281828459045
        x = 0
        for a in range(nb_actions):
            x = x + math.pow(e, (q_values[a] / eps))

        for a in range(nb_actions):
            p.append(math.pow(e, (q_values[a] / eps)) / x)

        action = random.choices(range(nb_actions), weights=p)

        return action[0]


class SmoothTimeResponse:

    @staticmethod
    def select_action(game: SGame, eps: float, p, s, t, epi):
        q_values = game.QL[p, s, 0:game.nums_actions[s, p]]
        nb_actions = q_values.shape[0]
        p = []
        e = 2.718281828459045
        x = 0
        t = t*eps/epi
        for a in range(nb_actions):
            x = x + math.pow(e, (q_values[a] * t))

        for a in range(nb_actions):
            p.append(math.pow(e, (q_values[a] * t)) / x)

        action = random.choices(range(nb_actions), weights=p)

        return action[0]


class SmoothTimeRootResponse:

    @staticmethod
    def select_action(game: SGame, eps: float, p, s, t, epi):
        q_values = game.QL[p, s, 0:game.nums_actions[s, p]]
        nb_actions = q_values.shape[0]
        e = 2.718281828459045
        t = math.sqrt(t)*eps/math.sqrt(epi)
        p = []
        x = 0
        for a in range(nb_actions):
            x = x + math.pow(e, (q_values[a] * t))

        for a in range(nb_actions):
            p.append(math.pow(e, (q_values[a] * t)) / x)

        action = random.choices(range(nb_actions), weights=p)

        return action[0]
