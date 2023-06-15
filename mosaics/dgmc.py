# Implementation of distributed version of MOSAiCS.
# Inspiration: https://doi.org/10.1016/j.csda.2008.10.025 (frankly could only read freely available parts due to lack of access and would appreciate a PDF)
# TODO: For now only implemented loky parallelization which is confined to one machine. If someone needs many-machine parallelization
# it should be possible to implement mpi4py-based object treating several RandomWalkEnsemble objects the same way RandomWalkEnsemble treats RandomWalk objects.
import numpy as np
from copy import deepcopy
from .random_walk import TrajectoryPoint, RandomWalk, default_minfunc_name
from loky import get_reusable_executor
from itertools import repeat
from sortedcontainers import SortedList
import random, os
from subprocess import run as terminal_run


class SubpopulationPropagationIntermediateResults:
    def __init__(self, rw):
        """
        Update/store intermediate data.
        More might be added as the code becomes more sophisticated.
        """
        self.nsteps = 0
        self.num_replicas = rw.num_replicas
        self.sum_minfunc = np.zeros((self.num_replicas,))
        self.sum_minfunc2 = np.zeros((self.num_replicas,))

    def update(self, rw):
        """
        Update information stored in the object.
        """
        self.nsteps += 1
        for rep_id, cur_tp in enumerate(rw.cur_tps):
            cur_minfunc = cur_tp.calculated_data[rw.min_function_name]
            self.sum_minfunc[rep_id] += cur_minfunc
            self.sum_minfunc2[rep_id] += cur_minfunc**2


def gen_subpopulation_propagation_result(
    init_tps,
    betas,
    random_rng_state,
    numpy_rng_state,
    num_global_steps=1,
    misc_random_walk_kwargs={},
    global_step_kwargs={},
    synchronization_signal_file=None,
    synchronization_check_frequency=None,
):
    # Create the random walk for propagation.
    rw = RandomWalk(init_tps=init_tps, betas=betas, **misc_random_walk_kwargs)
    # Initialize the two random number generators.
    if random_rng_state is not None:
        if isinstance(random_rng_state, int):
            random.seed(random_rng_state)
            np.random.seed(numpy_rng_state)
        else:
            random.setstate(random_rng_state)
            np.random.set_state(numpy_rng_state)
    intermediate_results = SubpopulationPropagationIntermediateResults(rw)

    check_other_finished = (synchronization_signal_file is not None) and (
        synchronization_check_frequency is not None
    )
    # Propagate.
    for step_id in range(num_global_steps):
        rw.global_random_change(**global_step_kwargs)
        intermediate_results.update(rw)
        if check_other_finished:
            if step_id % synchronization_check_frequency == 0:
                if os.path.isfile(synchronization_signal_file):
                    break
    if check_other_finished:
        terminal_run(["touch", synchronization_signal_file])
    return rw, intermediate_results, random.getstate(), np.random.get_state()


class RandomWalkEnsemble:
    def __init__(
        self,
        num_processes=1,
        num_subpopulations=1,
        num_internal_global_steps=1,
        betas=None,
        init_egcs=None,
        init_tps=None,
        min_function=None,
        min_function_name=default_minfunc_name,
        randomized_change_params={},
        global_step_params={},
        save_logs=False,
        saved_candidates_max_difference=None,
        num_saved_candidates=1,
        previous_saved_candidates=None,
        synchronization_signal_file=None,
        synchronization_check_frequency=None,
        subpopulation_propagation_seed=None,
        subpopulation_random_rng_states=None,
        subpopulation_numpy_rng_states=None,
        debug=False,
    ):
        # Simulation parameters.
        self.num_subpopulations = num_subpopulations
        self.betas = betas
        self.num_replicas = len(self.betas)
        self.num_internal_global_steps = num_internal_global_steps
        self.min_function = min_function
        self.min_function_name = min_function_name
        self.random_walk_kwargs = {
            "randomized_change_params": deepcopy(randomized_change_params),
            "debug": debug,
            "min_function": self.min_function,
            "min_function_name": self.min_function_name,
        }
        self.global_step_params = global_step_params
        self.subpopulation_indices_list = None
        self.num_internal_global_steps = num_internal_global_steps
        # Related to synchronization between different walkers.
        self.synchronization_signal_file = synchronization_signal_file
        self.synchronization_check_frequency = synchronization_check_frequency
        # Random number generator states used inside subpopulations.
        self.subpopulation_propagation_seed = subpopulation_propagation_seed
        self.subpopulation_random_rng_states = subpopulation_random_rng_states
        self.subpopulation_numpy_rng_states = subpopulation_numpy_rng_states
        if self.subpopulation_random_rng_states is None:
            self.subpopulation_random_rng_states = self.default_init_rng_states()
        if self.subpopulation_numpy_rng_states is None:
            self.subpopulation_numpy_rng_states = self.default_init_rng_states()
        # Parallelization.
        self.num_processes = num_processes
        # Initial conditions.
        if init_tps is None:
            assert init_egcs is not None
            init_tps = [TrajectoryPoint(egc=egc) for egc in init_egcs]
        self.current_trajectory_points = init_tps
        # Logs of relevant information.
        self.save_logs = save_logs
        if self.save_logs:
            self.propagation_logs = []
        else:
            self.propagation_logs = None

        # For storing most important temporary data.
        # Saving best candidate molecules.
        self.saved_candidates = previous_saved_candidates
        if self.saved_candidates is None:
            self.saved_candidates = SortedList()
        self.num_saved_candidates = num_saved_candidates
        self.saved_candidates_max_difference = saved_candidates_max_difference
        # Related to distribution of the optimized quantity.
        self.av_minfunc = np.empty((self.num_replicas,))
        self.av_minfunc2 = np.empty((self.num_replicas,))
        self.minfunc_stddev = np.empty((self.num_replicas,))

        # Correct random walk keyword arguments.
        self.random_walk_kwargs["num_saved_candidates"] = self.num_saved_candidates
        self.random_walk_kwargs[
            "saved_candidates_max_difference"
        ] = self.saved_candidates_max_difference
        # Make sure betas and initial conditions are not accidentally defined twice for the random walk.
        for del_key in ["betas", "init_tps", "init_egcs"]:
            if del_key in self.random_walk_kwargs:
                del self.random_walk_kwargs[del_key]

    def default_init_rng_states(self):
        if self.subpopulation_propagation_seed is None:
            return list(repeat(None, self.num_subpopulations))
        else:
            return [
                i + int(self.subpopulation_propagation_seed)
                for i in range(self.num_subpopulations)
            ]

    def divide_into_subpopulations(self, indices_list):
        shuffled_indices_list = deepcopy(indices_list)
        random.shuffle(shuffled_indices_list)
        subpopulation_indices = [[] for _ in range(self.num_subpopulations)]
        for shuffled_order, id in enumerate(shuffled_indices_list):
            subpopulation_indices[shuffled_order % self.num_subpopulations].append(id)
        return subpopulation_indices

    def generate_subpopulation_indices(self):
        """
        Randomly generate indices of replicas in different subpopulations.
        Both greedy and exploratin replicas are divided equally among replicas.
        """
        all_greedy_indices = []
        all_exploration_indices = []
        for replica_id, beta in enumerate(self.betas):
            if beta is None:
                all_greedy_indices.append(replica_id)
            else:
                all_exploration_indices.append(replica_id)
        subpopulation_greedy_indices_list = self.divide_into_subpopulations(
            all_greedy_indices
        )
        subpopulation_exploration_indices_list = self.divide_into_subpopulations(
            all_exploration_indices
        )
        self.subpopulation_indices_list = [
            np.array(greedy_indices + exploration_indices)
            for greedy_indices, exploration_indices in zip(
                subpopulation_greedy_indices_list,
                subpopulation_exploration_indices_list,
            )
        ]

    def update_saved_candidates(self, new_candidate):
        """
        Include Candidate object into saved_candidates list.
        """
        # TODO Has a lot in common with what appears in random_walk.py, could be combined in another object?
        if new_candidate in self.saved_candidates:
            return
        new_minfunc_val = new_candidate.func_val
        starting_num_candidates = len(self.saved_candidates)
        if (self.num_saved_candidates is not None) and (
            starting_num_candidates >= self.num_saved_candidates
        ):
            if new_minfunc_val > self.saved_candidates[-1].func_val:
                return
        if self.saved_candidates_max_difference is not None:
            if (
                new_minfunc_val - self.saved_candidates[-1]
                > self.saved_candidates_max_difference
            ):
                return
        self.saved_candidates.add(deepcopy(new_candidate))
        if (self.num_saved_candidates is not None) and (
            starting_num_candidates >= self.num_saved_candidates
        ):
            del self.saved_candidates[self.num_saved_candidates :]

    def update_temporary_data(
        self, subpopulation_indices, cur_random_walk, cur_intermediate_results
    ):
        # Update saved candidates list.
        for candidate in cur_random_walk.saved_candidates:
            self.update_saved_candidates(candidate)
        for internal_replica_id, true_replica_id in enumerate(subpopulation_indices):
            self.current_trajectory_points[true_replica_id] = cur_random_walk.cur_tps[
                internal_replica_id
            ]
            self.av_minfunc[true_replica_id] = (
                cur_intermediate_results.sum_minfunc[internal_replica_id]
                / cur_intermediate_results.nsteps
            )
            self.av_minfunc2[true_replica_id] = (
                cur_intermediate_results.sum_minfunc2[internal_replica_id]
                / cur_intermediate_results.nsteps
            )
            self.minfunc_stddev[true_replica_id] = (
                self.av_minfunc2[true_replica_id]
                - self.av_minfunc[true_replica_id] ** 2
            )

    def subpopulation_propagation_inputs(self):
        """
        Arguments used in loky's map.
        """
        all_init_tps = []
        all_betas = []
        for subpopulation_indices in self.subpopulation_indices_list:
            # The initial trajectory points and the beta values.
            cur_init_tps = []
            cur_betas = []
            for i in subpopulation_indices:
                cur_init_tps.append(deepcopy(self.current_trajectory_points[i]))
                cur_betas.append(self.betas[i])
            all_init_tps.append(cur_init_tps)
            all_betas.append(cur_betas)

        input_list = [
            all_init_tps,
            all_betas,
            self.subpopulation_random_rng_states,
            self.subpopulation_numpy_rng_states,
        ]
        for other_arg in [
            self.num_internal_global_steps,
            self.random_walk_kwargs,
            self.global_step_params,
            self.synchronization_signal_file,
            self.synchronization_signal_file,
        ]:
            input_list.append(repeat(other_arg, self.num_subpopulations))
        return input_list

    def propagate_subpopulations(self):
        executor = get_reusable_executor(max_workers=self.num_processes)
        if self.synchronization_signal_file is not None:
            terminal_run(["rm", "-f", self.synchronization_signal_file])
        propagation_results_iterator = executor.map(
            gen_subpopulation_propagation_result,
            *self.subpopulation_propagation_inputs()
        )
        if self.save_logs:
            subpopulation_propagation_latest_logs = []
        for subpopulation_index, propagation_results in enumerate(
            propagation_results_iterator
        ):
            random_walk = propagation_results[0]
            intermediate_results = propagation_results[1]
            subpopulation_indices = self.subpopulation_indices_list[subpopulation_index]
            if self.save_logs:
                # Save subpopulation indices, RandomWalk object, and
                subpopulation_propagation_latest_logs.append(
                    (subpopulation_indices, random_walk, intermediate_results)
                )
            self.update_temporary_data(
                subpopulation_indices, random_walk, intermediate_results
            )
            self.subpopulation_random_rng_states[
                subpopulation_index
            ] = propagation_results[2]
            self.subpopulation_numpy_rng_states[
                subpopulation_index
            ] = propagation_results[3]
        if self.save_logs:
            self.subpopulation_propagation_logs.append(
                subpopulation_propagation_latest_logs
            )

    def propagate(self):
        self.generate_subpopulation_indices()
        self.propagate_subpopulations()
