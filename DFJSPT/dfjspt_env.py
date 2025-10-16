from typing import List

# from memory_profiler import profile
# @profile
# def func():
#     print("2")

from gymnasium.spaces import Box, Discrete, Dict
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import matplotlib as mpl
import networkx as nx
import pandas as pd
import random
import cv2
from DFJSPT import dfjspt_params
from DFJSPT.dfjspt_data.dfjspt_data_generator import generate_a_complete_instance
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from DFJSPT.dfjspt_rule.dfjspt_rule9_FDDMTWR_EET_EET import rule9_a_makespan

# from ray.rllib.utils.spaces.repeated import Repeated

# Constraints on the Repeated space.
MAX_JOBS = dfjspt_params.max_n_jobs
MAX_MACHINES = dfjspt_params.max_n_machines
MAX_TRANSBOTS = dfjspt_params.max_n_transbots
# ENV_COUNT = 0

# A Multi-agent Environment for Dynamic Flexible Job-shop Scheduling Problem with Transportation
class DfjsptMaEnv(MultiAgentEnv):

    def __init__(self, env_config):
        super().__init__()

        # ------ render parameters ------ #
        self.width = 15
        self.height = 10
        self.dpi = 70
        self.dummy_task_color = "tab:gray"
        self.transbot_node_color = "#8C7AB6"
        self.c_map1 = "rainbow"
        self.c_map2 = "Greys"
        self.c_map3 = "rainbow"
        self.color_scheduled = "#DAF7A6"
        self.color_not_scheduled = "#FFC300"
        self.color_job_edge = "tab:gray"
        self.node_drawing_kwargs = {
            "node_size": 800,
            "linewidths": 5
        }
        self.edge_drawing_kwargs = {
            "arrowsize": 30
        }
        self.critical_path_drawing_kwargs = {
            "edge_color": 'r',
            "width": 20,
            "alpha": 0.1,
        }
        self.drawMachineToPrcsEdges = False if dfjspt_params.drawMachineToPrcsEdges is None else dfjspt_params.drawMachineToPrcsEdges

        # # generate colors for machines
        # c_map1 = plt.cm.get_cmap(self.c_map1)  # select the desired cmap
        # arr1 = np.linspace(0, 1, n_machines,
        #                    dtype=float)  # create a list with numbers from 0 to 1 with n items
        # self.machine_colors = {m_id: c_map1(val) for m_id, val in enumerate(arr1)}
        # # generate colors for transbots
        # c_map2 = plt.cm.get_cmap(self.c_map2)  # select the desired cmap
        # arr2 = np.linspace(0.1, 0.5, n_transbots,
        #                    dtype=float)  # create a list with numbers from 0 to 1 with n items
        # self.transbot_colors = {t_id: c_map2(val) for t_id, val in enumerate(arr2)}
        # # concatenate 2 color dicts
        # self.machine_transbot_colors = {**self.machine_colors, **self.transbot_colors}
        #
        # # generate colors for jobs
        # c_map3 = plt.cm.get_cmap(self.c_map3)  # select the desired cmap
        # arr3 = np.linspace(0, 1, n_jobs,
        #                    dtype=float)  # create a list with numbers from 0 to 1 with n items
        # self.job_colors = {j_id: c_map3(val) for j_id, val in enumerate(arr3)}

        if dfjspt_params.default_visualisations is None:
            self.default_visualisations = ["gantt_window"]
        else:
            self.default_visualisations = dfjspt_params.default_visualisations
        # y axis of Gantt chart ("Jobs" or "Machines & Transbots")
        self.y_axis = dfjspt_params.gantt_y_axis

        # ------ instance size parameters ------ #
        self.n_jobs = None
        self.n_machines = None
        self.n_transbots = None
        self.max_n_operations = None
        self.min_n_operations = None
        self.n_total_tasks = None
        self.n_processing_tasks = None
        self.n_total_nodes = None
        self.n_job_features = 8
        self.n_machine_features = 7
        self.n_transbot_features = 7
        makespan_upper_bound = 10 * MAX_JOBS * MAX_MACHINES * (dfjspt_params.max_prcs_time + dfjspt_params.max_tspt_time)

        if env_config["train_or_eval_or_test"] == "train":
            self.start_of_instance_pool = 0
            self.end_of_instance_pool = dfjspt_params.n_instances_for_training
        elif env_config["train_or_eval_or_test"] == "eval":
            self.start_of_instance_pool = dfjspt_params.n_instances_for_training
            self.end_of_instance_pool = dfjspt_params.n_instances_for_training + dfjspt_params.n_instances_for_evaluation
        elif env_config["train_or_eval_or_test"] == "test":
            self.start_of_instance_pool = dfjspt_params.n_instances - dfjspt_params.n_instances_for_testing
            self.end_of_instance_pool = dfjspt_params.n_instances
        else:
            self.start_of_instance_pool = 0
            self.end_of_instance_pool = dfjspt_params.n_instances

        # self.my_instance_pool = env_config["global_instance_pool"]
        # if "rule_makespan_results" in env_config:
        #     self.rule_makespan_results = env_config["rule_makespan_results"]
        #     # self.rule_makespan_results = env_config["rule_makespan_results"]
        # else:
        #     self.rule_makespan_results = None
        del env_config

        self.Graph = None
        # self.instance = None
        self.source_task = -1
        # self.sink_task = self.n_total_tasks - 1
        self.sink_task = None

        self.reward_this_step = None
        self.stage = None
        self.chosen_job = -1
        self.chosen_machine = -1
        self.chosen_transbot = -1
        self.operation_id = -1
        self.tspt_task_id = -1
        self.prcs_task_id = -1
        self.perform_left_shift_if_possible = dfjspt_params.perform_left_shift_if_possible
        self.schedule_done = False

        self.job_features = None
        self.machine_features = None
        self.transbot_features = None
        self.job_action_mask = None
        self.machine_action_mask = None
        self.transbot_action_mask = None
        self.env_current_time = 0
        self.final_makespan = 0
        self.drl_minus_rule = 0
        self.prev_cmax = None
        self.curr_cmax = None
        self.result_start_time_for_jobs = None
        self.result_finish_time_for_jobs = None
        self.current_instance_id = 0
        self.instance_count = 0

        # Hierarchical MARL: Multi-objective reward tracking
        self.current_preference_vector = np.array([1/3, 1/3, 1/3])  # Default: equal weights
        self.multi_obj_reward = np.array([0.0, 0.0, 0.0])  # [efficiency, cost, delivery]
        self.total_energy_cost = 0.0
        self.total_transport_cost = 0.0
        self.total_wear_cost = 0.0
        self.total_tardiness = 0.0
        self.job_due_times = None  # Will be set in reset
        
        # EMA normalization tracking
        self.reward_ema_mean = np.array([0.0, 0.0, 0.0])
        self.reward_ema_std = np.array([1.0, 1.0, 1.0])
        self.ema_initialized = False

        # 'routes' of the machines. indicates in which order a machine processes tasks
        self.machine_routes = None
        # 'routes' of the transbots. indicates in which order a machine processes tasks
        self.transbot_routes = None

        # agent0: operation selection agent
        # agent1: machine selection agent
        # agent2: transbot selection agent
        self.agents = {"agent0", "agent1", "agent2"}
        self._agent_ids = set(self.agents)
        self.terminateds = set()
        self.truncateds = set()

        # self.job_obs = Box(low=-1, high=makespan_upper_bound,
        #                    shape=(self.n_job_features,),
        #                    dtype=np.float64)
        #
        # self.machine_obs = Box(low=-1, high=makespan_upper_bound,
        #                        shape=(self.n_machine_features,),
        #                        dtype=np.float64)
        #
        # self.transbot_obs = Box(low=-1, high=makespan_upper_bound,
        #                        shape=(self.n_transbot_features,),
        #                        dtype=np.float64)

        self.observation_space = Dict({
            "agent0": Dict({
                "action_mask": Box(0, 1,
                                   shape=(MAX_JOBS,),
                                   dtype=np.int64),
                # "observation": Repeated(self.job_obs, max_len=MAX_JOBS),
                "observation": Box(-1, makespan_upper_bound,
                                   shape=(MAX_JOBS, self.n_job_features),
                                   dtype=np.float64),
            }),

            "agent1": Dict({
                "action_mask": Box(0, 1,
                                   shape=(MAX_MACHINES,),
                                   dtype=np.int64),
                # "observation": Repeated(self.machine_obs, max_len=MAX_MACHINES),
                "observation": Box(-1, makespan_upper_bound,
                                   shape=(MAX_MACHINES, self.n_machine_features),
                                   dtype=np.float64),
            }),

            "agent2": Dict({
                "action_mask": Box(0, 1,
                                   shape=(MAX_TRANSBOTS,),
                                   dtype=np.int64),
                # "observation": Repeated(self.transbot_obs, max_len=MAX_TRANSBOTS),
                "observation": Box(-1, makespan_upper_bound,
                                   shape=(MAX_TRANSBOTS, self.n_transbot_features),
                                   dtype=np.float64),
            }),
        })

        self.action_space = Dict({
            "agent0": Discrete(MAX_JOBS),
            "agent1": Discrete(MAX_MACHINES),
            "agent2": Discrete(MAX_TRANSBOTS)
        })

        self.resetted = False

    def _get_info(self):
        if self.stage == 0:
            return {
                "agent0": {
                    "current_stage": self.stage
                },
            }
        elif self.stage == 1:
            return {
                "agent1": {
                    "current_stage": self.stage
                },
            }
        else:
            return {
                "agent2": {
                    "current_stage": self.stage
                },
            }

    def _compute_efficiency_reward(self):
        """
        Compute efficiency reward based on makespan improvement and machine utilization.
        Returns a positive reward for better efficiency.
        """
        if not dfjspt_params.use_multi_objective_reward:
            return 0.0
        
        # Makespan improvement (normalized by rule makespan)
        makespan_improvement = (self.prev_cmax - self.curr_cmax) / max(self.rule_makespan_for_current_instance, 1.0)
        
        # Machine utilization (if scheduling is done)
        if self.schedule_done:
            total_processing_time = np.sum(self.machine_features[:self.n_machines, 4])
            makespan = self.curr_cmax
            n_machines = self.n_machines
            utilization = total_processing_time / max(makespan * n_machines, 1.0)
            efficiency_reward = makespan_improvement + utilization * 0.1
        else:
            efficiency_reward = makespan_improvement
        
        return efficiency_reward * dfjspt_params.efficiency_weight
    
    def _compute_cost_reward(self):
        """
        Compute cost reward based on energy consumption, transportation, and machine wear.
        Returns a negative reward (cost is bad) that should be minimized.
        """
        if not dfjspt_params.use_multi_objective_reward:
            return 0.0
        
        # Calculate incremental costs since last step
        current_machine_time = np.sum(self.machine_features[:self.n_machines, 4])
        current_transport_time = np.sum(self.transbot_features[:self.n_transbots, 5])
        current_operations = np.sum(self.machine_features[:self.n_machines, 2])
        
        # Energy cost (machine running time)
        energy_cost = current_machine_time * dfjspt_params.energy_cost_per_unit_time
        
        # Transport cost
        transport_cost = current_transport_time * dfjspt_params.transport_cost_per_unit_time
        
        # Machine wear cost
        wear_cost = current_operations * dfjspt_params.machine_wear_cost
        
        total_cost = energy_cost + transport_cost + wear_cost
        
        # Normalize by a reasonable upper bound
        max_possible_cost = (
            self.n_jobs * self.max_n_operations * dfjspt_params.max_prcs_time * 
            dfjspt_params.energy_cost_per_unit_time * 2
        )
        
        # Return negative reward (lower cost is better)
        cost_reward = -total_cost / max(max_possible_cost, 1.0)
        
        return cost_reward * dfjspt_params.cost_weight
    
    def _compute_delivery_reward(self):
        """
        Compute delivery reward based on tardiness (lateness penalty).
        Returns a negative reward for late jobs.
        """
        if not dfjspt_params.use_multi_objective_reward:
            return 0.0
        
        tardiness = 0.0
        if self.schedule_done:
            for job_id in range(self.n_jobs):
                completion_time = self.result_finish_time_for_jobs[job_id, :, :].max()
                due_time = self.job_due_times[job_id]
                if completion_time > due_time:
                    tardiness += (completion_time - due_time)
            
            # Normalize tardiness
            max_tardiness = np.sum(self.job_due_times) * dfjspt_params.tardiness_penalty_factor
            delivery_reward = -tardiness / max(max_tardiness, 1.0)
        else:
            delivery_reward = 0.0
        
        return delivery_reward * dfjspt_params.delivery_weight
    
    def _normalize_rewards(self, raw_rewards):
        """
        Normalize multi-dimensional rewards using EMA-based standardization.
        Args:
            raw_rewards: numpy array of shape (3,) with [efficiency, cost, delivery] rewards
        Returns:
            normalized_rewards: numpy array of shape (3,)
        """
        if dfjspt_params.reward_normalization_method == "none":
            return raw_rewards
        
        elif dfjspt_params.reward_normalization_method == "ema":
            # Update EMA statistics
            if not self.ema_initialized:
                self.reward_ema_mean = raw_rewards.copy()
                self.reward_ema_std = np.ones(3)
                self.ema_initialized = True
            else:
                alpha = dfjspt_params.ema_alpha
                self.reward_ema_mean = (1 - alpha) * self.reward_ema_mean + alpha * raw_rewards
                squared_diff = (raw_rewards - self.reward_ema_mean) ** 2
                self.reward_ema_std = np.sqrt((1 - alpha) * self.reward_ema_std ** 2 + alpha * squared_diff)
            
            # Standardize
            normalized = (raw_rewards - self.reward_ema_mean) / (self.reward_ema_std + 1e-8)
            return normalized
        
        else:
            # Default: return raw rewards
            return raw_rewards
    
    def _compute_multi_objective_reward(self):
        """
        Compute all three reward components and return as numpy array.
        Returns:
            rewards: numpy array of shape (3,) with [efficiency, cost, delivery]
        """
        r_efficiency = self._compute_efficiency_reward()
        r_cost = self._compute_cost_reward()
        r_delivery = self._compute_delivery_reward()
        
        return np.array([r_efficiency, r_cost, r_delivery])
    
    def _scalarize_reward(self, multi_reward):
        """
        Scalarize multi-objective reward using current preference vector.
        Args:
            multi_reward: numpy array of shape (3,)
        Returns:
            scalar_reward: float
        """
        return np.dot(self.current_preference_vector, multi_reward)

    def _get_obs(self):
        if self.stage == 0:
            return {
                "agent0": {
                    "action_mask": self.job_action_mask,
                    # "observation": [
                    #     self.job_features[job_id,:]
                    #     for job_id in range(self.n_jobs)
                    # ],
                    "observation": self.job_features,
                },
            }
        elif self.stage == 1:
            return {
                "agent1": {
                    "action_mask": self.machine_action_mask,
                    # "observation": [
                    #     self.machine_features[machine_id,:]
                    #     for machine_id in range(self.n_machines)
                    # ],
                    "observation": self.machine_features,
                },
            }
        else:
            return {
                "agent2": {
                    "action_mask": self.transbot_action_mask,
                    # "observation": [
                    #     self.transbot_features[transbot_id,:]
                    #     for transbot_id in range(self.n_transbots)
                    # ],
                    "observation": self.transbot_features,
                }
            }

    def get_strategy_obs(self):
        """
        Get global observation for strategy layer.
        
        Returns aggregated statistics from jobs, machines, and transbots
        to help strategy layer decide preference vector.
        """
        # Job statistics
        n_jobs_total = self.n_jobs
        n_jobs_completed = np.sum(self.job_features[:self.n_jobs, 4])
        n_jobs_remaining = n_jobs_total - n_jobs_completed
        avg_job_progress = np.mean(self.job_features[:self.n_jobs, 1] / 
                                   np.maximum(self.n_operations_for_jobs, 1))
        total_remaining_ops = np.sum(self.n_operations_for_jobs - self.job_features[:self.n_jobs, 1])
        avg_remaining_processing_time = np.mean(self.job_features[:self.n_jobs, 7])
        
        # Machine statistics
        n_machines = self.n_machines
        avg_machine_utilization = np.mean(self.machine_features[:self.n_machines, 2]) / max(n_jobs_total, 1)
        avg_machine_busy_time = np.mean(self.machine_features[:self.n_machines, 4])
        max_machine_busy_time = np.max(self.machine_features[:self.n_machines, 4])
        
        # Transbot statistics
        n_transbots = self.n_transbots
        avg_transbot_utilization = np.mean(self.transbot_features[:self.n_transbots, 2]) / max(n_jobs_total, 1)
        avg_transbot_busy_time = np.mean(self.transbot_features[:self.n_transbots, 5])
        
        # Time and makespan statistics
        current_makespan = self.curr_cmax
        normalized_makespan = current_makespan / max(self.rule_makespan_for_current_instance, 1.0)
        
        # Multi-objective reward components (if available)
        if hasattr(self, 'multi_obj_reward'):
            r_efficiency = self.multi_obj_reward[0]
            r_cost = self.multi_obj_reward[1]
            r_delivery = self.multi_obj_reward[2]
        else:
            r_efficiency = 0.0
            r_cost = 0.0
            r_delivery = 0.0
        
        # Aggregate all features into a single vector
        strategy_obs = np.array([
            # Job features (6)
            n_jobs_remaining / max(n_jobs_total, 1),
            avg_job_progress,
            total_remaining_ops / max(n_jobs_total * self.max_n_operations, 1),
            avg_remaining_processing_time / max(self.rule_makespan_for_current_instance, 1),
            n_jobs_completed / max(n_jobs_total, 1),
            n_jobs_total / MAX_JOBS,
            
            # Machine features (5)
            avg_machine_utilization,
            avg_machine_busy_time / max(current_makespan, 1),
            max_machine_busy_time / max(current_makespan, 1),
            n_machines / MAX_MACHINES,
            np.std(self.machine_features[:self.n_machines, 4]) / max(current_makespan, 1),
            
            # Transbot features (3)
            avg_transbot_utilization,
            avg_transbot_busy_time / max(current_makespan, 1),
            n_transbots / MAX_TRANSBOTS,
            
            # Makespan features (2)
            normalized_makespan,
            (current_makespan - self.prev_cmax) / max(self.rule_makespan_for_current_instance, 1),
            
            # Multi-objective reward components (3)
            r_efficiency,
            r_cost,
            r_delivery,
        ], dtype=np.float32)
        
        return strategy_obs

    def reset(self, seed=None, options=None):
        self.resetted = True
        self.terminateds = set()
        self.truncateds = set()

        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # self.instance = random.choice(self.my_instance_pool)
        if options is not None and "instance_id" in options:
            self.current_instance_id = options["instance_id"]
        else:

            if dfjspt_params.randomly_select_instance:
                self.current_instance_id = np.random.randint(low=0,
                                                             high=self.end_of_instance_pool - self.start_of_instance_pool)
            else:
                # if options is None or "instance_id" not in options:
                #     pass
                # else:
                #     self.instance_count = options["instance_id"]
                self.current_instance_id = self.instance_count
                    # self.current_instance_id += 1
                # if self.current_instance_id >= self.end_of_instance_pool - self.start_of_instance_pool:
                #     self.current_instance_id = 0
        # print(f"current_instance_id = {self.current_instance_id}")
        # instance = self.my_instance_pool[self.current_instance_id]
        if options is not None and "n_jobs" in options:
            generate_n_jobs = options["n_jobs"]
        else:
            generate_n_jobs = dfjspt_params.n_jobs
        instance = generate_a_complete_instance(
            seed=dfjspt_params.instance_generator_seed + self.current_instance_id + self.start_of_instance_pool,
            n_jobs=generate_n_jobs,
            n_machines=dfjspt_params.n_machines,
            n_transbots=dfjspt_params.n_transbots,
        )
        self.n_operations_for_jobs = np.array(instance[0])
        self.job_arrival_time = np.array(instance[1])
        self.processing_time_matrix = np.array(instance[2])
        self.empty_moving_time_matrix = np.array(instance[3])
        self.transport_time_matrix = np.array(instance[4])
        self.machine_quality = np.array(instance[5])
        self.transbot_quality = np.array(instance[6])

        # self.instance[0]: n_operations_for_jobs
        # self.instance[1]: processing_time_baseline
        # self.instance[2]: processing_time_matrix
        # self.instance[3]: empty_moving_time_matrix
        # self.instance[4]: transport_time_matrix
        # self.instance[5]: machine_quality
        # self.instance[6]: transbot_quality

        self.env_current_time = 0
        self.n_jobs = len(self.n_operations_for_jobs)
        self.max_n_operations = np.max(self.n_operations_for_jobs)
        self.min_n_operations = np.min(self.n_operations_for_jobs)
        self.n_machines = len(self.machine_quality[0])
        self.n_transbots = len(self.transbot_quality[0])
        self.n_total_tasks = np.sum(self.n_operations_for_jobs) + 2
        self.n_total_nodes = self.n_total_tasks + self.n_machines + self.n_transbots
        self.sink_task = self.n_total_tasks - 2

        # self.min_max_total_time_for_a_job = self.n_machines * (
        #         dfjspt_params.max_prcs_time + 2 * dfjspt_params.max_tspt_time) / 2
        

        # reset start_time and finish_time.
        self.result_start_time_for_jobs = np.zeros(shape=(self.n_jobs, self.max_n_operations, 2), dtype=float)
        self.result_finish_time_for_jobs = np.zeros(shape=(self.n_jobs, self.max_n_operations, 2), dtype=float)
        self.mean_processing_time_of_operations = np.zeros(shape=(self.n_jobs, self.max_n_operations), dtype=float)
        for job_id in range(self.n_jobs):
            for operation_id in range(self.n_operations_for_jobs[job_id]):
                processing_time_of_operations = self.processing_time_matrix[job_id][operation_id][
                    np.where(self.processing_time_matrix[job_id][operation_id] > 0)]
                # self.machine_mask[job_id][operation_id] = \
                #     [0 if value <= 0 else 1 for value in self.processing_time_matrix[job_id][operation_id]]
                self.mean_processing_time_of_operations[job_id][operation_id] = np.mean(processing_time_of_operations)
        self.mean_cumulative_processing_time_of_jobs = np.cumsum(self.mean_processing_time_of_operations, axis=1)
        # self.rule_makespan_for_current_instance = np.max(self.mean_cumulative_processing_time_of_jobs)

        # if self.rule_makespan_results:
        #     self.rule_makespan_for_current_instance = self.rule_makespan_results[self.current_instance_id]
        # else:
        #     self.rule_makespan_for_current_instance = np.max(self.mean_cumulative_processing_time_of_jobs)
        self.rule_makespan_for_current_instance = rule9_a_makespan(instance)
        del instance

        # generate colors for machines
        c_map1 = plt.cm.get_cmap(self.c_map1)  # select the desired cmap
        arr1 = np.linspace(0, 1, self.n_machines,
                           dtype=float)  # create a list with numbers from 0 to 1 with n items
        self.machine_colors = {m_id: c_map1(val) for m_id, val in enumerate(arr1)}
        # generate colors for transbots
        c_map2 = plt.cm.get_cmap(self.c_map2)  # select the desired cmap
        arr2 = np.linspace(0.1, 0.5, self.n_transbots,
                           dtype=float)  # create a list with numbers from 0 to 1 with n items
        self.transbot_colors = {t_id: c_map2(val) for t_id, val in enumerate(arr2)}
        # concatenate 2 color dicts
        self.machine_transbot_colors = {**self.machine_colors, **self.transbot_colors}

        # generate colors for jobs
        c_map3 = plt.cm.get_cmap(self.c_map3)  # select the desired cmap
        arr3 = np.linspace(0, 1, self.n_jobs,
                           dtype=float)  # create a list with numbers from 0 to 1 with n items
        self.job_colors = {j_id: c_map3(val) for j_id, val in enumerate(arr3)}
        self.initialize_disjunctive_graph(self.n_operations_for_jobs)

        # reset machine routes dict and transbot routes dict
        self.machine_routes = {m_id: np.array([], dtype=int)
                               for m_id in range(self.n_machines)}
        self.transbot_routes = {t_id: np.array([], dtype=int)
                                for t_id in range(self.n_transbots)}

        # remove is_scheduled flags and is_valid flag.
        self.job_features = np.zeros((MAX_JOBS, self.n_job_features), dtype=float)
        for job_id in range(self.n_jobs):
            self.job_features[job_id, 0] = job_id
        self.job_features[:self.n_jobs, 2] = self.job_arrival_time
        self.job_features[:, 3] = self.n_machines
        self.job_features[self.n_jobs:, 4] = 1
        self.job_features[:self.n_jobs, 6] = self.mean_processing_time_of_operations[:, 0]
        self.job_features[:self.n_jobs, 7] = self.mean_cumulative_processing_time_of_jobs[:, -1]

        self.machine_features = np.zeros((MAX_MACHINES, self.n_machine_features), dtype=float)
        for machine_id in range(self.n_machines):
            self.machine_features[machine_id, 0] = machine_id
        self.machine_features[:self.n_machines, 1] = self.machine_quality
        self.machine_features[:self.n_machines, 5:] = -1

        self.transbot_features = np.zeros((MAX_TRANSBOTS, self.n_transbot_features), dtype=float)
        for transbot_id in range(self.n_transbots):
            self.transbot_features[transbot_id, 0] = transbot_id
        self.transbot_features[:self.n_transbots, 1] = self.transbot_quality
        self.transbot_features[:self.n_transbots, 4] = self.n_machines
        self.transbot_features[:self.n_transbots, 6] = -1

        # Initial Action mask
        self.job_action_mask = np.zeros((MAX_JOBS,), dtype=int)
        for job_id in range(self.n_jobs):
            if self.job_arrival_time[job_id] == 0:
                self.job_action_mask[job_id] = 1

        self.machine_action_mask = np.zeros((MAX_MACHINES,), dtype=int)
        self.machine_action_mask[:self.n_machines] = 1

        self.transbot_action_mask = np.zeros((MAX_TRANSBOTS,), dtype=int)
        self.transbot_action_mask[:self.n_transbots] = 1

        self.prev_cmax = 0
        self.curr_cmax = 0
        self.reward_this_step = 0.0
        self.schedule_done = False
        
        # Initialize multi-objective tracking
        self.multi_obj_reward = np.array([0.0, 0.0, 0.0])
        self.total_energy_cost = 0.0
        self.total_transport_cost = 0.0
        self.total_wear_cost = 0.0
        self.total_tardiness = 0.0
        
        # Calculate due times for jobs (based on mean processing time)
        self.job_due_times = np.zeros(self.n_jobs)
        for job_id in range(self.n_jobs):
            self.job_due_times[job_id] = (
                self.job_arrival_time[job_id] + 
                self.mean_cumulative_processing_time_of_jobs[job_id, -1] * 
                dfjspt_params.delivery_due_time_factor
            )
        
        self.stage = 0
        observations = self._get_obs()
        info = self._get_info()

        return observations, info
    
    def set_preference_vector(self, preference_vector):
        """
        Set preference vector for multi-objective reward scalarization.
        
        This allows external controllers (e.g., strategy meta-controller) to 
        set preference without being part of the RLlib agent framework.
        
        Args:
            preference_vector: numpy array of shape (3,) with weights for
                              [efficiency, cost, delivery]. Should sum to 1.0.
        """
        if not isinstance(preference_vector, np.ndarray):
            preference_vector = np.array(preference_vector, dtype=np.float32)
        
        # Normalize to ensure sum = 1.0
        preference_sum = np.sum(preference_vector)
        if preference_sum > 0:
            preference_vector = preference_vector / preference_sum
        else:
            # Fallback to balanced if invalid
            preference_vector = np.array([1/3, 1/3, 1/3], dtype=np.float32)
        
        self.current_preference_vector = preference_vector

    def step(self, action):
        observations, reward, terminated, truncated, info = {}, {}, {}, {}, {}
        self.reward_this_step = 0.0

        if self.stage == 0:
            self.chosen_job = action["agent0"]

            # invalid job_id
            if self.chosen_job >= self.n_jobs or self.chosen_job < 0:
                self.schedule_done = min(self.job_features[:, 4]) >= 1
                if self.schedule_done:
                    self.final_makespan = self.curr_cmax
                    self.reward_this_step = self.reward_this_step + 1
                for i in range(3):
                    reward["agent{}".format(i)] = self.reward_this_step
                    terminated["agent{}".format(i)] = self.schedule_done
                    if terminated["agent{}".format(i)]:
                        self.terminateds.add("agent{}".format(i))
                    truncated["agent{}".format(i)] = False
                observations = self._get_obs()
                info = self._get_info()
                terminated["__all__"] = len(self.terminateds) == len(self.agents)
                truncated["__all__"] = False
                return observations, reward, terminated, truncated, info

            self.operation_id = int(self.job_features[self.chosen_job, 1])
            self.tspt_task_id = sum(self.n_operations_for_jobs[:self.chosen_job]) + self.operation_id
            self.prcs_task_id = self.tspt_task_id

            # invalid operation_id
            if self.operation_id >= self.n_operations_for_jobs[self.chosen_job] or self.operation_id < 0:
                self.schedule_done = min(self.job_features[:, 4]) >= 1
                if self.schedule_done:
                    self.final_makespan = self.curr_cmax
                    self.reward_this_step = self.reward_this_step + 1
                for i in range(3):
                    reward["agent{}".format(i)] = self.reward_this_step
                    terminated["agent{}".format(i)] = self.schedule_done
                    if terminated["agent{}".format(i)]:
                        self.terminateds.add("agent{}".format(i))
                    truncated["agent{}".format(i)] = False
                observations = self._get_obs()
                info = self._get_info()
                terminated["__all__"] = len(self.terminateds) == len(self.agents)
                truncated["__all__"] = False
                return observations, reward, terminated, truncated, info

            # update machines' state
            for machine_id in range(self.n_machines):
                # The expected time required for this machine to process the chosen job
                self.machine_features[machine_id, 5] = max(-1.0, self.processing_time_matrix[
                    self.chosen_job, self.operation_id, machine_id] / self.machine_features[machine_id, 1])
                # The expected time to transport the chosen job to this machine
                self.machine_features[machine_id, 6] = self.transport_time_matrix[
                    int(self.job_features[self.chosen_job, 3]),
                    int(self.machine_features[machine_id, 0])]
                if self.machine_features[machine_id, 5] > 0:
                    self.machine_action_mask[machine_id] = 1
                else:
                    self.machine_action_mask[machine_id] = 0

            self.stage = 1

        elif self.stage == 1:
            self.chosen_machine = action["agent1"]

            # invalid machine_id
            if self.chosen_machine >= self.n_machines or self.chosen_machine < 0 or self.machine_features[self.chosen_machine, 5] <= 0:
                self.schedule_done = min(self.job_features[:, 4]) >= 1
                if self.schedule_done:
                    self.final_makespan = self.curr_cmax
                    self.reward_this_step = self.reward_this_step + 1
                for i in range(3):
                    reward["agent{}".format(i)] = self.reward_this_step
                    terminated["agent{}".format(i)] = self.schedule_done
                    if terminated["agent{}".format(i)]:
                        self.terminateds.add("agent{}".format(i))
                    truncated["agent{}".format(i)] = False
                observations = self._get_obs()
                info = self._get_info()
                terminated["__all__"] = len(self.terminateds) == len(self.agents)
                truncated["__all__"] = False
                return observations, reward, terminated, truncated, info

            for transbot_id in range(self.n_transbots):
                self.transbot_features[transbot_id, 6] = max(-1.0,
                    (self.empty_moving_time_matrix[
                    int(self.transbot_features[transbot_id, 4]),
                    int(self.job_features[self.chosen_job, 3])]) / self.transbot_features[transbot_id, 1])

            self.stage = 2

        else:
            self.chosen_transbot = action["agent2"]

            tspt_result = self._schedule_tspt_task(
                job_id=self.chosen_job,
                task_id=self.tspt_task_id,
                transbot_id=self.chosen_transbot,
                machine_id=self.chosen_machine
            )

            prcs_result = self._schedule_prcs_task(
                job_id=self.chosen_job,
                task_id=self.prcs_task_id,
                machine_id=self.chosen_machine
            )

            # self.perform_left_shift_if_possible = True
            self.env_current_time = max(self.machine_features[:self.n_machines,3])
            for job_id in range(self.n_jobs):
                if (self.job_arrival_time[job_id] < self.env_current_time) and self.job_features[job_id, 1] == 0:
                    self.job_action_mask[job_id] = 1

            self.machine_features[:self.n_machines, 5:] = -1
            self.transbot_features[:self.n_transbots, 6] = -1

            self.prev_cmax = self.curr_cmax
            self.curr_cmax = self.result_finish_time_for_jobs.max()

            # Compute reward (original or multi-objective)
            if dfjspt_params.use_multi_objective_reward:
                # Compute multi-objective rewards
                self.multi_obj_reward = self._compute_multi_objective_reward()
                # Normalize rewards
                normalized_rewards = self._normalize_rewards(self.multi_obj_reward)
                # Scalarize using preference vector
                tactical_reward = self._scalarize_reward(normalized_rewards)
                self.reward_this_step = self.reward_this_step + tactical_reward
            else:
                # Original single-objective reward (makespan-based)
                self.reward_this_step = self.reward_this_step + 1.0 * (self.prev_cmax - self.curr_cmax) / self.rule_makespan_for_current_instance

            self.schedule_done = min(self.job_features[:, 4]) == 1
            if self.schedule_done:
                self.final_makespan = self.curr_cmax
                self.drl_minus_rule = self.final_makespan - self.rule_makespan_for_current_instance
                self.reward_this_step = self.reward_this_step + 1
                self.instance_count += 1
                if self.instance_count >= self.end_of_instance_pool - self.start_of_instance_pool:
                    self.instance_count = 0

            for i in range(3):
                reward["agent{}".format(i)] = self.reward_this_step
                terminated["agent{}".format(i)] = self.schedule_done
                if terminated["agent{}".format(i)]:
                    self.terminateds.add("agent{}".format(i))
                truncated["agent{}".format(i)] = False

            self.stage = 0

        observations = self._get_obs()
        info = self._get_info()
        terminated["__all__"] = len(self.terminateds) == len(self.agents)
        truncated["__all__"] = False

        return observations, reward, terminated, truncated, info

    def render(self, mode="human", show: List[str] = None):
        df = None
        colors = None

        if mode not in ["human", "rgb_array", "console"]:
            raise ValueError(f"mode '{mode}' is not defined. allowed modes are: 'human' and 'rgb_array'.")

        if show is None:
            if mode == "rgb_array":
                show = [s for s in self.default_visualisations if "window" in s]
            elif mode == "console":
                show = [s for s in self.default_visualisations if "console" in s]
            else:
                show = self.default_visualisations

        if "gantt_window" in show:
            df = self.network_as_dataframe()
            if self.y_axis == "Job":
                colors1 = {f"Machine {m_id + 1}": (r, g, b)
                           for m_id, (r, g, b, a) in self.machine_colors.items()
                           }
                colors2 = {f"Transbot {m_id + 1}": (r, g, b)
                           for m_id, (r, g, b, a) in self.transbot_colors.items()
                           }
                colors = dict(colors1, **colors2)
            else:
                colors = {f"Job {j_id + 1}": (r, g, b)
                          for j_id, (r, g, b, a) in self.job_colors.items()
                          }
            if mode == "human":
                self.render_gantt_in_window(df=df, colors=colors)
            elif mode == "rgb_array":
                return self.gantt_chart_rgb_array(df=df, colors=colors)

    # *********************************
    # Schedule An Transportation Task
    # *********************************
    def _schedule_tspt_task(self, job_id: int, task_id: int, transbot_id: int, machine_id: int) -> dict:
        """
        schedules a transport task/node in the graph representation if the task can be scheduled.

        This adding one or multiple corresponding edges (multiple when performing a left shift) and updating the
        information stored in the nodes.

        :param task_id:     the transport task or node that shall be scheduled.
        :return:            a dict with additional information for the `DisjunctiveGraphJssEnv.step`-method.
        """

        if self.Graph.nodes[task_id]["node_type"] != "operation" or \
                job_id != self.Graph.nodes[task_id]["job_id"] or \
                self.Graph.nodes[task_id]["operation_id"] != self.job_features[job_id][1]:
            return {
                "schedule_success": False,
            }

        if machine_id < 0 or machine_id >= self.n_machines or transbot_id < 0 or transbot_id >= self.n_transbots:
            return {
                "schedule_success": False,
            }

        transbot_location = int(self.transbot_features[transbot_id, 4])
        job_location = int(self.job_features[job_id, 3])
        machine_location = int(self.machine_features[machine_id, 0])

        previous_operation_finish_time = float(self.job_features[job_id, 2])

        expected_duration = self.empty_moving_time_matrix[transbot_location, job_location] + \
                            self.transport_time_matrix[job_location, machine_location]
        if self.transbot_quality[0, transbot_id] > 0:
            if random.random() <= self.transbot_quality[0, transbot_id]:
                actual_duration = expected_duration
            else:
                actual_duration = random.uniform(expected_duration,
                                                 expected_duration / self.transbot_quality[0, transbot_id])
        else:
            actual_duration = 999

        len_transbot_routes = len(self.transbot_routes[transbot_id])
        if len_transbot_routes:
            if self.perform_left_shift_if_possible:
                j_lower_bound_st = previous_operation_finish_time
                j_lower_bound_ft = j_lower_bound_st + actual_duration

                # check if task can be scheduled between src and first task
                transbot_firt_task = self.transbot_routes[transbot_id][0]
                transbot_first_task_start_time = self.result_start_time_for_jobs[
                    self.task_job_operation_dict[transbot_firt_task][0],
                    self.task_job_operation_dict[transbot_firt_task][1], 0]

                if j_lower_bound_ft <= transbot_first_task_start_time:
                    self.transbot_routes[transbot_id] = np.insert(self.transbot_routes[transbot_id], 0, task_id)
                    start_time = previous_operation_finish_time
                    finish_time = start_time + actual_duration

                    self.result_start_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 0] = start_time
                    self.result_finish_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 0] = finish_time
                    self.Graph.nodes[task_id]["t_start_time"] = start_time
                    self.Graph.nodes[task_id]["t_finish_time"] = finish_time
                    self.Graph.nodes[task_id]["transbot_id"] = transbot_id
                    self.transbot_features[transbot_id][2] += 1
                    self.transbot_features[transbot_id][3] = finish_time
                    self.transbot_features[transbot_id][4] = machine_location
                    self.transbot_features[transbot_id][5] += actual_duration

                    self.job_features[job_id][2] = finish_time
                    self.job_features[job_id][3] = machine_location

                    return {
                        "schedule_success": True,
                        "actual_duration": actual_duration,
                        "start_time": start_time,
                        "finish_time": finish_time,
                    }

                elif len_transbot_routes == 1:
                    self.transbot_routes[transbot_id] = np.append(self.transbot_routes[transbot_id], task_id)
                    transbot_previous_task_finish_time = float(self.transbot_features[transbot_id, 3])
                    start_time = max(previous_operation_finish_time, transbot_previous_task_finish_time)
                    finish_time = start_time + actual_duration

                    self.result_start_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 0] = start_time
                    self.result_finish_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 0] = finish_time
                    self.Graph.nodes[task_id]["t_start_time"] = start_time
                    self.Graph.nodes[task_id]["t_finish_time"] = finish_time
                    self.Graph.nodes[task_id]["transbot_id"] = transbot_id
                    self.transbot_features[transbot_id][2] += 1
                    self.transbot_features[transbot_id][3] = finish_time
                    self.transbot_features[transbot_id][4] = machine_location
                    self.transbot_features[transbot_id][5] += actual_duration

                    self.job_features[job_id][2] = finish_time
                    self.job_features[job_id][3] = machine_location

                    return {
                        "schedule_success": True,
                        "actual_duration": actual_duration,
                        "start_time": start_time,
                        "finish_time": finish_time,
                    }

                # check if task can be scheduled between two tasks
                for i, (m_prev, m_next) in enumerate(
                        zip(self.transbot_routes[transbot_id], self.transbot_routes[transbot_id][1:])):
                    m_temp_prev_ft = self.result_finish_time_for_jobs[
                        self.task_job_operation_dict[m_prev][0],
                        self.task_job_operation_dict[m_prev][1], 0]
                    m_temp_next_st = self.result_start_time_for_jobs[
                        self.task_job_operation_dict[m_next][0],
                        self.task_job_operation_dict[m_next][1], 0]

                    if j_lower_bound_ft > m_temp_next_st:
                        continue

                    m_gap = m_temp_next_st - m_temp_prev_ft
                    if m_gap < actual_duration:
                        continue

                    # at this point the task can fit in between two already scheduled task
                    start_time = max(j_lower_bound_st, m_temp_prev_ft)
                    finish_time = start_time + actual_duration
                    # insert task at the corresponding place in the machine routes list
                    self.transbot_routes[transbot_id] = np.insert(self.transbot_routes[transbot_id], i + 1, task_id)

                    self.result_start_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 0] = start_time
                    self.result_finish_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 0] = finish_time
                    self.Graph.nodes[task_id]["t_start_time"] = start_time
                    self.Graph.nodes[task_id]["t_finish_time"] = finish_time
                    self.Graph.nodes[task_id]["transbot_id"] = transbot_id
                    self.transbot_features[transbot_id][2] += 1
                    self.transbot_features[transbot_id][3] = finish_time
                    self.transbot_features[transbot_id][4] = machine_location
                    self.transbot_features[transbot_id][5] += actual_duration

                    self.job_features[job_id][2] = finish_time
                    self.job_features[job_id][3] = machine_location

                    return {
                        "schedule_success": True,
                        "actual_duration": actual_duration,
                        "start_time": start_time,
                        "finish_time": finish_time,
                    }

                self.transbot_routes[transbot_id] = np.append(self.transbot_routes[transbot_id], task_id)
                transbot_previous_task_finish_time = float(self.transbot_features[transbot_id, 3])
                start_time = max(previous_operation_finish_time, transbot_previous_task_finish_time)
                finish_time = start_time + actual_duration

            else:
                self.transbot_routes[transbot_id] = np.append(self.transbot_routes[transbot_id], task_id)
                transbot_previous_task_finish_time = float(self.transbot_features[transbot_id, 3])
                start_time = max(previous_operation_finish_time, transbot_previous_task_finish_time)
                finish_time = start_time + actual_duration
        else:
            self.transbot_routes[transbot_id] = np.insert(self.transbot_routes[transbot_id], 0, task_id)
            start_time = previous_operation_finish_time
            finish_time = start_time + actual_duration

        # self.reward_this_step = self.reward_this_step + 0.1 * (max_tspt_time - actual_duration) / self.min_max_total_time_for_a_job
        # self.reward_this_step = self.reward_this_step + 0.1 * (previous_operation_finish_time - start_time) / self.min_max_total_time_for_a_job

        self.result_start_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 0] = start_time
        self.result_finish_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 0] = finish_time
        self.Graph.nodes[task_id]["t_start_time"] = start_time
        self.Graph.nodes[task_id]["t_finish_time"] = finish_time
        self.Graph.nodes[task_id]["transbot_id"] = transbot_id
        self.transbot_features[transbot_id][2] += 1
        self.transbot_features[transbot_id][3] = finish_time
        self.transbot_features[transbot_id][4] = machine_location
        self.transbot_features[transbot_id][5] += actual_duration

        self.job_features[job_id][2] = finish_time
        self.job_features[job_id][3] = machine_location

        return {
            "schedule_success": True,
            "actual_duration": actual_duration,
            "start_time": start_time,
            "finish_time": finish_time,
        }

    # *********************************
    # Schedule An Processing Task
    # *********************************
    def _schedule_prcs_task(self, job_id: int, task_id: int, machine_id: int) -> dict:
        """
        schedules a process task/node in the graph representation if the task can be scheduled.

        This adding one or multiple corresponding edges (multiple when performing a left shift) and updating the
        information stored in the nodes.

        :param task_id:     the process task or node that shall be scheduled.
        :param machine_id:     the process machine that shall be allocated.
        :return:            a dict with additional information for the `DisjunctiveGraphJssEnv.step`-method.
        """

        if self.Graph.nodes[task_id]["node_type"] != "operation" or \
                job_id != self.Graph.nodes[task_id]["job_id"] or \
                self.Graph.nodes[task_id]["operation_id"] != self.job_features[job_id][1]:
            return {
                "schedule_success": False,
            }

        if machine_id < 0 or machine_id >= self.n_machines:
            return {
                "schedule_success": False,
            }

        # previous_operation_finish_time = self.observations["job_attrs"][job_id][2] * self.min_max_total_time_for_a_job
        previous_operation_finish_time = float(self.job_features[job_id][2])

        expected_duration = self.processing_time_matrix[job_id, self.Graph.nodes[task_id]["operation_id"], machine_id]
        if expected_duration <= 0:
            return {
                "schedule_success": False,
            }
        if self.machine_quality[0, machine_id] > 0:
            if random.random() <= self.machine_quality[0, machine_id]:
                actual_duration = expected_duration
            else:
                actual_duration = random.uniform(expected_duration,
                                                 expected_duration / self.machine_quality[0, machine_id])
        else:
            actual_duration = 999

        len_machine_routes = len(self.machine_routes[machine_id])
        if len_machine_routes:
            if self.perform_left_shift_if_possible:
                j_lower_bound_st = previous_operation_finish_time
                j_lower_bound_ft = j_lower_bound_st + actual_duration

                # check if task can be scheduled between src and first task
                machine_firt_task = self.machine_routes[machine_id][0]
                machine_first_task_start_time = self.result_start_time_for_jobs[
                    self.task_job_operation_dict[machine_firt_task][0],
                    self.task_job_operation_dict[machine_firt_task][1], 1]

                if j_lower_bound_ft <= machine_first_task_start_time:
                    self.machine_routes[machine_id] = np.insert(self.machine_routes[machine_id], 0, task_id)
                    start_time = previous_operation_finish_time
                    finish_time = start_time + actual_duration

                    self.result_start_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 1] = start_time
                    self.result_finish_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 1] = finish_time
                    self.Graph.nodes[task_id]["m_start_time"] = start_time
                    self.Graph.nodes[task_id]["m_finish_time"] = finish_time
                    self.Graph.nodes[task_id]["machine_id"] = machine_id
                    self.machine_features[machine_id][2] += 1
                    self.machine_features[machine_id][3] = finish_time
                    self.machine_features[machine_id][4] += actual_duration

                    self.job_features[job_id][1] += 1
                    self.job_features[job_id][2] = finish_time
                    self.job_features[job_id][5] += actual_duration
                    self.job_features[job_id][7] -= self.job_features[job_id, 6]
                    if self.job_features[job_id][7] < 0:
                        self.job_features[job_id][7] = 0
                    if self.job_features[job_id][1] >= self.n_operations_for_jobs[job_id]:
                        self.job_features[job_id, 4] = 1
                        self.job_features[job_id, 6] = -1
                        self.job_action_mask[job_id] = 0
                    else:
                        self.job_features[job_id, 6] = self.mean_processing_time_of_operations[job_id][
                            int(self.job_features[job_id][1])]

                    return {
                        "schedule_success": True,
                        "actual_duration": actual_duration,
                        "start_time": start_time,
                        "finish_time": finish_time,
                    }

                elif len_machine_routes == 1:
                    self.machine_routes[machine_id] = np.append(self.machine_routes[machine_id], task_id)
                    machine_previous_task_finish_time = float(self.machine_features[machine_id, 3])
                    start_time = max(previous_operation_finish_time, machine_previous_task_finish_time)
                    finish_time = start_time + actual_duration

                    self.result_start_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 1] = start_time
                    self.result_finish_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 1] = finish_time
                    self.Graph.nodes[task_id]["start_time"] = start_time
                    self.Graph.nodes[task_id]["finish_time"] = finish_time
                    self.Graph.nodes[task_id]["machine_id"] = machine_id
                    self.machine_features[machine_id][2] += 1
                    self.machine_features[machine_id][3] = finish_time
                    self.machine_features[machine_id][4] += actual_duration

                    self.job_features[job_id][1] += 1
                    self.job_features[job_id][2] = finish_time
                    self.job_features[job_id][5] += actual_duration
                    self.job_features[job_id][7] -= self.job_features[job_id, 6]
                    if self.job_features[job_id][7] < 0:
                        self.job_features[job_id][7] = 0
                    if self.job_features[job_id][1] >= self.n_operations_for_jobs[job_id]:
                        self.job_features[job_id, 4] = 1
                        self.job_features[job_id, 6] = -1
                        self.job_action_mask[job_id] = 0
                    else:
                        self.job_features[job_id, 6] = self.mean_processing_time_of_operations[job_id][
                            int(self.job_features[job_id][1])]

                    return {
                        "schedule_success": True,
                        "actual_duration": actual_duration,
                        "start_time": start_time,
                        "finish_time": finish_time,
                    }

                # check if task can be scheduled between two tasks
                for i, (m_prev, m_next) in enumerate(
                        zip(self.machine_routes[machine_id], self.machine_routes[machine_id][1:])):
                    m_temp_prev_ft = self.result_finish_time_for_jobs[
                        self.task_job_operation_dict[m_prev][0],
                        self.task_job_operation_dict[m_prev][1], 1]
                    m_temp_next_st = self.result_start_time_for_jobs[
                        self.task_job_operation_dict[m_next][0],
                        self.task_job_operation_dict[m_next][1], 1]

                    if j_lower_bound_ft > m_temp_next_st:
                        continue

                    m_gap = m_temp_next_st - m_temp_prev_ft
                    if m_gap < actual_duration:
                        continue

                    # at this point the task can fit in between two already scheduled task
                    start_time = max(j_lower_bound_st, m_temp_prev_ft)
                    finish_time = start_time + actual_duration
                    # insert task at the corresponding place in the machine routes list
                    self.machine_routes[machine_id] = np.insert(self.machine_routes[machine_id], i + 1, task_id)

                    self.result_start_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 1] = start_time
                    self.result_finish_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 1] = finish_time
                    self.Graph.nodes[task_id]["m_start_time"] = start_time
                    self.Graph.nodes[task_id]["m_finish_time"] = finish_time
                    self.Graph.nodes[task_id]["machine_id"] = machine_id
                    self.machine_features[machine_id][2] += 1
                    self.machine_features[machine_id][3] = finish_time
                    self.machine_features[machine_id][4] += actual_duration

                    self.job_features[job_id][1] += 1
                    self.job_features[job_id][2] = finish_time
                    self.job_features[job_id][5] += actual_duration
                    self.job_features[job_id][7] -= self.job_features[job_id, 6]
                    if self.job_features[job_id][7] < 0:
                        self.job_features[job_id][7] = 0
                    if self.job_features[job_id][1] >= self.n_operations_for_jobs[job_id]:
                        self.job_features[job_id, 4] = 1
                        self.job_features[job_id, 6] = -1
                        self.job_action_mask[job_id] = 0
                    else:
                        self.job_features[job_id, 6] = self.mean_processing_time_of_operations[job_id][
                            int(self.job_features[job_id][1])]

                    return {
                        "schedule_success": True,
                        "actual_duration": actual_duration,
                        "start_time": start_time,
                        "finish_time": finish_time,
                    }

                self.machine_routes[machine_id] = np.append(self.machine_routes[machine_id], task_id)
                machine_previous_task_finish_time = float(self.machine_features[machine_id][3])
                start_time = max(previous_operation_finish_time, machine_previous_task_finish_time)
                finish_time = start_time + actual_duration

            else:
                self.machine_routes[machine_id] = np.append(self.machine_routes[machine_id], task_id)
                machine_previous_task_finish_time = float(self.machine_features[machine_id][3])
                start_time = max(previous_operation_finish_time, machine_previous_task_finish_time)
                finish_time = start_time + actual_duration
        else:
            self.machine_routes[machine_id] = np.insert(self.machine_routes[machine_id], 0, task_id)
            start_time = previous_operation_finish_time
            finish_time = start_time + actual_duration

        # self.reward_this_step = self.reward_this_step + 0.1 * (max_prcs_time - actual_duration) / self.min_max_total_time_for_a_job
        # self.reward_this_step = self.reward_this_step + 0.1 * (previous_operation_finish_time - start_time) / self.min_max_total_time_for_a_job

        self.result_start_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 1] = start_time
        self.result_finish_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 1] = finish_time
        self.Graph.nodes[task_id]["m_start_time"] = start_time
        self.Graph.nodes[task_id]["m_finish_time"] = finish_time
        self.Graph.nodes[task_id]["machine_id"] = machine_id
        self.machine_features[machine_id][2] += 1
        self.machine_features[machine_id][3] = finish_time
        self.machine_features[machine_id][4] += actual_duration

        self.job_features[job_id][1] += 1
        self.job_features[job_id][2] = finish_time
        self.job_features[job_id][5] += actual_duration
        self.job_features[job_id][7] -= self.job_features[job_id, 6]
        if self.job_features[job_id][7] < 0:
            self.job_features[job_id][7] = 0
        if self.job_features[job_id][1] >= self.n_operations_for_jobs[job_id]:
            self.job_features[job_id, 4] = 1
            self.job_features[job_id, 6] = -1
            self.job_action_mask[job_id] = 0
        else:
            self.job_features[job_id, 6] = self.mean_processing_time_of_operations[job_id][int(self.job_features[job_id][1])]

        return {
            "schedule_success": True,
            "actual_duration": actual_duration,
            "start_time": start_time,
            "finish_time": finish_time,
        }

    def initialize_disjunctive_graph(self,
                                     n_operations_for_jobs,
                                     # process_time_matrix,
                                     ) -> None:
        """
        Get a new disjunctive graph.
        """
        self.Graph = nx.DiGraph()

        # add nodes for processing machines
        for machine_id in range(self.n_machines):
            self.Graph.add_node(
                "machine" + str(machine_id),
                machine_id=machine_id,
                node_type="machine",
                shape='p',
                pos=(machine_id, -self.n_jobs - 2),
                color=self.machine_colors[machine_id],
            )

        # add nodes for transbots
        for transbot_id in range(self.n_transbots):
            self.Graph.add_node(
                "transbot" + str(transbot_id),
                transbot_id=transbot_id,
                node_type="transbot",
                shape='p',
                pos=(transbot_id, 0),
                color=self.transbot_colors[transbot_id],
            )

        # add src node
        self.Graph.add_node(
            self.source_task,
            node_type="dummy",
            shape='o',
            pos=(-2, int(-self.n_jobs * 0.5) - 1),
            color=self.dummy_task_color,
        )
        # add sink task at the end to avoid permutation in the adj matrix.
        self.Graph.add_node(
            self.sink_task,
            node_type="dummy",
            shape='o',
            pos=(2 * max(n_operations_for_jobs), int(-self.n_jobs * 0.5) - 1),
            color=self.dummy_task_color,
        )

        # add nodes for tasks
        task_id = -1
        self.task_job_operation_dict = {}
        for job_id in range(self.n_jobs):
            for operation_id in range(n_operations_for_jobs[job_id]):
                task_id += 1  # start from transportation task id 0, -1 is dummy starting task
                self.task_job_operation_dict[task_id] = [job_id, operation_id]
                # add a transportation task node
                self.Graph.add_node(
                    task_id,
                    node_type="operation",
                    job_id=job_id,
                    operation_id=operation_id,
                    shape='s',
                    pos=(2 * operation_id - 1, -job_id - 1),
                    color=self.job_colors[job_id],
                    # start_time=-1,
                    # finish_time=-1,
                    t_start_time=-1,
                    t_finish_time=-1,
                    transbot_id=-1,
                    m_start_time=-1,
                    m_finish_time=-1,
                    machine_id=-1,
                )
                if operation_id != 0:
                    # add a conjunctive edge from last processing (task_id - 1) to this transportation (task_id)
                    self.Graph.add_edge(
                        task_id - 1, task_id,
                        edge_type="conjunctive_arc",
                        weight=0,
                    )

                # task_id += 1
                # # add a processing task node
                # self.Graph.add_node(
                #     task_id,
                #     node_type="processing_operation",
                #     job_id=job_id,
                #     operation_id=operation_id + 1,
                #     shape='p',
                #     pos=(2 * operation_id, -job_id - 1),
                #     color=self.job_colors[job_id],
                #     start_time=0,
                #     finish_time=0,
                #     machine_id=0,
                # )
                # # add a conjunctive edge from last transportation (task_id - 1) to this processing (task_id)
                # self.Graph.add_edge(
                #     task_id - 1, task_id,
                #     edge_type="transportation_to_processing",
                #     weight=0,
                # )
                # for machine_node_id, machine_id in self.Graph.nodes(data="machine_id"):
                #     # if machine machine_id can process operation task_id:
                #     if process_time_matrix[job_id, operation_id, machine_id - 1] != 0:
                #         # add a conjunctive edge from machine (machine_node_id) to this processing (task_id)
                #         self.Graph.add_edge(
                #             machine_node_id, task_id,
                #             edge_type="machine_to_processing_operation",
                #             weight=process_time_matrix[job_id, operation_id, machine_id - 1],
                #         )
                # if operation_id == n_operations_for_jobs[job_id] - 1:
                #     # add edges from last processing tasks in job to sink task
                #     self.Graph.add_edge(
                #         task_id, self.sink_task,
                #         edge_type="prcs_to_sink",
                #         weight=0,
                #     )

    def network_as_dataframe(self) -> pd.DataFrame:
        """
        returns the current state of the environment in a format that is supported by Plotly gant charts.
        (https://plotly.com/python/gantt/)

        :return: the current state as pandas dataframe
        """
        if self.y_axis == "Job":
            dataframe1 = pd.DataFrame([
                {
                    'Task': f'Job {int(self.Graph.nodes[processing_task_id]["job_id"] + 1)}',
                    'Start': self.Graph.nodes[processing_task_id]["m_start_time"],
                    'Finish': self.Graph.nodes[processing_task_id]["m_finish_time"],
                    'Resource': f'Machine {int(self.Graph.nodes[processing_task_id]["machine_id"] + 1)}'
                }
                for processing_task_id in range(self.n_total_tasks - 2)
                if self.Graph.nodes[processing_task_id]["node_type"] == "operation"
            ])

            dataframe2 = pd.DataFrame([
                {
                    'Task': f'Job {int(self.Graph.nodes[transportation_task_id]["job_id"] + 1)}',
                    'Start': self.Graph.nodes[transportation_task_id]["t_start_time"],
                    'Finish': self.Graph.nodes[transportation_task_id]["t_finish_time"],
                    'Resource': f'Transbot {int(self.Graph.nodes[transportation_task_id]["transbot_id"] + 1)}'
                }
                for transportation_task_id in range(self.n_total_tasks - 2)
                if self.Graph.nodes[transportation_task_id]["node_type"] == "operation"
            ])
        else:
            dataframe1 = pd.DataFrame([
                {
                    'Task': f'Machine {int(self.Graph.nodes[processing_task_id]["machine_id"]) + 1}',
                    'Start': self.Graph.nodes[processing_task_id]["m_start_time"],
                    'Finish': self.Graph.nodes[processing_task_id]["m_finish_time"],
                    'Resource': f'Job {int(self.Graph.nodes[processing_task_id]["job_id"]) + 1}'
                }
                for processing_task_id in range(self.n_total_tasks - 2)
                if self.Graph.nodes[processing_task_id]["node_type"] == "operation"
            ])

            dataframe2 = pd.DataFrame([
                {
                    'Task': f'Transbot {int(self.Graph.nodes[transportation_task_id]["transbot_id"]) + 1}',
                    'Start': self.Graph.nodes[transportation_task_id]["t_start_time"],
                    'Finish': self.Graph.nodes[transportation_task_id]["t_finish_time"],
                    'Resource': f'Job {int(self.Graph.nodes[transportation_task_id]["job_id"]) + 1}'
                }
                for transportation_task_id in range(self.n_total_tasks - 2)
                if self.Graph.nodes[transportation_task_id]["node_type"] == "operation"
            ])

        dataframe = pd.concat([dataframe1, dataframe2], ignore_index=True)

        return dataframe

    @staticmethod
    def render_rgb_array(
            vis: np.ndarray,
            window_title: str = "Flexible Job Shop Scheduling",
            wait: int = 1
    ) -> None:
        """
        renders a rgb-array in an `cv2` window.
        the window will remain open for `:param wait:` ms or till the user presses any key.

        :param vis:             the rgb-array to render.
        :param window_title:    the title of the `cv2`-window
        :param wait:            time in ms that the `cv2`-window is open.
                                if `None`, then the window will remain open till a keyboard occurs.

        :return:
        """
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_title, vis)
        # https://stackoverflow.com/questions/64061721/opencv-to-close-the-window-on-a-specific-key
        k = cv2.waitKey(wait) & 0xFF
        if k == 27:  # wait for ESC key to exit
            cv2.destroyAllWindows()

    def gantt_chart_rgb_array(
            self,
            df: pd.DataFrame,
            colors: dict
    ) -> np.ndarray:
        """

        wrapper for `plotly` gantt chart function. turn a gantt chart into a rgb array.

        see: https://plotly.com/python/gantt/

        :param df:      dataframe according to `plotly` specification (https://plotly.com/python/gantt/).

        :param colors:  a dict that maps resources to color values. see example below.
                        Note: make sure that the key match the resources specified in `:param df:`

        :return:        a `plotly` gantt chart as rgb array.

        color example

            import numpy as np

            c_map = plt.cm.get_cmap("jet")  # select the desired cmap
            arr = np.linspace(0, 1, 10)  # create a list with numbers from 0 to 1 with n items (n = 10 here)
            colors = {resource: c_map(val) for resource, val in enumerate(arr)}
            <<pass `colors` as parameter>>

        """
        plt.figure(dpi=self.dpi)
        plt.axis("off")
        plt.tight_layout()

        fig = mpl.pyplot.gcf()
        fig.set_size_inches(self.width, self.height)

        # Gantt chart
        width, height = fig.canvas.get_width_height()
        if not len(df):
            if self.y_axis == "Job":
                df = pd.DataFrame([{"Task": "Job 1", "Start": 0, "Finish": 0, "Resource": "Transbot 1"}])
            else:
                df = pd.DataFrame([{"Task": "Transbot 1", "Start": 0, "Finish": 0, "Resource": "Job 1"}])
        fig = ff.create_gantt(df=df, show_colorbar=True, index_col='Resource', group_tasks=True, colors=colors)
        fig.update_layout(xaxis_type='linear')

        img_str = fig.to_image(format="jpg", width=width, height=height)

        nparr = np.frombuffer(img_str, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # clear current frame
        plt.clf()
        plt.close('all')
        return img

    def render_gantt_in_window(
            self,
            df: pd.DataFrame,
            colors: dict
    ) -> None:
        """
        wrapper for the `gantt_chart_rgb_array`- and `render_rgb_array`-methods

        :param df:              parameter for `gantt_chart_rgb_array`
        :param colors:          parameter for `gantt_chart_rgb_array`

        :return:                None
        """
        vis = self.gantt_chart_rgb_array(df=df, colors=colors)
        self.render_rgb_array(vis)








