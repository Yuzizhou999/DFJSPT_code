from gymnasium.spaces import Dict
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
from DFJSPT import dfjspt_params
from DFJSPT.dfjspt_generate_a_sample_batch import generate_sample_batch

torch, nn = try_import_torch()

# GNN encoder imports (conditional)
if dfjspt_params.use_gnn_encoder:
    try:
        from DFJSPT.dfjspt_gnn_encoder import WorkshopGNN, TORCH_GEOMETRIC_AVAILABLE
        from DFJSPT.gnn_obs_converter import observation_to_graph, batch_observations_to_graphs
        GNN_AVAILABLE = TORCH_GEOMETRIC_AVAILABLE
    except ImportError as e:
        print(f"Warning: Could not import GNN encoder: {e}")
        print("Falling back to MLP encoder.")
        GNN_AVAILABLE = False
else:
    GNN_AVAILABLE = False


class JobActionMaskModel(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel with optional GNN encoder."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observation" in orig_space.spaces
        )

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        # Choose encoder based on configuration
        if dfjspt_params.use_gnn_encoder and GNN_AVAILABLE:
            print(f"✅ {name}: Using GNN encoder")
            # GNN encoder
            node_feature_dims = {
                'job': orig_space["observation"].shape[1],  # Number of features per job
                'machine': 0,  # Not used in job agent
                'transbot': 0,  # Not used in job agent
                'operation': 0,  # Not used in job agent
            }
            
            self.gnn_encoder = WorkshopGNN(
                node_feature_dims=node_feature_dims,
                edge_feature_dim=4,
                hidden_dim=dfjspt_params.gnn_hidden_dim,
                num_layers=dfjspt_params.gnn_num_layers,
                gnn_type=dfjspt_params.gnn_type,
                pooling=dfjspt_params.gnn_pooling,
                dropout=dfjspt_params.gnn_dropout,
            )
            
            # Policy and value heads on top of GNN
            self.policy_head = nn.Sequential(
                nn.Linear(dfjspt_params.gnn_hidden_dim, dfjspt_params.gnn_hidden_dim),
                nn.ReLU(),
                nn.Linear(dfjspt_params.gnn_hidden_dim, num_outputs),
            )
            
            self.value_head = nn.Sequential(
                nn.Linear(dfjspt_params.gnn_hidden_dim, dfjspt_params.gnn_hidden_dim),
                nn.ReLU(),
                nn.Linear(dfjspt_params.gnn_hidden_dim, 1),
            )
            
            self.use_gnn = True
            self.internal_model = None
            self._last_value = None
            
        else:
            print(f"ℹ️  {name}: Using MLP encoder (GNN disabled or unavailable)")
            # Traditional MLP encoder
            self.internal_model = TorchFC(
                orig_space["observation"],
                action_space,
                num_outputs,
                model_config,
                name + "_internal",
            )
            self.use_gnn = False
            self.gnn_encoder = None
            self.policy_head = None
            self.value_head = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]
        observation = input_dict["obs"]["observation"]

        if self.use_gnn and self.gnn_encoder is not None:
            # GNN forward pass
            # observation shape: [batch_size, n_jobs, n_features]
            batch_size = observation.shape[0] if len(observation.shape) > 2 else 1
            
            if batch_size == 1 and len(observation.shape) == 2:
                # Single observation, add batch dimension
                observation = observation.unsqueeze(0)
            
            # Convert observation to graph structure
            graph_dict = batch_observations_to_graphs(observation, agent_type='job')
            
            # Pass through GNN
            gnn_output = self.gnn_encoder(
                node_features=graph_dict['node_features'],
                node_types=graph_dict['node_types'],
                edge_index=graph_dict['edge_index'],
                edge_features=graph_dict['edge_features'],
                batch=graph_dict['batch'],
            )
            
            # Get job-specific embedding
            job_embedding = gnn_output['job_embedding']  # [batch_size, hidden_dim]
            
            # Generate logits
            logits = self.policy_head(job_embedding)  # [batch_size, num_actions]
            
            # Compute value for value_function()
            self._last_value = self.value_head(job_embedding).squeeze(-1)  # [batch_size]
            
        else:
            # MLP forward pass
            logits, _ = self.internal_model({"obs": observation})
        
        # Apply action mask
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    @override(ModelV2)
    def value_function(self):
        if self.use_gnn:
            return self._last_value if self._last_value is not None else torch.zeros(1)
        else:
            return self.internal_model.value_function()

    @override(ModelV2)
    def custom_loss(self, policy_loss, loss_inputs=None):
        if dfjspt_params.use_custom_loss is True:
            # Get the next batch from our input files.
            batch = generate_sample_batch(batch_type="job")

            obs = restore_original_dimensions(
                torch.from_numpy(batch["obs_flat"]).float().to(policy_loss[0].device),
                self.obs_space,
                tensorlib="torch",
            )
            logits, _ = self.forward({"obs": obs}, [], None)

            action_mask = obs["action_mask"]
            # Convert action_mask into a [0.0 || -inf]-type mask.
            inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
            logits = logits + inf_mask

            # Compute the IL loss.
            action_dist_1 = TorchCategorical(logits, self.model_config)

            imitation_loss_1 = torch.mean(
                -action_dist_1.logp(
                    # torch.from_numpy(batch["actions"]).to(policy_loss[0].device)
                    torch.from_numpy(batch["actions"]).to(policy_loss[0].device)
                )
            )
            imitation_loss = imitation_loss_1
            # self.imitation_loss_metric = imitation_loss.item()
            # self.policy_loss_metric = np.mean([loss.item() for loss in policy_loss])

            # Add the imitation loss to each already calculated policy loss term.
            # Alternatively (if custom loss has its own optimizer):
            # return policy_loss + [10 * self.imitation_loss]
            return [loss_ + dfjspt_params.il_loss_weight * imitation_loss for loss_ in policy_loss]
        elif dfjspt_params.use_custom_loss is False:
            return policy_loss
        else:
            raise RuntimeError('Invalid "use_custom_loss" value!')


class MachineActionMaskModel(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel with optional GNN encoder."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observation" in orig_space.spaces
        )

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        # Choose encoder based on configuration
        if dfjspt_params.use_gnn_encoder and GNN_AVAILABLE:
            print(f"✅ {name}: Using GNN encoder")
            # GNN encoder
            node_feature_dims = {
                'job': 0,
                'machine': orig_space["observation"].shape[1],
                'transbot': 0,
                'operation': 0,
            }
            
            self.gnn_encoder = WorkshopGNN(
                node_feature_dims=node_feature_dims,
                edge_feature_dim=4,
                hidden_dim=dfjspt_params.gnn_hidden_dim,
                num_layers=dfjspt_params.gnn_num_layers,
                gnn_type=dfjspt_params.gnn_type,
                pooling=dfjspt_params.gnn_pooling,
                dropout=dfjspt_params.gnn_dropout,
            )
            
            self.policy_head = nn.Sequential(
                nn.Linear(dfjspt_params.gnn_hidden_dim, dfjspt_params.gnn_hidden_dim),
                nn.ReLU(),
                nn.Linear(dfjspt_params.gnn_hidden_dim, num_outputs),
            )
            
            self.value_head = nn.Sequential(
                nn.Linear(dfjspt_params.gnn_hidden_dim, dfjspt_params.gnn_hidden_dim),
                nn.ReLU(),
                nn.Linear(dfjspt_params.gnn_hidden_dim, 1),
            )
            
            self.use_gnn = True
            self.internal_model = None
            self._last_value = None
            
        else:
            print(f"ℹ️  {name}: Using MLP encoder")
            self.internal_model = TorchFC(
                orig_space["observation"],
                action_space,
                num_outputs,
                model_config,
                name + "_internal",
            )
            self.use_gnn = False
            self.gnn_encoder = None
            self.policy_head = None
            self.value_head = None

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]
        observation = input_dict["obs"]["observation"]

        if self.use_gnn and self.gnn_encoder is not None:
            # GNN forward pass
            batch_size = observation.shape[0] if len(observation.shape) > 2 else 1
            
            if batch_size == 1 and len(observation.shape) == 2:
                observation = observation.unsqueeze(0)
            
            # Convert observation to graph structure
            graph_dict = batch_observations_to_graphs(observation, agent_type='machine')
            
            # Pass through GNN
            gnn_output = self.gnn_encoder(
                node_features=graph_dict['node_features'],
                node_types=graph_dict['node_types'],
                edge_index=graph_dict['edge_index'],
                edge_features=graph_dict['edge_features'],
                batch=graph_dict['batch'],
            )
            
            # Get machine-specific embedding
            machine_embedding = gnn_output['machine_embedding']
            
            # Generate logits
            logits = self.policy_head(machine_embedding)
            
            # Compute value
            self._last_value = self.value_head(machine_embedding).squeeze(-1)
            
        else:
            # MLP forward pass
            logits, _ = self.internal_model({"obs": observation})

        # Apply action mask
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        if self.use_gnn:
            return self._last_value if self._last_value is not None else torch.zeros(1)
        else:
            return self.internal_model.value_function()

    @override(ModelV2)
    def custom_loss(self, policy_loss, loss_inputs=None):
        if dfjspt_params.use_custom_loss is True:
            # Get the next batch from our input files.
            batch = generate_sample_batch(batch_type="machine")

            # Define a secondary loss by building a graph copy with weight sharing.
            # obs = batch["obs"]
            obs = restore_original_dimensions(
                torch.from_numpy(batch["obs_flat"]).float().to(policy_loss[0].device),
                self.obs_space,
                tensorlib="torch",
            )
            logits, _ = self.forward({"obs": obs}, [], None)
            action_mask = obs["action_mask"]
            # Convert action_mask into a [0.0 || -inf]-type mask.
            inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
            logits = logits + inf_mask

            # Compute the IL loss.
            action_dist_1 = TorchCategorical(logits, self.model_config)

            imitation_loss_1 = torch.mean(
                -action_dist_1.logp(
                    # torch.from_numpy(batch["actions"]).to(policy_loss[0].device)
                    torch.from_numpy(batch["actions"]).to(policy_loss[0].device)
                )
            )
            imitation_loss = imitation_loss_1
            # self.imitation_loss_metric = imitation_loss.item()
            # self.policy_loss_metric = np.mean([loss.item() for loss in policy_loss])

            # Add the imitation loss to each already calculated policy loss term.
            # Alternatively (if custom loss has its own optimizer):
            # return policy_loss + [10 * self.imitation_loss]
            return [loss_ + dfjspt_params.il_loss_weight * imitation_loss for loss_ in policy_loss]
        else:
            return policy_loss


class TransbotActionMaskModel(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel with optional GNN encoder."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
                isinstance(orig_space, Dict)
                and "action_mask" in orig_space.spaces
                and "observation" in orig_space.spaces
        )
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        # Choose encoder based on configuration
        if dfjspt_params.use_gnn_encoder and GNN_AVAILABLE:
            print(f"✅ {name}: Using GNN encoder")
            # GNN encoder
            node_feature_dims = {
                'job': 0,
                'machine': 0,
                'transbot': orig_space["observation"].shape[1],
                'operation': 0,
            }
            
            self.gnn_encoder = WorkshopGNN(
                node_feature_dims=node_feature_dims,
                edge_feature_dim=4,
                hidden_dim=dfjspt_params.gnn_hidden_dim,
                num_layers=dfjspt_params.gnn_num_layers,
                gnn_type=dfjspt_params.gnn_type,
                pooling=dfjspt_params.gnn_pooling,
                dropout=dfjspt_params.gnn_dropout,
            )
            
            self.policy_head = nn.Sequential(
                nn.Linear(dfjspt_params.gnn_hidden_dim, dfjspt_params.gnn_hidden_dim),
                nn.ReLU(),
                nn.Linear(dfjspt_params.gnn_hidden_dim, num_outputs),
            )
            
            self.value_head = nn.Sequential(
                nn.Linear(dfjspt_params.gnn_hidden_dim, dfjspt_params.gnn_hidden_dim),
                nn.ReLU(),
                nn.Linear(dfjspt_params.gnn_hidden_dim, 1),
            )
            
            self.use_gnn = True
            self.internal_model = None
            self._last_value = None
            
        else:
            print(f"ℹ️  {name}: Using MLP encoder")
            self.internal_model = TorchFC(
                orig_space["observation"],
                action_space,
                num_outputs,
                model_config,
                name + "_internal",
            )
            self.use_gnn = False
            self.gnn_encoder = None
            self.policy_head = None
            self.value_head = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"]
        observation = input_dict["obs"]["observation"]

        if self.use_gnn and self.gnn_encoder is not None:
            # GNN forward pass
            batch_size = observation.shape[0] if len(observation.shape) > 2 else 1
            
            if batch_size == 1 and len(observation.shape) == 2:
                observation = observation.unsqueeze(0)
            
            # Convert observation to graph structure
            graph_dict = batch_observations_to_graphs(observation, agent_type='transbot')
            
            # Pass through GNN
            gnn_output = self.gnn_encoder(
                node_features=graph_dict['node_features'],
                node_types=graph_dict['node_types'],
                edge_index=graph_dict['edge_index'],
                edge_features=graph_dict['edge_features'],
                batch=graph_dict['batch'],
            )
            
            # Get transbot-specific embedding
            transbot_embedding = gnn_output['transbot_embedding']
            
            # Generate logits
            logits = self.policy_head(transbot_embedding)
            
            # Compute value
            self._last_value = self.value_head(transbot_embedding).squeeze(-1)
            
        else:
            # MLP forward pass
            logits, _ = self.internal_model({"obs": observation})

        # Apply action mask
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    @override(ModelV2)
    def value_function(self):
        if self.use_gnn:
            return self._last_value if self._last_value is not None else torch.zeros(1)
        else:
            return self.internal_model.value_function()

    @override(ModelV2)
    def custom_loss(self, policy_loss, loss_inputs=None):
        if dfjspt_params.use_custom_loss is True:
            # Get the next batch from our input files.
            batch = generate_sample_batch(batch_type="transbot")

            obs = restore_original_dimensions(
                torch.from_numpy(batch["obs_flat"]).float().to(policy_loss[0].device),
                self.obs_space,
                tensorlib="torch",
            )
            logits, _ = self.forward({"obs": obs}, [], None)
            action_mask = obs["action_mask"]
            # Convert action_mask into a [0.0 || -inf]-type mask.
            inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
            logits = logits + inf_mask

            # Compute the IL loss.
            action_dist = TorchCategorical(logits, self.model_config)

            imitation_loss = torch.mean(
                -action_dist.logp(
                    # torch.from_numpy(batch["actions"]).to(policy_loss[0].device)
                    torch.from_numpy(batch["actions"]).to(policy_loss[0].device)
                )
            )
            # self.imitation_loss_metric = imitation_loss.item()
            # self.policy_loss_metric = np.mean([loss.item() for loss in policy_loss])

            # Add the imitation loss to each already calculated policy loss term.
            # Alternatively (if custom loss has its own optimizer):
            # return policy_loss + [10 * self.imitation_loss]
            return [loss_ + dfjspt_params.il_loss_weight * imitation_loss for loss_ in policy_loss]
        elif dfjspt_params.use_custom_loss is False:
            return policy_loss
        else:
            raise RuntimeError('Invalid "use_custom_loss" value!')

class StrategyModel(TorchModelV2, nn.Module):
    """
    Strategy layer model that outputs preference vector for multi-objective optimization.
    
    Input: Global state (aggregated features from jobs, machines, transbots)
    Output: Preference vector [p_efficiency, p_cost, p_delivery] (sums to 1)
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        # Extract observation dimension
        if hasattr(obs_space, 'original_space'):
            obs_dim = obs_space.original_space.shape[0]
        else:
            obs_dim = obs_space.shape[0]

        # Get hidden layer sizes from config
        hiddens = model_config.get("fcnet_hiddens", [128, 128])
        activation = model_config.get("fcnet_activation", "tanh")

        # Build network layers
        layers = []
        in_size = obs_dim
        
        for hidden_size in hiddens:
            layers.append(nn.Linear(in_size, hidden_size))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Tanh())
            in_size = hidden_size
        
        # Output layer for preference vector (3 dimensions)
        layers.append(nn.Linear(in_size, num_outputs))
        
        self.policy_network = nn.Sequential(*layers)
        
        # Value network (for critic)
        value_layers = []
        in_size = obs_dim
        for hidden_size in hiddens:
            value_layers.append(nn.Linear(in_size, hidden_size))
            if activation == "tanh":
                value_layers.append(nn.Tanh())
            elif activation == "relu":
                value_layers.append(nn.ReLU())
            else:
                value_layers.append(nn.Tanh())
            in_size = hidden_size
        value_layers.append(nn.Linear(in_size, 1))
        
        self.value_network = nn.Sequential(*value_layers)
        
        self._value_out = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        Forward pass to compute preference vector logits.
        
        The logits will be passed through softmax by the action distribution
        to ensure the preference vector sums to 1.
        """
        obs = input_dict["obs"]
        
        # Ensure obs is 2D (batch_size, obs_dim)
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        
        # Policy network output (logits for preference vector)
        logits = self.policy_network(obs.float())
        
        # Value network output
        self._value_out = self.value_network(obs.float())
        
        return logits, state

    @override(ModelV2)
    def value_function(self):
        """Return the value function output from the last forward pass."""
        if self._value_out is None:
            return torch.zeros(1)
        return self._value_out.squeeze(-1)
