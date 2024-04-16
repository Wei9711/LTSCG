
from .q_learner import QLearner
from components.episode_buffer import EpisodeBatch
import torch as th
import torch.nn as nn
from torch.optim import RMSprop
from modules.graphlearner.GTSModel import GTSModel
from components.attention_module import AttentionModule
from components.gcn_module import GCNModule
from torch_geometric.nn import global_mean_pool
import sys

class LTSCG_InferLearner(QLearner):

    def __init__(self, mac, scheme, logger, args):
        super(LTSCG_InferLearner, self).__init__(mac, scheme, logger, args)
        self.temperature = 0.5
        self._model_kwargs = args.gtsmodel
        self._model_kwargs['input_dim'] = self.args.obs_shape
        self._model_kwargs['output_dim'] = self.args.obs_shape
        self._model_kwargs['num_nodes'] = self.args.n_agents
        self._model_kwargs['dim_fc'] = self.args.obs_shape * (args.episode_limit + 1)
        graph_learner = GTSModel(self.temperature, self.logger, self._model_kwargs)
        self.graph_learner = graph_learner.cuda() if th.cuda.is_available() else graph_learner
        input_shape = args.obs_shape + 1
        self.n_gcn_layers = args.number_gcn_layers
        self.graph_emb_dim = args.state_shape
        self.graph_emb_hid = args.graph_emb_hid
        self.graph_input_obs_MLP = self._mlp(input_shape, self.graph_emb_hid, self.graph_emb_dim)
        self.attention_layer = AttentionModule((self.graph_emb_dim), attention_type='general')
        self.gcn_layers = nn.ModuleList([GCNModule(in_features=(self.graph_emb_dim), out_features=(self.graph_emb_dim), bias=True, id=i) for i in range(self.n_gcn_layers)])
        
        self.mlp_emb_hid = args.mlp_emb_hid
        self.mlp_out = args.mlp_out
        self.graph_output_MLP = self._mlp(self.graph_emb_dim, self.mlp_emb_hid, self.mlp_out)
        self.state_MLP = self._mlp(args.state_shape, self.mlp_emb_hid, self.mlp_out)

        if th.cuda.is_available():
            self.graph_input_obs_MLP.cuda()
            self.attention_layer.cuda()
            self.gcn_layers.cuda()
            self.graph_output_MLP.cuda()
            self.state_MLP.cuda()

    def train(self, episode_sample: EpisodeBatch, max_ep_t, t_env: int, episode_num: int):
        batch = episode_sample[:, :max_ep_t]
        rewards = batch['reward'][:, :-1]
        actions = batch['actions'][:, :-1]
        terminated = batch['terminated'][:, :-1].float()
        mask = batch['filled'][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch['avail_actions']


        obs = episode_sample['obs'][:, :]
        obs = obs.permute(2, 0, 1, 3)
        episode_obs = obs.reshape(self.args.n_agents, batch.batch_size, -1)
        input_obs = batch['obs'][:, :-1].transpose(1, 0)
        input_last_a = batch['actions'][:, :-1].transpose(1, 0)
        obs_with_action = th.cat([input_obs, input_last_a], dim=3)


        encoder_input = obs_with_action.reshape(max_ep_t - 1, batch.batch_size, -1)
        graph = batch['graph'][:, :]
        self.graph_learner = self.graph_learner.train()
        gumbel_adj,_ = self.graph_learner(1, encoder_input, episode_obs, graph, self.temperature, 1, 1, 1)
        learn_graph = th.unsqueeze(gumbel_adj, dim=0)
        learn_graph_batch = learn_graph.repeat(batch.batch_size, 1, 1)
        batch['graph'][:, :] = learn_graph_batch

        
        graph_embeddings_collection = []
        graph_embeddings_0 = self.graph_input_obs_MLP.forward(obs_with_action)
        graph_embeddings_collection.append(graph_embeddings_0)
        attention_weights = self.attention_layer.forward(graph_embeddings_0)
        for i_layer, gcn_layer in enumerate(self.gcn_layers):
            embeddings_gcn = gcn_layer.forward(graph_embeddings_collection[i_layer], learn_graph_batch.unsqueeze(0) * attention_weights)
            graph_embeddings_collection.append(embeddings_gcn)

        avg_pool = th.mean((graph_embeddings_collection[-1]), dim=2).transpose(1, 0)

        graph_obs_emb = self.graph_output_MLP.forward(avg_pool) 
        state_emb = self.state_MLP(batch['state'][:, :-1])
        g_loss = nn.MSELoss()
        graph_loss = g_loss(graph_obs_emb, state_emb)

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, Atten_graph = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)

        mac_out = th.stack(mac_out, dim=1)
        chosen_action_qvals = th.gather((mac_out[:, :-1]), dim=3, index=actions).squeeze(3)
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs,_ = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        target_mac_out = th.stack((target_mac_out[1:]), dim=1)
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999
        if self.args.double_q:
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch['state'][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch['state'][:, 1:])
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        td_error = chosen_action_qvals - targets.detach()
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask

        loss = (masked_td_error ** 2).sum() / mask.sum()
        total_loss = loss + self.args.graph_loss_weight * graph_loss
        self.optimiser.zero_grad()
        total_loss.backward()

        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat('QMIX loss', loss.item(), t_env)
            self.logger.log_stat('Graph loss', graph_loss.item(), t_env)
            self.logger.log_stat('Total loss', total_loss.item(), t_env)
            self.logger.log_stat('grad_norm', grad_norm.cpu(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat('td_error_abs', masked_td_error.abs().sum().item() / mask_elems, t_env)
            self.logger.log_stat('q_taken_mean', (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat('target_mean', (targets * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            
            self.logger.log_matrix("gumbel_adj", gumbel_adj, t_env)
            self.logger.log_matrix("Atten_adj", Atten_graph[0], t_env)

            self.log_stats_t = t_env

    @staticmethod
    def _mlp(input, hidden_dims, output):
        """ Creates an MLP with the specified input and output dimensions and (optional) hidden layers. """
        hidden_dims = [] if hidden_dims is None else hidden_dims
        hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else hidden_dims
        dim = input
        layers = []
        for d in hidden_dims:
            layers.append(nn.Linear(dim, d))
            layers.append(nn.ReLU())
            dim = d

        layers.append(nn.Linear(dim, output))
        return (nn.Sequential)(*layers)