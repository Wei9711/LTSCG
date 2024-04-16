from .q_learner import QLearner
from components.episode_buffer import EpisodeBatch
import torch as th
import torch.nn as nn
from torch.optim import RMSprop
from modules.graphlearner.GTSModel import GTSModel
import sys
import os
import json

class LTSCG_PredictLearner(QLearner):

    def __init__(self, mac, scheme, logger, args):
        super(LTSCG_PredictLearner, self).__init__(mac, scheme, logger, args)

        ###################
        # Initial graph learner class
        self.temperature = 0.5
        self._model_kwargs = args.gtsmodel
        self._model_kwargs['input_dim'] = self.args.obs_shape
        self._model_kwargs['output_dim'] = self.args.obs_shape
        self._model_kwargs['num_nodes'] = self.args.n_agents
        self._model_kwargs['dim_fc'] = self.args.obs_shape * (args.episode_limit+1)
        # print(self._model_kwargs)
        graph_learner = GTSModel(self.temperature, self.logger, self._model_kwargs)
        self.graph_learner = graph_learner.cuda() if th.cuda.is_available() else graph_learner
        # graph_leaner -> based on Qlearner
        self.mlp_emb_hid = args.mlp_emb_hid
        self.mlp_out = args.mlp_out
        self.graph_obs_MLP = self._mlp(args.obs_shape, self.mlp_emb_hid, self.mlp_out)
        self.next_obs_MLP = self._mlp(args.obs_shape, self.mlp_emb_hid, self.mlp_out)
        ####################
        if th.cuda.is_available():
            self.graph_obs_MLP.cuda()
            self.next_obs_MLP.cuda()
        
    def train(self, episode_sample: EpisodeBatch,max_ep_t, t_env: int, episode_num: int):
        batch = episode_sample[:, :max_ep_t]
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        #############################
        ### Total episode obs to generate node_features
        # obs.shape [32, 401, 6, 156] 
        obs = episode_sample["obs"][:, :] 
        # [6, 32, 401, 156] [n_agents, batch,max_ep,obs_feature]
        # [10,32, 151, 105]
        obs = obs.permute(2, 0, 1, 3) 
        episode_obs = obs.reshape(self.args.n_agents,batch.batch_size,-1)
        # episode_obs.shape [6, 32, 62556]
        # [10,32,15855]

        ### input_obs as encoder input
        input_obs = batch["obs"][:, :-1].transpose(1,0)
        input_last_a = batch["actions"][:, :-1].transpose(1,0)
        # max_ep_t: 53
        # input_obs.shape [53, 32, 6, 156]


        # cancate obs and last action
        encoder_input = th.cat([input_obs,input_last_a],dim = 3)
        # encoder_input.shape [53, 32, 6, 156+1]


        encoder_input = encoder_input.reshape(max_ep_t-1,batch.batch_size,-1)
        
        graph = batch["graph"][:, :]
        # graph.shape [32, 6, 6]
        self.graph_learner = self.graph_learner.train()

        # input: [actual sequence lenght, bs, num_node*obs] 
        gumbel_adj,graph_decoder_out = self.graph_learner(1, encoder_input, episode_obs,graph, self.temperature, 1, 1, 1)
        # encoder_output.shape [39, 32, 936]  936 = 6 * 156
        # mid_output.shape [6, 6]
        graph_decoder_out = graph_decoder_out.reshape(max_ep_t-1,batch.batch_size,self.args.n_agents,-1)
        # graph_decoder_out.shape [39, 32, 6, 156]
        learn_graph = th.unsqueeze(gumbel_adj, dim=0)
        learn_graph = learn_graph.repeat(batch.batch_size, 1, 1)
        batch["graph"][:, :] = learn_graph
        # calculate |t| (obs + delta obs) and |t+1| obs
        graph_obs_emb = self.graph_obs_MLP.forward(input_obs + graph_decoder_out) 
        next_obs_emb = self.next_obs_MLP.forward(batch["obs"][:, 1:].transpose(1,0)) 
        g_loss = nn.MSELoss()
        graph_loss = g_loss(graph_obs_emb, next_obs_emb)
        #############################
        mac_out = []
        """
        batch.batch_size =32
        self.mac.init_hidden() -> torch.Size([32, 5, 64]) all 0 (same initial parameters)
        """
        self.mac.init_hidden(batch.batch_size)
        """
        t: time slice in batch.max_seq_length
        batch.max_seq_length -> number of max step in episode <= buffer.max_seq_length 120

        batch["actions"] -> torch.Size([32, 66, 5, 1])
        batch["actions"][:, :-1] -> torch.Size([32, 65, 5, 1]) -> current actions: time 0 to t 
        """
        for t in range(batch.max_seq_length):
            """
            for t time slice
            agent_outs -> torch.Size([32, 5, 11])
            [32: smaple bacth size, 5: agents, 11: actions num]
            """
            agent_outs, Atten_graph = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        """
        Agent output Q_i over time max_seq_length
        mac_out : torch.Size([32, 66, 5, 11])
        """
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # Pick the Q-Values for the actions taken by each agent / accoding actions take Q-values
        """
        Already have all Q_i for all action,
        mac_out : torch.Size([32, 66, 5, 11])

        Get actual Q_values according to actions the agents taken
        actions: torch.Size([32, 66, 5, 1])

        chosen_action_qvals -> torch.Size([32, 65, 5])
        """
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs,_ = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q: # QMIX double_q = True
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-o  ut the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum() 
        total_loss =loss + self.args.graph_loss_weight * graph_loss

        # Optimise
        self.optimiser.zero_grad()
        total_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("QMIX loss", loss.item(), t_env)
            self.logger.log_stat("Graph loss", graph_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.cpu(), t_env)
            self.logger.log_stat('Total loss', total_loss.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            
            # Log gumbel_adj
            self.logger.log_matrix("gumbel_adj", gumbel_adj, t_env)
            self.logger.log_matrix("Atten_adj", Atten_graph[0], t_env)
            
            self.log_stats_t = t_env

            # self.save_gumbel_adj(t_env,gumbel_adj.cpu())

    # def save_gumbel_adj(self,t_env,gumbel_adj):
    #     if os.path.exists('adj.json'):
    #         with open('adj.json', 'r') as file:
    #             data = json.load(file)
    #     else:
    #         data = {}

    #     data[t_env] = gumbel_adj.tolist()

    #     with open('adj.json', 'w') as file:
    #         json.dump(data, file)

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



        

