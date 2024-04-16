from .basic_controller import BasicMAC
from modules.agents import REGISTRY as agent_REGISTRY
from modules.action_encoders import REGISTRY as action_encoder_REGISTRY
import torch as th
import torch.nn as nn
import numpy as np
import utils.constructor as constructor
import itertools
import copy


# This multi-agent controller shares parameters between agents
class SopcgMAC(BasicMAC):

    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.n_actions = args.n_actions
        if args.construction == 'matching':
            self.solver = constructor.MatchingSolver(args)
        elif args.construction == 'tree':
            self.solver = constructor.TreeSolver(args)
        elif args.construction == 'line':
            self.solver = constructor.LineSolver(args)
        elif args.construction == 'star':
            self.solver = constructor.StarSolver(args)
        else:
            raise Exception("unimplemented method")

        self.input_shape = self.args.rnn_hidden_dim

        # action representation
        self.use_action_repr = args.use_action_repr
        if self.use_action_repr:
            # ObsRewardEncoder
            self.action_encoder = action_encoder_REGISTRY[args.action_encoder](args)
            self.action_repr = th.ones(self.n_actions, self.args.action_latent_dim).to(args.device)
            input_i = self.action_repr.unsqueeze(1).repeat(1, self.n_actions, 1)
            input_j = self.action_repr.unsqueeze(0).repeat(self.n_actions, 1, 1)
            self.p_action_repr = th.cat([input_i, input_j], dim=-1).view(self.n_actions * self.n_actions, -1).t().unsqueeze(0)

        self.single_q = self._mlp(self.input_shape, args.single_q_hidden_dim, self.n_actions)
        if self.use_action_repr:
            self.pairwise_q = self._mlp(2 * self.input_shape, args.pairwise_q_hidden_dim, 2 * self.args.action_latent_dim)
        else:
            self.pairwise_q = self._mlp(2 * self.input_shape, args.pairwise_q_hidden_dim, self.n_actions ** 2)
        
        self.privileged_bias = args.privileged_bias
        if self.privileged_bias:
            self.state_value = self._mlp(int(np.prod(args.state_shape)), [args.state_embed_dim], 1)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        f, g, graphs = self.forward(ep_batch, t_ep, test_mode=test_mode, select_graph=True)
        graphs, chosen_actions = self.action_selector.select_action(f[bs], g[bs], avail_actions[bs], graphs[bs], t_env, test_mode=test_mode)
        return graphs, chosen_actions

    def forward(self, ep_batch, t, test_mode=False, select_graph=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        self.hidden_states = self.agent(agent_inputs, self.hidden_states).view(-1, self.n_agents, self.args.rnn_hidden_dim)

        inputs = self.hidden_states
        f = self.single_q(inputs.view(-1, self.input_shape))
        f = f.view(-1, self.n_agents, self.n_actions)
        #=============payoff of DCG============
        input_i = inputs.unsqueeze(2).repeat(1, 1, self.n_agents, 1)
        input_j = inputs.unsqueeze(1).repeat(1, self.n_agents, 1, 1)
        inputs = th.cat([input_i, input_j], dim=-1).view(-1, 2 * self.input_shape)
        if self.use_action_repr:
            key = self.pairwise_q(inputs).view(-1, self.n_agents * self.n_agents, 2 * self.args.action_latent_dim)
            g = th.bmm(key, self.p_action_repr.repeat(f.shape[0], 1, 1)) / self.args.action_latent_dim / 2
        else:
            g = self.pairwise_q(inputs)
        #=============payoff of DCG============
        
        g = g.view(-1, self.n_agents, self.n_agents, self.n_actions, self.n_actions)
        g = (g + g.permute(0, 2, 1, 4, 3)) / 2.

        if self.args.privileged_bias and test_mode == False:
            f[:, 0, :] += self.state_value(ep_batch['state'][:, t])

        if select_graph == False:
            return f, g

        graphs = self.solver.solve(f, g, avail_actions, self.args.device)
        return f, g, graphs

    def update_action_repr(self):
        action_repr = self.action_encoder()

        self.action_repr = action_repr.detach().clone()

        # Pairwise Q (|A|, al) -> (|A|, |A|, 2*al)
        input_i = self.action_repr.unsqueeze(1).repeat(1, self.n_actions, 1)
        input_j = self.action_repr.unsqueeze(0).repeat(self.n_actions, 1, 1)
        self.p_action_repr = th.cat([input_i, input_j], dim=-1).view(self.n_actions * self.n_actions, -1).t().unsqueeze(0)
    
    #SAME AS BasicMAC
    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
    #SAME AS BasicMAC
    def parameters(self):
        return itertools.chain(self.agent.parameters(), self.single_q.parameters(), self.pairwise_q.parameters())
    
    def load_state(self, other_mac):
        #SAME AS BasicMAC
        self.agent.load_state_dict(other_mac.agent.state_dict())
        #SPOCG
        self.single_q.load_state_dict(other_mac.single_q.state_dict())
        self.pairwise_q.load_state_dict(other_mac.pairwise_q.state_dict())
        if self.args.use_action_repr:
            self.action_repr = copy.deepcopy(other_mac.action_repr)
            self.p_action_repr = copy.deepcopy(other_mac.p_action_repr)

    def cuda(self):
        #SAME AS BasicMAC
        self.agent.cuda()
        #SPOCG
        self.single_q.cuda()
        self.pairwise_q.cuda()
        if self.privileged_bias:
            self.state_value.cuda()
        if self.use_action_repr:
            self.action_encoder.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.single_q.state_dict(), "{}/single_q.th".format(path))
        th.save(self.pairwise_q.state_dict(), "{}/pairwise_q.th".format(path))
        if self.args.privileged_bias:
            th.save(self.state_value.state_dict(), "{}/agent.th".format(path))
        if self.args.use_action_repr:
            th.save(self.action_repr, "{}/action_repr.pt".format(path))
            th.save(self.p_action_repr, "{}/p_action_repr.pt".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.single_q.load_state_dict(th.load("{}/single_q.th".format(path), map_location=lambda storage, loc: storage))
        self.pairwise_q.load_state_dict(
            th.load("{}/pairwise_q.th".format(path), map_location=lambda storage, loc: storage))
        if self.args.privileged_bias:
            self.state_value.load_state_dict(
                th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        if self.args.use_action_repr:
            self.action_repr = th.load("{}/action_repr.pt".format(path),
                                    map_location=lambda storage, loc: storage).to(self.args.device)
            self.p_action_repr = th.load("{}/p_action_repr.pt".format(path),
                                        map_location=lambda storage, loc: storage).to(self.args.device)

    def action_encoder_params(self):
        return list(self.action_encoder.parameters())

    def action_repr_forward(self, ep_batch, t):
        return self.action_encoder.predict(ep_batch["obs"][:, t], ep_batch["actions_onehot"][:, t])

    # ========================= Private methods =========================

    @staticmethod
    def _mlp(input, hidden_dims, output):
        dim = input
        layers = []
        for d in hidden_dims:
            layers.append(nn.Linear(dim, d))
            layers.append(nn.ReLU())
            dim = d
        layers.append(nn.Linear(dim, output))
        return nn.Sequential(*layers)
