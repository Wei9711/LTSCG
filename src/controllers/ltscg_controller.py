
import sys
from .basic_controller import BasicMAC
import torch as th
import torch.nn as nn
import numpy as np, contextlib, itertools, torch_scatter
from math import factorial
from random import randrange

from modules.agents import REGISTRY as agent_REGISTRY
from modules.action_encoders import REGISTRY as action_encoder_REGISTRY
from components.attention_module import AttentionModule
from components.gcn_module import GCNModule
import copy

class LTSCG_GraphMAC(BasicMAC):

    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        # self.residual = args.residual
        self.args = args
        self.n_gcn_layers = args.number_gcn_layers
        self.dicg_layers = []
        self.dicg_emb_hid = args.dicg_emb_hid
        # original input shape: obs + actions + agents : 10m vs 11m :105+17+10=132
        org_input_shape = self._get_input_shape(scheme) 
        self.gcn_message_dim = args.gcn_message_dim 
        self.concate_mlp_dim = args.concate_mlp_dim 
        agent_input_shape = org_input_shape
        if self.args.concate_gcn:
            agent_input_shape = agent_input_shape + self.gcn_message_dim
        if self.args.concate_gcn and self.args.concate_mlp:
            agent_input_shape = agent_input_shape + self.concate_mlp_dim
        # print("agent_input_shape",agent_input_shape)
        self._build_agents(agent_input_shape)

        self.dicg_emb_dim = org_input_shape
        self.dicg_encoder = self._mlp(org_input_shape, self.dicg_emb_hid, self.concate_mlp_dim)
        self.dicg_layers.append(self.dicg_encoder)
        self.attention_layer = AttentionModule((self.concate_mlp_dim), attention_type='general')
        self.dicg_layers.append(self.attention_layer)
        self.gcn_layers = nn.ModuleList([
                                GCNModule(in_features=(self.concate_mlp_dim), out_features=(self.gcn_message_dim), bias=True, id=0),
                                GCNModule(in_features=(self.gcn_message_dim), out_features=(self.gcn_message_dim), bias=True, id=1)
                                ])
        self.dicg_layers.extend(self.gcn_layers)
        # self.dicg_aggregator = self._mlp(self.gcn_message_dim, self.dicg_emb_hid, self.gcn_message_dim)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch['avail_actions'][:, t_ep]
        agent_outputs , _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action((agent_outputs[bs]), (avail_actions[bs]), t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        org_agent_inputs = self.build_dicg_inputs(ep_batch, t)
        # [1,10,132]
        # [32,10,132]
        # org_agent_inputs-> batch size *self.n_agents, -1
        # avail_actions : [1,151,10,17]: [batch=1,max_episode=150+1,agent:10,actions:17]
        avail_actions = ep_batch['avail_actions'][:, t]
        
        embeddings_collection = []
        embeddings_0 = self.dicg_encoder.forward(org_agent_inputs)
        embeddings_collection.append(embeddings_0)
        # print("embeddings_0",embeddings_0.shape)
        # attention_weights [10,10] [agent,agent]
        attention_weights = self.attention_layer.forward(embeddings_0)
        # print("attention_weights",attention_weights.shape)
        # print("ep_batch['graph']",ep_batch['graph'].shape)

        # print(ep_batch['graph'])
        # print(ep_batch['graph'].shape)
        # print(ep_batch['graph']*attention_weights)
        # result_matrix = th.zeros((graph_bs * graph_size, graph_bs * graph_size)).to(ep_batch.device)
        # for i in range(graph_bs):
        #     start_idx = i * graph_size
        #     end_idx = start_idx + graph_size
        #     result_matrix[start_idx:end_idx, start_idx:end_idx] = ep_batch['graph'][i]

        # print("result_matrix", result_matrix.shape)
        for i_layer, gcn_layer in enumerate(self.gcn_layers):
            embeddings_gcn = gcn_layer.forward(embeddings_collection[i_layer], attention_weights)
            # print("embeddings_gcn",embeddings_gcn.shape)
            embeddings_collection.append(embeddings_gcn)
        
        if self.args.concate_gcn and self.args.concate_mlp:
            temp_org_input = org_agent_inputs.view(-1,org_agent_inputs.shape[-1])
            # print("temp_org_input",temp_org_input.shape)
            temp_mlp_message = embeddings_collection[0].view(-1,self.concate_mlp_dim)
            # print("temp_mlp_message",temp_mlp_message.shape)
            temp_gcn_message = embeddings_collection[-1].view(-1,self.gcn_message_dim)
            # print("temp_gcn_message",temp_gcn_message.shape)
            agent_input = th.cat([temp_org_input,temp_mlp_message, temp_gcn_message], dim=1)
        elif self.args.concate_gcn:
            temp_org_input = org_agent_inputs.view(-1,org_agent_inputs.shape[-1])
            # print("temp_org_input",temp_org_input.shape)
            temp_gcn_message = embeddings_collection[-1].view(-1,self.gcn_message_dim)
            # print("temp_gcn_message",temp_gcn_message.shape)
            agent_input = th.cat([temp_org_input, temp_gcn_message], dim=1)
        else:
            agent_input = org_agent_inputs.view(-1,org_agent_inputs.shape[-1])
        # print("agent_input",agent_input.shape)

        # dicg_agent_inputs = self.dicg_aggregator.forward(dicg_agent_inputs)
        agent_outs, self.hidden_states = self.agent(agent_input, self.hidden_states)
        if self.agent_output_type == 'pi_logits':
            if getattr(self.args, 'mask_before_softmax', True):
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -10000000000.0
            agent_outs = th.nn.functional.softmax(agent_outs, dim=(-1))
            # if not test_mode:
            #     epsilon_action_num = agent_outs.size(-1)
            #     if getattr(self.args, 'mask_before_softmax', True):
            #         epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()
            #     agent_outs = (1 - self.action_selector.epsilon) * agent_outs + th.ones_like(agent_outs) * self.action_selector.epsilon / epsilon_action_num
            #     if getattr(self.args, 'mask_before_softmax', True):
            #         agent_outs[reshaped_avail_actions == 0] = 0.0
        # sys.exit()
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), attention_weights

    def build_dicg_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        # for x in inputs:
        #     print(x.shape) 
        #     print(x.reshape(bs,self.n_agents, -1).shape) 
        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=2)
        return inputs

    def parameters(self):
        param = itertools.chain(BasicMAC.parameters(self), self.dicg_encoder.parameters(), self.attention_layer.parameters(), self.gcn_layers.parameters())
        return param

    def load_state(self, other_mac):
        BasicMAC.load_state(self, other_mac)
        self.dicg_encoder.load_state_dict(other_mac.dicg_encoder.state_dict())
        self.attention_layer.load_state_dict(other_mac.attention_layer.state_dict())
        self.gcn_layers.load_state_dict(other_mac.gcn_layers.state_dict())
        # self.dicg_aggregator.load_state_dict(other_mac.dicg_aggregator.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.dicg_encoder.cuda()
        self.attention_layer.cuda()
        self.gcn_layers.cuda()
        # self.dicg_aggregator.cuda()

    def save_models(self, path):
        BasicMAC.save(self, path)
        th.save(self.dicg_encoder.state_dict(), '{}/dicg_encoder.th'.format(path))
        th.save(self.attention_layer.state_dict(), '{}/attention_layer.th'.format(path))
        th.save(self.gcn_layers.state_dict(), '{}/gcn_layers.th'.format(path))
        # th.save(self.dicg_aggregator.state_dict(), '{}/dicg_aggregator.th'.format(path))

    def load_models(self, path):
        BasicMAC.load_state_dict(self, path)
        self.dicg_encoder.load_state_dict(th.load(('{}/dicg_encoder.th'.format(path)), map_location=(lambda storage, loc: storage)))
        self.attention_layer.load_state_dict(th.load(('{}/attention_layer.th'.format(path)), map_location=(lambda storage, loc: storage)))
        self.gcn_layers.load_state_dict(th.load(('{}/gcn_layers.th'.format(path)), map_location=(lambda storage, loc: storage)))
        # self.dicg_aggregator.load_state_dict(th.load(('{}/dicg_aggregator.th'.format(path)), map_location=(lambda storage, loc: storage)))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

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