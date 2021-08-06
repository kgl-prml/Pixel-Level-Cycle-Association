import torch.nn as nn
import torch

class DomainBN(nn.Module):
    def __init__(self, norm_layer, in_channels, num_domains=1):
        super(DomainBN, self).__init__()
        self.norm_layers = nn.ModuleList()
        for i in range(num_domains):
            self.norm_layers.append(norm_layer(in_channels))

        self.num_domains = num_domains
        self.cur_domain_id = None

    def init(self, state_dict=None):
        for i in range(self.num_domains):
            m = self.norm_layers[i]
            if state_dict is None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            else:
                m.load_state_dict(state_dict)

    @classmethod
    def fresh_parameters(cls, module, d0=0, d1=1):
        if isinstance(module, DomainBN):
            assert(d0 < module.num_domains), d0
            assert(d1 < module.num_domains), d1
            module.norm_layers[d1].weight = module.norm_layers[d0].weight
            module.norm_layers[d1].bias = module.norm_layers[d0].bias
        for name, child in module.named_children():
            cls.fresh_parameters(child, d0, d1)

    @classmethod
    def set_domain_id(cls, module, domain_id=0):
        if isinstance(module, DomainBN):
            assert(domain_id < module.num_domains), domain_id
            module.cur_domain_id = domain_id
        for name, child in module.named_children():
            cls.set_domain_id(child, domain_id)

    @classmethod
    def freeze_domain_bn(cls, module, domain_id=0):
        if isinstance(module, DomainBN):
            assert(domain_id < module.num_domains), domain_id
            module.norm_layers[domain_id].weight.requires_grad = False
            module.norm_layers[domain_id].bias.requires_grad = False

        for name, child in module.named_children():
            cls.freeze_domain_bn(child, domain_id)
    
    def forward(self, x):
        assert(self.cur_domain_id is not None)
        return self.norm_layers[self.cur_domain_id](x) 

    @classmethod
    def convert_domain_batchnorm(cls, module, num_domains=1):
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module_output = DomainBN(module.__class__, module.num_features, num_domains)
            # set the parameters
            module_output.init(module.state_dict())

        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_domain_batchnorm(child, num_domains))
        del module
        return module_output
