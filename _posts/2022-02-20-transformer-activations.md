---
title: "Visualizing Pytorch's transformer activations"
date: 2022-02-20
permalink: /posts/2022/02/20/transformer-activations
tags:
  - pytorch  
  - transformers  
  - visualization  
  - hooks  
---
Transformers are a trendy architecture nowadays. The paralellism and possibility of working with sequences of different lenghts allowed this architecture to achieve 
awesome results in different fields. Today we are gonna learn how to visualize the attention probabilities when using pytorch's official transformer modules.  


## Catching the activations via hooks  
Forwards hooks are a well known tool in pytorch toolset. Despite Pytorch developed a more complex system, [Torch FX](https://pytorch.org/blog/FX-feature-extraction-torchvision/),
this system works under certain contrains which may be frustrating. We are going to use pytorch hooks are code-independent. To do so I've written a simple wrapper which 
allow us to enable hooks within a single line of code.

```
class Model(object):
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = {}
        self._hook_handlers = {}

    def set_forward_hook(self, name: str, layer: nn.Module, fn=None):
        """
        Set a forward hook which stores the input and output for a given layer.
        Check https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
        for additional info.

        The funtion fn should be written as official pytorch docs state.

        :param name: Unique name for the hook.
        :type name: str
        :param layer: module to be hooked
        :type layer: nn.Module
        :param fn: Optional function to modify the hook.
        :param fn: Callable
        :return: torch.utils.hooks.RemovableHandle
        a handle that can be used to remove the added hook by calling handle.remove()

        """
        if name in self._hook_handlers:
            raise ValueError(f'Hook name {name} already in use')

        handler = layer.register_forward_hook(self._fwd_hook_fn)
        self._hook_handlers[name] = {'type': 'forward_hook', 'handler': handler, 'fn': fn}
        layer._flerken_fwd_hook_name = name

        return handler

    def set_backward_hook(self, name: str, layer: nn.Module, fn=None):
        """
        Set a backward hook which stores the input gradient and output gradient for a given layer.
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_backward_hook
        for additional info.

        The funtion fn should be written as official pytorch docs state.

        :param name: Unique name for the hook.
        :type name: str
        :param layer: module to be hooked
        :type layer: nn.Module
        :param fn: Optional function to modify the hook.
        :param fn: Callable
        :return: torch.utils.hooks.RemovableHandle
        a handle that can be used to remove the added hook by calling handle.remove()
        """
        if name in self._hook_handlers:
            raise ValueError(f'Hook name {name} already in use')

        handler = layer.register_backward_hook(self._bwd_hook_fn)
        self._hook_handlers[name] = {'type': 'backward_hook', 'handler': handler, 'fn': fn}
        layer._flerken_bwd_hook_name = name

        return handler

    def set_forward_pre_hook(self, name: str, layer: nn.Module, fn=None):
        """
        Set a pre-hook (which is like a forward hook but called before calling the layer)
         which stores the input for a given layer.
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook
        for additional info.

        The funtion fn should be written as official pytorch docs state.

        :param name: Unique name for the hook.
        :type name: str
        :param layer: module to be hooked
        :type layer: nn.Module
        :param fn: Optional function to modify the hook.
        :param fn: Callable
        :return: torch.utils.hooks.RemovableHandle
        a handle that can be used to remove the added hook by calling handle.remove()
        """
        if name in self._hook_handlers:
            raise ValueError(f'Hook name {name} already in use')

        handler = layer.register_forward_pre_hook(self._fwd_hook_pre_fn)
        self._hook_handlers[name] = {'type': 'forward_pre_hook', 'handler': handler, 'fn': fn}
        layer._flerken_fwd_pre_hook_name = name

        return handler

    def del_hook(self, name):
        self._hook_handlers[name]['handler'].remove()
        self._hook_handlers.pop(name)

    def _fwd_hook_fn(self, module, input, output):
        name = module._flerken_fwd_hook_name
        if self._hook_handlers[name]['fn'] is not None:
            module, input, output = self._hook_handlers[name]['fn'](module, input, output)
        self.hooks[name] = {'input': input, 'output': output}

    def _fwd_hook_pre_fn(self, module, input):
        name = module._flerken_fwd_pre_hook_name
        if self._hook_handlers[name]['fn'] is not None:
            module, input = self._hook_handlers[name]['fn'](module, input)
        self.hooks[name] = {'input': input}

    def _bwd_hook_fn(self, module, grad_in, grad_out):
        name = module._flerken_bwd_hook_name
        if self._hook_handlers[name]['fn'] is not None:
            module, grad_in, grad_out = self._hook_handlers[name]['fn'](module, grad_in, grad_out)
        self.hooks[name] = {'input': grad_in, 'output': grad_out}
        if self._hook_handlers[name]['fn'] is not None:
            return grad_in
```  
Then, we just simply need to write a function which returns the nn.layers from which we want to catch the self.attention activations:  
In this case we are gonna grab decoder's attention modules.
```
def get_decoder_self_att(n):
    return eval(f'model.model.transformer.decoder.layers._modules["{n}"].self_attn')
```
## Modifying the Pytorch's trasnformer class  
Pytorch's transformer nn.Module is almost coded to make this task really simple. The major problem is there is no way to return the activations by passing arguments to 
the constructor. To achieve this we need to inherit pytorch's trnasformer class and to overwrite a simple flag. This is `need_weights=True`.  
```
from torch.nn import TransformerEncoderLayer as TorchTFL
class TransformerEncoderLayer(TorchTFL):
    def _sa_block(self, x,
                  attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)[0]
        return self.dropout1(x)
```
Once this is done we just need to build the encoder or decoder with our custom encoder/decoder layers:  
```
from torch.nn import TransformerEncoder, Transformer
layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation='gelu')
encoder = TransformerEncoder(layer, num_encoder_layers)
transformer = Transfomer(custom_encoder=encoder)
```
From now onwards, the activations will be computer automatically and now we just need to grab them with the hooks.  

## Setting the hooks!  
Assume that we have already instantiated our model. Then we just need to wrap it with the provided class which powers the hooks:  
```
n_decoders = 9
model = Transformer(custom_ecoder=encoder,num_decoder_layers=n_decoders,...)
model = Model(model)
```  
Lastly we just need to set the hooks:  
```
for i in range(n_decoders):
    model.set_forward_hook(f'decoder_attn{i}', get_decoder_self_att(i))
```
And we will have the activations in `model.hooks[f'decoder_attn{i}']['output'][1]`.
We can apply the same logic for the encoder, self attention or any module.  

