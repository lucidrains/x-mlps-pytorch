## x-mlps

Just a repository that will house MLPs, from garden variety to the exotic, so as to avoid having to reimplement them again and again for different projects (especially RL)

Will also be the repository I use for testing out [Jules](https://jules.google.com/) and other AI assisted tools.


## Install

```bash
$ pip install x-mlps-pytorch
```

## Usage

```python
import torch
from x_mlps import MLP

actor = MLP(10, 16, 5)

critic = MLP(10, 32, 16, 1)

state = torch.randn(10)

action_logits = actor(state) # (5,)

values = critic(state) # (1,)
```
