## x-mlps-pytorch

Just a repository that will house MLPs for Pytorch, from garden variety to the exotic, so as to avoid having to reimplement them again and again for different projects (especially RL)

Will also be the repository I use for testing out [Jules](https://jules.google.com/) and other AI assisted tools.


## Install

```bash
$ pip install x-mlps-pytorch
```

## Usage

```python
import torch
from x_mlps_pytorch import MLP

actor = MLP(10, 16, 5)

critic = MLP(10, 32, 16, 1)

state = torch.randn(10)

action_logits = actor(state) # (5,)

values = critic(state) # (1,)
```

## Citations

```bibtex
@article{So2021PrimerSF,
    title   = {Primer: Searching for Efficient Transformers for Language Modeling},
    author  = {David R. So and Wojciech Ma'nke and Hanxiao Liu and Zihang Dai and Noam M. Shazeer and Quoc V. Le},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2109.08668},
    url     = {https://api.semanticscholar.org/CorpusID:237563187}
}
```

```bibtex
@article{Zhang2024ReLU2WD,
    title   = {ReLU2 Wins: Discovering Efficient Activation Functions for Sparse LLMs},
    author  = {Zhengyan Zhang and Yixin Song and Guanghui Yu and Xu Han and Yankai Lin and Chaojun Xiao and Chenyang Song and Zhiyuan Liu and Zeyu Mi and Maosong Sun},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2402.03804},
    url     = {https://api.semanticscholar.org/CorpusID:267499856}
}
```

```bibtex
@inproceedings{Horuz2025TheRO,
    title   = {The Resurrection of the ReLU},
    author  = {Cocsku Can Horuz and Geoffrey Kasenbacher and Saya Higuchi and Sebastian Kairat and Jendrik Stoltz and Moritz Pesl and Bernhard A. Moser and Christoph Linse and Thomas Martinetz and Sebastian Otte},
    year    = {2025},
    url     = {https://api.semanticscholar.org/CorpusID:278959515}
}
```

```bibtex
@article{Loshchilov2024nGPTNT,
    title   = {nGPT: Normalized Transformer with Representation Learning on the Hypersphere},
    author  = {Ilya Loshchilov and Cheng-Ping Hsieh and Simeng Sun and Boris Ginsburg},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2410.01131},
    url     = {https://api.semanticscholar.org/CorpusID:273026160}
}
```

```bibtex
@article{Lee2025HypersphericalNF,
    title   = {Hyperspherical Normalization for Scalable Deep Reinforcement Learning},
    author  = {Hojoon Lee and Youngdo Lee and Takuma Seno and Donghu Kim and Peter Stone and Jaegul Choo},
    journal = {ArXiv},
    year    = {2025},
    volume  = {abs/2502.15280},
    url     = {https://api.semanticscholar.org/CorpusID:276558261}
}
```
