## x-mlps-pytorch

Just a repository that will house MLPs for Pytorch, from garden variety to the exotic, so as to avoid having to reimplement them again and again for different projects (especially RL)

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

```bibtex
@inproceedings{wang2025,
    title   = {1000 Layer Networks for Self-Supervised {RL}: Scaling Depth Can Enable New Goal-Reaching Capabilities},
    author  = {Kevin Wang and Ishaan Javali and Micha{\l} Bortkiewicz and Tomasz Trzcinski and Benjamin Eysenbach},
    booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year    = {2025},
    url     = {https://openreview.net/forum?id=s0JVsx3bx1}
}
```

```bibtex
@misc{chen2026postlayernormbackstableexpressive,
    title   = {Post-LayerNorm Is Back: Stable, ExpressivE, and Deep},
    author  = {Chen Chen and Lai Wei},
    year    = {2026},
    eprint  = {2601.19895},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2601.19895},
}
```

```bibtex
@inproceedings{dorovatas2025autocompressing,
    title  = {Auto-Compressing Networks},
    author = {Vaggelis Dorovatas and Georgios Paraskevopoulos and Alexandros Potamianos},
    booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year    = {2025},
    url     = {https://openreview.net/forum?id=eIDa6pd9iQ}
}
```

```bibtex
@inproceedings{Lin2025ContinualLV,
    title    = {Continual Learning via Sparse Memory Finetuning},
    author   = {Jessy Lin and Luke S. Zettlemoyer and Gargi Ghosh and Wen-tau Yih and Aram H. Markosyan and Vincent-Pierre Berges and Barlas Ouguz},
    year     = {2025},
    url      = {https://api.semanticscholar.org/CorpusID:282203348},
    blog_url = {https://jessylin.com/2025/10/20/continual-learning/}
}
```

```bibtex
@misc{kimiteam2026attentionresiduals,
    title   = {Attention Residuals},
    author  = {Kimi Team and Guangyu Chen and Yu Zhang and Jianlin Su and Weixin Xu and Siyuan Pan and Yaoyu Wang and Yucheng Wang and Guanduo Chen and Bohong Yin and Yutian Chen and Junjie Yan and Ming Wei and Y. Zhang and Fanqing Meng and Chao Hong and Xiaotong Xie and Shaowei Liu and Enzhe Lu and Yunpeng Tai and Yanru Chen and Xin Men and Haiqing Guo and Y. Charles and Haoyu Lu and Lin Sui and Jinguo Zhu and Zaida Zhou and Weiran He and Weixiao Huang and Xinran Xu and Yuzhi Wang and Guokun Lai and Yulun Du and Yuxin Wu and Zhilin Yang and Xinyu Zhou},
    year    = {2026},
    eprint  = {2603.15031},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL},
    url     = {https://arxiv.org/abs/2603.15031},
}
```
