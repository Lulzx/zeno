# Zeno

[![CI](https://github.com/lulzx/zeno/actions/workflows/ci.yml/badge.svg)](https://github.com/lulzx/zeno/actions/workflows/ci.yml)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black)](https://developer.apple.com/metal/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

Zeno is a batched rigid-body simulator for Apple Silicon. The physics runs in
Metal. The runtime is written in Zig. Python reads and writes shared
`MTLBuffer` storage through NumPy without a staging copy.

The intended use is simple: run thousands of reinforcement-learning
environments on the GPU already in a Mac.

This is a research engine. It is not a drop-in or physics-equivalent MuJoCo
replacement. The supported contact set and the measured limits are written
down in [Status and scope](docs/status.md).

## Numbers

Full `World.step`, real repository MJCF models, 1,024 environments, 1,000
steps. Median of five runs on an M4 Pro (16-core GPU, 24 GiB), macOS 26.7,
Zig 0.16.0:

| Model | Environment steps/s |
|---|---:|
| Pendulum | 2.46M |
| Cartpole | 2.50M |
| Pusher | 1.38M |
| Ant | 1.44M |
| Humanoid | 0.34M |

Ant scales from 0.196M environment steps/s at 64 environments to 5.24M at
16,384. These are Zeno throughput measurements, not claims of equal work or
equal physics across simulators. See the [method, raw results, and narrower
correctness checks](docs/reference/performance.md).

## Build

Requires macOS 13 or newer, Apple Silicon, and Zig 0.16.x.

```sh
git clone https://github.com/lulzx/zeno.git
cd zeno
zig build -Doptimize=ReleaseFast
zig build test
```

For Python:

```sh
python3 -m pip install -e python/
```

```python
import numpy as np
import zeno

env = zeno.make("ant.xml", num_envs=1024)
obs = env.reset()

for _ in range(1000):
    actions = np.zeros((env.num_envs, env.action_dim), dtype=np.float32)
    obs, rewards, dones, info = env.step(actions)

env.close()
```

The API also supports asynchronous submission, masked stepping, fused reset,
GPU-side task outputs, Gymnasium vector environments, and Stable-Baselines3.

## Read next

- [Documentation index](docs/index.md)
- [Install](docs/getting-started/installation.md)
- [Five-minute start](docs/getting-started/quickstart.md)
- [Status and scope](docs/status.md)
- [Python API](docs/guide/python-api.md)
- [Zig API](docs/guide/zig-api.md)
- [Architecture](docs/reference/architecture.md)
- [Performance](docs/reference/performance.md)
- [Physics model](docs/reference/physics.md)

## License

[MIT](LICENSE)
