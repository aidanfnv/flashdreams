Draft response for https://github.com/NVIDIA/flashdreams/pull/413#discussion_r3717338137

Good point. `rollout_binding` was not clear terminology.

I renamed it to `rollout_context`. I did not use `per_step` or `every_step` because the examples here are rollout-wide context that may be consumed across steps, not necessarily a fresh value supplied at each individual step. The docs now distinguish `global_conditioning`, `rollout_context`, and per-step conditioning more explicitly.
