"""Small existing-environment GPU kernel witness, no weights or downloads."""
import json
import jax
import jax.numpy as jnp

devices=jax.devices()
assert devices and all(d.platform=='gpu' for d in devices),devices
x=jax.random.normal(jax.random.PRNGKey(7),(8,8))
y=jax.jit(lambda v:v@v.T)(x)
y.block_until_ready()
assert bool(jnp.isfinite(y).all())
print('WITNESS_JAX_GPU',json.dumps({'jax':jax.__version__,'shape':list(y.shape),'devices':[str(d) for d in devices],'sum':float(y.sum())}),flush=True)
