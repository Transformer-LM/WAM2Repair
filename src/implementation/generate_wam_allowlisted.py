"""Generate an action-conditioned FastWAM future from one allowlisted input.

This program intentionally has no target/bank argument and rejects any input
other than an exact SHA-bound ``current_rgb, proprio, actions`` NPZ.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from PIL import Image

ROOT = Path('__WAM2REPAIR_ROOT__'); FASTWAM = ROOT / 'workspace/FastWAM'
sys.path.insert(0, str(FASTWAM))

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--input-sha256',required=True)
    p.add_argument('--language',required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--steps',type=int,default=2); p.add_argument('--seed',type=int,default=7)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    if a.steps < 1: raise ValueError('--steps must be positive')
    if not a.input.is_file() or sha(a.input)!=a.input_sha256: raise ValueError('allowlisted input file SHA mismatch')
    with np.load(a.input,allow_pickle=False) as d:
        if set(d.files)!={'current_rgb','proprio','actions'}: raise ValueError('input keys violate allowlist contract')
        rgb=d['current_rgb'].copy(); proprio=d['proprio'].copy(); actions=d['actions'].copy()
    if rgb.shape!=(224,448,3) or proprio.shape!=(8,) or actions.shape!=(8,7): raise ValueError('input tensor contract mismatch')
    if not np.isfinite(proprio).all() or not np.isfinite(actions).all(): raise ValueError('nonfinite input')
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    cache=ROOT/'cache/fastwam/text_embeds/p027c_libero'/(hashlib.sha256(DEFAULT_PROMPT.format(task=a.language).encode()).hexdigest()+'.t5_len128.wan22ti2v5b.pt')
    if not cache.is_file(): raise FileNotFoundError(cache)
    with initialize_config_dir(version_base=None,config_dir=str(FASTWAM/'configs')):
        cfg=compose(config_name='sim_libero',overrides=['task=p027c_libero_videoac_learn500','model.skip_dit_load_from_pretrain=false','model.load_text_encoder=false','model.action_dit_pretrained_path=__FASTWAM_ACTION_DIT_CHECKPOINT__'])
    model=instantiate(cfg.model,model_dtype=torch.bfloat16,device='cuda:0').eval()
    if model.model_paths['video_dit']=='SKIPPED_PRETRAIN': raise ValueError('pretrained video DiT was skipped')
    ckpt=FASTWAM/'runs/policy_relevant_imagined_state_repair_5000/checkpoints/weights/step_005000_trainable_only.pt'
    if not ckpt.is_file(): raise FileNotFoundError(ckpt)
    state=torch.load(ckpt,map_location='cpu',weights_only=False)
    incompat=model.load_state_dict(state['state_dict'],strict=False)
    if incompat.unexpected_keys: raise ValueError(incompat.unexpected_keys)
    processor=instantiate(cfg.data.train.processor).eval(); processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(ROOT/'results/israc/P027C_LIBERO_COMBINED_NORM_STATS.json')))
    norm=processor.normalizer.forward({'action':{'default':torch.from_numpy(actions)[None]},'state':{'default':torch.from_numpy(proprio)[None]}})
    text=torch.load(cache,map_location='cpu',weights_only=False); context=text['context'].clone(); context[~text['mask'].bool()]=0
    image=torch.from_numpy(rgb).permute(2,0,1)[None].to('cuda',torch.bfloat16)/127.5-1
    start=time.time()
    with torch.inference_mode():
        pred=model.infer_joint(prompt=None,input_image=image,num_video_frames=5,action_horizon=8,action=norm['action']['default'].to('cuda',torch.bfloat16),proprio=norm['state']['default'].to('cuda',torch.bfloat16),context=context,context_mask=torch.ones_like(text['mask'],dtype=torch.bool),num_inference_steps=a.steps,seed=a.seed,rand_device='cpu',tiled=False,test_action_with_infer_action=False)
    frames=np.stack([np.asarray(x) for x in pred['video']])
    if frames.shape!=(5,224,448,3) or not np.isfinite(frames).all(): raise ValueError(f'bad WAM video {frames.shape}')
    video=a.out/'raw_wam_video.npz'; np.savez_compressed(video,frames=frames)
    Image.fromarray(np.concatenate(list(frames),axis=0)).save(a.out/'raw_wam_frames.png')
    report={'schema':'wam2repair-allowlisted-wam-generation-v1','status':'GENERATED_INPUT_ONLY_REQUIRES_FACTUAL_EVAL','scope':'one action-conditioned WAM generation; no accuracy/repair/VLA claim','input_path':str(a.input),'input_sha256':sha(a.input),'input_allowlist':[str(a.input)],'privileged_target_access':'none: CLI has no target/bank argument','language':a.language,'action_horizon':8,'video_frames':5,'steps':a.steps,'seed':a.seed,'checkpoint_sha256':sha(ckpt),'config_root_sha256':sha(FASTWAM/'configs/sim_libero.yaml'),'normalizer_stats_sha256':sha(ROOT/'results/israc/P027C_LIBERO_COMBINED_NORM_STATS.json'),'text_cache_sha256':sha(cache),'model_paths':model.model_paths,'missing_key_count':len(incompat.missing_keys),'video':video.name,'video_sha256':sha(video),'elapsed_seconds':time.time()-start,'code_sha256':sha(Path(__file__))}
    (a.out/'result.json').write_text(json.dumps(report,indent=2,default=str)); print(json.dumps(report,default=str),flush=True)
if __name__=='__main__': main()
