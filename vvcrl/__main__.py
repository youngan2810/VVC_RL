from pathlib import Path
import argparse,json
from .config import Profile,RewardConfig,PPOConfig

def main():
    p=argparse.ArgumentParser(description='VVC QPA residual PPO; training is opt-in and full-preflight gated')
    p.add_argument('command',choices=['native-smoke','preflight','train-mini','diagnose','export','import'])
    p.add_argument('--output',required=True);p.add_argument('--toolchain');p.add_argument('--input',default='/kaggle/input');p.add_argument('--tools',default='/kaggle/working/vvc_rl_tools')
    p.add_argument('--frames',type=int,default=256);p.add_argument('--preflight');p.add_argument('--updates',type=int,default=10);p.add_argument('--val-every',type=int,default=5);p.add_argument('--hours',type=float,default=10)
    p.add_argument('--reward-version',choices=['task-budget-v2','paired-linear-v1'],default='task-budget-v2')
    p.add_argument('--rate-tolerance',type=float,default=.02);p.add_argument('--seed',type=int,default=20261120)
    p.add_argument('--gamma',type=float,default=1.0);p.add_argument('--gae-lambda',type=float,default=.95)
    p.add_argument('--train-schedule',choices=['blocked','mixed'],default='mixed')
    p.add_argument('--resume');p.add_argument('--package');p.add_argument('--start-mini-training',action='store_true');a=p.parse_args()
    if a.command=='export':
        from .artifacts import export_run
        export_run(a.output);return
    if a.command=='import':
        from .artifacts import import_run
        if not a.package:p.error('--package required')
        import_run(a.package,a.output);return
    if not a.toolchain:p.error('--toolchain required')
    t=json.loads(Path(a.toolchain).read_text());root=Path(__file__).resolve().parents[1]
    if a.command=='native-smoke':
        from .native_smoke import run_smoke
        print(json.dumps(run_smoke(t,a.output),indent=2));return
    profile=Profile(frames=a.frames).validate();reward_config=RewardConfig(a.reward_version,a.rate_tolerance).validate()
    ppo_config=PPOConfig(gamma=a.gamma,gae_lambda=a.gae_lambda).validate()
    if a.command=='preflight':
        from .preflight import full_preflight
        full_preflight(root,a.output,t,a.input,a.tools,profile,seed=a.seed,reward_config=reward_config,ppo_config=ppo_config,train_schedule=a.train_schedule);return
    if not a.preflight:p.error('--preflight required')
    if a.command=='diagnose':
        if not a.resume:p.error('--resume requires a trained checkpoint')
        from .diagnose import diagnose
        diagnose(root,a.output,a.preflight,t,a.input,a.tools,a.resume,profile,reward_config=reward_config,ppo_config=ppo_config,train_schedule=a.train_schedule);return
    if not 0<a.hours<=11.75:p.error('--hours must be in (0,11.75]')
    from .train import mini_train
    print(mini_train(root,a.output,a.preflight,t,a.input,a.tools,profile=profile,updates=a.updates,val_every=a.val_every,resume=a.resume,authorized=a.start_mini_training,hours=a.hours,seed=a.seed,reward_config=reward_config,ppo_config=ppo_config,train_schedule=a.train_schedule))

if __name__=='__main__':main()
