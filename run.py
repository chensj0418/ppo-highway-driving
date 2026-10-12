"""Run: python run.py smoke|train|evaluate|demo."""

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from continuous_highway_env import ContinuousHighwayEnv
from datetime import datetime


ROOT = Path(__file__).resolve().parent
# MODEL = ROOT / "models" / "ppo_continuous"
# MODEL = ROOT / "models" / "ppo_continuous_mixed"


MODEL = None
CONFIG_PATH = None


def select_model(mode, model_name=None):
    """训练时创建新名称；测试时选择指定模型或最新模型。"""
    global MODEL, CONFIG_PATH

    folder = ROOT / "models"
    folder.mkdir(exist_ok=True)

    if mode == "train":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        MODEL = folder / f"ppo_lane_{timestamp}.zip"

    elif mode in ("demo", "evaluate"):
        if model_name:
            MODEL = folder / model_name

            if not MODEL.is_file():
                raise FileNotFoundError(f"找不到模型：{MODEL}")

        else:
            # 只选择具有配套配置文件的模型
            candidates = [
                path
                for path in folder.glob("ppo_lane_*.zip")
                if path.with_name(
                    f"{path.stem}_config.json"
                ).is_file()
            ]

            if not candidates:
                raise FileNotFoundError(
                    "没有找到新格式的模型及配置，请先运行 train"
                )

            MODEL = max(
                candidates,
                key=lambda path: path.stat().st_mtime_ns,
            )

    else:
        # smoke 不需要模型
        return

    CONFIG_PATH = MODEL.with_name(
        f"{MODEL.stem}_config.json"
    )

    print("本次模型：", MODEL.name)
    print("配套配置：", CONFIG_PATH.name)



def set_front_slow_scenario(env):
    """将现有车辆调整为前方慢车、后方较快车辆。"""
    base = env.unwrapped
    ego = base.vehicle

    # 获取自车当前车道上的前后车辆
    front, rear = base.road.neighbour_vehicles(
        ego, ego.lane_index
    )

    if front is None or rear is None:
        raise RuntimeError("当前车道缺少前车或后车")

    lane = base.road.network.get_lane(ego.lane_index)
    ego_s, _ = lane.local_coordinates(ego.position)

    # 前车距自车60米，速度20；自车初始速度30
    front.position = lane.position(ego_s + 60, 0)
    front.heading = lane.heading_at(ego_s + 60)
    front.speed = 20.0
    front.target_speed = 20.0
    front.enable_lane_change = False

    # 后车距自车100米，速度35，增加变道时观察后方的必要性
    rear.position = lane.position(ego_s - 100, 0)
    rear.heading = lane.heading_at(ego_s - 100)
    rear.speed = 35.0
    rear.target_speed = 35.0

    # 修改了车辆状态，必须重新计算观测
    obs = base.observation_type.observe()
    info = base._info(obs, action=1)
    return obs, info








class MixedTrainingEnv(ContinuousHighwayEnv):
    """每轮随机选择普通交通或固定慢车场景。"""
    def _info(self, obs, action=None):
        info = super()._info(obs, action)

        # 构造环境时也会调用此方法，因此提供默认值
        info["scenario_type"] = getattr(
            self, "scenario_type", "normal"
        )
        return info

    def reset(self, *, seed=None, options=None):
        # 首先按照原环境生成道路、车辆和初始观测
        obs, info = super().reset(seed=seed, options=options)

        # 这个参数表示慢车场景的概率，默认50%
        probability = self.config.get("slow_front_probability", 0.5)

        # 使用环境自带随机数生成器，保证种子可复现
        is_slow_front = bool(self.np_random.random() < probability)

        if is_slow_front:
            # 调整已有前后车，并重新生成观测
            obs, info = set_front_slow_scenario(self)

        # 记录这一轮属于哪种场景，方便检查
        self.scenario_type = (
            "front_slow" if is_slow_front else "normal"
        )
        info["scenario_type"] = self.scenario_type

        return obs, info










def make_env(render_mode=None, saved_config=False):
    config = {}

    if saved_config:
        config = json.loads(
            CONFIG_PATH.read_text(encoding="utf-8")
        )["env_config"]


    # 展示时覆盖训练保存的无窗口设置
    if render_mode == "human":
        config["offscreen_rendering"] = False
        config["real_time_rendering"] = True

    return ContinuousHighwayEnv(
        config=config,
        render_mode=render_mode,
    )


def smoke():
    env = ContinuousHighwayEnv(
        config={
            "duration": 120,             # 每轮最多120秒仿真时间
            "real_time_rendering": True, # 按接近现实时间播放
        },
        render_mode="human",
    )

    try:
        # 连续展示5轮，每轮使用不同场景种子
        for episode in range(5):
            obs, info = env.reset(seed=42 + episode)
            print(f"\n第 {episode + 1} 轮开始")
            step = 0

            while True:
                # IDLE：保持目标车道和速度，不是停车
                obs, reward, terminated, truncated, info = env.step(1)

                step += 1

                if step % 10 == 0:
                    print(
                        f"仿真时间={env.time:.1f}秒，"
                        f"速度={info['speed']:.1f}m/s，"
                        f"前车数={info['traffic_front']}，"
                        f"后车数={info['traffic_rear']}"
                    )

                if terminated or truncated:
                    if info["crashed"]:
                        reason = "发生碰撞"
                    elif terminated:
                        reason = "驶离道路"
                    else:
                        reason = "达到时间上限"

                    print(f"本轮结束：{reason}")
                    break
    finally:
        env.close()


def train(steps, seed):
    (ROOT / "models").mkdir(exist_ok=True)
    logs = ROOT / "results"

    logs.mkdir(exist_ok=True)
    # env = Monitor(make_env(), str(logs / f"train_{seed}"))

    training_env = MixedTrainingEnv(
        config={"slow_front_probability": 0.0}
    )

    env = Monitor(
        training_env,
        str(logs / f"{MODEL.stem}_train"),
        info_keywords=("scenario_type",),
    )




    model = PPO(
        "MlpPolicy", env, learning_rate=3e-4, n_steps=1024,
        batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01, verbose=1, seed=seed, device="cpu",
    )
    try:
        model.learn(total_timesteps=steps)

        # 保存 PPO 模型：网络参数以及模型相关设置
        model.save(str(MODEL))

        # 保存对应的环境配置，保证测试时使用相同的环境设置
        CONFIG_PATH.write_text(
            json.dumps({
                "seed": seed,
                "actual_steps": model.num_timesteps,
                "env_config": env.unwrapped.config,
            }, indent=2),
            encoding="utf-8",
        )

        print("训练完成，模型与配置已保存")
        # (ROOT / "models" / "config_mixed.json").write_text(
        #     json.dumps({"seed": seed, "actual_steps": model.num_timesteps,
        #                 "env_config": env.unwrapped.config}, indent=2),
        #     encoding="utf-8",
        # )
    finally:
        env.close()


def evaluate(episodes):
    model = PPO.load(str(MODEL), device="cpu")
    results = {}
    for method in ("random", "idle", "ppo"):
        env = make_env(saved_config=True)
        rows = []
        for seed in range(1000, 1000 + episodes):
            # obs, info = env.reset(seed=seed)
            # env.action_space.seed(seed)

            obs, info = env.reset(seed=seed)

            # 在每轮开始时设置相同类型的风险场景
            # obs, info = set_front_slow_scenario(env)

            env.action_space.seed(seed)



            rewards, speeds, violations, dangers = [], [], [], []
            while True:
                if method == "ppo":
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(action)
                else:
                    action = env.action_space.sample() if method == "random" else 1
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)
                speeds.append(info["speed"])
                violations.append(info["speed_violation"])
                dangers.append(info["rewards"]["front_risk"] > 0)
                if terminated or truncated:
                    break
            rows.append({
                "seed": seed, "return": float(sum(rewards)),
                "mean_speed": float(np.mean(speeds)),
                "speed_violation_fraction": float(np.mean(violations)),
                "front_risk_fraction": float(np.mean(dangers)),
                "crashed": bool(info["crashed"]),
                "duration_seconds": env.unwrapped.time,
                "distance": float(info["distance"]),
                "success": bool(info["success"]),
            })
        env.close()
        results[method] = rows
        print(method, {
            key: round(float(np.mean([row[key] for row in rows])), 3)
            for key in rows[0] if key != "seed"
        })
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    # (output / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # report_path = output / f"evaluation_front_slow_{timestamp}.json"
    report_path = output / f"evaluation_random_traffic_{timestamp}.json"

    report_path.write_text(
        json.dumps({
            "model": MODEL.name,
            "config_file": CONFIG_PATH.name,
            # "scenario": "front_slow",
            "scenario": "random_traffic",
            "results": results,
        }, indent=2),
        encoding="utf-8",
    )

    print("结果保存到：", report_path)


def demo(seed):
    env = make_env("human", saved_config=True)

    try:
        model = PPO.load(str(MODEL), device="cpu")

        # 创建初始交通，再设置前方慢车场景
        obs, info = env.reset(seed=seed)
        # obs, info = set_front_slow_scenario(env)

        while True:
            # 根据当前观测选择一个动作
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            # 执行动作，获得新观测、奖励和结束标志
            obs, reward, terminated, truncated, info = env.step(action)

            print(
                f"动作={env.unwrapped.action_type.actions[action]}，"
                f"速度={info['speed']:.1f}，"
                f"前车风险={info['rewards']['front_risk']:.2f}，"
                f"当前车道={env.unwrapped.vehicle.lane_index[2]}，"
                f"目标车道={env.unwrapped.vehicle.target_lane_index[2]}"
                f"剩余时间={info['remaining_time']:.1f}秒，"
                f"所需均速={info['required_average_speed']:.1f}m/s，"
                f"低速计时={info['slow_lane_seconds']:.1f}秒，"
                f"低速占道处罚={info['slow_lane_stay_penalty']:.2f}，"
            )

            # 碰撞、任务完成或达到时间上限后停止
            if terminated or truncated:
                print("crashed:", info["crashed"])
                print("success:", info.get("success", False))
                break
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["smoke", "train", "evaluate", "demo"])
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="测试时指定 models 文件夹中的模型文件名",
    )
    args = parser.parse_args()
    select_model(args.mode, args.model)
    if args.steps <= 0 or args.episodes <= 0:
        parser.error("steps and episodes must be positive")
    if args.mode == "smoke":
        smoke()
    elif args.mode == "train":
        train(args.steps, args.seed)
    elif args.mode == "evaluate":
        evaluate(args.episodes)
    else:
        demo(args.seed)
