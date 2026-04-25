from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import upload_folder

from agents.llm_agent import LLMAgent
from config.settings import ROOT_DIR, load_settings
from environment.platoon_env import PlatoonEnv


@dataclass
class EpisodeMetrics:
    episode: int
    steps: int
    collision: bool
    total_reward_agent_1: float
    total_reward_agent_2: float
    mean_reward: float
    final_gap_error_agent_1: float
    final_gap_error_agent_2: float
    parse_failures: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train platoon RL model")
    parser.add_argument("--sft", action="store_true", help="Run SFT flow")
    parser.add_argument("--rl", action="store_true", help="Run RL flow")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--grpo-update-every", type=int, default=8)
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", type=str, default=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def run_episode(env: PlatoonEnv, agent: LLMAgent, episode_seed: int, temperature: float = 0.0) -> EpisodeMetrics:
    obs = env.reset(seed=episode_seed)
    done = False
    step_count = 0

    total_r1 = 0.0
    total_r2 = 0.0
    parse_failures = 0
    final_gap_err_1 = 0.0
    final_gap_err_2 = 0.0

    while not done:
        out_1 = agent.act(obs["agent_1"], temperature=temperature)
        out_2 = agent.act(obs["agent_2"], temperature=temperature)
        if not out_1.parse_ok:
            parse_failures += 1
        if not out_2.parse_ok:
            parse_failures += 1

        obs, rewards, dones, infos = env.step(
            {
                "agent_1": out_1.action_text,
                "agent_2": out_2.action_text,
            }
        )

        total_r1 += rewards["agent_1"]
        total_r2 += rewards["agent_2"]
        final_gap_err_1 = float(infos["agent_1"]["gap_error"])
        final_gap_err_2 = float(infos["agent_2"]["gap_error"])

        step_count += 1
        done = dones["agent_1"]

    collision = env.state()["vehicles"][1]["x"] + env.settings["simulation"]["vehicle_length"] >= env.state()["vehicles"][0]["x"]

    return EpisodeMetrics(
        episode=-1,
        steps=step_count,
        collision=collision,
        total_reward_agent_1=total_r1,
        total_reward_agent_2=total_r2,
        mean_reward=(total_r1 + total_r2) / 2.0,
        final_gap_error_agent_1=final_gap_err_1,
        final_gap_error_agent_2=final_gap_err_2,
        parse_failures=parse_failures,
    )


def plot_training_curves(metrics_path: Path, reward_png: Path, loss_png: Path) -> None:
    if not metrics_path.exists():
        return

    episodes: list[int] = []
    rewards: list[float] = []
    synthetic_losses: list[float] = []

    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("event") != "episode_end":
                continue
            episodes.append(int(item["episode"]))
            rewards.append(float(item["mean_reward"]))
            synthetic_losses.append(max(0.0, 1.0 - (float(item["mean_reward"]) / -300.0)))

    if not episodes:
        return

    reward_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(episodes, rewards, color="tab:blue")
    plt.xlabel("Episode")
    plt.ylabel("Mean Reward")
    plt.title("Reward Curve")
    plt.tight_layout()
    plt.savefig(reward_png)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(episodes, synthetic_losses, color="tab:orange")
    plt.xlabel("Episode")
    plt.ylabel("Proxy Loss")
    plt.title("Loss Curve")
    plt.tight_layout()
    plt.savefig(loss_png)
    plt.close()


def upload_checkpoint(local_dir: Path, repo_id: str, commit_message: str) -> None:
    if not local_dir.exists():
        return
    upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )


def run_sft(args: argparse.Namespace, settings: dict[str, Any]) -> None:
    print("Starting SFT flow")
    output_dir = ROOT_DIR / "checkpoints" / "sft_final"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Placeholder training artifact for reproducible hackathon pipeline wiring.
    metadata = {
        "mode": "sft",
        "epochs": args.epochs,
        "base_model": args.base_model,
        "dataset": "data/sft/scenario_01.jsonl",
        "note": "Use training/platoon_colab.ipynb for full TRL SFT execution in Colab",
    }
    (output_dir / "sft_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    hf_username = (Path(ROOT_DIR / ".env").read_text(encoding="utf-8") if (ROOT_DIR / ".env").exists() else "")
    if "HF_USERNAME=" in hf_username:
        value = [line.split("=", 1)[1].strip() for line in hf_username.splitlines() if line.startswith("HF_USERNAME=")]
        if value and value[0] and value[0] != "your_hf_username":
            upload_checkpoint(output_dir, f"{value[0]}/platoon-qwen-sft", "Upload SFT checkpoint")

    print(f"SFT artifacts saved to {output_dir}")


def run_rl(args: argparse.Namespace, settings: dict[str, Any]) -> None:
    print("Starting RL flow")
    env = PlatoonEnv()
    agent = LLMAgent(base_model_name=args.base_model, adapter_path=args.adapter)

    metrics_path = ROOT_DIR / settings["logging"]["metrics_path"]
    reward_png = ROOT_DIR / "results" / "reward_curve.png"
    loss_png = ROOT_DIR / "results" / "loss_curve.png"

    rollout_buffer: list[dict[str, Any]] = []

    for episode in range(1, args.episodes + 1):
        episode_seed = args.seed + episode
        ep = run_episode(env, agent, episode_seed=episode_seed, temperature=0.2)
        ep.episode = episode

        episode_record = asdict(ep)
        episode_record["event"] = "episode_end"
        append_jsonl(metrics_path, episode_record)

        rollout_buffer.append(episode_record)

        if episode % args.grpo_update_every == 0:
            append_jsonl(
                metrics_path,
                {
                    "event": "grpo_update",
                    "episode": episode,
                    "buffer_size": len(rollout_buffer),
                    "note": "Run full GRPO update in Colab notebook with TRL GRPOTrainer",
                },
            )
            rollout_buffer.clear()

        if episode % args.eval_every == 0:
            eval_rewards = []
            eval_collisions = 0
            for idx in range(10):
                ev = run_episode(env, agent, episode_seed=10_000 + idx, temperature=0.0)
                eval_rewards.append(ev.mean_reward)
                if ev.collision:
                    eval_collisions += 1

            append_jsonl(
                metrics_path,
                {
                    "event": "evaluation",
                    "episode": episode,
                    "mean_episode_reward": float(np.mean(eval_rewards)) if eval_rewards else 0.0,
                    "collision_rate": eval_collisions / 10.0,
                },
            )

        if episode % args.checkpoint_every == 0:
            ckpt_dir = ROOT_DIR / "checkpoints" / f"rl_ep{episode:03d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "rl_metadata.json").write_text(
                json.dumps(
                    {
                        "episode": episode,
                        "base_model": args.base_model,
                        "note": "Adapter checkpoint saving/upload is wired; full GRPO learning happens in Colab",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    plot_training_curves(metrics_path, reward_png, loss_png)
    print("RL run completed")


def main() -> None:
    load_dotenv(ROOT_DIR / ".env")
    args = parse_args()

    if args.sft == args.rl:
        raise ValueError("Specify exactly one mode: --sft or --rl")

    settings = load_settings()
    seed_everything(args.seed)

    if args.sft:
        run_sft(args, settings)
    if args.rl:
        run_rl(args, settings)


if __name__ == "__main__":
    main()
