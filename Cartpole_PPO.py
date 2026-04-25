import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

def main():
    # Create the environment
    env_id = "CartPole-v1"
    train_env = make_vec_env(env_id, n_envs=4)

    # Create the agent
    model = PPO(
        policy="MlpPolicy",
        env = train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2
    )

    # Train the agent
    total_timesteps = 50_000
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    model.save("cartpole_ppo_2")

    # Evaluate the agent
    eval_env = gym.make(env_id)
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=10, 
        deterministic=True
    )

    print(f"Mean reward: {mean_reward} - Std: {std_reward}")
    eval_env.close()

    # Test it
    render_env = gym.make(env_id, render_mode="human")
    obs, _ = render_env.reset()
    for _ in range(5_000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = render_env.step(action)
        
        if terminated or truncated:
            obs, _ = render_env.reset()

    render_env.close()

if __name__ == "__main__":
    main()