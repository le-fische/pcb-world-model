"""CLI Entrypoint for Silent Trench."""
import argparse
import sys
import uvicorn

from silent_trench.train import PPOTrainer


def main():
    parser = argparse.ArgumentParser(description="Silent Trench: Submarine Warfare & Neural AI")
    subparsers = parser.add_subparsers(dest="command")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Launch interactive web server")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port number")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train neural network via self-play PPO")
    train_parser.add_argument("--iters", type=int, default=10, help="Number of self-play iterations")
    train_parser.add_argument("--games", type=int, default=15, help="Games per iteration")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate current neural agent against baseline bot")
    eval_parser.add_argument("--games", type=int, default=20, help="Evaluation matches")

    args = parser.parse_args()

    if args.command == "serve" or args.command is None:
        port = getattr(args, "port", 8000)
        host = getattr(args, "host", "127.0.0.1")
        print(f"\n=======================================================")
        print(f"🌊 SILENT TRENCH: Submarine Warfare Server")
        print(f"👉 Web Console: http://{host}:{port}")
        print(f"=======================================================\n")
        uvicorn.run("silent_trench.server:app", host=host, port=port, reload=False)

    elif args.command == "train":
        trainer = PPOTrainer(checkpoint_dir="checkpoints")
        trainer.train(total_iterations=args.iters, games_per_iter=args.games)

    elif args.command == "eval":
        trainer = PPOTrainer(checkpoint_dir="checkpoints")
        winrate = trainer.evaluate_against_bot(num_games=args.games)
        print(f"\nNeural Agent Win Rate vs Kapitän Bot ({args.games} games): {winrate * 100:.1f}%\n")


if __name__ == "__main__":
    main()
