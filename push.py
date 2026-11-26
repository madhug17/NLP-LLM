import subprocess
import sys

def git_push():
    # Accept message from command line argument or prompt the user
    if len(sys.argv) > 1:
        commit_msg = " ".join(sys.argv[1:])
    else:
        commit_msg = input("Enter commit message: ").strip()

    if not commit_msg:
        print("Error: Commit message cannot be empty.")
        sys.exit(1)

    commands = [
        ["git", "add", "."],
        ["git", "commit", "-m", commit_msg],
        ["git", "push", "origin", "main"]
    ]

    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Stopped: Command failed with exit code {result.returncode}")
            sys.exit(result.returncode)

if __name__ == "__main__":
    git_push()
