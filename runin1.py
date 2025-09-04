# run_all_bots.py
import subprocess

bots = [
    "D:\\Work\\cbktest.py",
    "D:\\Work\\jitest.py",
    "D:\\Work\\wikitest.py",
    "D:\\Work\\yictest.py"
]

processes = []

try:
    for bot in bots:
        print(f"🚀 Starting {bot}...")
        p = subprocess.Popen(["python", bot])
        processes.append(p)

    # Wait for all bots to run forever
    for p in processes:
        p.wait()
except KeyboardInterrupt:
    print("\n🛑 Stopping all bots...")
    for p in processes:
        p.terminate()