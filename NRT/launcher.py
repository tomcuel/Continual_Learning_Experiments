import glob
import os 
import subprocess
import sys


def run_test_files():
    """
    Run all test_*.py files in the current directory to proceed to the NRT tests
    """
    test_files = glob.glob(os.path.join(os.path.dirname(__file__), "test_*.py"))
    for test_file in test_files:
        print(f"\n=== Running {os.path.basename(test_file)} ===")
        result = subprocess.run([sys.executable, test_file], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)


if __name__ == "__main__":
    run_test_files()

