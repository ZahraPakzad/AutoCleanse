import subprocess
import concurrent.futures
from multiprocessing import Process

def run_main(args):
    formatted_args = [f"-l {args[0]}", f"-w {args[1]}"]
    command = ["python", "AutoEncoder/main.py"] + formatted_args
    with open(f"{args[0]}_{args[1]}.txt", "w") as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.PIPE, text=True)
        _, stderr = process.communicate()
        if stderr:
            print(f"Error occurred for args {args}: {stderr}")

if __name__ == "__main__":
    # Define the list of argument sets
    argument_sets = [
        ["1024,128", "1,1"], 
        ["1024,128", "1,5"],
        ["1024,128", "1,20"],
        ["1024,128", "1,40"],
    ]

    # Create a list to hold the processes
    processes = []

    # Start a separate process for each set of arguments
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(run_main, argument_sets)