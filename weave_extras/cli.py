from pathlib import Path
from streamlit.web import cli

def main():
    entrypoint = Path(__file__).parent / "main.py"
    cli.main_run([str(entrypoint), "--server.port", "6637",
        "--server.port=6637", "--server.address=0.0.0.0"])

if __name__ == "__main__":
    main()