"""Install the CLI and MCP server in a local Python virtual environment."""
import argparse
import platform
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intel-gpu", action="store_true", help="Install Intel OpenVINO dependencies")
    parser.add_argument("--register-claude", action="store_true", help="Register Claude Desktop on Windows or macOS")
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.error("Python 3.10 or newer is required.")
    system = platform.system()
    if args.intel_gpu and (system not in ("Windows", "Linux")
                           or platform.machine().lower() not in ("amd64", "x86_64")):
        parser.error("--intel-gpu requires x86-64 Windows or Linux.")
    if args.register_claude and system not in ("Windows", "Darwin"):
        parser.error("Automatic Claude Desktop registration supports Windows and macOS.")

    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if system == "Windows" else "bin/python")
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(environment)

    def run(*arguments):
        subprocess.run([str(python), *arguments], cwd=ROOT, check=True)

    run("-m", "pip", "install", "-r", str(ROOT / "requirements.txt"))
    torch_packages = ("torch", "torchvision") if args.intel_gpu else ("torch",)
    if system in ("Windows", "Linux") and platform.machine().lower() in ("amd64", "x86_64"):
        run("-m", "pip", "install", *torch_packages, "--index-url", "https://download.pytorch.org/whl/cpu")
    else:
        run("-m", "pip", "install", *torch_packages)
    if args.intel_gpu:
        run("-m", "pip", "install", "openvino-genai", "optimum-intel[openvino]", "transformers>=4.51,<5")
    if args.register_claude:
        run(str(ROOT / "register_claude.py"))

    print(f'Run: "{python}" "{ROOT / "transcribe.py"}" --choose-model')


if __name__ == "__main__":
    main()
