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
    parser.add_argument("--nvidia-gpu", action="store_true", help="Install CUDA libraries for NVIDIA GPUs")
    parser.add_argument("--skip-models", action="store_true",
                        help="Do not download/convert the Korean model now (it is prepared on first use)")
    parser.add_argument("--register-claude", action="store_true", help="Register Claude Desktop on Windows or macOS")
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.error("Python 3.10 or newer is required.")
    system = platform.system()
    if args.intel_gpu and (system not in ("Windows", "Linux")
                           or platform.machine().lower() not in ("amd64", "x86_64")):
        parser.error("--intel-gpu requires x86-64 Windows or Linux.")
    if args.nvidia_gpu and (system not in ("Windows", "Linux")
                            or platform.machine().lower() not in ("amd64", "x86_64")):
        parser.error("--nvidia-gpu requires x86-64 Windows or Linux.")
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
    if args.nvidia_gpu:
        # CTranslate2(faster-whisper)용 cuBLAS·cuDNN. CUDA Toolkit을 따로 설치하지 않아도 됨
        run("-m", "pip", "install", "nvidia-cublas-cu12", "nvidia-cudnn-cu12>=9,<10")
    if not args.skip_models:
        # 첫 변환 때 몇 분씩 멈춘 것처럼 보이지 않도록, 한국어 모델을 미리 받아서 변환해 둠
        run("-c", "import transcribe as t; t.ensure_ko_model()")
        if args.intel_gpu:
            run("-c", "import transcribe as t; d = t.ov_device(); print('OpenVINO GPU:', d); "
                      "d and t.ensure_ko_ov_model()")
        if args.nvidia_gpu:
            run("-c", "import transcribe as t; print('CUDA GPU:', t.cuda_available())")
    if args.register_claude:
        run(str(ROOT / "register_claude.py"))

    print(f'Run: "{python}" "{ROOT / "transcribe.py"}" --choose-model')


if __name__ == "__main__":
    main()
