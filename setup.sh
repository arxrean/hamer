conda create --name hamer python=3.10

conda activate hamer

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install ninja

sudo dnf install cuda-toolkit-12-8

pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"

pip install --no-build-isolation "git+https://github.com/mattloper/chumpy"

pip install Cython

pip install -e .[all] --no-build-isolation

pip install -v -e third-party/ViTPose

pip install --no-build-isolation --force-reinstall --no-deps --no-binary xtcocotools xtcocotools

bash fetch_demo_data.sh
