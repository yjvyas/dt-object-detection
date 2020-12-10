FROM pytorch/pytorch
RUN apt-get update && apt install libgl1-mesa-glx libglib2.0-0 -y
RUN pip install opencv-python numpy pillow tqdm
WORKDIR /workdir
COPY model/coco_eval.py coco_eval.py
COPY model/coco_utils.py coco_utils.py
COPY model/dataset.py dataset.py
COPY model/model.py model.py
COPY model/weights weights

RUN ls

# Don't forget to instantiate your model in case it needs to download a pretrained backend!
RUN python3 -c "from model import Wrapper; wrapper = Wrapper();"