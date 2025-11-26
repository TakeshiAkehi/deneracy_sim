FROM python:3.12

RUN apt-get update && apt-get install --no-install-recommends -y \
    libegl1 \
    libgl1 \
    libgomp1

RUN pip install numpy scipy matplotlib pandas tqdm open3d sophuspy