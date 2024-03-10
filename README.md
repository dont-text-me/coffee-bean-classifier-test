# This project contains a python script allowing to make a prediction about a bean's defects.

## Getting started

### Check presence of pip
```shell
python3 -m ensurepip
```

### Install dependencies
```shell
pip install -r requirements.txt
```
### Run model
```shell
python3 predict-bean.py --weights-file model.pt --input-image data/burnt/brazil-catuai-nat-burnt-0-0.png
```
### 