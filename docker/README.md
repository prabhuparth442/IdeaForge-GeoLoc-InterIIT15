# docker/ : the team image

Environment only (the evaluation mounts `code/` = this pipeline folder, `models/`, `preprocess/` at run time). Based on the starter kit's Dockerfile:
torch 2.8.0 / torchvision 0.23.0 Jetson GPU wheels, numpy 1.26.4, OpenCV 4.10.0.84, frozen by `PIP_CONSTRAINT`, plus `requirements.txt` here.
It must be built `FROM` the published base `uav-geoloc-base:jp6-arm64` (load it first, see `Submission_Files/TeamKit/README.md`).

## Build

On the Jetson (needs Docker with buildx):

```
docker load -i uav-geoloc-base_jp6-arm64.tar.gz
docker build -t myteam:dev Main_Pipeline_Headless/docker
```

On a PC, cross-build for the Orin (needs the one-time setup in `TeamKit/base/BUILD_WITHOUT_JETSON.md`, run from `TeamKit/`):

```
base/build.sh starter-arm64 --context ../Main_Pipeline_Headless/docker --tag myteam:dev
```

## Check on the Jetson

```
docker run --rm --runtime nvidia --entrypoint python3 myteam:dev /opt/starter/gpu_check.py
```

## Run the harness

```
python3 harness/evaluate.py --image myteam:dev --code Main_Pipeline_Headless --models Main_Pipeline_Headless/models \
    --preprocess <store folder> --map <map> --dataset <dataset> --limit 30
```

Not tested yet: the build itself (no Docker/Jetson here), and the pipeline on numpy 1.26.4 / OpenCV 4.10 (it was tested with numpy 1.23.5 / OpenCV 4.8.1).
The `docker/` folder is inside `code/`, so it ships in `solution.tar`; it is small.
