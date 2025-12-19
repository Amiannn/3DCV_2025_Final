 # 3D reconstruction

```bash

docker run --rm --gpus all --net=host \
  --entrypoint /bin/bash \
  -v ~/Desktop/mast3r:/mast3r \
  -v ~/Desktop/mast3r/docker/files/checkpoints:/mast3r/checkpoints \
  -w /mast3r \
  docker-mast3r-demo \
  -c "python3 -u demo.py --weights checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth --device cuda --local_network"

```
url : http://0.0.0.0:7860
# Image Pair Key points matching

```bash

docker run --rm --gpus all --net=host \
 --entrypoint /bin/bash \
 -v ~/Desktop/mast3r:/mast3r  \
 -v ~/Desktop/mast3r/docker/files/checkpoints:/mast3r/checkpoints  \
 -w /mast3r   docker-mast3r-demo  \
 -c "python3 -u pair_test.py"

```


```bash

docker run --rm --gpus all --net=host   --entrypoint /bin/bash   -v ~/Desktop/mast3r:/mast3r   -v ~/Desktop/mast3r/docker/files/checkpoints:/mast3r/checkpoints   -w /mast3r   docker-mast3r-demo   -c "python3 heatmap.py"
```


python eval_rel_pose.py mapfree/outputs_json_corr_v7/   --output aggregated_stats.json   --plot-dir results_folder/plots   --pair-csv results_folder/errors.csv