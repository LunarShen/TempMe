# [ICLR 2025 🔥] TempMe: Video Temporal Token Merging for Efficient Text-Video Retrieval
  
#### MSRVTT
For MSRVTT, the official data and video links can be found in [link](http://ms-multimedia-challenge.com/2017/dataset).

For the convenience, the splits and captions can be found in sharing from [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip/),

```shell
wget https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip
```

Besides, the raw videos can be found in sharing from [Frozen in Time](https://github.com/m-bain/frozen-in-time), i.e.,

```shell
wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
```

###  Train on MSR-VTT

We conduct experiments on 4 A100x40G GPUs on MSR-VTT, in 2.2.0+cu118 Pytorch.

```shell
bash scripts/MSRVTT.sh
```
