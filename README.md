# DPR ZOO of Sentinel-based segmentation models

Repository used for training the DPR Zoo Models.

- [Amazon and Atlantic Forest dataset](https://zenodo.org/records/4498086#.Y6LPLuzP1hE)
- [SATLAS Pretrain dataset](https://huggingface.co/allenai/satlas-pretrain)

## Structure


- `dataset2.py` - Creating a dataset from Satlas pretrained data
- `fat_pretrain_ms.py` - Fine-tuning the Satlas pretrained model on new data


## Preparing Satlas for fine-tuning

Preparing Satlas dataset for training:

1. Unpack all label datasets (or those you want) into `.data/` - [Link to Satlas pretrain datasets](https://huggingface.co/allenai/satlas-pretrain/tree/main/dataset) (make sure to download static labels - e.g. `satlaspretrain_dataset_labels_static_0000.tar`)
2. Unpack a image dataset into `.data/` (e.g. `satlaspretrain_dataset_sentinel2_a_0042.tar`)
3. Run the following command:
```
python .data/prepare_dataset.py --root_dataset_folder .data/satlaspretrain_dataset_sentinel2_a_XXXX/sentinel2 --root_label_folder .data/ --output_csv satlas_sentinel2_a_XXXX.csv
```


> **DPR Team**, 2025
> 
> Made as part of [Arnes Hackathon 2025](https://hackathon.si/).