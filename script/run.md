python script/train.py --cfg_file /home/duyushi/NSM-modified/config/train_all_baseline.yml

python script/train.py --cfg_file /home/duyushi/NSM-modified/config/train_all_vnn.yml

python script/train.py --cfg_file /home/duyushi/NSM-modified/config/train_bag_vnn.yml

python script/vis.py --cfg_file /home/duyushi/NSM-modified/config/train_vnn.yml \
--weight log/vnn/last.ckpt