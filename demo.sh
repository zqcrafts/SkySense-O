cd /gruntdata/rs_nas/workspace/xingsu.zq/SkySense-O         
pip install -r require.txt -i https://artifacts.antgroup-inc.cn/simple/
pip install accelerate -U
git pull

python demo/demo.py --dist-url auto --config-file configs/skysense_o_demo.yaml --dist-url 'auto' --eval-only --num-gpus 4