# Install requirements
cd /gruntdata/rs_nas/workspace/xingsu.zq/SkySense-O         
pip install -r require.txt -i https://artifacts.antgroup-inc.cn/simple/
pip install accelerate -U
git pull

# Config
num_gpus=4
workers=4
debug=false
train_once=true
batch_per_gpu=1
eval_period=4000
model_name=skysense_o
config_path=configs/$model_name.yaml
ims_per_batch=$((batch_per_gpu * num_gpus))
ims_per_batch=4
name="base"
weights=None
scale=1
vc_regular=0.0  # 0.4
region_contrast=0.0  # 0.4
retrieval_aug=False
sampling_hyper="[1,3,9,10]"
text_batch=100

# Get opt
while getopts "n:v:r:s:w:adh:t:" opt; do
  case $opt in
    n) name="$OPTARG" ;;
    v) vc_regular="$OPTARG" ;;
    r) region_contrast="$OPTARG" ;;
    s) scale="$OPTARG" ;;
    w) weights="$OPTARG" ;;
    a) retrieval_aug=True ;;
    d) debug=true ;;
    h) sampling_hyper="$OPTARG" ;;
    t) text_batch="$OPTARG" ;;
    \?) echo "Not find opt: -$OPTARG" >&2 ;;
  esac
done

# Training
if [ "$debug" = "true" ]; then
    workers=1
    num_gpus=1
    eval_period=8
    model_name=$model_name\_debug
    ims_per_batch=$((batch_per_gpu * num_gpus))
fi

python train_net.py --dist-url 'auto' \
                    --config-file $config_path \
                    --num-gpus $num_gpus \
                    OUTPUT_DIR output/$name \
                    SOLVER.IMS_PER_BATCH $ims_per_batch \
                    DATASETS.TRAIN \(\'skysa\_graph\_train\',\) \
                    DATASETS.TEST \(\'isaid_test\',\) \
                    DATALOADER.NUM_WORKERS $workers \
                    TEST.EVAL_PERIOD $eval_period \
                    REGULAR_WEIGHT $vc_regular \
                    REGION_CONTRAST_WEIGHT $region_contrast \
                    CONTRAST_SCALE $scale \
                    DATASETS.RETRIEVAL_AUGMENTATION $retrieval_aug \
                    MODEL.WEIGHTS $weights \
                    SAMPLING_HYPER $sampling_hyper \
                    TEXT_BATCH $text_batch
                    

# DATASETS.TEST \(\'isaid_test\',\'potsdam_test\',\'samrs_fast_test\',\'samrs_sior_test\',\'samrs_sota_test\',\'uavid_test\',\'floodnet_test\',\'loveda_test\',\'vaihingen_test\',\'oem_test\',\) \
# DATASETS.TRAIN \(\'skysa\_graph\_train\',\) \
# DATASETS.TRAIN \(\'sky5k_train\',\) \
