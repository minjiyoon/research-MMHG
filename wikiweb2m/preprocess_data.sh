ulimit -c unlimited
module load cuda-11.1.1

export PYTHONPATH=.

python wikiweb2m/preprocess_data.py
