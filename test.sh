CHECKPOINTS_DIR=../checkpoints/
RESULTS_DIR=../results/

cd colorization-pytorch

python test.py --name siggraph_reg2 \
               --checkpoints_dir $CHECKPOINTS_DIR \
               --results_dir $RESULTS_DIR