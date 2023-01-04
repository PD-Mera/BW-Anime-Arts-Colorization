CHECKPOINTS_DIR=../checkpoints/

cd colorization-pytorch

mkdir -p $CHECKPOINTS_DIR

# Train classification network on small training set first
python train.py --name siggraph_class_small \
                --sample_p 1.0 \
                --niter 100 \
                --niter_decay 0 \
                --classification \
                --phase train_small \
                --checkpoints_dir $CHECKPOINTS_DIR

# Train classification network first
mkdir -p $CHECKPOINTS_DIR/siggraph_class
cp $CHECKPOINTS_DIR/siggraph_class_small/latest_net_G.pth $CHECKPOINTS_DIR/siggraph_class/
python train.py --name siggraph_class \
                --sample_p 1.0 \
                --niter 15 \
                --niter_decay 0 \
                --classification \
                --load_model \
                --phase train \
                --checkpoints_dir $CHECKPOINTS_DIR

# Train regression model (with color hints)
mkdir -p $CHECKPOINTS_DIR/siggraph_reg
cp $CHECKPOINTS_DIR/siggraph_class/latest_net_G.pth $CHECKPOINTS_DIR/siggraph_reg/
python train.py --name siggraph_reg \
                --sample_p .125 \
                --niter 10 \
                --niter_decay 0 \
                --lr 0.00001 \
                --load_model \
                --phase train \
                --checkpoints_dir $CHECKPOINTS_DIR

# Turn down learning rate to 1e-6
mkdir -p $CHECKPOINTS_DIR/siggraph_reg2
cp $CHECKPOINTS_DIR/siggraph_reg/latest_net_G.pth $CHECKPOINTS_DIR/siggraph_reg2/
python train.py --name siggraph_reg2 \
                --sample_p .125 \
                --niter 5 \
                --niter_decay 0 \
                --lr 0.000001 \
                --load_model \
                --phase train \
                --checkpoints_dir $CHECKPOINTS_DIR
