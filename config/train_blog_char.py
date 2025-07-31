# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

dataset  = 'dataset/blog_char'
out_dir  = 'out-blog'

eval_interval = 100
eval_iters = 200
log_interval = 10 # don't print too too often

always_save_checkpoint = True

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

gradient_accumulation_steps = 1
batch_size = 32 
block_size = 128

# baby GPT model :)
n_layer = 4
n_head  = 4
n_embd  = 256
dropout = 0.3

learning_rate = 2e-3
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-4
weight_decay = 0.1

beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

device = 'cpu'
compile = False
