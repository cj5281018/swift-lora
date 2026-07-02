import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.model import get_model_processor
from swift.dataset import load_dataset, EncodePreprocessor
from swift import get_template
from swift.utils import (add_version_to_work_dir, find_all_linears, get_logger,
                           get_model_parameter_info, plot_images, seed_everything)
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from functools import partial

logger = get_logger()
seed_everything(42)

# 模型
model_id_or_path = 'models/Qwen2.5-7B-Instruct'  # model_id or model_path
system = 'You are a helpful assistant.'
output_dir = 'output'  # 如果要和 CLI 一样加时间戳，改为例如 'output/v9-self-cog'

# 数据集
dataset = ['AI-ModelScope/alpaca-gpt4-data-zh#500', 'AI-ModelScope/alpaca-gpt4-data-en#500',
           'swift/self-cognition#500']  # dataset_id or dataset_path
data_seed = 42
max_length = 2048
split_dataset_ratio = 0.01  # 切分验证集
num_proc = 4  # 预处理的进程数
# 替换自我认知数据集中的填充符：{{NAME}}, {{AUTHOR}}
model_name = ['小羽毛', 'Xiao Yumao']  # 模型的中文名和英文名
model_author = ['陈清河', 'Chen Qinghe']  # 模型作者的中文名和英文名

# lora
lora_rank = 8
lora_alpha = 32

# 训练超参数
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_checkpointing=True,
    weight_decay=0.1,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    report_to=['tensorboard'],
    logging_first_step=True,
    save_strategy='steps',
    save_steps=50,
    eval_strategy='steps',
    eval_steps=50,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    metric_for_best_model='loss',
    save_total_limit=2,
    logging_steps=5,
    dataloader_num_workers=1,
    data_seed=data_seed,
)

output_dir = add_version_to_work_dir(output_dir)  # 与 CLI 一致，自动加 v<N>-时间戳
output_dir = os.path.abspath(os.path.expanduser(output_dir))
training_args.output_dir = output_dir
training_args.logging_dir = os.path.join(output_dir, 'runs')
logger.info(f'output_dir: {output_dir}')

model, processor = get_model_processor(model_id_or_path)
logger.info(f'model_info: {model.model_info}')
template = get_template(processor, default_system=system, max_length=max_length)
template.set_mode('train')

target_modules = find_all_linears(model)
lora_config = LoraConfig(task_type='CAUSAL_LM', r=lora_rank, lora_alpha=lora_alpha,
                         target_modules=target_modules)
model = Swift.prepare_model(model, lora_config)
logger.info(f'lora_config: {lora_config}')

# 打印模型结构和训练的参数量
logger.info(f'model: {model}')
model_parameter_info = get_model_parameter_info(model)
logger.info(f'model_parameter_info: {model_parameter_info}')

train_dataset, val_dataset = load_dataset(dataset, split_dataset_ratio=split_dataset_ratio, num_proc=num_proc,
        model_name=model_name, model_author=model_author, seed=data_seed)

logger.info(f'train_dataset: {train_dataset}')
logger.info(f'val_dataset: {val_dataset}')
logger.info(f'train_dataset[0]: {train_dataset[0]}')

train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=num_proc)
val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=num_proc)
logger.info(f'encoded_train_dataset[0]: {train_dataset[0]}')

# 打印一条样本
template.print_inputs(train_dataset[0])

model.enable_input_require_grads()  # 兼容gradient checkpointing
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    template=template,
)
trainer.train()

last_model_checkpoint = trainer.state.last_model_checkpoint
logger.info(f'last_model_checkpoint: {last_model_checkpoint}')


images_dir = os.path.join(output_dir, 'images')
logger.info(f'images_dir: {images_dir}')
plot_images(images_dir, os.path.join(output_dir, 'runs'), ['train/loss'], 0.9)  # 保存图片

# 展示图片
loss_plot = os.path.join(images_dir, 'train_loss.png')
if os.path.exists(loss_plot):
    from PIL import Image
    image = Image.open(loss_plot)
    logger.info(f'loss plot saved to {images_dir}')
else:
    logger.warning(f'loss plot not found at {loss_plot}, check tensorboard logs at {os.path.join(output_dir, "runs")}')