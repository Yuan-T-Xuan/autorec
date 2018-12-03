from openrec import ModelTrainer
from openrec.utils import Dataset
from BPR import BPR
from openrec.utils.evaluators import AUC
from openrec.utils.samplers import RandomPairwiseSampler
from openrec.utils.samplers import EvaluationSampler
import dataloader

#raw_data = dataloader.load_citeulike()
raw_data = dataloader.load_amazon_book()
dim_embed = CHANGE_DIM_HERE
total_iter = 200000
batch_size = 10000
eval_iter = total_iter
save_iter = eval_iter

train_dataset = Dataset(raw_data['train_data'], raw_data['total_users'], raw_data['total_items'], name='Train')
val_dataset = Dataset(raw_data['val_data'], raw_data['total_users'], raw_data['total_items'], name='Val', num_negatives=500)
test_dataset = Dataset(raw_data['test_data'], raw_data['total_users'], raw_data['total_items'], name='Test', num_negatives=500)

train_sampler = RandomPairwiseSampler(batch_size=batch_size, dataset=train_dataset, num_process=5)
val_sampler = EvaluationSampler(batch_size=batch_size, dataset=val_dataset)
test_sampler = EvaluationSampler(batch_size=batch_size, dataset=test_dataset)

bpr_model = BPR(batch_size=batch_size, total_users=train_dataset.total_users(), total_items=train_dataset.total_items(), 
                l2_reg=CHANGE_L2_REG_HERE,
                dim_user_embed=dim_embed, dim_item_embed=dim_embed, save_model_dir='bpr_recommender/', train=True, serve=True)

model_trainer = ModelTrainer(model=bpr_model)

auc_evaluator = AUC()
model_trainer.train(total_iter=total_iter, eval_iter=eval_iter, save_iter=save_iter, train_sampler=train_sampler, 
                    eval_samplers=[val_sampler], evaluators=[auc_evaluator])

