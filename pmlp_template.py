from openrec import ModelTrainer
from openrec.utils import Dataset
from PMLP import PMLP
from openrec.utils.evaluators import AUC, Recall
from openrec.utils.samplers import StratifiedPointwiseSampler
from openrec.utils.samplers import EvaluationSampler
import dataloader

#raw_data = dataloader.load_citeulike()
raw_data = dataloader.load_tradesy()
dim_embed = CHANGE_DIM_HERE
total_iter = 20000
batch_size = 1000
eval_iter = total_iter
save_iter = eval_iter

train_dataset = Dataset(raw_data['train_data'], raw_data['total_users'], raw_data['total_items'], name='Train')
val_dataset = Dataset(raw_data['val_data'], raw_data['total_users'], raw_data['total_items'], name='Val', num_negatives=500)
test_dataset = Dataset(raw_data['test_data'], raw_data['total_users'], raw_data['total_items'], name='Test', num_negatives=500)

train_sampler = StratifiedPointwiseSampler(pos_ratio=0.2, batch_size=batch_size, dataset=train_dataset, num_process=5)
val_sampler = EvaluationSampler(batch_size=batch_size, dataset=val_dataset)
test_sampler = EvaluationSampler(batch_size=batch_size, dataset=test_dataset)

model = PMLP(batch_size=batch_size, total_users=train_dataset.total_users(), total_items=train_dataset.total_items(), 
             l2_reg=CHANGE_L2_REG_HERE,
             mlp_dims=CHANGE_MLP_DIMS_HERE, dim_user_embed=dim_embed, dim_item_embed=dim_embed, save_model_dir='pmlp_recommender/', train=True, serve=True)

auc_evaluator = AUC()
recall_evaluator = Recall(recall_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])    
model_trainer = ModelTrainer(model=model)

model_trainer.train(total_iter=total_iter, eval_iter=eval_iter, save_iter=save_iter, train_sampler=train_sampler, 
                    eval_samplers=[val_sampler], evaluators=[auc_evaluator])

