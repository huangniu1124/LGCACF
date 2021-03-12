import world
import dataloader
import model
import utils
from pprint import pprint

if world.dataset in ['movielens']:
    dataset = dataloader.Movie(path="../data/"+world.dataset)
elif world.dataset in ['TaoBao']:
    dataset = dataloader.TaoBao(path="../data/"+world.dataset)
elif world.dataset == 'amazon-electronic':
    dataset = dataloader.Amazon(path="../data/"+world.dataset)

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'lgn': model.LightGCN,
    'lgcacf': model.LGCACF
}