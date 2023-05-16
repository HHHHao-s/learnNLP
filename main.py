from forgebox.imports import *
from gc_utils import *
import pytorch_lightning as pl
from tqdm import tqdm
from pathlib import Path
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel
)
import random
from typing import List
import re
from jieba import cut
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# 读取
def read_json(path):
    return json.loads(Path(path).read_text())


json_list = list(Path(".\\data").rglob("ci.song.*.json"))
# 读取词放入dict
shi_dict = dict((str(i), pd.read_json(i))
                for i in tqdm(json_list))

# 获取一个n行3列的列表，
# author  paragraphs  rhythmic
#  ....    ........    ......
all_df = pd.concat(list(shi_dict.values()))[["author", "paragraphs", "rhythmic"]].reset_index(drop=True)

# 消除杂项
all_df = all_df.sample(frac=1.).reset_index(drop=True)
para = list(all_df["paragraphs"])[0]


def extract(paragraphs: List[str], puncts="，。？！?,.!"):
    '''
    函数功能：将paragraphs的前2-4句的首字提取出来，换成[CLS]，返回被提取出来的几个字和替换完的paragraphs合起来
    '''
    text = "".join(paragraphs)
    num_head = random.choice([2, 3, 4])
    heads = ""
    return_text = ""
    last_is_break = True
    for i, c in enumerate(text):
        if last_is_break:
            heads += c
            return_text += "[CLS]"
        else:
            return_text += c
        if len(heads) >= num_head:
            return_text += text[i + 1:]
            break
        if c in puncts:
            last_is_break = True
        else:
            last_is_break = False
    return heads, return_text


extract(para)


# 用jieba将诗句切分，将标点替换成空格
def cutting(text):
    return list(i for i in cut(re.sub(r'[^\w\s]', ' ', text), HMM=True, ) if i != ' ')


cutting("会当凌绝顶，一览众山小")


# 将li随机排序，取min_n到max_n的元素
def pick_and_shuffle(li, min_n: int = 0, max_n: int = None):
    if max_n is None:
        max_n = int(len(li) * .7)
    n = min_n + random.randint(0, min(max_n - min_n, 10))
    random.shuffle(li)
    return list(set(li[:n]))


def create_each_word(text):
    '''
    将text分词，返回随机抽取的词
    '''
    return pick_and_shuffle(cutting(text))


# heads, headless = extract(para)

# heads, create_each_word(headless.replace('[CLS]',""))


tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")  # 分词器


class PoetDataset(Dataset):
    def __init__(
            self,
            df,
            tokenizer,
            p_head: float = .2,
    ):
        self.df = df.sample(frac=1).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.p_head = p_head
        self.cn_num_dict = dict((i + 1, f"『{c}』") for i, c in enumerate("一二三四"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        '''

        :param idx: index
        :return:  word-word......<<head>> 『len(head)』 [CLS]... [CLS]... .......
        e.g. :眉黛-无限-船头-儿-一时-青-滩头-愁-眼波-惊鱼《新女》『二』[CLS]妇滩头眉黛愁。[CLS]儿浦口眼波秋。惊鱼错认月沈钩。青箬笠前无限事，绿蓑衣底一时休。斜风吹雨转船头。
        '''
        row = self.df.loc[idx]
        paragraphs = row.paragraphs
        heads, headless = extract(paragraphs)
        words = '-'.join(create_each_word(headless.replace('[CLS]', "")))
        return f"{words}《{heads}》{self.cn_num_dict.get(len(heads))}{headless}"

    def collate_fn(self, batch):
        '''

        :param batch: getitem函数返回的数据项的batch形成的列表
        :return: 编码后的batch
        '''
        texts = list(batch)

        # 将text进行编码
        batch = self.tokenizer(
            list(texts),
            max_length=256,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )

        labels = batch['input_ids'].clone()
        labels[labels == 0] = -100
        batch['labels'] = labels
        return batch

    def dataloader(self, batch_size=32, shuffle=True):
        return DataLoader(
            self,
            batch_size=batch_size,  # 一批处理的词的数量
            shuffle=shuffle,
            collate_fn=self.collate_fn
        )

    def split(self, val_ratio=.05):  # 随机分句
        df = self.df.sample(frac=1).reset_index(drop=True)
        train_df = df[:int(len(df) * (1 - val_ratio))]
        val_df = df[int(len(df) * (1 - val_ratio)):]
        return PoetDataset(train_df, tokenizer=self.tokenizer), \
               PoetDataset(val_df, tokenizer=self.tokenizer)


poet_ds = PoetDataset(all_df, tokenizer)
dl = poet_ds.dataloader(2)
batch = next(iter(dl))  # 迭代器

model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-poem") # 加载模型

# 分成两个模块，可以在不知道模型的情况下开发数据集

#  pytorch 的数据模块
class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=32):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def setup(self, stage=None):
        # 分配训练/验证数据集，以便在数据加载器中使用
        self.train_dataset, self.val_dataset = self.dataset.split()

    def train_dataloader(self):
        return self.train_dataset.dataloader(
            batch_size=self.batch_size,
            shuffle=True)

    def val_dataloader(self):
        return self.val_dataset.dataloader(
            batch_size=self.batch_size * 2,
            shuffle=False)

#  训练模块
class CausalLMModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch.labels,
        )
        loss = outputs.loss
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch.labels,
        )
        loss = outputs.loss
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)


data_module = DataModule(poet_ds, batch_size=4)

module = CausalLMModule(model)




save = pl.callbacks.ModelCheckpoint(
    '/GCI/transformers/weights/kw_leading_po',
    save_top_k=2,
    verbose=True,
    monitor='val_loss',
    mode='min',
)

trainer = pl.Trainer(
    gpus=[0],
    max_epochs=6,
    callbacks=[save],
)

#  开始训练
torch.cuda.empty_cache()
trainer.fit(module, datamodule=data_module)




module.load_state_dict(
        torch.load(str(save.best), map_location="cpu")['state_dict'])
model = module.model
model = model.cpu()
model = model.eval()
model.save_pretrained("./model")


def inference(lead):
    leading = f"《{lead}》"
    input_ids = tokenizer(leading, return_tensors='pt', ).input_ids
    with torch.no_grad():
        pred = model.generate(
            input_ids,
            max_length=256,
            num_beams=3,
#             do_sample=True,
#             top_p=.6,
            bos_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.sep_token_id,
        )
    print(pred)
    return tokenizer.batch_decode(pred, skip_special_tokens=True)




