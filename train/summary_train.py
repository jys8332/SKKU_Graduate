from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from transformers import T5TokenizerFast, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import Dataset
# 데이터셋 불러오기
train_data = pd.read_csv('train.csv')
val_data = pd.read_csv('val.csv')
model_name = "paust/pko-t5-large"


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# 데이터셋 클래스 정의
class PassageSummaryDataset(Dataset):
    def __init__(self, data_df, tokenizer):
        self.data = data_df
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        passage = self.data['script'][idx]
        summary = self.data['summary'][idx]
        
        script_inputs = self.tokenizer.encode_plus(
            passage,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        summary_labels = self.tokenizer.encode_plus(
            summary,
            padding="max_length",
            truncation=True,
            max_length=200,
            return_tensors="pt",
        )
        
        return {
            "input_ids": script_inputs["input_ids"].squeeze(),
            "attention_mask": script_inputs["attention_mask"].squeeze(),
            "label_ids": summary_labels["input_ids"].squeeze()
        }

# 모델 초기화
# tokenizer = MT5Tokenizer.from_pretrained(model_name) # google/mt5
tokenizer = T5TokenizerFast.from_pretrained(model_name) # polyglot-ko

# model = MT5ForConditionalGeneration.from_pretrained(model_name) # google/mt5
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.to(DEVICE)

# 데이터셋 전처리 함수 정의
def preprocess_data(csv_file):
    inputs = csv_file['script']
    targets = csv_file['summary']
    return {'input_text': inputs, 'target_text': targets}


# 데이터셋과 데이터로더 초기화
train_dataset = PassageSummaryDataset(train_data, tokenizer)
eval_dataset = PassageSummaryDataset(val_data, tokenizer)


# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
# 훈련 파라미터 설정
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    logging_steps=2,
    output_dir="./results",
    overwrite_output_dir=True,
    warmup_steps=500,
    save_steps=1000,
    save_total_limit=2,
)

# 트레이너 초기화
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 모델 훈련
trainer.train()
# 모델 저장
model.save_pretrained("summary_model.pt")