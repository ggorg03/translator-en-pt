from transformers import (MT5Config, 
                          MT5ForConditionalGeneration,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer,
                          DataCollatorForSeq2Seq,
                          MT5Tokenizer)
from datasets import DatasetDict, Dataset
import evaluate
import torch
import numpy as np


class MT5Model:
  METRIC = evaluate.load("sacrebleu")
  PREFIX = "translate English to Portuguese: "

  def __init__(self):
    self.ORGINAL_MODEL = "google/mt5-small"
    self.CONFIG = MT5Config.from_pretrained(self.ORGINAL_MODEL)
    self.TOKENIZER = MT5Tokenizer.from_pretrained(self.ORGINAL_MODEL)
  
    self.model = MT5ForConditionalGeneration.from_pretrained(
                    self.ORGINAL_MODEL,
                    config=self.CONFIG,
                    ignore_mismatched_sizes=True
                )
    self.data_collator = DataCollatorForSeq2Seq(
                          tokenizer=self.TOKENIZER,
                          model=self.ORGINAL_MODEL
                      )
    
    self.tokenized_data = None
    self.trainer = None

  def train(self):
    self.trainer.train()

  def set_to_train(self, data : DatasetDict, epochs:int = 1) -> None:
    self.__build_tokenized_data(data)
    self.__build_trainer(epochs)
  
  def set_to_test(self, data : DatasetDict, checkpoint_path:str = None) -> None:
    self.__build_tokenized_data(data)
    self.__build_test(self, checkpoint_path)
  
  def infer(self, en_text:str) -> str:
    tokenizer=self.TOKENIZER
    en_text = MT5Model.PREFIX + en_text

    inputs = tokenizer(en_text, return_tensors="pt").input_ids
    outputs = self.model.generate(inputs.cuda(), max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)

    pt_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return pt_text
  
  def __build_tokenized_data(self, data: DatasetDict) -> None:
    self.tokenized_data = self.__tokenize_data(data)

  def __build_trainer(self, epochs:int = 1) -> None:
    training_args = Seq2SeqTrainingArguments(
                      output_dir="./dimap-mt5-en-pt_checkpoints",
                      evaluation_strategy="epoch",
                      learning_rate=2e-5,
                      per_device_train_batch_size=16,
                      per_device_eval_batch_size=16,
                      weight_decay=0.01,
                      save_total_limit=3,
                      num_train_epochs=epochs,
                      predict_with_generate=True,
                      fp16=False,
                      push_to_hub=False,
                    )

    self.trainer = Seq2SeqTrainer(
                      model=self.model,
                      args=training_args,
                      train_dataset=self.tokenized_data["train"],
                      eval_dataset=self.tokenized_data["validation"],
                      tokenizer=self.TOKENIZER,
                      data_collator=self.data_collator,
                      compute_metrics=self.__compute_metrics,
                    )
  
  def __build_test(self, checkpoint_path:str = None) -> None:
    checkpoint_path = checkpoint_path | '/content/drive/MyDrive/deeplearning-ufrm-2023/checkpoint-15500'
    # Load the model state from checkpoint
    checkpoint = torch.load(checkpoint_path + '/pytorch_model.bin')
    self.model.load_state_dict(checkpoint)

  def __tokenize_data(self, data: DatasetDict) -> DatasetDict:
    return data.map(self.__preprocess_function, batched=True)

  def __preprocess_function(self, samples: Dataset) -> Dataset:
    source_lang = "en"
    target_lang = "pt"
    prefix = MT5Model.PREFIX

    inputs = [prefix + sample for sample in samples[source_lang]]
    targets = [sample for sample in samples[target_lang]]
    model_inputs = self.TOKENIZER(inputs, text_target=targets, max_length=128, truncation=True)

    return model_inputs

  def __postprocess_text(self, preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

  def __compute_metrics(self, eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = self.TOKENIZER.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, self.TOKENIZER.pad_token_id)
    decoded_labels = self.TOKENIZER.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = self.__postprocess_text(decoded_preds, decoded_labels)

    result = MT5Model.METRIC.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != self.TOKENIZER.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result
