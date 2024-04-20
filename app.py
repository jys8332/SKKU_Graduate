from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import whisper
import openai
import os
import time
# # -*- coding: utf-8 -*-
from sentence_transformers import SentenceTransformer, util
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import asyncio
from tqdm import tqdm
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from hanspell import spell_checker

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



print(os.getcwd())
# # model = whisper.load_model("base")
# model = whisper.load_model("large-v2")

# sum_model_name="traintogpb/pko-t5-large-kor-for-colloquial-summarization-finetuned"
# sum_model = AutoModelForSeq2SeqLM.from_pretrained(sum_model_name)
# sum_model.to(device)
# tokenizer = AutoTokenizer.from_pretrained(sum_model_name)
# par_model = SentenceTransformer("jhgan/ko-sroberta-multitask")



# gen_model_name = "traintogpb/mt5-large-kor-qa-generation-finetuned"
# tokenizer = MT5Tokenizer.from_pretrained(gen_model_name)
# gen_model = MT5ForConditionalGeneration.from_pretrained(gen_model_name)
# gen_model.to(device)





def generate_qna(summary_list, tokenizer, model, device, n_beams):
    question_list= []
    answer_list = []
    for summary in summary_list:
      input_ids = tokenizer.encode_plus(
          summary,
          padding="max_length",
          truncation=True,
          max_length=512,
          return_tensors="pt"
      ).input_ids.to(device)

      generated_ids = model.generate(
          input_ids,
          num_beams=n_beams,
          max_length=128,
          early_stopping=True,
          num_return_sequences=n_beams,
      )

      generated_qnas = []
      for generated_id in generated_ids:
          qa = tokenizer.decode(generated_id, skip_special_tokens=True)

          generated_qnas.append(qa)
      if len(generated_qnas)>0:
        if len(generated_qnas[0].split("?")) >1:
          question =generated_qnas[0].split("?")[0]
          answer = generated_qnas[0].split("?")[1]
          question_list.append(question+"?")
          answer_list.append(answer)


    file_path = "qna.txt"
    with open(file_path, "w", encoding='utf-8') as file:
      for idx in range(len(question_list)):
        file.write(f"Question {idx+1} : {question_list[idx]}\n")
        file.write(f"Answer {idx+1} : {answer_list[idx]}\n")
    return question_list, answer_list


def make_paragraph(text, model, min_score):
    print("text is ", text)
    sen_list = text.split(".")
    cos_score_list =[]
    for idx in range(1, len(sen_list)):
      prev_sen = sen_list[idx-1]
      now_sen = sen_list[idx]
      prev_embeddings = model.encode(prev_sen, convert_to_tensor=True)
      now_embedding = model.encode(now_sen, convert_to_tensor=True)
      cos_scores = util.pytorch_cos_sim(prev_embeddings, now_embedding)[0]
      cos_scores = cos_scores.cpu()
      cos_score_list.append(cos_scores[0])
    par_count = 0
    temp_par_list = {0:[sen_list[0]]}
    for idx in range(len(cos_score_list)):
      if cos_score_list[idx]>min_score :
        temp_par_list[par_count].append(sen_list[idx+1])
      else:
        par_count+=1
        temp_par_list[par_count] = [sen_list[idx+1]]
    par_list =[]
    for num in range(par_count+1):
        if(len(temp_par_list[num][0])>0):
          par_list.append(".".join(temp_par_list[num]))
    return par_list

def make_summary(par_list, model, tokenizer):
    sum_list = []
    for paragraph in par_list:
      input_ids = tokenizer.encode(paragraph, return_tensors="pt")
      summary_tokens = model.generate(
          input_ids=input_ids,
          max_length=400,
          no_repeat_ngram_size=2,
          num_beams=4
      )
      summary_text = tokenizer.decode(summary_tokens[0], skip_special_tokens=True)
      if len(summary_text) >0:
        sum_list.append(summary_text)


    return sum_list

def STT(model, sound_file):
  print("STT Start!")
  result = model.transcribe(sound_file)
  file_path = "STT.txt"
  with open(file_path, "w", encoding='utf-8') as file:
    file.write(result["text"])
  print("STT Finish!")
  return result["text"]

def summarize(model, stt_result, tokenizer):
    print("Start make paragraph!")
    print(stt_result)
    min_score = 0.3
    par_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    paragraph_list = make_paragraph(stt_result,par_model, min_score)
    print("Finish make paragraph!")

    print("Start Summarize!")
    summary_list = make_summary(paragraph_list, model, tokenizer)
    print("Finish Summarize!")

    file_path = "summary.txt"
    with open(file_path, "w", encoding='utf-8') as file:
      file.write("-".join(summary_list))

    return summary_list

def generation(model, summary_list, tokenizer):
    print("Start Generation!")
    question_list, answer_list = generate_qna(summary_list,tokenizer, model, device,  10 )
    print("Finish Generation!")
    return question_list, answer_list
app = Flask(__name__)


@app.route('/')

def upload_mp3():

    return render_template('upload.html')



@app.route('/lecture' , methods=['GET', 'POST'])

def script():
    if request.method=='POST':
        if os.path.isfile("STT.txt"):

          file = open("STT.txt", 'r', encoding="utf-8")    
          STT_text = file.read()

          return render_template('lecture.html', script=STT_text)
        else:
          openai.api_key = 'YOUR_API_KEY'
          # model = whisper.load_model("base")
          model = whisper.load_model("large-v2")
          f = request.files['record_file']
          f.save(secure_filename("sound_file.mp3"))
          STT_text = STT(model, "sound_file.mp3")
          return render_template('lecture.html', script=STT_text)
    return render_template('lecture.html', script="Upload Failed")


@app.route('/summary' , methods=['GET', 'POST'])

def summary():
    
    # if request.method=='POST':
        file_path = "summary.txt"
        if os.path.isfile(file_path):
            file = open(file_path, 'r', encoding="utf-8")    
            text = file.read()   
            sum_list =text.split("-")
        else:
            sum_model_name="traintogpb/pko-t5-large-kor-for-colloquial-summarization-finetuned"
            sum_model = AutoModelForSeq2SeqLM.from_pretrained(sum_model_name)
            sum_model.to(device)
            tokenizer = AutoTokenizer.from_pretrained(sum_model_name)
            
            file_path = "STT.txt"
            file = open(file_path, 'r', encoding="utf-8")    # hello.txt 占쎈솁占쎌뵬占쎌뱽 占쎌뵭疫뀐옙 筌뤴뫀諭�(r)嚥∽옙 占쎈였疫뀐옙. 占쎈솁占쎌뵬 揶쏆빘猿� 獄쏆꼹�넎
            stt_result = file.read()     
            sum_list = summarize(sum_model, stt_result, tokenizer)
        sum_result="\n".join(sum_list)
        return render_template('summary.html', summary=sum_result)
    # return render_template('summary.html', summary="Summary Failed")

@app.route('/qna' , methods=['GET', 'POST'])

def qna():
    # if request.method=='POST':
        file_path = "qna.txt"
        if os.path.isfile(file_path):
            file = open(file_path, 'r', encoding="utf-8")    
            qna_result = file.read()  
        else:
            gen_model_name = "traintogpb/mt5-large-kor-qa-generation-finetuned"
            tokenizer = MT5Tokenizer.from_pretrained(gen_model_name)
            gen_model = MT5ForConditionalGeneration.from_pretrained(gen_model_name)
            gen_model.to(device)

            file_path = "summary.txt"
            file = open(file_path, 'r', encoding="utf-8")    
            text = file.read()   
            summary_list =text.split("-")
            question_list, answer_list = generation(gen_model, summary_list, tokenizer)
            qna_result = ""
            for idx in range(len(question_list)):
                qna_result +=f"Question {idx+1} : {question_list[idx]}\n"
                qna_result +=f"Answer {idx+1} : {answer_list[idx]}\n"
        return render_template('qna.html', qna=qna_result)
    # return render_template('qna.html', qna="QNA Generation Failed")




if __name__ == '__main__':

    app.run(debug=True)