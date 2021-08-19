from flask import Flask
from flask import request, jsonify, g
app=Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
import pandas as pd
import numpy as np
import tflite_runtime.interpreter as tflite
import sys
import json

market_list=['Asandh',
     'Attabira',
     'Badnawar',
     'Balugaon',
     'Barwala',
     ' Bhadrak',
     'Bhuna',
     'Bhuntar',
     'Chamba',
     'Chamkaur Sahib',
     'Cherthalai',
     'Dhanotu (Mandi)',
     'Dharamkot',
      'Dhenkanal',
     'Dhuri',
     'Digapahandi',
      'Dudhansadhan',
      'Fatehabad',
     'Firozepur City',
     'Garobadha',
     'Godabhaga',
     'Gosala',
     'Gunpur',
     'Gunupur(Maniguda)',
     'Gurdaspur',
     'Hamirpur',
     'Jalandhar City',
     'Kangra',
     ' Kangra(Baijnath)',
     'Kangra(Jassour)',
     'Kangra(Nagrota Bagwan)',
     'Kantabaji',
     'Kathua',
     'Katpadi(Uzhavar Santhai)',
     'Kesinga',
     'Kot ise Khan',
     'Kottarakka',
     'Machhiwara',
     'Malout',
     'Mohindergarh',
     'Nahan',
     'Nakodar',
     'Naraingarh',
     'Narwal Jammu (F&V)',
     'Nawan Shahar(Subzi Mandi)',
     'Nawanshahar',
     'Nobarangpur',
     'Palampur',
     'Paonta Sahib',
     'Pataudi',
     'Pathankot',
     'Phillaur(Apra Mandi)',
     'Quadian',
     'Rajouri (F&V)',
     'Rajpura',
     'Rishikesh',
     'Roorkee',
     'Samalkha',
     'Samana',
     'Samba',
     'Sambalpur',
     'Shahdara',
     'Sirhind',
     'Una',
     'kalanwali']

def predict(date):
  print(date)
  month=date["date"][5:7]
  year= date["date"][:4] 
  predict_input=np.zeros(67)
  predict_input[0]=month
  predict_input[1]=year
  predict_input[market_list.index(date["market"])+2]=1
  predict_input=np.array([predict_input], dtype=np.float32)
  # print(predict_input.shape)
  interpreter = tflite.Interpreter(model_path="converted_model.tflite")
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  # print(input_details)
  # print(output_details)
  interpreter.set_tensor(input_details[0]['index'], predict_input)
  interpreter.invoke()
  output_data = interpreter.get_tensor(output_details[0]['index'])
  print(output_data)
  return output_data[0][0]

@app.route("/",methods=["POST"])
def getPrice():
  List=[]
  query=request.get_json()
  print(query)
  y=predict(query)
  List=[]
  List.append(str(y))
  return json.dumps(List)







