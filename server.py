

from flask import Flask
from flask import Flask,jsonify,request
from service_streamer import ThreadStreamer
from app.ner.albert_ner import albert_ner

app = Flask(__name__)

@app.route('./predications',methods=["POST"])
def index():
    if request.method == "POST":
        text = request.get_json()
        text = text['texts']
        outputs = app.streamer.predict(text)
        return jsonify(outputs)

if __name__ == "__main__":
    streamer = ThreadStreamer(albert_ner.predict,batch_size=64,max_latency = 0.1)
    app.streamer = streamer
    app.run(host="0.0.0.0",port=5005,debug=True)









