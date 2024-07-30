from flask import Flask, render_template, jsonify

import ai

app = Flask(__name__)
app.secret_key = r'4c44309dd5a080a06e7d67c91cd53fa30012e6296e6258ceb06276a8c06c5e01'


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/data')
def data():
    data1, data2 = ai.main()
    return jsonify({
        "data1": data1,
        "data2": data2
    })


@app.route('/graphs')
def graphs():
    data1, data2 = ai.main()
    return jsonify({
        "data1": ai.generate_plot(data1),
        "data2": ai.generate_plot(data2)
    })


if __name__ == '__main__':
    app.run(debug=True)
