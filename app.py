from flask import Flask, render_template, jsonify

import ai

app = Flask(__name__)
app.secret_key = r'4c44309dd5a080a06e7d67c91cd53fa30012e6296e6258ceb06276a8c06c5e01'


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/data')
def data():
    data1, data2, _ = ai.main()
    # Transpose datasets so that we have a list of series
    # instead of a list of y-values
    data1 = list(zip(*data1))
    data2 = list(zip(*data2))
    return jsonify({
        "Layer 1": [
            {"title": "Apical MP 1", "data": data1[0]},
            {"title": "Apical MP 2", "data": data1[1]}
        ],
        "Layer 2": [
            {"title": "Apical MP 1", "data": data2[0]},
            {"title": "Apical MP 2", "data": data2[1]},
            {"title": "Apical MP 3", "data": data2[2]}
        ]
    })


if __name__ == '__main__':
    app.run(debug=True)
