from flask import Flask, render_template, jsonify

import ai

app = Flask(__name__)
app.secret_key = r'4c44309dd5a080a06e7d67c91cd53fa30012e6296e6258ceb06276a8c06c5e01'


@app.route('/')
def home():
    """
    Called when the user accesses the root URL of the web app, the Flask app
    routes this request to the "home" function b/c of the route decorator.
    """
    return render_template('index.html')


@app.route('/data')
def data():
    """ Right half """
    data1, data2, data3 = ai.main()
    # Transpose datasets so that we have a list of series
    # instead of a list of y-values
    data1 = list(zip(*data1))
    data2 = list(zip(*data2))
    data3 = list(zip(*data3))
    return jsonify({
        "Layer 1 Apical MPs": [
            {"title": "Apical MP 1", "data": data1[0]},
            {"title": "Apical MP 2", "data": data1[1]}
        ],
        "Layer 2 Apical MPs": [
            {"title": "Apical MP 1", "data": data2[0]},
            {"title": "Apical MP 2", "data": data2[1]},
            {"title": "Apical MP 3", "data": data2[2]}
        ],
        "Learning Rule PP_FF": [
          {"title": "Soma act", "data": data3[0]},
          {"title": "Basal act", "data": data3[1]},
          {"title": "Post value", "data": data3[2]},
          {"title": "Soma mp", "data": data3[3]},
          {"title": "Basal mp", "data": data3[4]},
        ]
    })


if __name__ == '__main__':
    app.run(debug=True)  # starts the Flask web server.
