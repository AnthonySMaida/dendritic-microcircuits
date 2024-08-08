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
    datasets = ai.main()
    # Transpose datasets so that we have a list of series
    # instead of a list of y-values
    data1, data2, data3, data4 = list(map(lambda dataset: list(zip(*dataset)), datasets))
    return jsonify({
        "Layer 1 Apical MPs": {
            "precision": 2,
            "series": [
                {"title": "Apical MP 1", "data": data1[0]},
                {"title": "Apical MP 2", "data": data1[1]}
            ],
            "xaxis": "Training steps",
            "yaxis": "Membrane potential (mV)"
        },
        "Layer 2 Apical MPs": {
            "precision": 2,
            "series": [
                {"title": "Apical MP 1", "data": data2[0]},
                {"title": "Apical MP 2", "data": data2[1]},
                {"title": "Apical MP 3", "data": data2[2]}
            ],
            "xaxis": "Training steps",
            "yaxis": "Membrane potential (mV)"
        },
        "Learning Rule PP_FF Triggers": {
            "precision": 2,
            "series": [
                {"title": "Soma act", "data": data3[0]},
                {"title": "Basal act", "data": data3[1]},
                {"title": "Post value", "data": data3[2]},
                {"title": "Soma mp", "data": data3[3]},
                {"title": "Basal mp", "data": data3[4]},
            ],
            "xaxis": "Training steps",
            "yaxis": "..."
        },
        "Learning Rule PP_FF wts": {
            "precision": 2,
            "series": [
                {"title": "Weight value", "data": data4[0]},
            ],
            "xaxis": "Training steps",
            "yaxis": "..."
        }
    })


if __name__ == '__main__':
    app.run(debug=True)  # starts the Flask web server.
