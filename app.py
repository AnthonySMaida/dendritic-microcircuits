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
    return jsonify({
        "data1": data1,
        "data2": data2,
        "data3": data3
    })


@app.route('/graphs')
def graphs():
    """ Left half. """
    data1, data2 = ai.main()
    return jsonify({
        "data1": ai.generate_plot(data1),
        "data2": ai.generate_plot(data2)
    })


if __name__ == '__main__':
    app.run(debug=True)  # starts the Flask web server.
