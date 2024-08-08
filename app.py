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
    return jsonify(ai.main())


if __name__ == '__main__':
    app.run(debug=True)  # starts the Flask web server.
