from flask import Flask, render_template, request, jsonify

import ai

app = Flask(__name__)
app.secret_key = r'4c44309dd5a080a06e7d67c91cd53fa30012e6296e6258ceb06276a8c06c5e01'


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        return jsonify(ai.main())
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
