from flask import Flask, render_template, jsonify, request

import ai
from ai.experiments import EXPERIMENTS, KEYS

app = Flask(__name__)
app.secret_key = r'4c44309dd5a080a06e7d67c91cd53fa30012e6296e6258ceb06276a8c06c5e01'


@app.route('/')
def list_experiments():
    """
    Called when the user accesses the root URL of the web app, the Flask app
    routes this request to the "home" function b/c of the route decorator.
    """
    return render_template('list.html',
                           title='Dendritic Microcircuits',
                           experiments=EXPERIMENTS)


@app.route('/experiments/<experiment_name>')
def get_experiment_form(experiment_name: KEYS):
    exp = EXPERIMENTS[experiment_name]
    return render_template(f'experiments/{experiment_name}.html',
                           title=exp.title,
                           description=exp.long_description,
                           key=experiment_name)


@app.route('/data/<experiment_name>')
def get_experiment_data(experiment_name: KEYS):
    return jsonify(ai.main(experiment_name, request.args))


if __name__ == '__main__':
    app.run(debug=True)  # starts the Flask web server.
