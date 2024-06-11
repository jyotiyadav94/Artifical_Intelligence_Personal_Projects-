import os
import numpy as np
from prediction_service import prediction
from flask import Flask, render_template, request, jsonify

webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if request.form:
                dict_req = dict(request.form)
                print('dict_req',dict_req)
                response = prediction.form_response(dict_req)
                print(response)
                return render_template("index.html", response=response)
            elif request.json:
                response = prediction.api_response(request.json)
                return jsonify(response)

        except Exception as e:
            print(e)
            error = str(e)  # Assigning the exception message directly
            return render_template("error.html", error=error)  # Using an appropriate error template
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)