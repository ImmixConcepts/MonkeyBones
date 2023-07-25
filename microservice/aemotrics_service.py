import aemotrics
from flask import Flask, request
from werkzeug.utils import secure_filename
import os
import queue
import threading

app = Flask(__name__)
ALLOWED_EXTENSIONS = {"m4v", "mp4", "avi"}
upload_dir = "/aemotrics/videos"
SECRET_KEY = os.urandom(14).hex()
app.config["SECRET_KEY"] = SECRET_KEY
config = os.path.split(os.path.realpath(__file__))[0] + "/Aemotrics_V3-Nate-2021-12-20_pruned"
predictQueue = queue.Queue()


def _worker():
    while True:
        item = predictQueue.get()
        aemotrics.analysis.analyze(item, dlc_model_path=config)
        # upload file to storage
        predictQueue.task_done()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["POST"])
def post_analyze():
    results = []
    for key, file in request.files.items():
        app.logger.info(f"file = {file.filename}")
        if file.filename == "":
            results.append({"success": False, "error": "uploaded video has no name"})
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            f_path = os.path.join(upload_dir, filename)
            file.save(f_path)
            app.logger.info("running analysis on " + f_path)
            if os.path.isfile(f_path):
                results.append({"success": True, "file_name": file.filename})
    return {"results": results}


if __name__ == "__main__":
    if not os.path.isfile(config + "/config.yaml"):
        raise AttributeError(f"config does not exit at {config}/config.yaml")
    if not os.path.isdir(upload_dir):
        os.mkdir(upload_dir)
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    app.secret_key = SECRET_KEY
    app.run(host="0.0.0.0")
