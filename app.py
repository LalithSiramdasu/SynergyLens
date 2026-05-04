from flask import Flask, jsonify, render_template
from werkzeug.exceptions import HTTPException

from backend.config import ensure_directories
from backend.routes.api_routes import api_bp


def create_app():
    """Create the SynergyLens Flask application."""
    ensure_directories()

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
    app.register_blueprint(api_bp)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.errorhandler(HTTPException)
    def handle_http_exception(error):
        message = (
            "Uploaded files must be 10 MB or smaller."
            if error.code == 413
            else error.description
        )
        return jsonify({
            "status": "error",
            "message": message,
            "error": message,
        }), error.code

    @app.errorhandler(Exception)
    def internal_error(error):
        return jsonify({
            "status": "error",
            "message": "An internal server error occurred. Check the Flask logs for details.",
            "error": "An internal server error occurred. Check the Flask logs for details.",
        }), 500

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
