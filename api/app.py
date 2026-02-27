"""Flask application factory."""

import os
import sys
from flask import Flask, send_from_directory
from flask_cors import CORS

# Ensure ml/ is on sys.path for model imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ML_PATH = os.path.join(ROOT, 'ml')
if ML_PATH not in sys.path:
    sys.path.insert(0, ML_PATH)


def create_app(dev: bool = False) -> Flask:
    """Create and configure the Flask application.

    Args:
        dev: When True, enable CORS for the Vite dev server on port 5173.

    Returns:
        Configured Flask application instance.
    """
    app = Flask(
        __name__,
        static_folder=os.path.join(ROOT, 'frontend', 'dist'),
        static_url_path='',
    )

    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
    app.config['UPLOAD_FOLDER'] = os.path.join(ROOT, 'outputs', 'uploads')
    app.config['OUTPUTS_DIR'] = os.path.join(ROOT, 'outputs')
    app.config['ML_PATH'] = ML_PATH
    app.config['DEV_MODE'] = dev

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(ROOT, 'outputs', 'runs'), exist_ok=True)

    if dev:
        CORS(app, resources={r'/api/*': {'origins': 'http://localhost:5173'}})

    # Register blueprints
    from api.routes.training import bp as training_bp
    from api.routes.inference import bp as inference_bp
    from api.routes.metrics import bp as metrics_bp
    from api.routes.models import bp as models_bp
    from api.routes.history import bp as history_bp

    app.register_blueprint(training_bp, url_prefix='/api/training')
    app.register_blueprint(inference_bp, url_prefix='/api/colorize')
    app.register_blueprint(metrics_bp, url_prefix='/api/metrics')
    app.register_blueprint(models_bp, url_prefix='/api/models')
    app.register_blueprint(history_bp, url_prefix='/api/history')

    # Serve Vue SPA for all non-API routes in production
    if not dev:
        @app.route('/', defaults={'path': ''})
        @app.route('/<path:path>')
        def serve_spa(path: str):
            dist = app.static_folder
            target = os.path.join(dist, path)
            if path and os.path.exists(target):
                return send_from_directory(dist, path)
            return send_from_directory(dist, 'index.html')

    return app
