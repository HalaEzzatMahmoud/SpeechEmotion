from flask import Flask
from .Routes.Deploy import deploy_bp


def create_app():
    app = Flask(__name__)
    #app.config.from_pyfile('../config.py')
    #db.init_app(app) 


    app.register_blueprint(deploy_bp)
    print("Deploy blueprint registered")  


    return app