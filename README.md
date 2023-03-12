# Simple Web Deployment for a Trained Deep Learning Model

This repo contains the code for a simple web deployment for a trained deep learning model for the upcoming Manning book on machine learning with tabular data. This is part of the code for the end-to-end and MLOps chapter.

Here are the key files in this repo:

- [flask_web_deploy.py](https://github.com/ryanmark1867/deep_learning_web_deployment/blob/master/flask_web_deploy.py) - the Flask server module that loads the model specified in the config file [flask_web_deploy_config.yml](https://github.com/ryanmark1867/deep_learning_web_deployment/blob/master/flask_web_deploy_config.yml)  and contains view functions to drive the home.html and show-prediction.html pages
- [flask_web_deploy_config.yml](https://github.com/ryanmark1867/deep_learning_web_deployment/blob/master/flask_web_deploy_config.yml) - contains the parameters to control the action of the Flask server module
- [home.html](https://github.com/ryanmark1867/deep_learning_web_deployment/blob/master/templates/home.html) main web page of the deployment where the user can enter details about the property for which they want to get a price prediction
- [show-prediction.html](https://github.com/ryanmark1867/deep_learning_web_deployment/blob/master/templates/show-prediction.html) web page to show the result of the model's prediction on the property whose details were input in [home.html](https://github.com/ryanmark1867/deep_learning_web_deployment/blob/master/templates/home.html)
- [main2.css](https://github.com/ryanmark1867/deep_learning_web_deployment/blob/master/static/css/main2.css) CSS file to control the rendering of the HTML pages