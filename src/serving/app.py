# import mlflow and fastapi
import mlflow
from fastapi import FastAPI
import pandas as pd
import uvicorn

run_id = '63d821254d6646508bcfa379865ffd32'

# create a function the takes the run_id and use it to create a full model serving using FastAPI
def create_app(run_id):
    # create the app
    app = FastAPI()
    # use mlflow to load the model
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    # create a prediction endpoint that accepts requests with a JSON body
    @app.post("/predict")
    def predict(data: dict):
        # convert the request body into a dataframe
        data_df = pd.DataFrame(data)
        # get the predictions from the model using the dataframe
        predictions = model.predict(data_df)
        # return the predictions as a JSON response
        return predictions.tolist()
    return app

if __name__ == '__main__':
    # run the app and define the port
    uvicorn.run("test:app", host="0.0.0.0", port = 8000, reload = True)



