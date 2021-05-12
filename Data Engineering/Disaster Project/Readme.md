
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        python models/train_classifier.py data/DisasterResponse.db classifier.pkl

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to https://classroom.udacity.com/nanodegrees/nd025/parts/3f1cdf90-8133-4822-ba56-934933b6b4bb/modules/fa5fdc5a-7de7-470f-81ba-ed37348de410/lessons/7a929d2c-6da9-49d4-9849-e725b8c6e7a2/concepts/65a088dd-681d-413f-963c-63dc5417da90

http://0.0.0.0:3001/
