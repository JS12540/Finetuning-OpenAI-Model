import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],
)

training_file_name = "training_data.jsonl"
validation_file_name = "validation_data.jsonl"

#  we upload the two datasets to the OpenAI developer account as follows:

training_file_id = client.files.create(
  file=open(training_file_name, "rb"),
  purpose="fine-tune"
)

validation_file_id = client.files.create(
  file=open(validation_file_name, "rb"),
  purpose="fine-tune"
)

print(f"Training File ID: {training_file_id}")
print(f"Validation File ID: {validation_file_id}")

'''
Output:

Training File ID: FileObject(id='file-04AeP4VWLNsYvfi4C9YovxLW', bytes=1320, created_at=1717313611, filename='training_data.jsonl', object='file', purpose='fine-tune', status='processed', status_details=None)
Validation File ID: FileObject(id='file-b86stzPjli3eUx4gVCKTUdIl', bytes=1044, created_at=1717313612, filename='validation_data.jsonl', object='file', purpose='fine-tune', status='processed', status_details=None)
'''

response = client.fine_tuning.jobs.create(
  training_file=training_file_id.id, 
  validation_file=validation_file_id.id,
  model="davinci-002", 
  hyperparameters={
    "n_epochs": 15,
	"batch_size": 3,
	"learning_rate_multiplier": 0.3
  }
)
job_id = response.id
status = response.status

print(f'Fine-tunning model with jobID: {job_id}.')
print(f"Training Response: {response}")
print(f"Training Status: {status}")

'''
Output:

Training File ID: FileObject(id='file-9kwkK9MlEu1E8z3bdlKIR7ar', bytes=1320, created_at=1717315092, filename='training_data.jsonl', object='file', purpose='fine-tune', status='processed', status_details=None)
Validation File ID: FileObject(id='file-wOa4CvLuP2KODqIDMY5SUK0g', bytes=1044, created_at=1717315093, filename='validation_data.jsonl', object='file', purpose='fine-tune', status='processed', status_details=None)

Fine-tunning model with jobID: ftjob-dNNxzHuTVTtmJzdy96n9GBmu.

Training Response: FineTuningJob(id='ftjob-dNNxzHuTVTtmJzdy96n9GBmu', created_at=1717315096, error=Error(code=None, message=None, param=None), fine_tuned_model=None, finished_at=None, hyperparameters=Hyperparameters(n_epochs=15, batch_size=3, learning_rate_multiplier=0.3), model='davinci-002', object='fine_tuning.job', organization_id='org-I13i7JRLQxQRg7pyMGFJe3nL', result_files=[], seed=718968671, status='validating_files', trained_tokens=None, training_file='file-9kwkK9MlEu1E8z3bdlKIR7ar', validation_file='file-wOa4CvLuP2KODqIDMY5SUK0g', estimated_finish=None, integrations=[], user_provided_suffix=None)    

Training Status: validating_files
'''